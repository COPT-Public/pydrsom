"""
- implement double backward: https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
"""
import os
from collections import deque
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector

RSOM_VERBOSE = int(os.environ.get('RSOM_VERBOSE', 0))


def _norm(alpha, tr):
  return (tr @ alpha).dot(alpha).sqrt()


def _compute_root(Q, c, delta, tr=torch.eye(2)):
  eps = 1e-2
  lsolve = torch.linalg.solve
  # solve an unconstrained quadratic function,
  #   if it satisfies the requirements,
  #   Lagrangian dual is 0
  alpha = lsolve(Q, -c)
  norm = _norm(alpha, tr)
  if norm < delta:
    return 0, 0, alpha, norm, False
  lmin = torch.linalg.eigvalsh(Q)[0]
  lb, ub = max(0, -lmin.item()), 10
  it = 0
  while True:
    lmb = (lb + ub) / 2
    alpha = lsolve(Q + tr * lmb, -c)
    norm = _norm(alpha, tr)
    if abs(norm - delta) < eps or it > 10:
      break
    if norm > delta + eps:
      lb = lmb
    else:
      ub = lmb
    it += 1
  return it, lmb, alpha, norm, True


class RSOM(torch.optim.Optimizer):
  """
  Implements the RSOM algorithm
  """
  
  def __init__(self, params, max_iter=1, option_tr='a', hessian_window=100, betas=(0.99, 0.999), eps=1e-8, delta=5.0):
    
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    
    defaults = dict(betas=betas, eps=eps)
    super(RSOM, self).__init__(params, defaults)
    
    self._params = self.get_params()
    for p in self._params:
      # keep momentum
      self.state[p]['momentum'] = torch.zeros_like(p.data, requires_grad=True)
    
    #
    self._numel_cache = None
    ##########################
    # RSOM only params
    ##########################
    self.max_iter = max_iter
    self.betas = betas
    self.option_tr = option_tr
    self.hessian_window = hessian_window
    
    ##########################
    # global averages & keepers
    ##########################
    
    self.Q: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    self.c: Optional[torch.Tensor] = torch.zeros(2, requires_grad=False)
    self.Qa = deque(maxlen=hessian_window)
    self.ca = deque(maxlen=hessian_window)
    if option_tr == 'p':
      self.G: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
      self.Ga = deque(maxlen=hessian_window)
    else:
      self.G = None
      self.Ga = None
    
    ##########################
    # scalar attrs
    ##########################
    # total number of runs acc. all steps
    self.iter = 0
    self.alpha: Optional[torch.TensorType] = None
    self.alpha_norm = 0.0
    self.delta = delta
    self.delta_max = 1e4
    self.active = False
    self.eta = 0.05
    self.logline = None
  
  def get_params(self):
    """
    gets all parameters in all param_groups with gradients requirements
    """
    return [p for group in self.param_groups for p in group['params'] if p.requires_grad]
  
  def _clone_param(self):
    return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
  
  @torch.no_grad()
  def _set_param(self, params_data):
    
    for p, pdata in zip(self._params, params_data):
      p.copy_(pdata)
  
  def _bool_grad_vanish(self, p):
    return p.grad is None or torch.linalg.norm(p.grad) < 1e-8
  
  @torch.no_grad()
  def _clear_momentum(self):
    # only has globally state
    for p in self._params:
      self.state[p]['momentum'].zero_()
      break
  
  def _apply_step(self, flat_p, flat_d):
    with torch.no_grad():
      offset = 0
      for p in self._params:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.copy_(flat_p[offset:offset + numel].view_as(p))
        self.state[p]['momentum'].copy_(flat_d[offset:offset + numel].view_as(p))
        offset += numel
      assert offset == self._numel()
  
  def _directional_evaluate(self, closure, flat_p, flat_d):
    self._apply_step(flat_p, flat_d)
    # evaluation
    loss = float(closure())
    return loss
  
  def _numel(self):
    if self._numel_cache is None:
      self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
    return self._numel_cache
  
  def _gather_flat_grad(self, _valid_params, target='self'):
    if target == 'grad':
      flat = torch.concat([p.grad.reshape(-1) for p in _valid_params])
    elif target == 'momt':
      flat = torch.concat([self.state[p]['momentum'].reshape(-1) for p in _valid_params])
    else:
      flat = torch.concat([p.reshape(-1) for p in _valid_params])
    
    return flat
  
  @torch.no_grad()
  def solve_alpha(self, Q, c, delta, tr=torch.eye(2)):
    # initialization
    it = 1
    active = True
    if self.iter == 0:
      lmd = 0
      alpha = torch.tensor([-c[0] / Q[0, 0], 0])
      norm = _norm(alpha, tr)
      if norm > delta:
        alpha = alpha / alpha.norm() * delta
      else:
        active = False
    else:
      # lmd = 1e-2
      # apply root-finding
      it, lmd, alpha, norm, active = _compute_root(Q, c, delta, tr)
    
    if RSOM_VERBOSE:
      self.logline = {
        'lambda': '{:+.2e}'.format(np.round(lmd, 2)),
        'c': np.round(c.tolist(), 2).__str__(),
        'Q': np.round(self.Q.tolist(), 2).__str__(),
        'alpha': np.round(alpha.tolist(), 2).__str__(),
        'root_it': it
      }
    return alpha, norm, active
  
  def est_dec(self):
    trs_est = - 1 / 2 * (self.Q @ self.alpha).dot(self.alpha) - self.c.dot(self.alpha)
    return trs_est
  
  def compute_step(self, option_tr='p'):
    # compute alpha
    if option_tr == 'a':
      self.alpha, self.alpha_norm, *_ = self.solve_alpha(
        self.Q, self.c, self.delta
      )
    elif option_tr == 'p':
      self.alpha, self.alpha_norm, *_ = self.solve_alpha(
        self.Q, self.c, self.delta, tr=self.G
      )
    else:
      raise ValueError(f"unknown option for trust-region option: {option_tr}")
    
    ####################################
    # compute estimate decrease
    ####################################
    trs_est = - 1 / 2 * (self.Q @ self.alpha).dot(self.alpha) - self.c.dot(self.alpha)
    
    return trs_est
  
  def update_trust_region(self, flat_p, flat_g, flat_d, option_tr='a'):
    with torch.enable_grad():
      __unused = flat_p
      gg = flat_g.dot(flat_g)
      gd = flat_d.dot(flat_g)
      
      Hg = self._gather_flat_grad(torch.autograd.grad(
        gg / 2, self._params,
        create_graph=True
      ), target='self')
      Hd = self._gather_flat_grad(torch.autograd.grad(
        gd, self._params,
        create_graph=True
      ), target='self')
      
      gHg = flat_g.dot(Hg)
      gHd = flat_g.dot(Hd)
      dHd = flat_d.dot(Hd)
      
      dd = flat_d.dot(flat_d)
      Q = torch.tensor([[gHg, -gHd], [-gHd, dHd]], requires_grad=False)
      c = torch.tensor([-gg, gd], requires_grad=False)
      
      # self.Q = Q
      # self.c = c
      
      beta1, beta2 = self.betas
      self.Qa.appendleft(Q)
      self.ca.appendleft(c)
      _total = len(self.Qa)
      b = torch.tensor([beta1 ** (k + 1) for k in range(len((self.Qa)))])
      b = b / b.sum()
      self.Q = sum(_Q * b[k] for k, _Q in enumerate(self.Qa))
      self.c = sum(_c * b[k] for k, _c in enumerate(self.ca))
      G = torch.tensor([[gg, -gd], [-gd, dd]])
      self.Ga.append(G)
      self.G = sum(_G * b[k] for k, _G in enumerate(self.Ga))
  
  def step(self, closure=None, restart_delta=False):
    """
    Performs a single optimization step.
    Arguments:
        closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
    """
    
    if closure is None:
      raise ValueError("must provide a closure for RSOM")
    closure = torch.enable_grad()(closure)
    if RSOM_VERBOSE:
      torch.autograd.set_detect_anomaly(True)
    n_iter = 0
    if restart_delta:
      self.delta = 5.0
    loss = closure()
    state = self.state[self._params[0]]
    for p in self._params:
      p.vanish = self._bool_grad_vanish(p)
    # _valid_params = [p for p in self._params if not p.vanish]
    flat_g = parameters_to_vector([p.grad for p in self._params])
    flat_p = parameters_to_vector(self._params)
    flat_d = self._gather_flat_grad([self.state[p]['momentum'] for p in self._params])
    self.update_trust_region(flat_p, flat_g, flat_d, option_tr=self.option_tr)
    
    while n_iter < self.max_iter:
      # solve alpha
      # trs_est = self.compute_step(option_tr=self.option_tr)
      trs_est = self.compute_step(option_tr='a')
      alpha1, alpha2 = self.alpha
      
      # build direction
      # p_copy = self._clone_param()
      flat_new_d = torch.zeros_like(flat_d, requires_grad=False)
      flat_new_d.add_(flat_g, alpha=-alpha1).add_(flat_d, alpha=alpha2)
      flat_p.add_(flat_new_d)
      
      # accept or not？
      loss_est = self._directional_evaluate(closure, flat_p, flat_new_d)
      loss_dec = loss - loss_est
      rho = loss_dec / trs_est
      
      # update the trust-region radius
      delta_inc = 0
      delta_old = self.delta
      if rho <= 0.25:
        self.delta *= 0.25
      else:
        if rho >= 0.75 and self.active:
          delta_inc = 1
          self.delta = min(2 * self.delta, self.delta_max)
      
      acc_step = rho > self.eta
      if not acc_step:
        # set back to old ~ trial step failed
        self._set_param(flat_p)
        self._clear_momentum()
      
      if RSOM_VERBOSE:
        self.logline['trs_dec'] = '{:+.2e}'.format(trs_est.item())
        self.logline['rho'] = '{:+.2e}'.format(rho.item())
        self.logline['acc_step'] = acc_step.item()
        self.logline['delta_inc'] = delta_inc
        self.logline['delta'] = '{:+.2e}'.format(self.delta)
        self.logline['delta_old'] = '{:+.2e}'.format(delta_old)
        self.logline['loss'] = '{:+.2e}'.format(loss.item())
        self.logline['loss_dec'] = '{:+.2e}'.format(loss_dec.item())
        self.logline['iter_inner'] = n_iter
        self.logline['iter_total'] = self.iter
        print(pd.DataFrame(data=[list(self.logline.values())], columns=self.logline.keys(), dtype=str))
      
      # loss = closure()
      self.iter += 1
      n_iter += 1
      if self.delta < 1e-2 or abs(loss_dec) < 1e-6:
        break
    return loss
