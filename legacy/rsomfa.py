"""
trust-region radius free rsom
- implement double backward: https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
"""
import os
from functools import reduce
from pprint import pprint
from typing import Optional
from collections import deque
import numpy
import pandas as pd
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
RSOM_VERBOSE = int(os.environ.get('RSOM_VERBOSE', 0))


def _norm(alpha, tr):
  return (tr @ alpha).dot(alpha).sqrt()


def _compute_root(Q, c, lmb, tr=torch.eye(2)):
  lsolve = torch.linalg.solve
  
  D, V = torch.linalg.eigh(Q)
  
  lmin, lmax = D
  lb = max(0, -lmin.item())
  lmax = lmax.item() if lmax > lb else lb + 1e4
  _lmb_this = lmb * lmax + max(1 - lmb, 0) * lb
  it = 0
  try:
    alpha = lsolve(Q + tr * _lmb_this, -c)
  except torch.linalg.LinAlgError as e:
    print(e)
    print(Q, tr, _lmb_this, -c)
    # todo, can do better
    alpha = lsolve(Q + tr * (_lmb_this + 1e-4), -c)
  
  norm = _norm(alpha, tr)
  
  return it, _lmb_this, alpha, norm, True


class RSOMF(torch.optim.Optimizer):
  """
  Implements the RSOM algorithm
  """
  
  def __init__(self, params, max_iter=1, option_tr='a', lmb=1e-6, theta1=5e1, theta2=30, hessian_window=100,
               betas=(0.99, 0.999), eps=1e-8):
    
    defaults = dict(betas=betas, eps=eps)
    super(RSOMF, self).__init__(params, defaults)
    
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
    self._max_iter_adj = 15
    self.option_tr = option_tr
    self.hessian_window = hessian_window
    
    ##########################
    # global averages & keepers
    ##########################
    
    self.Q: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    self.c: Optional[torch.Tensor] = torch.zeros(2, requires_grad=False)
    self.Qa = deque(maxlen=hessian_window)
    self.ca = deque(maxlen=hessian_window)
    
    self.G: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    self.Ga = deque(maxlen=hessian_window)
    
    ##########################
    # scalar attrs
    ##########################
    # total number of runs acc. all steps
    self.iter = 0
    self.alpha: Optional[torch.TensorType] = None
    self.alpha_norm = 0.0
    self.lmb = lmb
    self.delta_max = 1e1  # maximum step
    # self.active = False
    self.eta = 0.08
    self.betas = betas
    self.theta1 = theta1
    self.theta2 = theta2
    self.logline = None
    # other indicators
    self.ghg = 0.0

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
    loss = float(closure(backward=False))
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
  def solve_alpha(self, Q, c, tr=torch.eye(2)):
    # initialization
    it = 1
    if self.iter == 0 or self.Q[1, 1] < 1e-4:
      lmd = 0.0
      alpha = torch.tensor([-c[0] / Q[0, 0] / (1 + self.lmb), 0]) if Q[0, 0] > 0 else torch.tensor([1e-4, 0])
      norm = _norm(alpha, tr)
      if norm > self.delta_max:
        alpha = alpha / alpha.norm() * self.delta_max
    else:
      # lmd = 1e-2
      # apply root-finding
      it, lmd, alpha, norm, active = _compute_root(Q, c, self.lmb, tr)
    
    if RSOM_VERBOSE:
      self.logline = {
        '𝜆': '{:+.2e}'.format(lmd),
        'Q/c/G': np.round(np.vstack([self.Q, self.c, self.G]), 3),
        'a': np.round(alpha.tolist(), 3).reshape((2, 1)),
        'ghg': '{:+.2e}'.format(Q[0, 0]),
        'ghg-': '{:+.2e}'.format(self.ghg),
        # 'root_it': it
      }
    return alpha, norm
  

  def compute_step(self, option_tr='p'):
    # compute alpha
    if option_tr == 'a':
      self.alpha, self.alpha_norm = self.solve_alpha(
        self.Q, self.c,
      )
    elif option_tr == 'p':
      self.alpha, self.alpha_norm = self.solve_alpha(
        self.Q, self.c, tr=self.G
      )
    else:
      raise ValueError(f"unknown option for trust-region option: {option_tr}")
    
    ####################################
    # compute estimate decrease
    ####################################
    trs_est = - 1 / 2 * (self.Q @ self.alpha).dot(self.alpha) - self.c.dot(self.alpha)
    
    return trs_est

  def update_trust_region(self, flat_p, flat_g, flat_d, flat_g_eps, flat_d_eps, scale):
    with torch.enable_grad():
      __unused = flat_p
      gg = flat_g.dot(flat_g)
      gd = flat_d.dot(flat_g)
      ########################
      # comply with julia
      # x = state.x
      # _, b = Zygote.pullback(f, x)
      # g, = b(1)
      # _, b1 = Zygote.pullback(f, x + g ./ scale)
      # g1, = b1(1)
      # Hg = scale * (g1 - g)
      # Hd = g - state.∇fz
      dd = flat_d.dot(flat_d)
      Hg = scale * (flat_g_eps - flat_g)
      Hd = scale * (flat_d_eps - flat_g)
      gHg = flat_g.dot(Hg)
      gHd = flat_g.dot(Hd)
      dHd = flat_d.dot(Hd)

      # if False:
      #   Hg = self._gather_flat_grad(torch.autograd.grad(
      #     gg / 2, self._params,
      #     create_graph=True
      #   ), target='self')
      #   Hd = self._gather_flat_grad(torch.autograd.grad(
      #     gd, self._params,
      #     create_graph=True
      #   ), target='self')
      #
      #   gHg = flat_g.dot(Hg)
      #   gHd = flat_g.dot(Hd)
      #   dHd = flat_d.dot(Hd)

      Q = torch.tensor([[gHg, -gHd], [-gHd, dHd]], requires_grad=False)
      c = torch.tensor([-gg, gd], requires_grad=False)

      beta1, beta2 = self.betas
      self.Qa.appendleft(Q)
      self.ca.appendleft(c)
      _total = len(self.Qa)
      b = torch.tensor([beta1 ** (k + 1) for k in range(len((self.Qa)))])
      b = b / b.sum()
      self.Q = sum(_Q * b[k] for k, _Q in enumerate(self.Qa))
      self.c = sum(_c * b[k] for k, _c in enumerate(self.ca))
      # use generalized a'Ga <= delta
      G = torch.tensor([[gg, -gd], [-gd, dd]])
      self.Ga.append(G)
      self.G = sum(_G * b[k] for k, _G in enumerate(self.Ga))

  def step(self, closure=None, scale=1e3):
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
    
    loss = closure()
    flat_g = parameters_to_vector([p.grad for p in self._params])
    flat_p = parameters_to_vector(self._params)
    flat_d = self._gather_flat_grad([self.state[p]['momentum'] for p in self._params])
    # copy of it at last step
    p_copy = self._clone_param()

    ############################
    # trial steps
    ############################
    # compute g(x + g*eps)
    with torch.no_grad():
      for p in self._params:
        p.add_(p.grad, alpha=1 / scale)
      _ = closure()
    flat_g_eps = parameters_to_vector([p.grad for p in self._params])
    # revert to original
    self._set_param(p_copy)
    
    # compute g(x + d*eps)
    with torch.no_grad():
      for p in self._params:
        p.add_(self.state[p]['momentum'], alpha=1 / scale)
      _ = closure()
    flat_d_eps = parameters_to_vector([p.grad for p in self._params])
    
    # revert to original
    self._set_param(p_copy)
    ############################
    loss = closure()
    self.update_trust_region(flat_p, flat_g, flat_d, flat_g_eps, flat_d_eps, scale=scale)
    
    # adjust lambda: (and thus trust region radius)
    iter_adj = 1
    while iter_adj < self._max_iter_adj:
      
      # solve alpha
      trs_est = self.compute_step(option_tr=self.option_tr)
      if trs_est < 0:
        self.lmb = max(self.lmb * self.theta1, 1e-4)
        if RSOM_VERBOSE:
          pprint(self.logline)
        continue
      alpha1, alpha2 = self.alpha
      
      # build direction
      flat_new_d = torch.zeros_like(flat_d, requires_grad=False)
      flat_new_d.add_(flat_g, alpha=-alpha1).add_(flat_d, alpha=alpha2)
      flat_p.add_(flat_new_d)
      
      # accept or not？
      loss_est = self._directional_evaluate(closure, flat_p, flat_new_d)
      loss_dec = loss - loss_est
      rho = loss_dec / trs_est
      
      # update the trust-region radius (implicitly by lmb)
      lmb_dec = 0
      lmb_old = self.lmb
      if rho <= 0.25:
        self.lmb = max(self.lmb * self.theta1, 1e-4)
      else:
        if rho >= 0.75:
          lmb_dec = 1
          self.lmb = max(1e-12, min(self.lmb / self.theta2, np.log(self.lmb)))
      
      acc_step = rho > self.eta
      if RSOM_VERBOSE:
        self.logline['dQ'] = '{:+.2e}'.format(trs_est.item())
        self.logline['df'] = '{:+.2e}'.format(loss_dec.item())
        self.logline['rho'] = '{:+.2e}'.format(rho.item())
        self.logline['acc'] = int(acc_step.item())
        self.logline['acc-𝜆'] = lmb_dec
        self.logline['𝛄'] = '{:+.2e}'.format(self.lmb)
        self.logline['𝛄-'] = '{:+.2e}'.format(lmb_old)
        self.logline['f'] = '{:+.2e}'.format(loss.item())
        self.logline['k'] = '{:+6d}'.format(self.iter)
        self.logline['k0'] = iter_adj
        print(pd.DataFrame(data=[list(self.logline.values())], columns=self.logline.keys(), dtype=str).to_markdown(
          tablefmt="grid"))
      if not acc_step:
        # set back to old ~ trial step failed
        self._set_param(p_copy)
        
      else:
        break
      
      iter_adj += 1
    
    self.iter += 1
    n_iter += 1
    
    if not acc_step:
      # restart
      self._clear_momentum()
      self.lmb = 1e-6
      
    return loss
