"""
radius free (dimension-reduced trust-region method) DRSOM
@author: Chuwen Zhang<chuwzhang@gmail.com>, Yinyu Ye<yinyu-ye@stanford.edu>
@note:
  This is a vanilla implementation of (Mini-batch, Radius-Free) DRSOM.
  - the options to run DRSOM can be controlled by the environment variables, see `drsom_utils.py`
  - we treat the parameters as a set of tensors
"""
from functools import reduce
from pprint import pprint
from typing import Optional
import torch
import numpy as np
import pandas as pd

from torch.nn.utils import parameters_to_vector

from .drsom_utils import *


class DRSOMB(torch.optim.Optimizer):
  
  def __init__(
      self,
      model,
      max_iter=15,
      option_tr='a',
      gamma=1e-6,
      beta1=5e1,
      beta2=3e1,
      hessian_window=1,
      thetas=(0.99, 0.999), eps=1e-8
  ):
    """
    The DRSOM:
      Implementation of (Mini-batch) DRSOM (Dimension-Reduced Second-Order Method) in F (Radius-Free) style
    Args:
      model: model params
      max_iter: # of iteration for trust-region adjustment
      option_tr: option of trust-region, I or G?
               - if 'a'; G = eye(2)
               - if 'p'; G = [-g d]'[-g d]
      gamma: lower bound for gamma
      beta1: gamma + multiplier
      beta2: gamma - multiplier
      hessian_window: window size to keep last k hessian information
      thetas: weight decay params (like betas for Adam)
      eps: ...
    """
    
    defaults = dict(betas=thetas, eps=eps)
    params = model.parameters()
    super(DRSOMB, self).__init__(params, defaults)
    ##########################
    # AD hvps
    ##########################
    self.Hv = None
    self._params = self.get_params()
    for p in self._params:
      # keep momentum
      for k in DRSOM_DIRECTIONS:
        self.state[p][k] = torch.zeros_like(p.data, requires_grad=True)
    
    #
    self._numel_cache = None
    ##########################
    # DRSOM only params
    ##########################
    # frequency to update Hv (Hessian-vector product)
    self.freq = 1
    self._max_iter_adj = max_iter
    self.option_tr = option_tr
    
    ##########################
    # global averages & keepers
    ##########################
    self.Q: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    self.c: Optional[torch.Tensor] = torch.zeros(2, requires_grad=False)
    self.G: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    
    ##########################
    # scalar attrs
    ##########################
    # total number of runs acc. all steps
    self.iter = 0
    self.alpha: Optional[torch.TensorType] = None
    self.alpha_norm = 0.0
    # gamma & lower bound on gamma
    self.gamma = gamma
    self.gammalb = 1e-12
    # gamma increasing rules
    self.beta1 = beta1
    self.beta2 = beta2
    # maximum step size
    self.delta_max = 1e1
    ##########################
    # step acc rules
    ##########################
    self.eta = 0.08
    self.zeta1 = 0.25
    self.zeta2 = 0.75
    ##########################
    # weight decay of the past
    ##########################
    # other indicators
    self.ghg = 0.0
    # structured log line
    self.logline = None
    #########################
  
  def get_name(self):
    return f"drsom-b:d@{DRSOM_MODE}.{self.option_tr}:ad@{DRSOM_MODE_HVP}"
  
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
  
  @torch.no_grad()
  def _use_new_d(self, d_new):
    for p in self._params:
      p.add_(d_new[p])
  
  def _bool_grad_vanish(self, p):
    return p.grad is None or torch.linalg.norm(p.grad) < 1e-8
  
  @torch.no_grad()
  def _clear_momentum(self):
    # only has globally state
    for p in self._params:
      for k in DRSOM_DIRECTIONS:
        if k in self.state[p]:
          self.state[p][k].zero_()
  
  @drsom_timer
  def _save_momentum(self, vlist, key):
    """
    saving momentum
    """
    with torch.no_grad():
      for idx, p in enumerate(self._params):
        self.state[p][key] = vlist[p]
  
  @torch.no_grad()
  def solve_alpha(self, Q, c, tr):
    # initialization
    return TRS._solve_alpha(self, Q, c, tr)
  
  @drsom_timer
  def compute_step(self, option_tr='p'):
    
    self.alpha, self.alpha_norm = self.solve_alpha(
      self.Q, self.c, tr=self.G
    )
    
    ####################################
    # compute estimate decrease
    ####################################
    trs_est = - 1 / 2 * (self.Q @ self.alpha).dot(self.alpha) - self.c.dot(self.alpha)
    
    return trs_est
  
  @drsom_timer
  def hv(self, directions):
    """
    exact metric
    Args:
      p:
      v:

    Returns:

    """
    
    gv = [torch.mul(p.grad, directions[p]).sum() for p in self._params]
    return torch.autograd.grad(
      gv, self._params,
      # create_graph=True,
      retain_graph=True
    )
  
  @drsom_timer
  def update_trust_region(self, p_copy, directions, closure=None, style=DRSOM_MODE_HVP):
    st = time.time()
    with torch.enable_grad():
      __unused = p_copy
      # each direction is a list of tensors
      dim = len(directions)
      # construct G (the inner products)
      G = torch.zeros((dim, dim), requires_grad=False, device='cpu')
      for i in range(dim):
        for j in range(i, dim):
          for p in self._params:
            v = directions[i][p]
            u = directions[j][p]
            # compute G[i,j]
            G[i, j] += torch.mul(u, v).sum().detach().cpu()
      
      # keep symmetry
      for i in range(dim):
        for j in range(i):
          G[i, j] = G[j, i]
      
      # compute Hv for v in directions;
      #   assume directions[0] = g/|g|
      if self.iter % self.freq == 0:
        # @note:
        #   compute Hv:
        #   by analytic gv
        Q = torch.zeros((dim, dim), requires_grad=False, device='cpu')
        if self.Hv is None:
          self.Hv = [[] for _ in range(dim)]
        for i in range(dim):
          if style == 0:
            self.Hv[i] = self.hv(directions[i])
          # elif style == 1:
          #   self.hv_diff(p, g, v, closure, index=i)
        for i in range(dim):
          for j in range(i, dim):
            for idx, p in enumerate(self._params):
              u = directions[i][p]
              hv = self.Hv[j][idx]
              Q[i, j] += torch.mul(u, hv).sum().detach().cpu()
        for i in range(dim):
          for j in range(i):
            Q[i, j] = Q[j, i]
      else:
        # if set freq = 1
        #   this never happens
        # Q = torch.tensor([[G, 0.0], [0.0, dd]], requires_grad=False)
        raise ValueError("not handled yet")
      
      c = torch.zeros(dim, requires_grad=False, device='cpu')
      for i in range(dim):
        for p in self._params:
          u = directions[i][p]
          c[i] += torch.mul(p.grad, u).sum().cpu()
      
      self.ghg = (Q[0, 0] + self.ghg * self.iter) / (self.iter + 1)
      
      # compute Q/c/G
      self.Q = Q
      self.c = c
      # use generalized a'Ga <= delta
      self.G = G
    et = time.time()
  
  def normalize(self, v):
    v_norm = torch.linalg.norm(v)
    v_norm = 1 if v_norm == 0 else v_norm
    return v / v_norm
  
  def gather_normalized_grad(self):
    
    # todo, should we normalize?
    return {p: p.grad.detach().clone() for p in self._params}
  
  def gather_normalize(self, k):
    return {p: self.state[p][k] for p in self._params}
  
  def step(self, closure=None):
    """
    Performs a single optimization step.
    Arguments:
        closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
    """
    
    if closure is None:
      raise ValueError("must provide a closure for RSOM")
    closure = torch.enable_grad()(closure)
    if DRSOM_VERBOSE:
      torch.autograd.set_detect_anomaly(True)
    n_iter = 0
    
    loss = closure()
    
    # copy of it at last step
    p_copy = self._clone_param()
    
    # @note
    # no need to scale (even for gradient) since it is a new tensor.
    g_old = self.gather_normalized_grad()
    directions = [
      g_old,  # make sure -g is the first direction
      *(self.gather_normalize(k) for k in DRSOM_DIRECTIONS)
    ]
    
    self.update_trust_region(p_copy, directions, closure=closure, style=DRSOM_MODE_HVP)
    # accept or not?
    acc_step = False
    # adjust lambda: (and thus trust region radius)
    iter_adj = 1
    while iter_adj < self._max_iter_adj:
      # solve alpha
      trs_est = self.compute_step(option_tr=self.option_tr)
      if trs_est < 0:
        self.gamma = max(self.gamma * self.beta1, 1e-4)
        if DRSOM_VERBOSE:
          pprint(self.logline)
        continue
      alpha = self.alpha
      
      # build direction
      dim = len(directions)
      d_new = {p: torch.zeros_like(p, requires_grad=False) for p in self._params}
      for p in self._params:
        for i in range(dim):
          u = directions[i][p]
          d_new[p].add_(u, alpha=alpha[i])
      
      self._use_new_d(d_new)
      loss_est = float(closure(backward=False))
      # accept or not？
      loss_dec = loss - loss_est
      rho = loss_dec / trs_est
      
      # update the trust-region radius (implicitly by gamma/lambda)
      lmb_dec = 0
      gamma_old = self.gamma
      if rho <= self.zeta1:
        self.gamma = max(self.gamma * self.beta1, 1e-4)
      else:
        if rho >= self.zeta2:
          lmb_dec = 1
          self.gamma = max(self.gammalb, min(self.gamma / self.beta2, np.log(self.gamma)))
      
      acc_step = rho > self.eta
      if DRSOM_VERBOSE:
        self.logline['dQ'] = '{:+.2e}'.format(trs_est.item())
        self.logline['df'] = '{:+.2e}'.format(loss_dec.item())
        self.logline['rho'] = '{:+.2e}'.format(rho.item())
        self.logline['acc'] = int(acc_step.item())
        self.logline['acc-𝜆'] = lmb_dec
        self.logline['𝛄'] = '{:+.2e}'.format(self.gamma)
        self.logline['𝛄-'] = '{:+.2e}'.format(gamma_old)
        self.logline['f'] = '{:+.2e}'.format(loss.item())
        self.logline['k'] = '{:+6d}'.format(self.iter)
        self.logline['k0'] = iter_adj
        print(
          pd.DataFrame(
            data=[list(self.logline.values())], columns=self.logline.keys(), dtype=str
          ).to_markdown(
            tablefmt="grid"
          )
        )
      if not acc_step:
        # set back to old ~ trial step failed
        self._set_param(p_copy)
      
      else:
        if 'momentum_g' in DRSOM_DIRECTIONS:
          # compute momentum_g
          _loss = closure()
          raise ValueError("not implemented")
        elif 'momentum' in DRSOM_DIRECTIONS:
          self._save_momentum(d_new, 'momentum')
        else:
          # no momentum has to be kept
          pass
        break
      
      iter_adj += 1
    
    self.iter += 1
    n_iter += 1
    
    if not acc_step:
      # if this step is not acc. (after max # of iteration for adjustment)
      # consider restart the optimizer by clearing the momentum,
      #   just like a nonlinear conjugate gradient method.
      self._clear_momentum()
      self.gamma = self.gammalb
    
    return loss
