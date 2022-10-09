"""
DRSOM, (dimension-reduced second-order method)
@author: Chuwen Zhang<chuwzhang@gmail.com>, Yinyu Ye<yinyu-ye@stanford.edu>
@note:
   (Mini-batch, Radius-Free) DRSOM.
   block-DRSOM
  
"""
from functools import reduce
from pprint import pprint
from typing import Optional
import torch
import numpy as np
import pandas as pd

from torch.nn.utils import parameters_to_vector

from .drsom_utils import *
from .kfac_utils import *


class DRSOM(torch.optim.Optimizer):
  
  def __init__(
      self,
      model,
      max_iter=15,
      gamma=1e-6,
      beta1=5e1,
      beta2=3e1,
      thetas=(0.99, 0.999),
      eps=1e-8,
      TCov=5,
      som_decay=0.95,  # decay param for second-order info
      batch_averaged=True,
      **kwargs
  ):
    """
    The DRSOM:
      Implementation of (Mini-batch) DRSOM (Dimension-Reduced Second-Order Method) in F (Radius-Free) style
    Args:
      params: model params
      max_iter: # of iteration for trust-region adjustment
      gamma: lower bound for gamma
      beta1: gamma + multiplier
      beta2: gamma - multiplier
      hessian_window: window size to keep last k hessian information
      thetas: weight decay params (like betas for Adam)
      eps: ...
    """
    
    defaults = dict(betas=thetas, eps=eps)
    params = model.parameters()
    super(DRSOM, self).__init__(params, defaults)
    self.CovAHandler = ComputeCovA()
    self.CovGHandler = ComputeCovG()
    self.batch_averaged = batch_averaged
    
    self.known_modules = {'Linear', 'Conv2d'}
    
    self.modules = []
    self.grad_outputs = {}
    
    self.model = model
    self._prepare_model()
    
    self.steps = 0
    
    self.m_aa, self.m_gg = {}, {}
    self.Q_a, self.Q_g = {}, {}
    self.d_a, self.d_g = {}, {}
    self.som_decay = som_decay
    self.TCov = TCov
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
  
  def _save_input(self, module, input):
    if torch.is_grad_enabled() and self.steps % self.TCov == 0:
      aa = self.CovAHandler(input[0].data, module)
      # Initialize buffers
      if self.steps == 0:
        self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
      update_running_stat(aa, self.m_aa[module], self.som_decay)
  
  def _save_grad_output(self, module, grad_input, grad_output):
    # todo, fix this
    self.acc_stats = True
    # Accumulate statistics for Fisher matrices
    if self.acc_stats and self.steps % self.TCov == 0:
      gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
      # Initialize buffers
      if self.steps == 0:
        self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
      update_running_stat(gg, self.m_gg[module], self.som_decay)
  
  def _prepare_model(self):
    count = 0
    print(self.model)
    print("=> We keep following layers in KFAC. ")
    for module in self.model.modules():
      classname = module.__class__.__name__
      # print('=> We keep following layers in KFAC. <=')
      if classname in self.known_modules:
        self.modules.append(module)
        module.register_forward_pre_hook(self._save_input)
        module.register_backward_hook(self._save_grad_output)
        print('(%s): %s' % (count, module))
        count += 1
  
  def get_name(self):
    return f"drsom:d@{DRSOM_MODE}:ad@{DRSOM_MODE_HVP}"
  
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
      for k in DRSOM_DIRECTIONS:
        if k in self.state[p]:
          self.state[p][k].zero_()
  
  @drsom_timer
  def _apply_step(self, flat_p):
    with torch.no_grad():
      offset = 0
      for p in self._params:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.copy_(flat_p[offset:offset + numel].view_as(p))
        offset += numel
      assert offset == self._numel()
  
  @drsom_timer
  def _save_momentum(self, *args):
    """
    saving momentum
    Args:
      *args:
      0 - d(x)
      1 - d(g)
    Returns:

    """
    with torch.no_grad():
      
      offset = 0
      for p in self._params:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        if 'momentum' in DRSOM_DIRECTIONS:
          self.state[p]['momentum'].copy_(args[0][offset:offset + numel].view_as(p))
        if 'momentum_g' in DRSOM_DIRECTIONS:
          self.state[p]['momentum_g'].copy_(args[1][offset:offset + numel].view_as(p))
        offset += numel
      assert offset == self._numel()
  
  @drsom_timer
  def _directional_evaluate(self, closure, flat_p):
    self._apply_step(flat_p)
    # evaluation
    loss = float(closure(backward=False))
    return loss
  
  def _numel(self):
    if self._numel_cache is None:
      self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
    return self._numel_cache
  
  @drsom_timer
  def _gather_flat_grad(self, _valid_params, target='self'):
    if target == 'grad':
      flat = torch.concat([p.grad.reshape(-1) for p in _valid_params])
    elif target == 'momentum':
      flat = torch.concat([self.state[p]['momentum'].reshape(-1) for p in _valid_params])
    elif target == 'momentum_g':
      flat = torch.concat([self.state[p]['momentum_g'].reshape(-1) for p in _valid_params])
    else:
      flat = torch.concat([p.reshape(-1) for p in _valid_params])
    
    return flat
  
  @torch.no_grad()
  def solve_alpha(self, Q, c, tr):
    # initialization
    dim = c.size()[0]
    if self.iter == 0 or self.Q[dim - 1, dim - 1] < 1e-4:
      lmd = 0.0
      alpha = torch.zeros_like(c)
      alpha[0] = -c[0] / Q[0, 0] / (1 + self.gamma) if Q[0, 0] > 0 else 1e-4
      norm = TRS._norm(alpha, tr)
      if norm > self.delta_max:
        alpha = alpha / alpha.norm() * self.delta_max
    else:
      # apply root-finding
      it, lmd, alpha, norm, active = TRS._compute_root(Q, c, self.gamma, tr)
    
    if DRSOM_VERBOSE:
      self.logline = {
        '𝜆': '{:+.2e}'.format(lmd),
        'Q/c/G': np.round(np.vstack([self.Q, self.c, self.G]), 3),
        'a': np.round(alpha.tolist(), 3).reshape((dim, 1)),
        'ghg': '{:+.2e}'.format(Q[0, 0]),
        'ghg-': '{:+.2e}'.format(self.ghg),
      }
    return alpha, norm
  
  @drsom_timer
  def compute_step(self):
    # compute alpha
    
    self.alpha, self.alpha_norm = self.solve_alpha(
        self.Q, self.c, tr=self.G
    )
    
    ####################################
    # compute estimate decrease
    ####################################
    trs_est = - 1 / 2 * (self.Q @ self.alpha).dot(self.alpha) - self.c.dot(self.alpha)
    
    return trs_est
  
  @drsom_timer
  def hv(self, g, v, flag=0, index=0):
    """
    exact metric
    Args:
      g:
      v:
      flag:
      index:

    Returns:

    """
    mul = 0.5 if flag == 1 else 1
    gv = g.dot(v)
    self.Hv[index] = self._gather_flat_grad(torch.autograd.grad(
      gv * mul, self._params,
      # create_graph=True,
      retain_graph=True
    ), target='self')
  
  @drsom_timer
  def hv_diff(self, flat_p, g, v, closure, flag=0, index=0, eps=1e-8):
    """
    finite diff
    Args:
      g:
      v:
      flag:
      index:
      eps
    Returns:

    """
    mul = 0.5 if flag == 1 else 1
    scale = 1 / eps
    with torch.no_grad():
      self._apply_step(eps * v + flat_p)
    _ = closure()
    g_eps = parameters_to_vector([p.grad for p in self._params])
    self.Hv[index] = scale * (g_eps - g) * mul
  
  @drsom_timer
  def update_trust_region(self, flat_p, flat_g, directions, closure=None, style=DRSOM_MODE_HVP):
    st = time.time()
    with torch.enable_grad():
      __unused = flat_p
      
      dim = len(directions)
      # construct G (the inner products)
      G = torch.zeros((dim, dim), requires_grad=False, device='cpu')
      for i, v in enumerate(directions):
        for j in range(i, dim):
          u = directions[j]
          # compute G[i,j]
          G[i, j] = G[j, i] = v.dot(u).detach().cpu()
      
      # compute Hv for v in directions;
      #   assume directions[0] = g/|g|
      if self.iter % self.freq == 0:
        # @note:
        #   compute Hv:
        #   by analytic gv
        Q = torch.zeros((dim, dim), requires_grad=False, device='cpu')
        if self.Hv is None:
          self.Hv = [torch.empty_like(flat_g) for _ in directions]
        for i, v in enumerate(directions):
          if style == 0:
            self.hv(flat_g, v, index=i)
          elif style == 1:
            self.hv_diff(flat_p, flat_g, v, closure, index=i)
        for i, v in enumerate(directions):
          for j in range(i, dim):
            Q[i, j] = Q[j, i] = v.dot(self.Hv[j]).detach().cpu()
      else:
        # if set freq = 1
        #   this never happens
        # Q = torch.tensor([[G, 0.0], [0.0, dd]], requires_grad=False)
        raise ValueError("not handled yet")
      
      c = torch.tensor([flat_g.dot(v).detach().cpu() for v in directions], requires_grad=False)
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
  
  def gather_normalize(self, k):
    return self.normalize(self._gather_flat_grad([self.state[p][k] for p in self._params]))
  
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
    
    
    loss = closure()
    flat_p = parameters_to_vector(self._params)
    # copy of it at last step
    p_copy = self._clone_param()
    
    flat_g = parameters_to_vector([p.grad for p in self._params])
    
    # @note
    # no need to scale (even for gradient) since it is a new tensor.
    directions = [
      self.normalize(- flat_g),  # make sure -g is the first direction
      *(self.gather_normalize(k) for k in DRSOM_DIRECTIONS)
    ]
    
    self.update_trust_region(flat_p, flat_g, directions, closure=closure, style=DRSOM_MODE_HVP)
    # accept or not?
    acc_step = False
    # adjust lambda: (and thus trust region radius)
    iter_adj = 1
    while iter_adj < self._max_iter_adj:
      # solve alpha
      trs_est = self.compute_step()
      if trs_est < 0:
        self.gamma = max(self.gamma * self.beta1, 1e-4)
        if DRSOM_VERBOSE:
          pprint(self.logline)
        continue
      alpha = self.alpha
      
      # build direction
      
      flat_new_d = torch.zeros_like(flat_p, requires_grad=False)
      for aa, dd in zip(alpha, directions):
        flat_new_d.add_(dd, alpha=aa)
      
      # new trial points
      flat_new_p = torch.zeros_like(flat_p, requires_grad=False).copy_(flat_p).add_(flat_new_d)
      
      # accept or not？
      loss_est = self._directional_evaluate(closure, flat_new_p)
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
          # compute flat_momentum_g
          _loss = closure()
          flat_g_new = parameters_to_vector([p.grad for p in self._params])
          self._save_momentum(flat_new_d, flat_g_new - flat_g)
        else:
          self._save_momentum(flat_new_d)
        break
      
      iter_adj += 1
    
    self.iter += 1
    
    if not acc_step:
      # if this step is not acc. (after max # of iteration for adjustment)
      # consider restart the optimizer by clearing the momentum,
      #   just like a nonlinear conjugate gradient method.
      self._clear_momentum()
      self.gamma = self.gammalb
    
    return loss
