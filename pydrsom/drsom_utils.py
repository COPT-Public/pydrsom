import collections
import functools
import os
import time

import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DRSOM_VERBOSE = int(os.environ.get('DRSOM_VERBOSE', 0))
# QP construction (0-hvp, 1-interpolation)
DRSOM_MODE_QP = int(os.environ.get('DRSOM_MODE_QP', 1))
# hvp strategy
DRSOM_MODE_HVP = int(os.environ.get('DRSOM_MODE_HVP', 0))
# subproblem strategy
DRSOM_MODE_TRS = int(os.environ.get('DRSOM_MODE_TRS', 0))
# gamma and radius decay
DRSOM_MODE_DECAY = int(os.environ.get('DRSOM_MODE_DECAY', 1))
# direction
DRSOM_MODE = int(os.environ.get('DRSOM_MODE', 0))
if DRSOM_MODE == 0:
  DRSOM_DIRECTIONS = ['momentum']
elif DRSOM_MODE == 1:
  DRSOM_DIRECTIONS = ['momentum_g']
elif DRSOM_MODE == 2:
  DRSOM_DIRECTIONS = ['momentum_g', 'momentum']
elif DRSOM_MODE == 3:
  DRSOM_DIRECTIONS = []
else:
  raise ValueError("invalid selection of mode")

DRSOM_GLOBAL_PROFILE = {
  'count': collections.defaultdict(int),
  'total': collections.defaultdict(float)
}


def load_checkpoint(ckpt_name):
  print('==> Resuming from checkpoint..')
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  assert os.path.exists(ckpt_name), 'Error: checkpoint {} not found'.format(ckpt_name)
  return torch.load(ckpt_name)


def query_name(optimizer, name, args, ckpt):
  if name.startswith("drsom"):
    return f"{optimizer.get_name()}" if not args.resume else f"{optimizer.get_name()}-r-{ckpt['epoch']}"
  else:
    return name


def drsom_timer(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    st = time.time()
    cc = func(*args, **kwargs)
    et = time.time()
    DRSOM_GLOBAL_PROFILE['total'][func.__qualname__] += et - st
    DRSOM_GLOBAL_PROFILE['count'][func.__qualname__] += 1
    return cc
  
  return wrapper


##########################################
# TRS/Regularized QP solver
##########################################
class TRS:
  lsolve = torch.linalg.solve
  
  @staticmethod
  def _norm(alpha, G):
    return (G @ alpha).dot(alpha).sqrt().item()
  
  @staticmethod
  def _compute_root(Q, c, gamma, G):
    """
    This is the radius free mode.
      scale Lagrangian dual by gamma
    Args:
      gamma: is the scale param
      G:

    Returns:

    """
    
    if len(c) == 1 or G[1, 1] == 0:
      lmin = max(0, (-Q[0, 0] / G[0, 0]).item())
    else:
      lmin = torch.linalg.eigvalsh(Q)
      lmin = lmin[0]
    lb = max(0, lmin)
    lmax = lb + 1e4
    _lmb_this = gamma * lmax + max(1 - gamma, 0) * lb
    it = 0
    try:
      alpha = TRS.lsolve(Q + G * _lmb_this, -c)
    except torch.linalg.LinAlgError as e:
      print(e)
      print(Q, G, _lmb_this, -c)
      alpha = TRS.lsolve(Q + G * (_lmb_this + 1e-4), -c)
    
    norm = TRS._norm(alpha, G)
    
    return it, _lmb_this, alpha, norm, True
  
  @staticmethod
  def _compute_root_tr(Q, c, delta, G=torch.eye(2)):
    eps = 1e-2
    
    if len(c) == 1 or G[1, 1] == 0:
      lmin = max(0, (-Q[0, 0] / G[0, 0]).item())
    else:
      lmin = torch.linalg.eigvalsh(Q)
      lmin = lmin[0]
    lb = max(0, lmin)
    ub = lb + 1e1
    it = 0
    gamma = 1e-1
    while True:
      _lmb_this = lb * (1 - gamma) + gamma * ub
      try:
        alpha = TRS.lsolve(Q + G * _lmb_this, -c)
      except:
        print(Q, G, c)
      norm = TRS._norm(alpha, G)
      if norm < delta or it > 10:
        break
      gamma *= 5
      it += 1
    return it, _lmb_this, alpha, norm, True
  
  @staticmethod
  def _solve_alpha(optimizer, Q, c, G):
    dim = c.size()[0]
    if optimizer.iter == 0 or optimizer.G[dim - 1, dim - 1] < 1e-4:
      lmd = 0.0
      alpha = torch.zeros_like(c)
      if Q[0, 0] > 0:
        alpha[0] = - c[0] / Q[0, 0] / (1 + optimizer.gamma)
      else:
        alpha[0] = - 1e-4 / (1 + optimizer.gamma)
      norm = TRS._norm(alpha, G)
      if norm > optimizer.radius:
        alpha = alpha / alpha.norm() * optimizer.radius
    else:
      # apply root-finding
      if DRSOM_MODE_TRS:
        it, lmd, alpha, norm, active = TRS._compute_root_tr(Q, c, optimizer.radius, G)
      else:
        it, lmd, alpha, norm, active = TRS._compute_root(Q, c, optimizer.gamma, G)
    
    if DRSOM_VERBOSE:
      optimizer.logline = {
        **optimizer.logline,
        '𝜆': '{:+.2e}'.format(lmd),
        'Q/c/G': np.round(np.vstack([optimizer.Q, optimizer.c, optimizer.G]), 3),
        'a': np.array(alpha.tolist()).reshape((dim, 1)),
        'ghg': '{:+.2e}'.format(Q[0, 0]),
        'ghg-': '{:+.2e}'.format(optimizer.ghg),
      }
    return alpha, norm
