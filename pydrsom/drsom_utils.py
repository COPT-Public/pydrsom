import collections
import functools
import json
import os
import time
from argparse import ArgumentParser

import scipy.linalg
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DRSOM_VERBOSE = int(os.environ.get('DRSOM_VERBOSE', 0))
DRSOM_NORMALIZE = int(os.environ.get('DRSOM_NORMALIZE', 0))
# QP construction (0-hvp, 1-interpolation)
DRSOM_MODE_QP = int(os.environ.get('DRSOM_MODE_QP', 1))
# hvp or interpolation strategy
DRSOM_MODE_HVP = int(os.environ.get('DRSOM_MODE_HVP', 0))
DRSOM_MODE_DELTA = int(os.environ.get('DRSOM_MODE_DELTA', 0))
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


def update_running_stat(aa, m_aa, stat_decay):
  # using inplace operation to save memory!
  m_aa *= stat_decay / (1 - stat_decay)
  m_aa += aa
  m_aa *= (1 - stat_decay)


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
  eigvalsh = scipy.linalg.eigvalsh
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
    """
    
    if len(c) == 1 or G[1, 1] == 0:
      lmin = max(0, (-Q[0, 0] / G[0, 0]).item())
    else:
      lmin = TRS.eigvalsh(Q.numpy(), G.numpy())
      lmin = lmin[0]
    lb = max(0, -lmin)
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
      lmin = TRS.eigvalsh(Q.numpy(), G.numpy())
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


def add_parser_options(parser: ArgumentParser):
  ##############
  # for DRSOM
  ##############
  parser.add_argument("--itermax", required=False, type=int, default=15)
  parser.add_argument(
    "--option_tr", required=False, type=str, default="p", choices=["a", "p"]
  )
  parser.add_argument(
    "--drsom_beta1", default=50, type=float, help="DRSOM coefficients beta_1"
  )
  parser.add_argument(
    "--drsom_beta2", default=30, type=float, help="DRSOM coefficients beta_2"
  )
  ##############
  # "learning rate scheduler and hyperparames" for DRSOM
  parser.add_argument(
    "--drsom_qp_freq",
    default=1,
    type=int,
    help="the frequency of updating QP model",
  )
  
  parser.add_argument(
    "--drsom_decay_window",
    default=8000,
    type=int,
    help="""
      this is an analogue of `step size of the learning rate scheduler` for DRSOM,
        it basically says, in every `drsom_decay_window` steps:
        - we increase the lower bound on gamma via:
            gamma *= drsom_decay_step,
          such that the stepsize is decreased)
        - or decrease radius:
            radius *= drsom_decay_radius_step,
          if you choose to use the trust-region mode
          """,
  )
  parser.add_argument(
    "--drsom_decay_step", default=5e2, type=float, help="see drsom_decay_window"
  )
  parser.add_argument(
    "--drsom_decay_radius_step",
    default=2e-1,
    type=float,
    help="""
    see drsom_decay_window, this only works if you use the trust-region mode.
    """,
  )
  parser.add_argument(
    "--drsom_decay_sin_rel_max",
    default=1e8,
    type=float,
    help="""
      this is another stepsize choice of DRSOM,
      using the sin() update rules.
      """,
  )
  parser.add_argument(
    "--drsom_decay_sin_max_scope",
    default=37600,
    type=float,
    help="see drsom_decay_sin_rel_max",
  )
  parser.add_argument(
    "--drsom_decay_sin_min_scope",
    default=0,
    type=float,
    help="see drsom_decay_sin_rel_max",
  )


def render_args(args):
  drsom_decay_rules = DRSOMDecayRules(
    decay_qp_freq=args.drsom_qp_freq,
    decay_window=args.drsom_decay_window,
    decay_step=args.drsom_decay_step,
    decay_radius_step=args.drsom_decay_radius_step,
    decay_sin_rel_max=args.drsom_decay_sin_rel_max,
    decay_sin_min_scope=args.drsom_decay_sin_min_scope,
    decay_sin_max_scope=args.drsom_decay_sin_max_scope,
  )
  return dict(
    option_tr=args.option_tr,
    beta1=args.drsom_beta1,
    beta2=args.drsom_beta2,
    max_iter=args.itermax,
    decayrules=drsom_decay_rules,
  )


class DRSOMDecayRules(object):
  def __init__(self, **kwargs):
    # quadratic approx. rules
    self.qp_rate = 0.3
    self.qp_freq = kwargs.get("decay_qp_freq", 1)
    
    # linear rule
    self.decay_mode = DRSOM_MODE_DECAY
    self.decay_window = kwargs.get("decay_window", 8000)
    self.decay_step = kwargs.get("decay_step", 5e2)
    self.decay_radius_step = kwargs.get("decay_radius_step", 2e-1)
    # sin rule
    self.decay_sin_rel_max = kwargs.get("decay_sin_rel_max", 1e8)
    self.decay_sin_min_scope = kwargs.get("decay_sin_min_scope", 0)
    self.decay_sin_max_scope = kwargs.get("decay_sin_max_scope", 37600)
    print(self.print())
  
  def print(self):
    import json
    
    return json.dumps(self.__dict__, indent=2)
  
  def __str__(self):
    return f"[{DRSOM_MODE_DECAY}w@{self.decay_sin_rel_max:.2e}:{self.decay_sin_min_scope:.2e}-{self.decay_sin_max_scope:.2e}]"
