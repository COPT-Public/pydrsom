import collections
import functools
import os
import time

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DRSOM_VERBOSE = int(os.environ.get('DRSOM_VERBOSE', 0))
DRSOM_MODE = int(os.environ.get('DRSOM_MODE', 3))
DRSOM_MODE_HVP = int(os.environ.get('DRSOM_MODE_HVP', 0))
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
  @staticmethod
  def _norm(alpha, tr):
    return (tr @ alpha).dot(alpha).sqrt()

  @staticmethod
  def _compute_root(Q, c, gamma, tr=torch.eye(2)):
    lsolve = torch.linalg.solve
    
    D, V = torch.linalg.eigh(Q)
    
    lmin, lmax = min(D), max(D)
    lb = max(0, -lmin.item())
    lmax = lmax.item() if lmax > lb else lb + 1e4
    _lmb_this = gamma * lmax + max(1 - gamma, 0) * lb
    it = 0
    try:
      alpha = lsolve(Q + tr * _lmb_this, -c)
    except torch.linalg.LinAlgError as e:
      print(e)
      print(Q, tr, _lmb_this, -c)
      # todo, can we do better
      alpha = lsolve(Q + tr * (_lmb_this + 1e-4), -c)
    
    norm = TRS._norm(alpha, tr)
    
    return it, _lmb_this, alpha, norm, True
