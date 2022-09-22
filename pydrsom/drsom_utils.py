import argparse
import json
import os
import time
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
