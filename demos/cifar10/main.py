"""
A script for DRSOM on CIFAR dataset.
@author: Chuwen Zhang
@note:
  This script runs DRSOM and compares to Adam, SGD, and so forth.
  ################################################################
    usage:
      $ python main.py -h
  ################################################################
"""

from __future__ import print_function

import argparse
import json
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from pydrsom.drsomb import DRSOMB as DRSOM
from pydrsom.drsomk import DRSOMK
from pydrsom.drsom_utils import *
from .adabound import AdaBound
# from .torch_optimizer import Adahessian
from .models import *
from .util import *


def get_parser():
  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument(
    '--model', default='resnet', type=str, help='model',
    choices=[
      'resnet18', 'resnet34'
    ]
  )
  parser.add_argument(
    '--optim', default='sgd', type=str, help='optimizer',
    choices=['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound', 'adahessian',
             'drsom', 'drsomk']
  )
  parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
  parser.add_argument('--batch', default=128, type=int, help='batch size')
  # for adabound
  parser.add_argument('--final_lr', default=0.1, type=float,
                      help='final learning rate of AdaBound')
  parser.add_argument(
    '--gamma', default=1e-3, type=float,
    help='convergence speed term of AdaBound'
  )
  # sgd & adam
  parser.add_argument('--momentum', default=0.95, type=float, help='momentum term')
  parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
  parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
  parser.add_argument('--ckpt_name', type=str, help='resume from checkpoint')
  parser.add_argument('--epoch', '-e', default=200, type=int, help='num of epoches to run')
  parser.add_argument('--weight_decay', default=5e-4, type=float,
                      help='weight decay for optimizers')
  
  # for drsom
  parser.add_argument("--itermax", required=False, type=int, default=15)
  parser.add_argument("--option_tr", required=False, type=str, default='p', choices=['a', 'p'])
  parser.add_argument("--hessian_window", required=False, type=int, default=1)
  parser.add_argument('--drsom_beta1', default=50, type=float, help='DRSOM coefficients beta_1')
  parser.add_argument('--drsom_beta2', default=30, type=float, help='DRSOM coefficients beta_2')
  
  parser.add_argument('--tflogger', default="/tmp/", type=str, help='tf logger directory')
  
  # learning rate scheduler
  parser.add_argument(
    '--lrstep', default=20, type=int,
    help='step for learning rate scheduler\n'
         ' - for Adam/SGD, it corresponds to usual step definition for the lr scheduler\n'
  )
  parser.add_argument(
    '--lrcoeff', default=0.5, type=float,
    help='step for learning rate scheduler\n'
         ' - for Adam/SGD, it corresponds to usual step definition for the lr scheduler\n'
  )
  parser.add_argument(
    '--drsom_decay_window', default=8000, type=int,
    help="""
    this is an analogue of `step size of the learning rate scheduler` for DRSOM,
      it basically says, in every `drsom_decay_window` steps:
      - we increase the lower bound on gamma via:
          gamma *= drsom_decay_step,
        such that the stepsize is decreased)
      - or decrease radius:
          radius *= drsom_decay_radius_step,
        if you choose to use the trust-region mode
        """
  )
  parser.add_argument(
    '--drsom_decay_step', default=5e2, type=float,
    help='see drsom_decay_window'
  )
  parser.add_argument(
    '--drsom_decay_radius_step', default=2e-1, type=float,
    help='see drsom_decay_window, this only works if you use the trust-region mode.'
  )
  
  return parser


def build_dataset(args):
  print('==> Preparing data..')
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=transform_train)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True,
                                             num_workers=2)
  
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=transform_test)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)
  
  # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
  return train_loader, test_loader


def build_model(args, device, ckpt=None):
  print('==> Building model..')
  net = {
    'resnet18': ResNet18,
    'resnet34': ResNet34
  }[args.model]()
  print(f'==> Building model {args.model}')
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
  
  if ckpt:
    net.load_state_dict(ckpt['net'])
  
  return net


def create_optimizer(args, model, start_epoch=0):
  model_params = model.parameters()
  if args.optim == 'sgd':
    return optim.SGD(model_params, args.lr * args.lrcoeff ** (start_epoch // args.lrstep), momentum=args.momentum,
                     weight_decay=args.weight_decay)
  elif args.optim == 'adagrad':
    return optim.Adagrad(model_params, args.lr * args.lrcoeff ** (start_epoch // args.lrstep),
                         weight_decay=args.weight_decay)
  elif args.optim == 'adam':
    return optim.Adam(model_params, args.lr * args.lrcoeff ** (start_epoch // args.lrstep),
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay)
  elif args.optim == 'amsgrad':
    return optim.Adam(model_params, args.lr * args.lrcoeff ** (start_epoch // args.lrstep),
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay, amsgrad=True)
  elif args.optim == 'adabound':
    return AdaBound(model_params, args.lr * args.lrcoeff ** (start_epoch // args.lrstep),
                    betas=(args.beta1, args.beta2),
                    final_lr=args.final_lr, gamma=args.gamma,
                    weight_decay=args.weight_decay)
  # second-order method
  # elif args.optim == 'adahessian':
  #   return Adahessian(
  #     model_params,
  #     lr=1.0,
  #     betas=(0.9, 0.999),
  #     eps=1e-4,
  #     weight_decay=0.0,
  #     hessian_power=1.0,
  #   )
  # my second-order method
  elif args.optim == 'drsom':
    return DRSOM(
      model_params,
      hessian_window=args.hessian_window,
      option_tr=args.option_tr,
      beta1=args.drsom_beta1,
      beta2=args.drsom_beta2,
      max_iter=args.itermax,
      decay_window=args.drsom_decay_window,
      decay_step=args.drsom_decay_step,
      decay_radius_step=args.drsom_decay_radius_step
    )
  elif args.optim == 'drsomk':
    return DRSOMK(
      model,
      hessian_window=args.hessian_window,
      option_tr=args.option_tr,
      beta1=args.drsom_beta1,
      beta2=args.drsom_beta2,
      max_iter=args.itermax
    )
  else:
    raise ValueError(f"Optimizer {args.optim} not defined")


def main():
  parser = get_parser()
  args = parser.parse_args()
  print(json.dumps(args.__dict__, indent=2))
  train_loader, test_loader = build_dataset(args)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  if args.resume:
    ckpt = load_checkpoint(args.ckpt_name)
    start_epoch = ckpt['epoch'] + 1
  else:
    ckpt = None
    start_epoch = 0
  
  net = build_model(args, device, ckpt=ckpt)
  criterion = nn.CrossEntropyLoss()
  optimizer = create_optimizer(args, net, start_epoch=start_epoch)
  
  if args.optim.startswith("drsom"):
    # get a scheduler
    pass
  else:
    # get a scheduler
    scheduler = optim.lr_scheduler.StepLR(
      optimizer, step_size=args.lrstep, gamma=args.lrcoeff,
    )
  
  if args.optim.startswith("drsom"):
    log_name = f"[{args.model}]" + "-" + query_name(optimizer, args.optim, args, ckpt)
  else:
    log_name = get_ckpt_name(
      model=args.model, optimizer=args.optim, lr=args.lr,
      final_lr=args.final_lr, momentum=args.momentum,
      beta1=args.beta1, beta2=args.beta2, gamma=args.gamma
    )
  print(f"Using model: {args.model}")
  print(f"Using optimizer:\n {log_name}")
  
  writer = SummaryWriter(log_dir=os.path.join(args.tflogger, log_name))
  
  for epoch in range(start_epoch, start_epoch + args.epoch):
    try:
      if args.optim.startswith("drsom"):
        pass
      
      else:
        scheduler.step()
        print(f"lr scheduler steps: {scheduler.get_lr()}")
      
      train_acc, train_loss, = train(net, epoch, device, train_loader, args.optim, optimizer, criterion)
      test_acc, test_loss = test(net, device, test_loader, criterion)
      
      # writer
      writer.add_scalar("Acc/train", train_acc, epoch)
      writer.add_scalar("Loss/train", train_loss, epoch)
      writer.add_scalar("Acc/test", test_acc, epoch)
      writer.add_scalar("Loss/test", test_loss, epoch)
      
      # Save checkpoint.
      if epoch % 5 == 0:
        print('Saving..')
        
        state = {
          'net': net.state_dict(),
          'acc': test_acc,
          'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
          os.mkdir('checkpoint')
        
        torch.save(state, os.path.join('checkpoint', f"{log_name}-{epoch}"))
        
        ######################################
        # train_accuracies.append(train_acc)
        # test_accuracies.append(test_acc)
        # if not os.path.isdir('curve'):
        #   os.mkdir('curve')
        # torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
        #            os.path.join('curve', ckpt_name))
        #######################################
    
    
    except KeyboardInterrupt:
      print(f"Exiting at {epoch}")
      break
    ################
    # profile details
    ################
    if args.optim.startswith("drsom"):
      import pandas as pd
      print("|--- DRSOM COMPUTATION STATS ---")
      stats = pd.DataFrame.from_dict(DRSOM_GLOBAL_PROFILE)
      stats['avg'] = stats['total'] / stats['count']
      stats = stats.sort_values(by='total', ascending=False)
      print(stats.to_markdown())


if __name__ == '__main__':
  main()
