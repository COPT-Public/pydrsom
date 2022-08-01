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

from pydrsom.drsom import DRSOMF
from .adabound import AdaBound
from .models import *


def get_parser():
  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument(
    '--model', default='resnet', type=str, help='model',
    choices=[
      'resnet', 'resnet18',
    ]
  )
  parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                      choices=['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound', 'drsom'])
  parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
  # for adabound
  parser.add_argument('--final_lr', default=0.1, type=float,
                      help='final learning rate of AdaBound')
  parser.add_argument(
    '--gamma', default=1e-3, type=float,
    help='convergence speed term of AdaBound'
  )
  # sgd & adam
  parser.add_argument('--momentum', default=0.99, type=float, help='momentum term')
  parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
  parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
  parser.add_argument('--epoch', '-e', default=200, type=int, help='num of epoches to run')
  parser.add_argument('--weight_decay', default=5e-4, type=float,
                      help='weight decay for optimizers')
  
  # for drsom
  parser.add_argument("--itermax", required=False, type=int, default=15)
  parser.add_argument("--option_tr", required=False, type=str, default='p', choices=['a', 'p'])
  parser.add_argument("--hessian_window", required=False, type=int, default=1)
  parser.add_argument('--drsom_beta1', default=50, type=float, help='DRSOM coefficients beta_1')
  parser.add_argument('--drsom_beta2', default=30, type=float, help='DRSOM coefficients beta_2')
  parser.add_argument('--gamma_power', default=1e3, type=float,
                      help='gamma multiplier: (hyper) DRSOM coefficients for adjusting gamma lower bound')
  parser.add_argument('--tflogger', default="/tmp/", type=str, help='tf logger directory')
  
  # learning rate scheduler
  parser.add_argument(
    '--lrstep', default=10, type=int,
    help='step for learning rate scheduler\n'
         ' - for Adam/SGD, it corresponds to usual step definition for the lr scheduler\n'
         ' - for DRSOM, it says, in every `lrstep` step, we increase the lower bound on gamma via `gamma *= gamma_power` (such that the stepsize is decreased)'
  )
  return parser


def build_dataset():
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
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                             num_workers=2)
  
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=transform_test)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
  
  # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
  return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3):
  name = {
    'sgd': 'lr{}-momentum{}'.format(lr, momentum),
    'adagrad': 'lr{}'.format(lr),
    'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
    'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
    'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    'drsom': 'betas{}-{}'.format(beta1, beta2),
  }[optimizer]
  return '{}-{}-{}'.format(model, optimizer, name)


def load_checkpoint(ckpt_name):
  print('==> Resuming from checkpoint..')
  path = os.path.join('checkpoint', ckpt_name)
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
  return torch.load(path)


def build_model(args, device, ckpt=None):
  print('==> Building model..')
  net = {
    'resnet': ResNet34,
    'resnet18': ResNet18
  }[args.model]()
  print(f'==> Building model {args.model}')
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
  
  if ckpt:
    net.load_state_dict(ckpt['net'])
  
  return net


def create_optimizer(args, model_params):
  if args.optim == 'sgd':
    return optim.SGD(model_params, args.lr, momentum=args.momentum,
                     weight_decay=args.weight_decay)
  elif args.optim == 'adagrad':
    return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
  elif args.optim == 'adam':
    return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay)
  elif args.optim == 'amsgrad':
    return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay, amsgrad=True)
  elif args.optim == 'adabound':
    return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                    final_lr=args.final_lr, gamma=args.gamma,
                    weight_decay=args.weight_decay)
  elif args.optim == 'drsom':
    return DRSOMF(
      model_params,
      hessian_window=args.hessian_window,
      option_tr=args.option_tr,
      beta1=args.drsom_beta1,
      beta2=args.drsom_beta2,
      max_iter=args.itermax
    )
  else:
    raise ValueError(f"Optimizer {args.optim} not defined")


def train(net, epoch, device, data_loader, name, optimizer, criterion):
  print('\nEpoch: %d' % epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  size = len(data_loader.dataset)
  for batch_idx, (inputs, targets) in enumerate(data_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    
    def closure(backward=True):
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, targets)
      if not backward:
        return loss
      if name in {'drsom'}:
        loss.backward(create_graph=True)
      else:
        loss.backward()
      return loss
    
    loss = optimizer.step(closure=closure)
    outputs = net(inputs)
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    if batch_idx % 20 == 0:
      loss, current = loss.item(), batch_idx * len(inputs)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  
  accuracy = 100. * correct / total
  print('train acc %.3f' % accuracy)
  
  return accuracy, train_loss / len(data_loader)


def test(net, device, data_loader, criterion):
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)
      
      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
  
  accuracy = 100. * correct / total
  print(' test acc %.3f' % accuracy)
  
  return accuracy, test_loss / len(data_loader)


def main():
  parser = get_parser()
  args = parser.parse_args()
  print(json.dumps(args.__dict__, indent=2))
  writer = SummaryWriter(log_dir=args.tflogger)
  train_loader, test_loader = build_dataset()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  ckpt_name = get_ckpt_name(
    model=args.model, optimizer=args.optim, lr=args.lr,
    final_lr=args.final_lr, momentum=args.momentum,
    beta1=args.beta1, beta2=args.beta2, gamma=args.gamma
  )
  if args.resume:
    ckpt = load_checkpoint(ckpt_name)
    best_acc = ckpt['acc']
    start_epoch = ckpt['epoch']
  else:
    ckpt = None
    best_acc = 0
    start_epoch = -1
  
  net = build_model(args, device, ckpt=ckpt)
  criterion = nn.CrossEntropyLoss()
  optimizer = create_optimizer(args, net.parameters())
  if args.optim not in ['drsom']:
    scheduler = optim.lr_scheduler.StepLR(
      optimizer, step_size=args.lrstep, gamma=0.5,
      last_epoch=start_epoch
    )
  
  train_accuracies = []
  test_accuracies = []
  print(f"lambda increased factor {args.gamma_power}")
  gammabase = optimizer.gammalb
  for epoch in range(start_epoch + 1, args.epoch):
    if args.optim in ['drsom']:
      
      optimizer.gammalb = gammabase * args.gamma_power ** ((epoch + 1) // args.lrstep)
      print(f"gamma lower bound changed to {optimizer.gammalb}")
    
    else:
      scheduler.step()
    
    train_acc, train_loss, = train(net, epoch, device, train_loader, args.optim, optimizer, criterion)
    test_acc, test_loss = test(net, device, test_loader, criterion)
    
    # writer
    writer.add_scalars("Accuracy/train", {f'{args.optim}-{args.lrstep}': train_acc}, epoch)
    writer.add_scalars("Loss/train", {f'{args.optim}-{args.lrstep}': train_loss}, epoch)
    writer.add_scalars("Accuracy/test", {f"{args.optim}-{args.lrstep}": test_acc}, epoch)
    writer.add_scalars("Loss/test", {f"{args.optim}-{args.lrstep}": test_loss}, epoch)
    
    # Save checkpoint.
    if test_acc > best_acc:
      print('Saving..')
      state = {
        'net': net.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
      }
      if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
      torch.save(state, os.path.join('checkpoint', ckpt_name))
      best_acc = test_acc
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    if not os.path.isdir('curve'):
      os.mkdir('curve')
    torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
               os.path.join('curve', ckpt_name))


if __name__ == '__main__':
  main()
