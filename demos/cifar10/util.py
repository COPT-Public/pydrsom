import argparse

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pydrsom.drsom_utils import *
from pydrsom.drsom import DRSOMB as DRSOM
from pydrsom.drsom_utils import add_parser_options, DRSOMDecayRules
from pydrsom.drsomk import DRSOMK
from .models import *


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
  
  parser.add_argument('--tflogger', default="/tmp/", type=str, help='tf logger directory')
  ##############
  # sgd & adam
  ##############
  parser.add_argument('--momentum', default=0.95, type=float, help='momentum term')
  parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
  parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
  parser.add_argument('--ckpt_name', type=str, help='resume from checkpoint')
  parser.add_argument('--epoch', '-e', default=200, type=int, help='num of epoches to run')
  parser.add_argument('--weight_decay', default=5e-4, type=float,
                      help='weight decay for optimizers')
  ##############
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
  ##############
  # drsom
  ##############
  add_parser_options(parser)
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
  train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True,
                            num_workers=2)
  
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=transform_test)
  test_loader = DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)
  
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
    drsom_decay_rules = DRSOMDecayRules(
      decay_window=args.drsom_decay_window,
      decay_step=args.drsom_decay_step,
      decay_radius_step=args.drsom_decay_radius_step,
      decay_sin_rel_max=args.drsom_decay_sin_rel_max,
      decay_sin_min_scope=args.drsom_decay_sin_min_scope,
      decay_sin_max_scope=args.drsom_decay_sin_max_scope
    )
    return DRSOM(
      model_params,
      option_tr=args.option_tr,
      beta1=args.drsom_beta1,
      beta2=args.drsom_beta2,
      max_iter=args.itermax,
      decayrules=drsom_decay_rules
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


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3):
  """
  get checkpoint name for optimizers except for DRSOM.
  Args:
    model:
    optimizer:
    lr:
    final_lr:
    momentum:
    beta1:
    beta2:
    gamma:

  Returns:

  """
  name = {
    'sgd': 'lr{}-momentum{}'.format(lr, momentum),
    'adagrad': 'lr{}'.format(lr),
    'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
    'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
    'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
  }[optimizer]
  return '[{}]-{}-{}'.format(model, optimizer, name)


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
      if name.startswith('drsom'):
        if DRSOM_MODE_QP == 0 or DRSOM_VERBOSE == 1:
          # only need for hvp
          loss.backward(create_graph=True)
        else:
          loss.backward()
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
  print('train acc  %.3f' % accuracy)
  print('train loss %.3f' % (train_loss / len(data_loader)))
  
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
  print(' test acc  %.3f' % accuracy)
  print(' test loss %.3f' % (test_loss / len(data_loader)))
  return accuracy, test_loss / len(data_loader)
