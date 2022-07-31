"""
A quickstart script for DRSOM on Fashion-MNIST dataset.
@author: Chuwen Zhang
@note:
  This script runs DRSOM and compares to Adam, SGD, and so forth.

    usage: quickstart.py [-h] --model {cnn,simple} --optim {drsom,adam,sgd4} [--data_size DATA_SIZE] [--batch BATCH]
                         [--epoch EPOCH] [--interval INTERVAL] [--itermax ITERMAX] [--bool_decay BOOL_DECAY]
                         [--option_tr {a,p}] [--hessian_window HESSIAN_WINDOW] [--theta1 THETA1] [--theta2 THETA2]
                         [--tflogger TFLOGGER]
    
    optional arguments:
      -h, --help            show this help message and exit
      --model {cnn,simple}
      --optim {drsom,adam,sgd4}
      --data_size DATA_SIZE
      --batch BATCH
      --epoch EPOCH
      --interval INTERVAL
      --itermax ITERMAX
      --bool_decay BOOL_DECAY
      --option_tr {a,p}
      --hessian_window HESSIAN_WINDOW
      --theta1 THETA1       DRSOM coefficients theta_1 (default: 50)
      --theta2 THETA2       DRSOM coefficients theta_2 (default: 30)
      --tflogger TFLOGGER   tf logger (default: run)
      
    You can use an easy/complex model by option --model {simple,cnn}
"""

import argparse
import json
import os
import time
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pydrsom.drsom import DRSOMF

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", required=True, type=str, default='simple', choices=['cnn', 'simple'])
parser.add_argument("--optim", required=True, type=str, default='simple', choices=['drsom', 'adam', 'sgd4'])
parser.add_argument("--data_size", required=False, type=int, default=int(1e3))
parser.add_argument("--batch", required=False, type=int, default=128)
parser.add_argument("--epoch", required=False, type=int, default=5)
parser.add_argument("--interval", required=False, type=int, default=20)
parser.add_argument("--itermax", required=False, type=int, default=1)
parser.add_argument("--bool_decay", required=False, type=int, default=1)
parser.add_argument("--option_tr",
                    required=False,
                    type=str,
                    default='p',
                    choices=['a', 'p'])
parser.add_argument("--hessian_window", required=False, type=int, default=1)
parser.add_argument('--theta1',
                    default=50,
                    type=float,
                    help='DRSOM coefficients theta_1')
parser.add_argument('--theta2',
                    default=30,
                    type=float,
                    help='DRSOM coefficients theta_2')
parser.add_argument('--tflogger', default="run", type=str, help='tf logger')


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1,
                              out_channels=32,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=32,
                              out_channels=64,
                              kernel_size=3,
                              stride=1,
                              padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fcrelu = nn.ReLU()

        self.fc2 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)

        out = self.fc2(out)
        return out


class VanillaNetwork(nn.Module):
    def __init__(self):
        super(VanillaNetwork, self).__init__()
        self.flatten = nn.Flatten()

        # case II: seems ok:
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


def train(dataloader, name, model, loss_fn, optimizer, ninterval):
    st = time.time()
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total = 0
    avg_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        def closure(backward=True):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            if not backward:
                return loss
            if name.startswith('drsom'):
                loss.backward(create_graph=True)
            else:
                loss.backward()
            return loss

        # backpropagation
        loss = optimizer.step(closure=closure)
        avg_loss += loss.item()

        # compute prediction error
        outputs = model(X)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        if batch % ninterval == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    accuracy = 100. * correct / total
    avg_loss = avg_loss / len(dataloader)
    print('train acc %.3f' % accuracy)
    print('train avg_loss %.3f' % avg_loss)
    print('train batches: ', len(dataloader))

    et = time.time()
    return et - st, avg_loss, accuracy


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            test_loss += loss_fn(outputs, y).item()
            correct += predicted.eq(y).sum().item()
    test_loss /= num_batches
    correct /= size
    rstring = f"Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    print(rstring)
    result = {'acc': (100 * correct), 'avg_loss': test_loss}
    return result


if __name__ == '__main__':

    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.tflogger)
    # reproducibility
    # download training data from open datasets.
    try:
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=False,
            transform=ToTensor(),
        )
        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=False,
            transform=ToTensor(),
        )
    except:
        # not exists
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    training_data.data = training_data.data[:args.data_size]
    nbatch = args.batch
    nepoch = args.epoch
    ninterval = args.interval
    itermax = args.itermax
    option_tr = args.option_tr
    hessian_window = args.hessian_window
    betas = (0.96, 0.99) if args.bool_decay else (0, 0)

    # get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    loss_fn = F.cross_entropy
    methods = {
        'adam': torch.optim.Adam,
        'sgd1': torch.optim.SGD,
        'sgd2': torch.optim.SGD,
        'sgd3': torch.optim.SGD,
        'sgd4': torch.optim.SGD,
        'lbfgs': torch.optim.LBFGS,
        'drsom': DRSOMF,
    }
    method_kwargs = {
        'adam':
        dict(lr=0.001, betas=(0.99, 0.999)),
        'sgd1':
        dict(lr=0.001, momentum=0.95),
        'sgd2':
        dict(lr=0.001, momentum=0.90),
        'sgd3':
        dict(lr=0.001, momentum=0.85),
        'sgd4':
        dict(lr=0.001, momentum=0.99),
        'lbfgs':
        dict(line_search_fn='strong_wolfe', max_iter=5),
        'drsom':
        dict(max_iter=itermax,
             betas=betas,
             option_tr='a',
             hessian_window=hessian_window),
    }

    pprint(method_kwargs)

    results = {}
    name = args.optim
    # model
    print("{:^10s}".format(name))
    if args.model == 'cnn':
        model = CNNModel().to(device)
    elif args.model == 'simple':
        model = VanillaNetwork().to(device)
    else:
        raise ValueError(f"unknown model {args.model}")
    # create data loaders.
    train_dataloader = DataLoader(
        training_data,
        batch_size=nbatch,
        generator=torch.Generator().manual_seed(1))
    test_dataloader = DataLoader(
        test_data,
        batch_size=nbatch,
        generator=torch.Generator().manual_seed(1))
    # start
    st = time.time()
    func = methods[name]
    func_kwargs = method_kwargs.get(name, {})
    optimizer = func(model.parameters(), **func_kwargs)
    for t in range(nepoch):
        print(f"epoch {t}")
        _, avg_loss, acc = train(train_dataloader, name, model, loss_fn,
                                 optimizer, ninterval)
        if name in {'drsom'} and (t + 1) % 5 == 0:
            optimizer.lmblb *= 1e3
        print("-------------------------------")
        # train loss
        writer.add_scalars("Loss/train", {f'{name}': avg_loss}, t)
        writer.add_scalars("Accuracy/train", {f'{name}': acc}, t)
        # test loss
        rt = test(test_dataloader, model, loss_fn)
        writer.add_scalars("Loss/test", {f"{name}": rt['avg_loss']}, t)
        writer.add_scalars("Accuracy/test", {f"{name}": rt['acc']}, t)

    et = time.time()

    print("done!")
    subresult = {}
    subresult['info_train'] = test(train_dataloader, model, loss_fn)
    subresult['info_test'] = test(test_dataloader, model, loss_fn)
    subresult['time_train'] = et - st
    results[name] = subresult
    print(subresult)
    del model
    torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))
