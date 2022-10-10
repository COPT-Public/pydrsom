import torch


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
