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

import json

from .util import *

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
