for opt in adam sgd;do
  export DRSOM_MODE=3; export DRSOM_MODE_HVP=0; export DRSOM_VERBOSE=0;
  python -u -m demos.cifar10.main  --model resnet18 --optim $opt --epoch 50 --tflogger cifar &> cifar.resnet18.$opt.log
done

# 2d DRSOM
export DRSOM_MODE=0; export DRSOM_MODE_HVP=0; export DRSOM_VERBOSE=0; python -u -m demos.cifar10.main  --model resnet18 --optim drsom --epoch 50 --tflogger cifar &> cifar.resnet18.drsom.log

# 1d DRSOM block
export DRSOM_MODE=3; export DRSOM_MODE_HVP=0; export DRSOM_VERBOSE=0; python -u -m demos.cifar10.main  --model resnet18 --optim drsomk --epoch 50 --tflogger cifar &> cifar.resnet18.drsomk.log
