# resnet18
for net in resnet18 resnet34; do
  for md in 3 0; do
    export DRSOM_MODE=$md
    export DRSOM_MODE_QP=1
    export DRSOM_MODE_HVP=0
    export DRSOM_VERBOSE=0
    python -u -m demos.cifar10.main --model $net --optim drsom --epoch 80 --tflogger tf_logs/cifar &>cifar.$net.drsom.log
  done

  for opt in adam sgd; do
    export DRSOM_MODE=3
    export DRSOM_MODE_HVP=0
    export DRSOM_VERBOSE=0
    python -u -m demos.cifar10.main --model $net --optim $opt --epoch 80 --tflogger tf_logs/cifar &>cifar.$net.$opt.log
  done
done
