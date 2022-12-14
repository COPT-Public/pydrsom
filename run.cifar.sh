# resnet18
echo $'running CIFAR10\n'
read -rep $'Networks:' networks
read -rep $'DRSOM QP Mode:\n\t0-hvps\n\t1-interpolation\n' drsomqpmode
read -rep $'completing algorithms: (choices: sgd adam adagrad)\n' otheralgs
for net in $networks; do
  cmd="
      export DRSOM_MODE_DECAY=1;
      export DRSOM_MODE_DELTA=1;
      export DRSOM_MODE=0;
      export DRSOM_MODE_QP=$drsomqpmode;
      export DRSOM_MODE_HVP=0;
      export DRSOM_VERBOSE=0;
      python -u -m demos.cifar10.main --model $net --optim drsom \
      --epoch 80 --tflogger tf_logs/cifar
      --drsom_adjust_beta1 20 --option_tr a
             &>cifar.$net.drsom2d.log"
  echo $cmd
  cmd="
      export DRSOM_MODE_DECAY=0;
      export DRSOM_MODE_DELTA=1;
      export DRSOM_MODE=3;
      export DRSOM_MODE_QP=$drsomqpmode;
      export DRSOM_MODE_HVP=0;
      export DRSOM_VERBOSE=0;
      python -u -m demos.cifar10.main --model $net --optim drsom \
        --epoch 80 --tflogger tf_logs/cifar
             &>cifar.$net.drsom1d.log"
  echo $cmd

  for opt in $otheralgs; do
    cmd="python -u -m demos.cifar10.main --model $net --optim $opt \
        --lr 0.001 --lrstep 20 \
        --epoch 80 --tflogger tf_logs/cifar &>cifar.$net.$opt.log"
    echo $cmd
  done
done
