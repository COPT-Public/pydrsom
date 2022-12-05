# resnet18
echo $'running CIFAR10\n'
read -rep $'Networks:' networks
read -rep $'DRSOM Direction Mode:\n\t0-grad+momentum\n\t3-grad alone\n' drsommode
read -rep $'DRSOM QP Mode:\n\t0-hvps\n\t1-interpolation\n' drsomqpmode
read -rep $'completing algorithms: (choices: sgd adam adagrad)\n' otheralgs
for net in $networks; do
  for md in $drsommode; do
    for qpmd in $drsomqpmode; do
      cmd="export DRSOM_MODE=$md; export DRSOM_MODE_QP=$qpmd; export DRSOM_MODE_HVP=0; export DRSOM_VERBOSE=0;
         python -u -m demos.cifar10.main --model $net --optim drsom --epoch 80 --tflogger tf_logs/cifar &>cifar.$net.drsom.$md.log"
      echo $cmd
    done
  done

  for opt in $otheralgs; do
    cmd="python -u -m demos.cifar10.main --model $net --optim $opt --epoch 80 --tflogger tf_logs/cifar &>cifar.$net.$opt.log"
    echo $cmd
  done
done
