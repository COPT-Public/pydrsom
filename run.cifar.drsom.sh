# resnet18
echo $'running CIFAR10\n'
read -rep $'Networks:' networks
read -rep $'DRSOM Direction Mode:\n\t0-grad+momentum\n\t3-grad alone\n' drsommode
read -rep $'DRSOM QP Mode:\n\t0-hvps\n\t1-interpolation\n' drsomqpmode
for net in $networks; do
  for md in $drsommode; do
    for qpmd in $drsomqpmode; do
#      for window in 4000; do
#        for step in 2e2 5e2 8e2 1e3; do
#          cmd="export DRSOM_MODE=$md; export DRSOM_MODE_QP=$qpmd; export DRSOM_MODE_HVP=0; export DRSOM_VERBOSE=0;
#             python -u -m demos.cifar10.main --model $net --optim drsom --epoch 80 --tflogger tf_logs/cifar
#             --drsom_decay_window $window --drsom_decay_step $step
#             &>cifar.$net.drsom.$md.$window-$step.log"
#          echo $cmd
#        done
#      done
      for window in 3000; do
        for step in 25 30 45 50 100; do
          cmd="export DRSOM_MODE=$md; export DRSOM_MODE_QP=$qpmd; export DRSOM_MODE_HVP=0; export DRSOM_VERBOSE=0;
             python -u -m demos.cifar10.main --model $net --optim drsom --epoch 80 --tflogger tf_logs/cifar
             --drsom_decay_window $window --drsom_decay_step $step
             &>cifar.$net.drsom.$md.$window-$step.log"
          echo $cmd
        done
      done
    done
  done
done
