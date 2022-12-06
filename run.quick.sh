# resnet18

for md in 3; do
  export DRSOM_MODE=$md
  export DRSOM_MODE_QP=1
  export DRSOM_MODE_HVP=0
  export DRSOM_VERBOSE=0
  python -u quickstart.py --model cnn --optim drsom --epoch 30 --data_size 100000000 --tflogger tf_logs/quick &>quick.cnn.drsom.$md.log
done

for opt in adam sgd1 sgd2 sgd3 sgd4; do
  export DRSOM_MODE=3
  export DRSOM_MODE_HVP=0
  export DRSOM_VERBOSE=0
  python -u quickstart.py --model cnn --optim $opt --epoch 30 --data_size 100000000 --tflogger tf_logs/quick &>quick.cnn.$opt.log
done
