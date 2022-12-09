# final script to run quickstart
echo $'running QUICKSTART for fashion MNIST\n'

read -rep $'repeating seeds' seeds
for seed in $seeds; do

  export DRSOM_MODE=3
  export DRSOM_MODE_QP=1
  export DRSOM_MODE_HVP=0
  export DRSOM_VERBOSE=0
  python -u quickstart.py \
    --model cnn --seed $seed --optim drsom --epoch 40 --data_size 100000000 \
    --tflogger tf_logs/quick &>quick.cnn.drsom.3.log

  export DRSOM_MODE=0
  export DRSOM_MODE_QP=1
  export DRSOM_MODE_HVP=0
  export DRSOM_VERBOSE=0
  python -u quickstart.py \
    --model cnn --seed $seed --optim drsom --epoch 40 --data_size 100000000 \
    --option_tr a \
    --drsom_adjust_beta1 20 \
    --tflogger tf_logs/quick &>quick.cnn.drsom.0.log

  for opt in adam sgd1 sgd2 sgd3 sgd4; do
    python -u quickstart.py \
      --model cnn --seed $seed --optim $opt --epoch 60 --data_size 100000000 \
      --tflogger tf_logs/quick &>quick.cnn.$opt.log
  done
done
