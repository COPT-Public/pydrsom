runname=$1
for cat in Loss Acc; do
  for dat in train test; do
    #		 for f in adam sgd1 sgd2 sgd3 sgd4 rsomf; do
    #		for f in adam-30 adam-40 rsomf-100 rsomf-200 rsomf-500; do
    for f in \
      adam \
      sgd1 \
      sgd2 \
      sgd3 \
      sgd4 \
      'drsom-gd' \
      'drsom-g'; do
      cmd="curl \"http://localhost:6006/data/plugin/scalars/scalars?tag=$cat/$dat&run=$runname/${f}&format=csv\" -o $2/${cat}_${dat}_${f}.csv"
      echo $cmd;
#      eval $cmd;
    done
  done
done
