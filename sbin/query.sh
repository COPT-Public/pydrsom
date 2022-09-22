for cat in Loss Acc; do
	for dat in train test; do
#		 for f in adam sgd1 sgd2 sgd3 sgd4 rsomf; do
#		for f in adam-30 adam-40 rsomf-100 rsomf-200 rsomf-500; do
		for f in adam sgd4 \
  		'drsom-mode:0-p' \
  		'drsom-mode:1-p' \
  		'drsom-mode:3-p' \
  		'drsom-mode:3-p-r-20' \
  		; do
			cmd="curl http://localhost:6006/data/plugin/scalars/scalars\?tag\=$cat%2F$dat\&run\=${cat}_${dat}_${f}\&format\=csv > $1/${cat}_${dat}_${f}.csv";
			echo $cmd;
			eval $cmd;
done;
done;
done

