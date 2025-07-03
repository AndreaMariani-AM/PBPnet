#!/bin/bash

# Script to run parallel python kernels for concurrent training of different models
LOG_LOCATION=/data/mariani/specificity_bpnet/logs/training_logs

training_script="/data/mariani/specificity_bpnet/code/bpnet_train.py"

source /data/mariani/bpnet_lite/bin/activate

dict_path='/data/mariani/specificity_bpnet/config_dict'


for f in $dict_path/*.pkl; do
	if [[ $f == *n_layers* ]]; then
		echo "Traingin BPNet with $f dict file"
		python $training_script --dict_file "$f" &
	fi
done

wait