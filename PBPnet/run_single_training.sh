#!/bin/bash

# Script to runa single python kernel
LOG_LOCATION=/data/mariani/specificity_bpnet/logs/training_logs

training_script="/data/mariani/specificity_bpnet/code/bpnet_train.py"

source /data/mariani/bpnet_lite/bin/activate

dict_path='/data/mariani/specificity_bpnet/config_dict'

f='/data/mariani/specificity_bpnet/config_dict/train_dict_alpha_10.pkl'
echo "Traingin BPNet with $f dict file"
python $training_script --dict_file "$f" 