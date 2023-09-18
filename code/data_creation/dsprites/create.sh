#!/bin/sh

save_path=$1
test_ratio=$2
rule=$3
num_train=$4
num_test=$5

python3 data_creation/dsprites/create_data.py --save_path ${save_path} --train_ratio 0.0 0.2 0.4 0.6 --test_ratio ${test_ratio} --rule ${rule} --num_train ${num_train:=64000} --num_test ${num_test:=12800}

