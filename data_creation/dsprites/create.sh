#!/bin/sh

rule=$1
save_path=$2

train_ratio=$3
test_ratio=$4

num_train=$5
num_test=$6

python3 data_creation/dsprites/create_data.py --rule ${rule} --save_path ${save_path} --train_ratio ${train_ratio:="0.0 0.2 0.4 0.6"} --test_ratio ${test_ratio:="0.2"} --num_train ${num_train:=64000} --num_test ${num_test:=8000}

