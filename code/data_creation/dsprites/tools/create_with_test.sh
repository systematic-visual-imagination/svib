#!/bin/sh

save_path=$1
rule=$2
num_train=$3
num_test=$4

i=0.2

train_ratio=0.8

while [ "`echo "$i <= $train_ratio + 0.1" | bc`" -eq "1" ]
do
    python3 data_creation/dsprites/create_with_test.py --save_path ${save_path}  --test_path "tmp/dsprites-${rule}/test" --train_ratio $i  --rule ${rule} --num_train ${num_train}
    i=`echo "0.2 + $i" | bc`
done

