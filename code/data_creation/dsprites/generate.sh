#!/bin/sh

save_path=$1
test_ratio=$2
rule=$3
num_train=$4
num_test=$5

python3 data_creation/dsprites/create.py --save_path ${save_path} --test_ratio ${test_ratio} --rule ${rule} --num_train ${num_train:=64000} --num_test ${num_test:=12800}

i=0.2

train_ratio=`echo "1.0 - $test_ratio" | bc`

while [ "`echo "$i <= $train_ratio + 0.1" | bc`" -eq "1" ]
do
    python3 data_creation/dsprites/piece.py --save_path ${save_path}  --test_path "${save_path}/dsprites-${rule}-alpha-0.0/test" --train_ratio $i  --rule ${rule} --num_train ${num_train}
    i=`echo "0.2 + $i" | bc`
done

