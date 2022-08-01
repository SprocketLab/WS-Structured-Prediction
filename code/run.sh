#!/bin/bash

resdir=results/ranking
mkdir -p ${resdir}

for seed in 3 4 5 6 7 8 9 # 0 1 2
do 
    for n in 100 500 1_000 5_000 10_000 50_000 100_000 500_000 1_000_000
    do 
        python -u main.py --seed ${seed} --n ${n} | tee ${resdir}/n${n}_seed${seed}.log
    done
done
