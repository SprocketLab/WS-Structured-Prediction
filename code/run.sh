#!/bin/bash

resdir=results/ranking
mkdir -p ${resdir}

# NOTE set theta_star appropriately in main.py. 

# T recovery and expected squared distance recovery experiment
for seed in 0 1 2 3 4 5 6 7 8 9
do 
    for n in 100 500 1_000 5_000 10_000 50_000 100_000 500_000 1_000_000
    do 
        python -u main.py --seed ${seed} --n ${n} | tee ${resdir}/n${n}_seed${seed}.log
    done
done

# End model and label model experiment
mkdir -p ${resdir}/endmodel/
n=100_000
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for n in 200 400 600 800 1000 1200 1400 1600 1800 2000
    do
        python -u main.py --n ${n} --seed=${seed} | tee ${resdir}/endmodel/n${n}_seed${seed}.log
    done
done
