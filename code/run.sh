#!/bin/bash

resdir=results/ranking
mkdir -p ${resdir}

#for seed in 3 4 5 6 7 8 9 # 0 1 2
#do 
#    for n in 100 500 1_000 5_000 10_000 50_000 100_000 500_000 1_000_000
#    do 
#        python -u main.py --seed ${seed} --n ${n} | tee ${resdir}/n${n}_seed${seed}.log
#    done
#done

# mkdir -p ${resdir}/endmodel/
# n=100_000
# for seed in 0 1 2 3 4 5 6 7 8 9
# do
#     for t in 0.0 0.1 0.2 0.3 0.4 0.5
#     do
#         python -u main.py --n ${n} --theta_1 ${t} --seed=${seed} | tee ${resdir}/endmodel/n${n}_theta${t}_seed${seed}.log
#     done
# done

mkdir -p ${resdir}/endmodel/
n=100_000
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for n in 1000 1200 1400 1600 1800 2000  #100 200 300 400 500 600 700 800 900
    do
        python -u main.py --n ${n} --seed=${seed} | tee ${resdir}/endmodel/n${n}_seed${seed}.log
    done
done
