#!/bin/bash


for quantile in .99
do
  for be in graphs
  do
    echo "$quantile" " $be"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_coeff -q $quantile --be-name $be --comment "$be"_q"$quantile"_coeff4
  done
done



