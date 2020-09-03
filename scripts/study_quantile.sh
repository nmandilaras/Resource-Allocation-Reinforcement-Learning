#!/bin/bash


for quantile in .95 
do
  for be in graphs
  do
    echo "$quantile" " $be"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_quantile -q $quantile --be-name $be --comment "$be"_q"$quantile"_quantile
  done
done



