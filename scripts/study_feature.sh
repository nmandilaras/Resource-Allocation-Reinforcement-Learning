#!/bin/bash


for feature in MPKC MPKI Bandwidth IPC
do
  for be in in-memory graphs
  do
    echo "$feature" " $be"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_feature -f $feature --be-name $be --comment "$be"_"$feature"_feature
  done
done



