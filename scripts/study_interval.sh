#!/bin/bash


for interval in "200 0.002"
do
  set -- $interval
  for be in graphs
  do
    echo "$1" " $be" " $2"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_interval -d $2 -t $1 --be-name $be --comment "$be"_"$1"ms_"$2"decay_interval
  done
done

