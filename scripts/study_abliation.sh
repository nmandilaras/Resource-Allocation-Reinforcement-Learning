#!/bin/bash


  for be in in-memory
  do
    echo " $be"
    #time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_abliation_all --be-name $be --comment "$be"_abliation_all
    #time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_abliation_noper --be-name $be --comment "$be"_abliation_noper
    #time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_abliation_nodueling --be-name $be --comment "$be"_abliation_nodueling
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_abliation_nodouble --be-name $be --comment "$be"_ababliation_nodouble
  done



