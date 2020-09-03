#!/bin/bash


for way in 0 -1 
do
    echo "$way" 
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_measurements.py -c configs_sched/study_diff_bes_transfer --ways-be $way  --comment diff_bes
done

