#!/bin/bash


for i in {0..18}
do
  time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_measurements.py -c configs_sched/measurement_lc --ways-be $i --warm-up 150 --comment new_lc_profiling
done
    
