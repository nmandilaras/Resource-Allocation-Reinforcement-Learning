#!/bin/bash


for way in -1 
do
  for be in RandomForestClassifier # hmmer astar LogisticRegressionWithElasticNet DecisionTreeClassification LinearSVC in-memory graphs GradientBoostedTreeRegressor
  do
    echo "$way" " $be"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_measurements.py -c configs_sched/study_measurement --ways-be $way --be-name $be --comment "$be"_coex_alone
  done
done

