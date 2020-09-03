#!/bin/bash


for be in RandomForestClassifier #DecisionTreeClassification GradientBoostedTreeRegressor LinearSVC RandomForestClassifier astar hmmer LogisticRegressionWithElasticNet
do
    echo " $be"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_coex --be-name $be --comment "$be"_coex_agent
done

