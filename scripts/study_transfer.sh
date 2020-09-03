#!/bin/bash


# TODO na pairnei to checkpoint ws parameter
 
for be in astar in-memory RandomForestClassifier hmmer
do
    echo " $be"
    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_transfer --be-name $be --comment "$be"_transfer_from5NoFlush
done

#for be in astar in-memory graphs RandomForestClassifier
#  do
#    echo " $be"
#    time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_transfer_from4 --be-name $be --comment "$be"_transfer_from4
#done

