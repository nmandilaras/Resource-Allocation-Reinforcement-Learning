#!/bin/bash


time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_transfer_train --comment transfer_train_on_5Noflush

