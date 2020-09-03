#!/bin/bash


time taskset --cpu-list 18-19 python ~/reinforcement-learning/main_rdt.py -c configs_sched/study_diff_bes_transfer --comment transfer_train_on_5Noflush_diff_bes

