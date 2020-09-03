#!/bin/bash

for i in "50 0.0005" "e 5"
do
    set -- $i
    echo $1 and $2
done
