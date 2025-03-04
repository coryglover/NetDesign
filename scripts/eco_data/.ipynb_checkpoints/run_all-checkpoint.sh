#!/bin/sh
for i in $(seq 0 7);
do
    sbatch run_data.sh $i
done