#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="tobii_and_webcam_data_alignment"
PARTITION="batch"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
MEM=30G

srun -K\
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEM \
    --container-image=/netscratch/$USER/pupil-size-estimation-with-super-resolution/scripts/pip_dependencies.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --time=12:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}-nodes_${NODES}-tasks_${NTASKS}-cpus_${CPUS_PER_TASK}.log" \
    python /netscratch/shah/pupil-size-estimation-with-super-resolution/data_creation/eyedentify/run_data_alignment.py \
        --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/tobii_and_webcam_data_alignment.yml" 
