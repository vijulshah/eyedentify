#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="ResNetCifar56-depth-no-sr-left-iris-fold1"
PARTITION="batch"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
GPUS_PER_TASK=1
MEM=28G

export LOGLEVEL=INFO

srun -K\
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --gpus-per-task=$GPUS_PER_TASK \
    --gpu-bind=none \
    --mem=$MEM \
    --container-image=/netscratch/$USER/pupil-size-estimation-with-super-resolution/scripts/pip_dependencies_pl.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time=01-00:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}-nodes_${NODES}-tasks_${NTASKS}-cpus_${CPUS_PER_TASK}-gpus_${GPUS_PER_TASK}.log" \
    python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/pl_training/pl_train.py \
        --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/pl_train.yml" \
        --split_fold="fold1"
