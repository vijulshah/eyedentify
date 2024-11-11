#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="RegressionModelPro-bicubic_x4-left-pupil-fold1"
PARTITION="batch"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
GPUS_PER_TASK=1
MEM=32G

export LOGLEVEL=INFO

# --gpus-per-task=$GPUS_PER_TASK \
# --gpu-bind=none \

srun -K\
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --gpus-per-task=$GPUS_PER_TASK \
    --gpu-bind=none \
    --mem=$MEM \
    --container-image=/netscratch/$USER/eyedentify/scripts/pip_dependencies_pl.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time=01-00:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}-nodes_${NODES}-tasks_${NTASKS}-cpus_${CPUS_PER_TASK}-gpus_${GPUS_PER_TASK}.log" \
    python ./eyedentify/training/pt_training/pt_train.py \
        --config_file="./eyedentify/configs/pt_train.yml" \
        --split_fold="fold1"

