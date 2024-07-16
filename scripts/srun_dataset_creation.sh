#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="create-dataset-wo-sr"
PARTITION="RTXA6000"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
GPUS_PER_TASK=1
MEM=50G

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
    --container-image=/netscratch/$USER/eyedentify/scripts/pip_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time=3-00:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}.log" \
    python /netscratch/$USER/eyedentify/preprocessing/dataset_creation.py \
        --config_file="/netscratch/$USER/eyedentify/configs/dataset_creation.yml"
