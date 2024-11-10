#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="ResNet50-no-sr-right-eyes-loocv"
PARTITION="A100-40GB"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
GPUS_PER_TASK=1
MEM=40G

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
    --time=03-00:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}-nodes_${NODES}-tasks_${NTASKS}-cpus_${CPUS_PER_TASK}-gpus_${GPUS_PER_TASK}.log" \
    loocv/No_SR/ResNet50_right.sh
