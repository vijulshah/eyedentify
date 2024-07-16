#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="model-interpretation"
PARTITION="V100-16GB"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
GPUS_PER_TASK=1
MEM=50G

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
    --time=12:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}.log" \
    python /netscratch/$USER/eyedentify/inference_n_interpretation/xai.py \
        --config_file="/netscratch/$USER/eyedentify/configs/xai.yml" 