#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="create-dataset-celeba_eyes-x2"
PARTITION="A100-40GB"
NODES=1
NTASKS=1
CPUS_PER_TASK=2
GPUS_PER_TASK=0
MEM=32G

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
    --container-image=/netscratch/$USER/pupil-size-estimation-with-super-resolution/scripts/pip_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time=20:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}.log" \
    python /netscratch/shah/pupil-size-estimation-with-super-resolution/data_creatioon/celeba_ffhq/ds_creation.py \
        --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/celeba_ffhq_ds_creation.yml"
