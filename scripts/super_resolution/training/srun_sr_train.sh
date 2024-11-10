#!/bin/bash

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="celeba-ffhq-hat-train-x2"
PARTITION="RTXA6000"
NODES=2
NTASKS=2
CPUS_PER_TASK=4
GPUS_PER_TASK=1
MEM=22G
LAUNCHER="pytorch"

# NODES=8
# NTASKS=8
# CPUS_PER_TASK=4
# GPUS_PER_TASK=1
# MEM=22G

# Current: V100-32GB
# 2 machines, 4 processes, 4 gpus, 8 batch size

# Next try: A100-40GB (has 4 nodes available) or A100-80GB (has 4 nodes available) or RTXA6000 (has 8 nodes available)
# 4 machines, 8 processes (2 processes per machine), 8 gpus (1 gpu per process), 4 batch size
# OR
# 1 machine, 8 processes (8 processes per machine), 8 gpus (1 gpu per process), 4 batch size
# OR
# 8 machine, 8 processes (2 process per machine), 8 gpus (1 gpu per process), 4 batch size

export LOGLEVEL=INFO
# export NCCL_DEBUG=INFO

srun -K\
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --gpus-per-task=$GPUS_PER_TASK \
    --gpu-bind=none \
    --mem=$MEM \
    --container-image=/netscratch/$USER/HAT/scripts/pip_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time=1-00:00:00 \
    --output="./logs/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/shah/pupil-size-estimation-with-super-resolution/super_resolution/training/HAT/hat/train.py \
        -opt="/netscratch/shah/pupil-size-estimation-with-super-resolution/super_resolution/training/HAT/options/train/train_HAT_SRx2_from_scratch.yml" \
        --launcher=$LAUNCHER

# --task-prolog="`pwd`/install.sh" 
# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh # System available images
# --container-save : to save an image after installing dependencies in an existing image, use like this:
# --container-image=/netscratch/$USER/HAT/scripts/pip_dependencies.sqsh \

# To see available images:
# cd /netscratch/enroot/
# ls nvcr.io_nvidia_pytorch_* -l

# Run in interractive mode to use debugging:
# --pty /bin/bash \

# execution commands:
# --mem-per-cpu=5G \
# python -u /netscratch/shah/HAT/hat/train.py
# python -m torch.distributed.launch /netscratch/shah/HAT/hat/train.py
# torchrun /netscratch/shah/HAT/hat/train.py
# --launcher="pytorch" # "pytorch" or "slurm"

# https://slurm.schedmd.com/sbatch.html