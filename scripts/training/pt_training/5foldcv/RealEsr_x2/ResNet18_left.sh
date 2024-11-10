#!/bin/bash

base_command="python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/pt_training/pt_train.py \
  --config_file=/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/pt_train.yml \
  --data_path=data/EyeDentify/W_SR/RealEsr_x2/eyes/left_eyes \
  --selected_targets left_pupil \
  --exp_name=resnet18_left_eyes_realesr_x2 \
  --registered_model_name=ResNet18"

for fold in {1..5}; do
  full_command="$base_command --split_fold='fold$fold'"
  
  echo "Running: $full_command"
  $full_command
done
