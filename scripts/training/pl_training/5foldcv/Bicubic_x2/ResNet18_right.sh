#!/bin/bash

# Define the base command
base_command="python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/pl_training/pl_train.py \
  --config_file=/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/pl_train.yml \
  --data_path=data/EyeDentify/Wo_SR/eyes/left_eyes \
  --selected_targets left_pupil \
  --registered_model_name=ResNet18 \
  --img_size 32 64"

# Loop through participants 1 to 51
for fold in {1..5}; do
  # Construct the full command with dynamic participant values
  full_command="$base_command --split_fold $fold"
  
  # Execute the command
  echo "Running: $full_command"
  $full_command
done


python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/Wo_SR/eyes/right_eyes" \
  --selected_target="right_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold2"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/Wo_SR/eyes/right_eyes" \
  --selected_target="right_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold3"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/Wo_SR/eyes/right_eyes" \
  --selected_target="right_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold4"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/Wo_SR/eyes/right_eyes" \
  --selected_target="right_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold5"