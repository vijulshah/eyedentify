#!/bin/bash

# Define the base command
base_command="python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file='/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml' \
  --data_path='data/EyeDentify/Wo_SR/eyes/left_eyes' \
  --selected_target='left_pupil' \
  --registered_model_name='ResNet18'"

# Loop through fold1 to fold5
for fold in {1..5}; do
  # Construct the full command with the current fold
  full_command="$base_command --split_fold='fold$fold'"
  
  # Execute the command
  echo "Running: $full_command"
  $full_command
done
