#!/bin/bash

base_command="python ./eyedentify/training/pt_training/pt_train.py \
  --config_file=./eyedentify/configs/pt_train.yml \
  --data_path=data/EyeDentify/Wo_SR/eyes/right_eyes \
  --selected_targets right_pupil \
  --exp_name=resnet18_right_eyes_no_sr \
  --registered_model_name=ResNet18"

for fold in {1..5}; do
  full_command="$base_command --split_fold='fold$fold'"
  
  echo "Running: $full_command"
  $full_command
done
