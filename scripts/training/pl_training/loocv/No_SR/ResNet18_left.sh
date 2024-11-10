#!/bin/bash

base_command="python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/pl_training/pl_train.py \
  --config_file=/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/pl_train.yml \
  --data_path=data/EyeDentify/Wo_SR/eyes/left_eyes \
  --selected_targets left_pupil \
  --exp_name=resnet18_left_eyes_no_sr \
  --registered_model_name=ResNet18"

for participant in {1..51}; do
  full_command="$base_command --left_out_participants_for_val $participant --left_out_participants_for_test $participant"
  
  echo "Running: $full_command"
  $full_command
done
