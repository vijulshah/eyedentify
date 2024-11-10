#!/bin/bash

base_command="python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/pl_training/pl_train.py \
  --config_file=/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/pl_train.yml \
  --data_path=data/EyeDentify/Wo_SR/eyes/right_eyes \
  --selected_targets right_pupil \
  --exp_name=resnet50_right_eyes_no_sr \
  --registered_model_name=ResNet50"

for participant in {1..51}; do
  full_command="$base_command --left_out_participants_for_val $participant --left_out_participants_for_test $participant"
  
  echo "Running: $full_command"
  $full_command
done
