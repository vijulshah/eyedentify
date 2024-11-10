#!/bin/bash

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/RealEsr_x4/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet152" \
  --split_fold="fold1"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/RealEsr_x4/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet152" \
  --split_fold="fold2"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/RealEsr_x4/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet152" \
  --split_fold="fold3"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/RealEsr_x4/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet152" \
  --split_fold="fold4"

python /netscratch/shah/pupil-size-estimation-with-super-resolution/training/train.py \
  --config_file="/netscratch/shah/pupil-size-estimation-with-super-resolution/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/RealEsr_x4/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet152" \
  --split_fold="fold5"