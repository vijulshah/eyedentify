#!/bin/bash

python ./eyedentify/training/train.py \
  --config_file="./eyedentify/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/SRResNet_x2/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold1"

python ./eyedentify/training/train.py \
  --config_file="./eyedentify/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/SRResNet_x2/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold2"

python ./eyedentify/training/train.py \
  --config_file="./eyedentify/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/SRResNet_x2/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold3"

python ./eyedentify/training/train.py \
  --config_file="./eyedentify/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/SRResNet_x2/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold4"

python ./eyedentify/training/train.py \
  --config_file="./eyedentify/configs/train.yml" \
  --data_path="data/EyeDentify/W_SR/SRResNet_x2/eyes/left_eyes" \
  --selected_target="left_pupil" \
  --registered_model_name="ResNet18" \
  --split_fold="fold5"