#!/bin/bash
# put your install commands here (remove lines you don't need):
# make sure you run `chmod a+x install.sh` in cluster to give permission to this batch file
# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here (remove lines you don't need):
  apt update
  apt clean
  # pip install -r ../requirements.txt

  # If encountered issue with opencv lib, use one of the following fixes:

  # Fix-1.
  # pip install pytorch-lightning
  # pip install tqdm
  # pip install PyYAML
  # pip install numpy
  # pip install pandas
  # pip install matplotlib
  # pip install seaborn
  # pip install mlflow
  # pip install Pillow
  # pip install scikit_learn
  # pip install torch
  # pip install captum
  # pip install evaluate
  # pip install basicsr
  # pip install facexlib
  # pip install realesrgan
  # pip install opencv_python
  # pip install cmake
  # pip install dlib
  # pip install einops
  # pip install transformers
  # pip install gfpgan
  # pip install streamlit
  # pip install mediapipe
  # pip install imutils
  # pip install scipy
  # pip install torchvision==0.16.0
  # pip install torchcam
  # pip install cmake
  # pip install dlib
  # pip uninstall opencv-python
  # pip uninstall opencv-contrib-python
  # pip uninstall opencv-contrib-python-headless
  # pip install opencv-python==4.5.5.64
  # pip install opencv-contrib-python==4.5.5.64
  # pip install opencv-python-headless==4.5.5.64
  # pip install protobuf==3.20.*
  # pip install onnx2torch

  # OR
  
  # Fix-2.
  # pip install opencv-fixer==0.2.5
  # python3 -c "from opencv_fixer import AutoFix; AutoFix()"
  # pip3 install opencv-python --upgrade

  apt-get clean
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi