# `EyeDentify`: A Webcam Based Pupil Diameter Estimation Dataset ✨

<div style="display: flex; align-items: center; gap: 20px;">
    <a href="https://www.kaggle.com/datasets/vijuls/PupilDiameterDatasets" style="text-decoration: none; background: linear-gradient(to right, #5D5D5D 64%, #0274B4 46%); padding-left: 5px; padding-right: 5px; border-radius: 3px;">
    <span style="font-size: 12px; color: white;">🗂️ EyeDentify Dataset</span>
</a>
    |
    <a href="https://huggingface.co/spaces/vijulshah/pupilsense">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Huggingface Spaces" style="margin-top: 5px;">
    </a>
</div>

<br>

Welcome to the official repository of the EyeDentify: A Webcam Based Pupil Diameter Estimation Dataset! 

It contains code for:
1) Creation of the Dataset: `EyeDentify`.
2) Training Pipelines for Pupil Diameter Estimation.
3) A link to the WebApp: `PupilSense`

## Table of Contents 📒 
- [Getting Started](#getting-started)
- [Project Stages](#project-stages)
  - [1. Data Collection App: `Chemeleon View`](#1-data-collection)
  - [2. Data Creation and Processing: `EyeDentify`](#2-data-creation-and-processing)
  - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
  - [4. WebApp: `PupilSense`](#4-webapp)
- [Citations](#🌏-citations)
- [Contact](#📧-contact)

> **Note**: The project supports execution using SLURM workload manager. You can modify the scripts in the `./scripts` folder to match your preferred environment.

---

# Getting Started 🚀
Clone the repository:
```bash
git clone https://github.com/vijulshah/eyedentify.git
cd eyedentify
```

# Data Collection App: `Chemeleon View` 🖥️

Our custom-built React app, **Chemeleon View**, facilitates video data collection through a simple interface where participants click a central button to start three-second recordings. Each recording session produces a timestamp file (`<session_id>.csv`, as shown in the figure below) that helps synchronize with Tobii eye-tracker data for precise pupil diameter measurements.

- **Interface**: A central button initiates a 3-second webcam recording.
- **Synchronization**: The timestamps are key to aligning webcam and Tobii eye-tracker data.
- **Session Setup**: Each participant completes 50 sessions, with screen colors varying to evoke different pupil reactions. The first ten recordings are done with a white background (⬜), followed by 40 recordings alternating between black (⬛), red (🟥), blue (🟦), yellow (🟨), green (🟩), and gray (🌫️). Each color appears five times consecutively. The last ten recordings return to a white background (⬜).

**Steps to run the app**:
1. Navigate to `./data_collection/`.
2. Run `npm install` to install the dependencies.
3. Start the app with `npm start`.

# 2. Data Creation and Processing: `EyeDentify` 👀

This stage includes 2 parts:

#### Aligning Recordings and Frame Extraction

This stage involves synchronizing webcam and Tobii data and extracting recording frames.

- Align Tobii data and timestamp files (`<session_id>.csv`) to match the frame rate differences (Tobii at 90 Hz vs. webcam at 30 fps) as show in the figure below, ensuring accurate pupil diameter measurements.
- Extract image frames corresponding to each synchronized data point.

    **Relevant files**:
    - Configuration: `./configs/tobii_and_webcam_data_alignment.yml`
    - Python File: `./data_creation/eyedentify/run_data_alignment.py`
    - Execution: `./scripts/data_creation/srun_tobii_and_webcam_data_processing.sh`

#### Eye Cropping and Blink Detection

- **Eye Cropping** using [Mediapipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md) landmarks for focused analysis.
- **Blink Detection** using [Eye Aspect Ratio (EAR)](https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706) and a [pretrained ViT](https://huggingface.co/dima806/closed_eyes_image_detection) to ensure data quality.

    **Relevant files**:
    - Configuration: `./configs/eyedentify_ds_creation.yml`
    - Python File: `./data_creation/eyedentify/ds_creation.py`
    - Execution: `./scripts/data_creation/eyedentify_ds_creation.sh`

# 3. Model Training and Evaluation ⚙️

This section details on training and evaluation. You have the option to execute scripts that can run on:
1) pytorch with DDP support 
2) pytorch-lightning 

The training strategies incude:
(A) Val/Test split Cross-Validation
(B) Leave One Participant Out Cross Validation (LOPOCV)

**Relevant files**:
- Configuration:
    - For pytorch with DDP: `./configs/pt_train.yml`
    - For pytorch-lightning: `./configs/pl_train.yml`
- Python File: 
    - For pytorch with DDP: `./training/pt_training/pt_train.py`
    - For pytorch-lightning: `./training/pl_training/pl_train.py` 
- Execution:
    - For pytorch with DDP: 
        - Single Run: `./scripts/training/pt_training/srun_train_single_exp.sh`
        - Multiple Runs with Val/Test Split CV: `./scripts/training/pt_training/srun_pt_training_5foldcv.sh`
        - Multiple Runs with LOPOCV: `./scripts/training/pt_training/srun_pt_training_loocv.sh`
    - For pytorch-lightning: 
        - Single Run: `./scripts/training/pl_training/srun_train_single_exp.sh`
        - Multiple Runs with Val/Test Split CV: `./scripts/training/pl_training/srun_pl_training_5foldcv.sh`
        - Multiple Runs with LOPOCV: `./scripts/training/pl_training/srun_pl_training_loocv.sh` 

# 4. WebApp: `PupilSense` 👁️

`PupilSense` is created with streamlit and hosted on 🤗 Hugging Face Spaces. 

You can view the app 🤗 [here](https://huggingface.co/spaces/vijulshah/eyedentify) and the source code [here](https://huggingface.co/spaces/vijulshah/eyedentify/tree/main).

--- 

## 🌏 Citations

The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

If `EyeDentify` helps your research or work, please cite <a href="https://arxiv.org/abs/2407.11204">EyeDentify</a>.
``` latex
@article{shah2024eyedentify,
  title={Eyedentify: A dataset for pupil diameter estimation based on webcam images},
  author={Shah, Vijul and Watanabe, Ko and Moser, Brian B and Dengel, Andreas},
  journal={arXiv preprint arXiv:2407.11204},
  year={2024}
}
```

---
## 📜 License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

---
## 📧 Contact

If you have any questions, please email `vijul1904@gmail.com`