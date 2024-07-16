import os
import cv2
import math
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PupilDiameter(Dataset):

    def __init__(
        self,
        data,
        data_path,
        selected_targets,
        img_mode="RGB",
        img_size=None,
        means=None,
        stds=None,
        augment=False,
        augmentation_configs=None,
    ):
        self.data_path = data_path
        self.selected_targets = selected_targets
        self.img_mode = img_mode
        self.data_array = data
        self.augment = augment
        self.augmentation_configs = augmentation_configs

        self.preprocess_steps = [transforms.ToTensor()]
        self.preprocess_mask_steps = [transforms.ToTensor()]

        if img_size is not None:

            self.preprocess_steps.append(
                transforms.Resize(
                    [img_size[0], img_size[-1]],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            )

            self.preprocess_mask_steps.append(
                transforms.Resize(
                    [img_size[0], img_size[-1]],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )

        if means is not None and stds is not None:
            self.preprocess_steps.append(transforms.Normalize(means, stds))

        self.preprocess_function = transforms.Compose(self.preprocess_steps)
        self.preprocess_mask_function = transforms.Compose(self.preprocess_mask_steps)

    def __len__(self):
        return len(self.data_array)

    @staticmethod
    def augment_img(img, rotate, rotation_angle):
        if rotate:
            img = img.rotate(rotation_angle)
        return img

    @staticmethod
    def augment_target_data(target_data, rotate, rotation_angle):
        if rotate:
            # NOTE: To perform a counter-clockwise rotation, we just need to adjust the sign of the rotation angle (because in pillow's rotation method, the direction of rotation is counter-clockwise)
            rotation_angle_rad = math.radians(-rotation_angle)
            rotation_matrix = torch.tensor(
                [
                    [math.cos(rotation_angle_rad), -math.sin(rotation_angle_rad)],
                    [math.sin(rotation_angle_rad), math.cos(rotation_angle_rad)],
                ]
            )
            target_data = torch.matmul(target_data, rotation_matrix)
        return target_data

    @staticmethod
    def get_mask(img):

        img_np = np.array(img)

        # Convert RGB image to grayscale
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur for denoising
        img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Perform adaptive thresholding
        _, img_mask_np = cv2.threshold(
            img_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Invert the mask
        img_mask_np = cv2.bitwise_not(img_mask_np)

        # conver numpy image to PIL.Image.Image
        img_mask_pil = Image.fromarray(img_mask_np)

        return img_mask_pil

    def __getitem__(self, idx):

        data = self.data_array[idx]
        (
            participant_id,
            timestamp,
            session_id,
            left_pupil,
            right_pupil,
            pupil_diameter,
            frame_path,
        ) = data

        targets = []
        if "left_pupil" in self.selected_targets:
            targets.append(left_pupil)
        if "right_pupil" in self.selected_targets:
            targets.append(right_pupil)
        if "pupil_diameter" in self.selected_targets:
            targets.append(pupil_diameter)

        target_data = torch.tensor(targets).float()

        if self.augment:
            rotate = random.random() < 0.5
            rotations = self.augmentation_configs.get("rotations", [-5, 5])
            rotation_angle = random.randint(rotations[0], rotations[-1])

        # Note: Temporary fix - hardcoded ".png". TODO: Change that!!!!!!!
        image = Image.open(os.path.join(self.data_path, frame_path)).convert(
            self.img_mode
        )

        image = (
            self.augment_img(image, rotate, rotation_angle) if self.augment else image
        )
        img_mask = self.get_mask(image)

        image = self.preprocess_function(image)
        img_mask = self.preprocess_mask_function(img_mask).bool()

        img_id = frame_path.split("/")[-1].split(".")[0]
        participant_data = {
            "frame_path": frame_path,
            "participant_id": participant_id,
            "session_id": session_id,
            "img_id": img_id,
            "img": image,
            "target_data": target_data,
            "img_mask": img_mask,
        }

        return participant_data


print("Registered datasets in DATASET_REGISTRY:", DATASET_REGISTRY.keys())
