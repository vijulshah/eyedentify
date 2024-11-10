import os
import math
import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
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
        self.depth_data_path = "/".join(data_path.split("/")[:-1]) + "/depth"
        self.selected_targets = selected_targets
        self.img_mode = img_mode
        self.data_array = data
        self.augment = augment
        self.augmentation_configs = augmentation_configs

        self.preprocess_steps_rgb = []

        if img_size is not None:
            self.preprocess_steps_rgb.append(
                transforms.Resize(
                    [img_size[0], img_size[-1]],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            )

        self.preprocess_steps_rgb.append(transforms.ToTensor())

        # if means is not None and stds is not None:
        #     self.preprocess_steps_rgb.append(transforms.Normalize(mean=means, std=stds))

        # self.preprocess_steps_rgb.append(
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # )

        self.preprocess_function_rgb = transforms.Compose(self.preprocess_steps_rgb)

    def __len__(self):
        return len(self.data_array)

    @staticmethod
    def augment_img(
        img,
        brightness,
        brightness_value,
        contrast,
        contrast_value,
        saturation,
        saturation_value,
        gaussian_blur,
        gaussian_blur_value,
        rotate,
        rotation_angle,
    ):
        if brightness:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_value)
        if contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_value)
        if saturation:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_value)
        if gaussian_blur:
            img = img.filter(ImageFilter.GaussianBlur(gaussian_blur_value))
        if rotate:
            img = img.rotate(rotation_angle)
        return img

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

        # Determine if augmentation should be applied
        if self.augment:

            brightness = random.random() < 0.5
            brightness_range = self.augmentation_configs.get("brightness_range", [0.8, 1.2])
            brightness_value = random.uniform(brightness_range[0], brightness_range[-1])

            contrast = random.random() < 0.5
            contrast_range = self.augmentation_configs.get("contrast_range", [0.8, 1.2])
            contrast_value = random.uniform(contrast_range[0], contrast_range[-1])

            saturation = random.random() < 0.5
            saturation_range = self.augmentation_configs.get("saturation_range", [0.8, 1.2])
            saturation_value = random.uniform(saturation_range[0], saturation_range[-1])

            gaussian_blur = random.random() < 0.5
            gaussian_blur_range = self.augmentation_configs.get("gaussian_blur_range", [0.3, 1.5])
            gaussian_blur_value = random.uniform(gaussian_blur_range[0], gaussian_blur_range[-1])

            rotate = random.random() < 0.5
            rotations = self.augmentation_configs.get("rotations", [-5, 5])
            rotation_angle = random.uniform(rotations[0], rotations[-1])
        else:
            brightness = False
            brightness_value = 1

            contrast = False
            contrast_value = 1

            saturation = False
            saturation_value = 1

            gaussian_blur = False
            gaussian_blur_value = 0

            rotate = False
            rotation_angle = 0

        # Load and preprocess the RGB image
        image = Image.open(os.path.join(self.data_path, frame_path)).convert(self.img_mode)
        image = (
            self.augment_img(
                image,
                brightness,
                brightness_value,
                contrast,
                contrast_value,
                saturation,
                saturation_value,
                gaussian_blur,
                gaussian_blur_value,
                rotate,
                rotation_angle,
            )
            if self.augment
            else image
        )
        image = self.preprocess_function_rgb(image)  # Convert RGB image to a tensor (C, H, W)

        frame_path_npy = frame_path.replace(".png", ".npy")
        depth_map = np.load(os.path.join(self.depth_data_path, frame_path_npy))

        # Convert depth map to tensor and make sure it has the correct shape (1, H, W)
        depth_map = torch.tensor(depth_map).float()

        # Ensure the depth map has a single channel (1, H, W)
        if depth_map.ndim == 2:
            depth_map = depth_map.unsqueeze(0)  # Convert (H, W) to (1, H, W)

        # Augment (rotate) the depth map if the image was rotated
        if self.augment and rotate:
            # Rotate the depth map using the same angle as the RGB image
            depth_map = transforms.functional.rotate(depth_map, rotation_angle)

        # Resize depth map if the RGB image was resized
        if image.shape[1:] != depth_map.shape[1:]:
            depth_map = transforms.functional.resize(
                depth_map, image.shape[1:], interpolation=transforms.InterpolationMode.BICUBIC
            )

        # Concatenate the RGB image (3, H, W) with the depth map (1, H, W) -> (4, H, W)
        img_with_depth_map = torch.cat((image, depth_map), dim=0)

        img_id = frame_path.split("/")[-1].split(".")[0]
        participant_data = {
            "frame_path": frame_path,
            "participant_id": participant_id,
            "session_id": session_id,
            "img_id": img_id,
            "img": image,
            "depth_map": depth_map,
            "img_with_depth_map": img_with_depth_map,
            "target_data": target_data,
        }

        return participant_data


print("Registered datasets in DATASET_REGISTRY:", DATASET_REGISTRY.keys())
