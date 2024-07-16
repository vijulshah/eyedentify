import os
import sys
import yaml
import json
import torch
import random
import requests
import argparse
import warnings
import numpy as np
import os.path as osp
from PIL import Image
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    GuidedGradCam,
    LayerGradCam,
    LayerGradientXActivation,
)
import torch.nn as nn
from torchcam.methods import CAM
from torchcam import methods as torchcam_methods
from captum.attr import visualization as captum_viz
from torchvision import transforms
from captum import attr as captum_attribution_methods
from torchvision.transforms.functional import to_pil_image
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution

from matplotlib.colors import LinearSegmentedColormap

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

from training.train_utils import get_model, seed_everything
from registry_utils import import_registered_modules

import_registered_modules()


class Xai:

    def __init__(self, config_file):
        self.config_file = config_file
        self.model_configs = self.config_file["model_configs"]
        self.dataloader_configs = self.config_file.get("dataloader_configs", None)
        self.img_configs = self.config_file.get("img_configs", None)
        self.xai_configs = self.config_file["xai_configs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.available_attribution_methods = [
            "IntegratedGradients",
            "Saliency",
            "InputXGradient",
            "GuidedBackprop",
            "Deconvolution",
            "GuidedGradCam",
            "LayerGradCam",
            "LayerGradientXActivation",
        ]
        self.available_cam_methods = [
            "CAM",
            "ScoreCAM",
            "SSCAM",
            "ISCAM",
            "GradCAM",
            "GradCAMpp",
            "SmoothGradCAMpp",
            "XGradCAM",
            "LayerCAM",
        ]

    def load_model(self):
        model_configs = self.model_configs
        model_path = os.path.join(root_path, model_configs["model_path"])
        model_configs.pop("model_path")
        model_dict = torch.load(model_path, map_location=self.device)
        model, num_params, num_trainable_params, model_size = get_model(
            model_configs=model_configs
        )
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        model = model.eval()
        return model

    def get_data(self):
        input_img = None
        if self.dataloader_configs:
            dataloader_configs = self.dataloader_configs
            dataloader_path = os.path.join(
                root_path, dataloader_configs["dataloader_path"]
            )
            test_dataloader = torch.load(dataloader_path, map_location=self.device)

            selected_batch_index = self.dataloader_configs.get(
                "selected_batch_index", 0
            )
            selected_batch_index = min(selected_batch_index, len(test_dataloader) - 1)

            for batch, data in enumerate(test_dataloader):

                if batch == selected_batch_index:

                    batch_samples = data["img"].size(0)
                    sample_index_in_batch = self.dataloader_configs.get(
                        "sample_index_in_batch", None
                    )

                    if sample_index_in_batch is None:
                        sample_index_in_batch = random.randint(0, batch_samples - 1)
                    else:
                        sample_index_in_batch = min(
                            sample_index_in_batch, batch_samples - 1
                        )

                    input_img = (
                        data["img"][sample_index_in_batch]
                        .unsqueeze(0)
                        .to(device=self.device)
                    )
                    target_data = (
                        data["target_data"][sample_index_in_batch]
                        .unsqueeze(0)
                        .to(device=self.device)
                    )
                    print(
                        "participant_id = ",
                        data["participant_id"][sample_index_in_batch],
                    )
                    print("target_value = ", data["target_data"][sample_index_in_batch])
                    break

            return input_img, target_data
        elif self.img_configs:
            image_path = self.img_configs.get("image_path", None)
            if image_path is None:
                image_url = "https://drive.google.com/file/d/1y-XUPyFGdLHK3i0h_5o0y35Tfwwkc3OF/view?usp=drive_link"
                # user has not specified any image - we use our own image
                print(
                    "Please use the `image_path` argument in `img_configs` parameter to indicate the path of the image you wish to visualize."
                )
                print(
                    f"Since no image path have been provided, we take the image from `{image_url}`."
                )
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            elif type(image_path) == bytes:
                img = Image.open(BytesIO(image_path)).convert("RGB")
            elif os.path.isfile(image_path):
                img = Image.open(image_path).convert("RGB")
            else:
                print(f"Provided image path `{image_path}` is non valid.")
                sys.exit(1)

            preprocess_steps = [transforms.ToTensor()]

            image_size = self.img_configs.get("image_size", None)
            if image_size is not None:
                preprocess_steps.append(
                    transforms.Resize(
                        [image_size[0], image_size[-1]],
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                )

            means = self.img_configs.get("means", None)
            stds = self.img_configs.get("stds", None)
            if means is not None and stds is not None:
                preprocess_steps.append(transforms.Normalize(means, stds))

            preprocess_function = transforms.Compose(preprocess_steps)
            input_img = preprocess_function(img)
            input_data = input_img.unsqueeze(0).to(device=self.device)

            target_value = self.img_configs.get("target_value", None)
            target_data = (
                torch.tensor(target_value).float().unsqueeze(0).to(device=self.device)
            )

            return input_data, target_data
        else:
            raise Exception("Please specify correct dataloader / image configs")

    @staticmethod
    def visualize_attributions(attributions, input_img, viz_path):
        print(attributions.shape)
        print(attributions)
        cm = "jet"
        visualization = captum_viz.visualize_image_attr_multiple(
            attr=np.transpose(attributions[0].cpu().detach().numpy(), (1, 2, 0)),
            original_image=np.transpose(input_img[0].cpu().detach().numpy(), (1, 2, 0)),
            methods=["original_image", "blended_heat_map"],
            signs=["all", "all"],
            titles=["Original Image", "Overlay Heatmap"],
            cmap=cm,
            fig_size=(5, 3),
            alpha_overlay=0.55,
            show_colorbar=True,
        )
        # visualization = captum_viz.visualize_image_attr(
        #     attributions[0].cpu().permute(1, 2, 0).detach().numpy(), sign="all"
        # )
        visualization[0].savefig(viz_path)

    @staticmethod
    def visualize_activation_map(
        activation_map, input_img, viz_path, caption1=None, caption2=None
    ):
        cm = "jet"
        raw_cam = activation_map[0].squeeze(0)
        original_image = to_pil_image(input_img)
        overlay_image = overlay_mask(
            original_image,
            to_pil_image(raw_cam, mode="F"),
            colormap=cm,
            alpha=0.55,
        )

        fig, axes = plt.subplots(1, 2, figsize=(5, 3))
        font_size = 12

        # Display the original image
        axes[0].imshow(original_image)
        axes[0].axis("off")
        axes[0].set_title("Original Image")
        axes[0].text(
            0.5,
            -0.1,
            f"True: {caption1:.2f} mm",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=font_size,
        )

        # Display the overlay image
        im = axes[1].imshow(overlay_image, cmap=cm, vmin=0, vmax=1)
        axes[1].axis("off")
        axes[1].set_title("Overlay Heatmap")
        axes[1].text(
            0.5,
            -0.1,
            f"Predicted: {caption2:.2f} mm",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=font_size,
        )
        cbar = fig.colorbar(im, orientation="horizontal", ticks=[0, 1])

        plt.tight_layout()
        plt.savefig(viz_path, bbox_inches="tight")
        plt.close()

    def __call__(self, input_img=None, target_value=None):

        input_img, target_data = input_img if input_img else self.get_data()
        print("input_img = ", input_img.shape)

        curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        viz_path = self.xai_configs.get("viz_path", "./xai_vizualizations")
        viz_path = os.path.join(root_path, viz_path, curr_date_time)
        os.makedirs(viz_path, exist_ok=True)

        print("model = \n", self.model)

        true_values = None
        if target_value:
            print("target_value = ", target_value)
            true_values = target_value
        elif target_data:
            print("target_data - ", target_data)
            true_values = target_data

        num_classes = int(self.model_configs.get("num_classes", 1))

        for method in self.xai_configs.get("methods", ["CAM", "InputXGradient"]):
            print(f"\n====================== {method} ======================")
            for target_class in range(0, num_classes):
                if method in self.available_cam_methods:
                    if method == "CAM":
                        cam_extractor = torchcam_methods.__dict__[method](
                            self.model,
                            target_layer=self.model.resnet.layer4[-1].conv2,
                            fc_layer=self.model.resnet.fc,
                            input_shape=(3, 32, 64),
                        )
                        with torch.no_grad():
                            out = self.model(input_img)
                        activation_map = cam_extractor(class_idx=target_class)
                    elif method == "ScoreCAM":
                        cam_extractor = torchcam_methods.__dict__[method](
                            self.model,
                            target_layer=self.model.resnet.layer4[-1].conv2,
                            batch_size=1,
                            input_shape=(3, 32, 64),
                        )
                        with torch.no_grad():
                            out = self.model(input_img)
                        activation_map = cam_extractor(class_idx=target_class)
                    elif method == "SSCAM":
                        cam_extractor = torchcam_methods.__dict__[method](
                            self.model,
                            target_layer=self.model.resnet.layer4[-1].conv2,
                            batch_size=1,
                            num_samples=1,
                            std=1.0,
                            input_shape=(3, 32, 64),
                        )
                        with torch.no_grad():
                            out = self.model(input_img)
                        activation_map = cam_extractor(class_idx=target_class)
                    elif method == "ISCAM":
                        cam_extractor = torchcam_methods.__dict__[method](
                            self.model,
                            target_layer=self.model.resnet.layer4[-1].conv2,
                            batch_size=1,
                            num_samples=1,
                            input_shape=(3, 32, 64),
                        )
                        with torch.no_grad():
                            out = self.model(input_img)
                        activation_map = cam_extractor(class_idx=target_class)
                    elif method == "SmoothGradCAMpp":
                        cam_extractor = torchcam_methods.__dict__[method](
                            self.model,
                            target_layer=self.model.resnet.layer4[-1].conv2,
                            input_shape=(3, 32, 64),
                        )
                        out = self.model(input_img)
                        activation_map = cam_extractor(class_idx=target_class)
                    else:
                        cam_extractor = torchcam_methods.__dict__[method](
                            self.model,
                            target_layer=self.model.resnet.layer4[-1].conv2,
                            input_shape=(3, 32, 64),
                        )
                        out = self.model(input_img)
                        activation_map = cam_extractor(
                            class_idx=target_class, scores=out
                        )
                        if method == "LayerCAM":
                            activation_map = cam_extractor.fuse_cams(activation_map)

                    print("predicted value = ", out)
                    print("activation_map = ", activation_map)
                    print("activation_map = ", activation_map[0].shape)
                    self.visualize_activation_map(
                        activation_map,
                        input_img.squeeze(0),
                        viz_path=f"{viz_path}/{method}_{target_class}.png",
                        caption1=(
                            true_values.squeeze(0).item() if true_values else "N/A"
                        ),
                        caption2=out.squeeze(0).item(),
                    )
                elif method in self.available_attribution_methods:
                    if method == "GuidedGradCam" or "Layer" in method:
                        xai_method = getattr(captum_attribution_methods, method)(
                            self.model,
                            layer=self.model.resnet.layer4[-1].conv2,
                        )
                    else:
                        try:
                            xai_method = getattr(captum_attribution_methods, method)(
                                self.model
                            )
                        except AttributeError:
                            warnings.warn(
                                f"Skipping {method}. Class {method} not found or not defined."
                            )
                    if method == "LayerGradCam":
                        attributions = xai_method.attribute(
                            input_img,
                            # attr_dim_summation=False,
                            relu_attributions=False,
                            target=target_class if num_classes > 1 else None,
                        )
                    else:
                        attributions = xai_method.attribute(
                            input_img, target=target_class if num_classes > 1 else None
                        )

                    if method == "LayerGradCam" or method == "LayerGradientXActivation":
                        attributions = LayerAttribution.interpolate(
                            attributions, (32, 64)
                        )
                    self.visualize_attributions(
                        attributions,
                        input_img,
                        viz_path=f"{viz_path}/{method}_{target_class}.png",
                    )
                elif method == "attention_visualization":
                    # TODO: implement this!!!
                    warnings.warn(f"Skipping {method}. Development in progress.")
                else:
                    warnings.warn(
                        f"Skipping {method}. Class {method} not found or not defined."
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Explain predictions of the model using captum or torchcam methods."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "xai.yml"),
        required=False,
        help="Path to config file.",
    )

    args = parser.parse_args()
    print("args:\n", json.dumps(vars(args), sort_keys=True, indent=4), "\n")

    with open(args.config_file, mode="r") as f:
        config_file = yaml.safe_load(f)
        print("config_file = ", config_file, "\n")

    seed_val = config_file.get("seed", random.randint(1, 10000))
    config_file["seed"] = seed_val
    seed_everything(seed_val)

    Xai(config_file)()
