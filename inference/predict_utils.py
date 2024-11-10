import os
import torch
import warnings
import numpy as np
from torchcam.utils import overlay_mask
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    GuidedGradCam,
)
import matplotlib.pyplot as plt
from torchcam.methods import CAM
from torchcam import methods as torchcam_methods
from captum.attr import visualization as captum_viz
from captum import attr as captum_attribution_methods
from torchvision.transforms.functional import to_pil_image
from matplotlib.colors import LinearSegmentedColormap


def visualize_attributions(attributions, input_img, viz_path):
    visualization = captum_viz.visualize_image_attr_multiple(
        np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(input_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "blended_heat_map"],
        ["all", "all"],
        show_colorbar=True,
    )
    visualization[0].savefig(viz_path)


def visualize_activation_map(
    activation_map,
    input_img,
    viz_path,
    caption1=None,
    caption2=None,
):
    cm = "jet"
    font_size = 18

    raw_cam = activation_map[0].squeeze(0)
    original_image = to_pil_image(input_img)
    overlay_image = overlay_mask(
        original_image,
        to_pil_image(raw_cam, mode="F"),
        colormap=cm,
        alpha=0.55,
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

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


def create_boxplot(all_sources, all_targets, results_path=".", file_name="box_plot"):

    plt.figure(figsize=(10, 10))
    plt.boxplot([all_targets, all_sources], labels=["True Values", "Predicted Values"])
    plt.title("Box Plot - True vs Predicted Values", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Pupil Diameter", fontsize=20)
    plt.savefig(f"{results_path}/{file_name}.png")
    plt.close()


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)
    plt.setp(bp["fliers"], color=color, markeredgecolor=color)


def create_grouped_boxplot(all_sources, all_targets, results_path=".", file_name="grouped_box_plot"):
    plt.figure(figsize=(16, 8))

    bpl = plt.boxplot(
        all_targets,
        positions=np.array(range(len(all_targets))) * 2.0 - 0.4,
        # sym="", # to hide outliers
        widths=0.6,
    )
    bpr = plt.boxplot(
        all_sources,
        positions=np.array(range(len(all_sources))) * 2.0 + 0.4,
        # sym="", # to hide outliers
        widths=0.6,
    )
    set_box_color(bpl, "#2C7BB6")  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, "#D7191C")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#2C7BB6", label="True")
    plt.plot([], c="#D7191C", label="Predicted")
    plt.legend(fontsize=20)
    plt.title("Grouped Box Plot - True vs Predicted Values", fontsize=20)

    ticks = range(1, len(all_targets) + 1)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=20)
    plt.xlim(-2, len(ticks) * 2)
    plt.xlabel("Participant Id", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Pupil Diameter", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{results_path}/{file_name}.png")
    plt.close()


def regression_chart_line(all_sources, all_targets, results_path=".", file_name="line_plot"):

    plt.figure(figsize=(14, 10))
    plt.clf()
    plt.plot(all_sources, label="Predicted")
    plt.plot(all_targets, label="True")
    plt.title("Line Plot - True vs Predicted Values", fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel("Samples", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Pupil Diameter", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f"{results_path}/{file_name}.png")


def run_xai(
    self,
    input_img,
    target_value,
    participant_id,
    session_id,
    img_id,
    input_mask=None,
    viz_path=None,
):
    if viz_path is None:
        viz_path = self.results_path

    viz_path = os.path.join(viz_path, str(participant_id), str(session_id))
    os.makedirs(viz_path, exist_ok=True)

    if "CAM" in self.xai_methods:
        if self.registered_model_name == "ResNet18":
            target_layer = self.model.resnet.layer4[-1].conv2
        elif self.registered_model_name == "ResNet50":
            target_layer = self.model.resnet.layer4[-1].conv3
        elif self.registered_model_name == "ResNet152":
            target_layer = self.model.resnet.layer4[-1].conv3
        else:
            raise Exception(f"No target layer available for selected model: {self.registered_model_name}")

    B, C, H, W = input_img.shape

    for method in self.xai_methods:
        # print(f"\n====================== {method} ======================")
        for target_class in range(0, self.num_classes):
            if method in self.available_cam_methods:
                if method == "CAM":
                    cam_extractor = torchcam_methods.__dict__[method](
                        self.model,
                        target_layer=target_layer,
                        fc_layer=self.model.resnet.fc,
                        input_shape=(C, H, W),
                    )
                    with torch.no_grad():
                        if input_mask is not None:
                            out = self.model(input_img, input_mask)
                        else:
                            out = self.model(input_img)
                    activation_map = cam_extractor(class_idx=target_class)
                elif method == "ScoreCAM":
                    cam_extractor = torchcam_methods.__dict__[method](
                        self.model,
                        target_layer=target_layer,
                        batch_size=1,
                        input_shape=(C, H, W),
                    )
                    with torch.no_grad():
                        out = self.model(input_img)
                    activation_map = cam_extractor(class_idx=target_class)
                elif method == "SSCAM":
                    cam_extractor = torchcam_methods.__dict__[method](
                        self.model,
                        target_layer=target_layer,
                        batch_size=1,
                        num_samples=1,
                        std=1.0,
                        input_shape=(C, H, W),
                    )
                    with torch.no_grad():
                        out = self.model(input_img)
                    activation_map = cam_extractor(class_idx=target_class)
                elif method == "ISCAM":
                    cam_extractor = torchcam_methods.__dict__[method](
                        self.model,
                        target_layer=target_layer,
                        batch_size=1,
                        num_samples=1,
                        input_shape=(C, H, W),
                    )
                    with torch.no_grad():
                        out = self.model(input_img)
                    activation_map = cam_extractor(class_idx=target_class)
                elif method == "SmoothGradCAMpp":
                    cam_extractor = torchcam_methods.__dict__[method](
                        self.model,
                        target_layer=target_layer,
                        input_shape=(C, H, W),
                    )
                    out = self.model(input_img)
                    activation_map = cam_extractor(class_idx=target_class)
                else:
                    cam_extractor = torchcam_methods.__dict__[method](
                        self.model,
                        target_layer=target_layer,
                        input_shape=(C, H, W),
                    )
                    out = self.model(input_img)
                    activation_map = cam_extractor(class_idx=target_class, scores=out)
                visualize_activation_map(
                    activation_map,
                    input_img.squeeze(0),
                    viz_path=f"{viz_path}/{img_id}_{method}_{target_class}.png",
                    caption1=(target_value.squeeze(0).item() if target_value else "N/A"),
                    caption2=out.squeeze(0).item(),
                )
            elif method in self.available_attribution_methods:
                try:
                    if method == "GuidedGradCam" or "Layer" in method:
                        xai_method = GuidedGradCam(
                            self.model,
                            layer=target_layer,
                        )
                    else:
                        xai_method = getattr(captum_attribution_methods, method)(self.model)
                except AttributeError:
                    warnings.warn(f"Skipping {method}. Class {method} not found or not defined.")
                if method == "LayerGradCam":
                    attributions = xai_method.attribute(
                        input_img,
                        # attr_dim_summation=False,
                        relu_attributions=False,
                        target=target_class if self.num_classes > 1 else None,
                    )
                else:
                    attributions = xai_method.attribute(
                        input_img, target=target_class if self.num_classes > 1 else None
                    )

                # if method == "LayerGradCam" or method == "LayerGradientXActivation":
                #     attributions = LayerAttribution.interpolate(attributions, (H, W))

                visualize_attributions(
                    attributions,
                    input_img,
                    viz_path=f"{viz_path}/{img_id}_{method}_{target_class}.png",
                )
            elif method == "attention_visualization":
                warnings.warn(f"Skipping {method}. Development in progress.")
            else:
                warnings.warn(f"Skipping {method}. Class {method} not found or not defined.")
