import os
import sys
import yaml
import json
import torch
import random
import requests
import argparse
import numpy as np
import os.path as osp
from PIL import Image
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import transforms as pth_transforms

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

from training.train_utils import get_model, seed_everything
from registry_utils import import_registered_modules

import_registered_modules()


class TestModel:

    def __init__(self, config_file):
        self.config_file = config_file
        self.model_configs = self.config_file["model_configs"]
        self.dataloader_configs = self.config_file.get("dataloader_configs", None)
        self.img_configs = self.config_file.get("img_configs", None)
        self.test_configs = self.config_file["test_configs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    @torch.no_grad()
    def load_model(self):
        model_configs = self.model_configs
        model_path = os.path.join(root_path, model_configs["model_path"])
        model_configs.pop("model_path")
        model_dict = torch.load(model_path, map_location=self.device)
        model, _, _, _ = get_model(model_configs=model_configs)
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        model = model.eval()
        return model

    @torch.no_grad()
    def get_data(self):

        input_data = []

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

                if batch == selected_batch_index or selected_batch_index == -1:

                    batch_samples = data["img"].size(0)
                    samples_per_batch = self.dataloader_configs.get(
                        "samples_per_batch", batch_samples
                    )
                    samples_per_batch = min(samples_per_batch, batch_samples - 1)
                    input_obj = {
                        "input_imgs": data["img"][:samples_per_batch]
                        .unsqueeze(0)
                        .to(device=self.device),
                        "input_targets": data["target_data"][:samples_per_batch]
                        .unsqueeze(0)
                        .to(device=self.device),
                    }
                    input_data.append(input_obj)

                    if selected_batch_index != -1:
                        break

        elif self.img_configs:
            image_path = self.img_configs.get("image_path", None)
            if image_path is None:
                image_url = "https://dl.fbaipublicfiles.com/dino/img.png"
                # user has not specified any image - we use our own image
                print(
                    "Please use the `image_path` argument in `img_configs` parameter to indicate the path of the image you wish to visualize."
                )
                print(
                    f"Since no image path have been provided, we take the image from `{image_url}`."
                )
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                img = img.convert("RGB")
            elif type(image_path) == bytes:
                img = Image.open(BytesIO(image_path))
                img = img.convert("RGB")
            elif os.path.isfile(image_path):
                with open(image_path, "rb") as f:
                    img = Image.open(f)
                    img = img.convert("RGB")
            else:
                print(f"Provided image path `{image_path}` is non valid.")
                sys.exit(1)

            image_size = self.img_configs.get("image_size", (32, 64))
            # TODO: add normalization via saved file if sent true in img_configs.
            # TODO: add other image transformations if sent in img_configs.
            transform = pth_transforms.Compose(
                [
                    pth_transforms.Resize(image_size),
                    pth_transforms.ToTensor(),
                ]
            )
            input_img = transform(img)
            input_target = self.img_configs.get("input_target", None)
            input_target = torch.tensor(input_target, dtype=torch.float32)
            input_obj = {
                "input_imgs": input_img.unsqueeze(0).to(device=self.device),
                "input_targets": input_target.unsqueeze(0).to(device=self.device),
            }

        else:
            raise Exception("Please specify correct dataloader / image configs")

        return input_data

    @staticmethod
    def regression_chart_line(
        predictions_x, targets_x, results_path, predictions_y=None, targets_y=None
    ):
        plt.clf()
        plt.plot(predictions_x, label="X_predicted")
        plt.plot(targets_x, label="X_expected")
        if predictions_y and targets_y:
            plt.plot(predictions_y, label="Y_predicted")
            plt.plot(targets_y, label="Y_expected")
            line_label = "line_XY"
        else:
            line_label = "line_X"
        plt.ylabel("output")
        plt.legend()
        plt.savefig(f"{results_path}/{line_label}.png")

    @torch.no_grad()
    def __call__(self, input_data=[]):

        if len(input_data) == 0:
            input_data = self.get_data()

        curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_path = self.test_configs.get("results_path", "results")
        results_path = os.path.join(root_path, results_path, curr_date_time)
        os.makedirs(results_path, exist_ok=True)

        for input_obj in input_data:

            input_imgs = input_obj["input_imgs"]
            input_targets = input_obj.get("input_targets", None)

            targets = []
            predictions = []

            for i, input_img in enumerate(input_imgs):
                output = self.model(input_img)
                predictions.append(output.numpy())
                if input_targets is not None:
                    targets.append(input_targets[i].numpy())

            predictions = np.concatenate(predictions, axis=0)

            if len(targets) > 0:
                targets = np.concatenate(targets, axis=0)
                viz_methods = self.test_configs.get("viz_methods", None)
                viz_methods = (
                    ["scatter"]
                    if viz_methods is None or len(input_imgs) == 1
                    else viz_methods
                )
                self.regression_chart_line(
                    predictions_x=predictions[:, 0],
                    targets_x=targets[:, 0],
                    results_path=results_path,
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get predictions of the model on the passed image or test dataloader."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "test.yml"),
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

    TestModel(config_file)()
