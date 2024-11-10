import os
import sys
import yaml
import json
import torch
import mlflow
import random
import pickle
import evaluate
import argparse
import requests
import numpy as np
import pandas as pd
import os.path as osp
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision import transforms
from matplotlib import pyplot as plt
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

from inference.predict_utils import (
    create_boxplot,
    create_grouped_boxplot,
    regression_chart_line,
    run_xai,
)

from common_utils import (
    get_model,
    seed_everything,
    get_loss_function,
    initialize_metrics,
)
from registry_utils import import_registered_modules

import_registered_modules()


class PredictWithXai:

    def __init__(self, config_file):
        self.config_file = config_file
        self.data_path = os.path.join(root_path, self.config_file["dataset_configs"]["data_path"])
        self.selected_participants = self.config_file["dataset_configs"].get("selected_participants", [])
        self.img_size = self.config_file["dataset_configs"]["dataset_registry_params"].get("img_size", None)
        self.selected_targets = self.config_file["dataset_configs"]["dataset_registry_params"]["selected_targets"][-1]
        self.model_configs = self.config_file["model_configs"]
        self.registered_model_name = self.model_configs["registered_model_name"]
        self.test_configs = self.config_file["train_test_configs"]
        self.xai_configs = self.config_file["xai_configs"]
        self.img_configs = self.config_file.get("img_configs", None)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.num_params, self.num_trainable_params, self.model_size = self._load_model()
        # print("model = \n", self.model)

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

        self.xai_methods = self.xai_configs.get("methods", ["CAM", "InputXGradient"])
        self.num_classes = int(self.model_configs.get("num_classes", 1))
        self.log_mlflow_params = self.config_file.get("log_mlflow_params", True)

    @torch.no_grad()
    def _load_model(self):
        model_configs = self.model_configs.copy()
        model_path = os.path.join(root_path, model_configs["model_path"])
        model_configs.pop("model_path")
        model_dict = torch.load(model_path, map_location=self.device)
        model, num_params, num_trainable_params, model_size = get_model(
            model_configs=model_configs, device=self.device, use_ddp=False
        )
        if "state_dict" in model_dict.keys():
            state_dict = model_dict["state_dict"]
            model_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(model_dict)
        model = model.to(self.device)
        model = model.eval()
        return model, num_params, num_trainable_params, model_size

    @torch.no_grad()
    def get_single_img(self):

        input_data = []

        if self.img_configs:
            image_path = self.img_configs.get("image_path", None)
            if image_path is None:
                image_url = "https://drive.google.com/file/d/1y-XUPyFGdLHK3i0h_5o0y35Tfwwkc3OF/view?usp=drive_link"
                # user has not specified any image - we use our own image
                print(
                    "Please use the `image_path` argument in `img_configs` parameter to indicate the path of the image you wish to visualize."
                )
                print(f"Since no image path have been provided, we take the image from `{image_url}`.")
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

            image_size = self.img_configs.get("image_size", None)

            if image_size:
                # TODO: add normalization via saved file if sent true in img_configs.
                # TODO: add other image transformations if sent in img_configs.
                transform = transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                    ]
                )
                input_img = transform(img)
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
                input_img = transform(img)

            input_target = self.img_configs.get("input_target", None)
            input_target = torch.tensor(input_target, dtype=torch.float32)
            input_obj = {
                "input_imgs": input_img.unsqueeze(0).to(device=self.device),
                "input_targets": input_target.unsqueeze(0).to(device=self.device),
            }
            input_data.append(input_obj)
        else:
            raise Exception("Please specify correct dataloader / image configs")

        return input_data

    def evaluate_img_dir(self):

        self.metric_values = {}
        self.metrics = {}

        self.eval_metrics = self.test_configs.get("eval_metrics", [])

        for metric_name in self.eval_metrics:
            self.metrics[metric_name] = evaluate.load(metric_name)

        self.metric_values = initialize_metrics(EVAL_METRICS=self.eval_metrics, split="test", device=self.device)

        all_sources, all_targets = [], []
        all_sources_list_of_list, all_targets_list_of_list = [], []
        current_test_step, step_targets, step_preds = 0, 0, 0

        criterion = get_loss_function(loss_function_configs=self.config_file["loss_function_configs"].copy())

        if self.log_mlflow_params:
            mlflow.log_params(
                {
                    "model_configs": json.dumps(
                        {
                            "num_params": "{:.3f} K".format(self.num_params / 1000),
                            "num_trainable_params": "{:.3f} K".format(self.num_trainable_params / 1000),
                            "model_size": "{:.3f} MB".format(self.model_size),
                        }
                    ),
                    "xai_methods": self.xai_methods,
                    "loss_function": criterion.__class__.__name__,
                }
            )
        results_path = self.config_file["train_test_configs"]["results_path"]

        participants = sorted([int(folder_name) for folder_name in os.listdir(self.data_path) if folder_name.isdigit()])
        print("available participants = ", len(participants))
        print(
            "selected participants = ",
            (self.selected_participants if len(self.selected_participants) > 0 else len(participants)),
        )

        for pid in participants:

            if len(self.selected_participants) == 0 or (pid in self.selected_participants):
                pid = str(pid)
                print(f"\n==================== Processing Participant: {pid} ====================")
                pid_predictions = []
                pid_targets = []
                participant_folder = os.path.join(self.data_path, pid)
                participant_sessions_folders = sorted(
                    [int(folder) for folder in os.listdir(participant_folder) if folder.isdigit()]
                )

                for session_folder in map(str, participant_sessions_folders):

                    print(f"---------- Session: {session_folder} ----------")

                    session_path = os.path.join(participant_folder, session_folder)
                    session_data_csv_path = os.path.join(session_path, "session_data.csv")

                    df = pd.read_csv(session_data_csv_path)
                    df = df.dropna(axis=0, how="any")
                    df = df.reset_index(drop=True)

                    session_img_paths = df["frame_path"]
                    selected_targets_values = df[self.selected_targets]

                    for indx, frame_path in enumerate(session_img_paths):

                        frame_name = frame_path.split("/")[-1].split(".")[0]

                        input_img = Image.open(os.path.join(root_path, self.data_path, frame_path)).convert("RGB")

                        preprocess_steps_eyes = [transforms.ToTensor()]

                        if self.img_size is not None:
                            preprocess_steps_eyes.append(
                                transforms.Resize(
                                    [self.img_size[0], self.img_size[-1]],
                                    interpolation=transforms.InterpolationMode.BICUBIC,
                                    antialias=True,
                                )
                            )

                        preprocess_function_eyes = transforms.Compose(preprocess_steps_eyes)

                        input_img = preprocess_function_eyes(input_img)
                        input_img = input_img.unsqueeze(0).to(device=self.device)

                        target_value = selected_targets_values[indx]
                        target_value = torch.tensor(target_value).unsqueeze(0).to(device=self.device)

                        with torch.no_grad():
                            outputs = self.model(input_img)

                        outputs = outputs.squeeze(0)
                        loss = criterion(outputs, target_value)

                        self.metric_values["loss"] = torch.tensor(loss.item()).to(self.device)
                        self.metric_values["running_loss"] += torch.tensor(loss.item()).to(self.device)

                        self._calc_metric_scores(
                            predictions=outputs,
                            labels=target_value,
                            iteration_type="batch",
                            iteration_num=current_test_step,
                            results_path=results_path,
                            step_preds=step_preds,
                        )
                        current_test_step += 1

                        v = outputs.cpu().numpy()[0]
                        all_sources.append(v)
                        pid_predictions.append(v)
                        mlflow.log_metric(key="inference_predicted_values", value=v, step=step_preds)
                        step_preds += 1

                        w = target_value.cpu().numpy()[0]
                        all_targets.append(w)
                        pid_targets.append(w)
                        mlflow.log_metric(key="inference_target_values", value=w, step=step_targets)
                        step_targets += 1

                        run_xai(
                            self=self,
                            input_img=input_img,
                            target_value=target_value,
                            participant_id=pid,
                            session_id=session_folder,
                            img_id=frame_name,
                            input_mask=None,
                            viz_path=results_path,
                        )

                all_sources_list_of_list.append(pid_predictions)
                all_targets_list_of_list.append(pid_targets)

                create_boxplot(
                    pid_predictions,
                    pid_targets,
                    results_path,
                    file_name=f"p{pid}_box_plot",
                )
                # create_grouped_boxplot(
                #     [pid_predictions],
                #     [pid_targets],
                #     results_path,
                #     file_name=f"p{pid}_grouped_box_plot",
                # )
                regression_chart_line(
                    pid_predictions,
                    pid_targets,
                    results_path,
                    file_name=f"p{pid}_line_plot",
                )

                with open(f"{results_path}/p{pid}_predictions.pkl", "wb") as f:
                    pickle.dump(pid_predictions, f)

                with open(f"{results_path}/p{pid}_targets.pkl", "wb") as f:
                    pickle.dump(pid_targets, f)

        self._calc_metric_scores(
            predictions=all_sources,
            labels=all_targets,
            iteration_type="epoch",
            iteration_num=0,
            results_path=results_path,
            step_preds=step_preds,
        )

        create_boxplot(all_sources, all_targets, results_path)
        create_grouped_boxplot(all_sources_list_of_list, all_targets_list_of_list, results_path)
        regression_chart_line(all_sources, all_targets, results_path)

        with open(f"{results_path}/all_predictions.pkl", "wb") as f:
            pickle.dump(all_sources, f)

        with open(f"{results_path}/all_targets.pkl", "wb") as f:
            pickle.dump(all_targets, f)

    @torch.no_grad()
    def evaluate_img(self, input_data=[]):
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
                viz_methods = ["scatter"] if viz_methods is None or len(input_imgs) == 1 else viz_methods
                self.regression_chart_line(
                    predictions_x=predictions[:, 0],
                    targets_x=targets[:, 0],
                    results_path=results_path,
                )

    def _calc_metric_scores(self, predictions, labels, iteration_type, iteration_num, results_path, step_preds):

        if iteration_type == "batch":
            for key in self.eval_metrics:
                if key == "mse":
                    self.metric_values[key] = self.metrics[key].compute(
                        predictions=predictions.view(-1),
                        references=labels.view(-1),
                        squared=True,  # if false, then it returns RMSE
                    )[key]
                else:
                    self.metric_values[key] = self.metrics[key].compute(
                        predictions=predictions.view(-1), references=labels.view(-1)
                    )[key]
                self.metric_values[key] = torch.tensor(self.metric_values[key]).to(self.device)
                self.metric_values.update(
                    {f"running_{key}": self.metric_values[key] + self.metric_values[f"running_{key}"]}
                )

        for metric_name in self.metric_values.keys():
            log_value = None
            if iteration_type == "batch":
                if "running" not in metric_name:
                    log_value = self.metric_values[metric_name]
            else:
                if "running" in metric_name:
                    log_value = self.metric_values[metric_name] / step_preds

            if log_value is not None:

                metric_file_path = os.path.join(results_path, f"inference_{iteration_type}_{metric_name}_log.txt")

                with open(metric_file_path, "a") as f:
                    f.write(f"Iteration {iteration_num}: {log_value.item()}\n")

                mlflow.log_metrics(
                    {f"inference_{iteration_type}_{metric_name}": log_value.item()},
                    step=iteration_num,
                )

    @staticmethod
    def regression_chart_line(predictions_x, targets_x, results_path, predictions_y=None, targets_y=None):
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


def run_prediction_pipeline(args, config_file):

    seed_everything(seed=config_file["seed"])

    if config_file.get("log_mlflow_params", True):
        mlflow.start_run(run_id=None)

    results_path = config_file["train_test_configs"].get("results_path", None)
    if results_path:
        curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_path = os.path.join(root_path, results_path, curr_date_time)
        os.makedirs(results_path, exist_ok=True)
    else:
        active_run = mlflow.active_run()
        artifact_repo = get_artifact_repository(active_run.info.artifact_uri)
        config_file["train_test_configs"]["results_path"] = artifact_repo.__getattribute__("artifact_dir")

    if config_file.get("log_mlflow_params", True):
        config = vars(args)
        mlflow.log_params(config)
        mlflow.log_artifact(args.config_file)

    predictWithXai = PredictWithXai(config_file)
    predictWithXai.evaluate_img_dir()
    predictWithXai.evaluate_img()

    mlflow.end_run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get predictions of the model on the passed image or test dataloader.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "test_img_dir_with_xai.yml"),
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

    run_prediction_pipeline(args, config_file)
