import os
import sys
import yaml
import json
import torch
import mlflow
import random
import evaluate
import argparse
import os.path as osp
from tqdm import tqdm
from datetime import datetime
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

from inference_interpretation_utils import (
    create_grouped_boxplot,
    regression_chart_line,
    run_xai,
)

from training.train_utils import (
    get_model,
    seed_everything,
    get_loss_function,
    initialize_metrics,
)
from registry_utils import import_registered_modules

import_registered_modules()


class TestModelWithXaiOnTestDataloader:

    def __init__(self, config_file):
        self.config_file = config_file
        self.model_configs = self.config_file["model_configs"]
        self.registered_model_name = self.model_configs["registered_model_name"]
        self.dataloader_configs = self.config_file["dataloader_configs"]
        self.test_configs = self.config_file["test_configs"]
        self.xai_configs = self.config_file["xai_configs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.num_params, self.num_trainable_params, self.model_size = (
            self._load_model()
        )
        print("model = \n", self.model)

        self.available_attribution_methods = [
            "IntegratedGradients",
            "Saliency",
            "InputXGradient",
            "GuidedBackprop",
            "Deconvolution",
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

    @torch.no_grad()
    def _load_model(self):
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
        return model, num_params, num_trainable_params, model_size

    def __call__(self):

        dataloader_path = os.path.join(
            root_path, self.dataloader_configs["dataloader_path"]
        )
        test_dataloader = torch.load(dataloader_path, map_location=self.device)

        max_steps = self.dataloader_configs.get("max_steps", len(test_dataloader))
        self.max_testing_steps = min(max_steps, len(test_dataloader))
        num_test_steps = self.max_testing_steps

        self.metric_values = {}
        self.metrics = {}

        self.eval_metrics = self.test_configs.get("eval_metrics", [])

        for metric_name in self.eval_metrics:
            self.metrics[metric_name] = evaluate.load(metric_name)

        self.metric_values = initialize_metrics(
            EVAL_METRICS=self.eval_metrics, split="test", device=self.device
        )

        all_sources, all_targets = [], []
        current_test_step, step_targets, step_preds = 0, 0, 0

        criterion = get_loss_function(
            loss_function_configs=self.config_file["loss_function_configs"]
        )

        testing_progress_bar = tqdm(range(num_test_steps))

        mlflow.log_params(
            {
                "dataloaders_length": len(test_dataloader),
                "steps": num_test_steps,
                "model_configs": json.dumps(
                    {
                        "num_params": "{:.3f} K".format(self.num_params / 1000),
                        "num_trainable_params": "{:.3f} K".format(
                            self.num_trainable_params / 1000
                        ),
                        "model_size": "{:.3f} MB".format(self.model_size),
                    }
                ),
                "xai_methods": self.xai_methods,
                "loss_function": criterion.__class__.__name__,
            }
        )

        results_path = self.config_file["test_configs"]["results_path"]

        for batch, data in enumerate(test_dataloader):

            participant_id = data["participant_id"]
            session_id = data["session_id"]
            img_id = data["img_id"]

            input_img = data["img"].to(self.device)
            target_value = data["target_data"].float().to(self.device)
            input_mask = data.get("img_mask", None)

            if input_mask is not None:
                input_mask = input_mask.to(self.device)

            # print("input_img = ", input_img[0], input_img[0].shape, input_img.shape)
            # print("max = ", torch.max(input_img[0]))
            # print("min = ", torch.min(input_img[0]))

            # Calculate L1 distance (Manhattan distance)
            # l1_distance = torch.norm(input_img[0], p=1)
            # print(f"L1 Distance: {l1_distance.item()}")

            # Calculate L2 distance (Euclidean distance)
            # l2_distance = torch.norm(input_img[0], p=2)
            # print(f"L2 Distance: {l2_distance.item()}")
            # break

            with torch.no_grad():
                outputs = self.model(input_img, input_mask)

            loss = criterion(outputs, target_value)

            self.metric_values["loss"] = torch.tensor(loss.item()).to(self.device)
            self.metric_values["running_loss"] += torch.tensor(loss.item()).to(
                self.device
            )

            self._calc_metric_scores(
                predictions=outputs,
                labels=target_value,
                iteration_type="batch",
                iteration_num=current_test_step,
                results_path=results_path,
            )

            for o in outputs:
                v = o.cpu().numpy()[0]
                all_sources.append(v)
                mlflow.log_metric(key="predicted_values", value=v, step=step_preds)
                step_preds += 1

            for i, t in enumerate(target_value):
                v = t.cpu().numpy()[0]
                all_targets.append(v)
                mlflow.log_metric(key="target_values", value=v, step=step_targets)
                step_targets += 1

                run_xai(
                    self=self,
                    input_img=input_img[i].unsqueeze(0),
                    target_value=t.unsqueeze(0),
                    participant_id=participant_id[i],
                    session_id=session_id[i].item(),
                    img_id=img_id[i],
                    input_mask=input_mask[i].unsqueeze(0),
                    viz_path=results_path,
                )

            current_test_step += 1
            testing_progress_bar.update(1)

            if batch + 1 == self.max_testing_steps:
                break

        self._calc_metric_scores(
            predictions=all_sources,
            labels=all_targets,
            iteration_type="epoch",
            iteration_num=0,
            results_path=results_path,
        )

        create_grouped_boxplot(all_sources, all_targets, results_path)
        regression_chart_line(all_sources, all_targets, results_path)

    def _calc_metric_scores(
        self, predictions, labels, iteration_type, iteration_num, results_path
    ):

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
                self.metric_values[key] = torch.tensor(self.metric_values[key]).to(
                    self.device
                )
                self.metric_values.update(
                    {
                        f"running_{key}": self.metric_values[key]
                        + self.metric_values[f"running_{key}"]
                    }
                )

        for metric_name in self.metric_values.keys():
            log_value = None
            if iteration_type == "batch":
                if "running" not in metric_name:
                    log_value = self.metric_values[metric_name]
                    metric_file_path = os.path.join(
                        results_path, f"{metric_name}_batch_log.txt"
                    )
                    with open(metric_file_path, "a") as f:
                        f.write(f"Iteration {iteration_num}: {log_value.item()}\n")
            else:
                if "running" in metric_name:
                    max_steps = self.max_testing_steps
                    log_value = self.metric_values[metric_name] / max_steps
                    epoch_log_file_path = os.path.join(
                        results_path, "epoch_metrics_log.txt"
                    )
                    with open(epoch_log_file_path, "a") as f:
                        f.write(
                            f"Iteration {iteration_num}: {metric_name} = {log_value.item()}\n"
                        )
            if log_value is not None:
                mlflow.log_metrics(
                    {f"{iteration_type}_{metric_name}": log_value.item()},
                    step=iteration_num,
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get predictions of the model on the passed image or test dataloader."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "test_on_dataloader.yml"),
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

    mlflow.start_run(run_id=None)
    results_path = config_file["test_configs"].get("results_path", None)
    if results_path:
        curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_path = os.path.join(root_path, results_path, curr_date_time)
        os.makedirs(results_path, exist_ok=True)
    else:
        active_run = mlflow.active_run()
        artifact_repo = get_artifact_repository(active_run.info.artifact_uri)
        config_file["test_configs"]["results_path"] = artifact_repo.__getattribute__(
            "artifact_dir"
        )
    mlflow.log_artifact(args.config_file)

    TestModelWithXaiOnTestDataloader(config_file)()
