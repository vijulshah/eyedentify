import os
import sys
import json
import yaml
import mlflow
import random
import argparse
import evaluate
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from dataloader import prepare_dataloader
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from train_utils import (
    get_model,
    get_loss_function,
    get_optimizer,
    get_lr_scheduler,
    initialize_metrics,
    print_on_rank_zero,
    seed_everything,
    GradualWarmupScheduler,
)

import torch
from torch.distributed.nn import all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

# To join other directories with this file, append the main folder
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)
from registry_utils import import_registered_modules

import_registered_modules()

import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        test_dataloader,
        val_dataloader,
        train_dataloader,
        train_test_configs,
        early_stopping_configs,
        checkpointing_configs,
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        # self.local_word_size = torch.cuda.device_count()

        if torch.cuda.is_available():
            self.device = self.local_rank
            print_on_rank_zero("Training on GPU")
            print_on_rank_zero(
                "local_rank = ",
                self.local_rank,
                " | global_rank = ",
                self.global_rank,
                " | world_size = ",
                self.world_size,
            )
        else:
            self.device = "cpu"
            print_on_rank_zero("Training on CPU")

        self.epochs_run = 0
        self.current_train_step = 0
        self.current_val_step = 0
        self.current_test_step = 0
        self.tracking_metric_best_score = 0.0
        self.early_stop_counter = 0
        self.early_stop = False

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.early_stopping_configs = early_stopping_configs

        self.max_epochs = train_test_configs["epochs"]
        self.max_training_steps = train_test_configs["max_training_steps"]
        self.max_validation_steps = train_test_configs["max_validation_steps"]
        self.max_testing_steps = train_test_configs["max_testing_steps"]
        num_training_steps = train_test_configs["num_training_steps"]
        num_val_steps = train_test_configs["num_val_steps"]
        num_test_steps = train_test_configs["num_test_steps"]
        self.training_progress_bar = tqdm(range(num_training_steps))
        self.validation_progress_bar = tqdm(range(num_val_steps))
        self.testing_progress_bar = tqdm(range(num_test_steps))

        self.clip_grad_norm = train_test_configs.get("clip_grad_norm", None)
        self.eval_metrics_average = train_test_configs.get("eval_metrics_average")
        eval_metrics = train_test_configs["eval_metrics"]
        self.eval_metrics = eval_metrics
        self.metric_values = {}
        self.metrics = {}
        self.best_model = None

        for metric_name in eval_metrics:
            self.metrics[metric_name] = evaluate.load(metric_name)

        if checkpointing_configs.get("save_every_n_epoch") is None:
            self.save_every_n_epoch = self.max_epochs
        else:
            self.save_every_n_epoch = checkpointing_configs.get("save_every_n_epoch")

        self.snapshot_path = checkpointing_configs.get("checkpoint_path")
        if checkpointing_configs.get("resume"):
            if os.path.exists(self.snapshot_path):
                print_on_rank_zero("Loading snapshot")
                self._load_snapshot(self.snapshot_path)

        if self.device != "cpu":
            self.model = DDP(self.model, device_ids=[self.local_rank])
            # self.model = DDP(
            #     self.model, device_ids=[self.local_rank], find_unused_parameters=True
            # )

    def _load_snapshot(self, snapshot_path):
        map_location = (
            f"cuda:{self.local_rank}" if self.device != "cpu" else self.device
        )
        snapshot = torch.load(snapshot_path, map_location=map_location)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.current_train_step = snapshot["CURRENT_TRAIN_STEP"]
        self.current_val_step = snapshot["CURRENT_VAL_STEP"]
        self.current_test_step = snapshot["CURRENT_TEST_STEP"]
        self.training_progress_bar.update(self.current_train_step)
        self.validation_progress_bar.update(self.current_val_step)
        self.testing_progress_bar.update(self.current_test_step)
        print_on_rank_zero(
            f"Resuming training from snapshot at Epoch {self.epochs_run}"
        )

    def _calc_metric_scores(
        self, predictions, labels, split, iteration_type, iteration_num
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

        if self.device != "cpu":
            barrier()
            for metric_name in self.metric_values.keys():
                if iteration_type == "batch":
                    if "running" not in metric_name:
                        all_reduce(self.metric_values[metric_name], op=ReduceOp.SUM)
                else:
                    if "running" in metric_name:
                        all_reduce(self.metric_values[metric_name], op=ReduceOp.SUM)

        for metric_name in self.metric_values.keys():
            log_value = None
            if iteration_type == "batch":
                if "running" not in metric_name:
                    log_value = self.metric_values[metric_name] / self.world_size
            else:
                if "running" in metric_name:
                    if split == "train":
                        max_steps = self.max_training_steps
                    elif split == "val":
                        max_steps = self.max_validation_steps
                    else:
                        max_steps = self.max_testing_steps
                    log_value = (
                        self.metric_values[metric_name] / max_steps
                    ) / self.world_size

            if log_value is not None and self.global_rank == 0:
                mlflow.log_metrics(
                    {f"{split}_{iteration_type}_{metric_name}": log_value.item()},
                    step=iteration_num,
                )

        if split == "train" and self.global_rank == 0:
            curr_lr = (
                self.scheduler.optimizer.param_groups[0]["lr"]
                if self.scheduler
                else self.optimizer.param_groups[0]["lr"]
            )
            mlflow.log_metrics(
                {f"{iteration_type}_lr": curr_lr},
                step=iteration_num,
            )

    def _run_train_epoch(self, epoch):

        split = "train"
        self.model.train()
        all_sources, all_targets = [], []

        if self.device == "cpu":
            print_on_rank_zero(
                f"[CPU] Epoch: {epoch} | Train_Dataloader: {len(self.train_dataloader)}"
            )
        else:
            self.train_dataloader.sampler.set_epoch(epoch)

        for batch, data in enumerate(self.train_dataloader):

            self.optimizer.zero_grad()

            source_img = data["img"].to(self.device)
            target_data = data["target_data"].float().to(self.device)
            img_mask = data.get("img_mask", None)

            if img_mask is not None:
                img_mask = img_mask.to(self.device)
                if batch == 0 and epoch == 0:
                    print_on_rank_zero("source_img = ", source_img.shape)
                    print_on_rank_zero("target_data = ", target_data.shape)
                    print_on_rank_zero("img_mask = ", img_mask.shape)
                    print_on_rank_zero("img_mask = ", img_mask)
                outputs = self.model(source_img, img_mask)
            else:
                outputs = self.model(source_img)

            loss = self.criterion(outputs, target_data)
            loss.backward()

            # Calculate gradient norm
            total_norm = 0.0
            parameters = [
                p
                for p in self.model.parameters()
                if p.grad is not None and p.requires_grad
            ]
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            self.metric_values["grad_norm"] = torch.tensor(total_norm).to(self.device)
            self.metric_values["running_grad_norm"] += torch.tensor(total_norm).to(
                self.device
            )

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.optimizer.step()

            # absolute_loss = torch.abs(outputs - target_data)
            # percentage_loss = (absolute_loss / torch.abs(target_data)) * 100
            # total_percentage_loss = torch.sum(percentage_loss) / len(percentage_loss)

            self.metric_values["loss"] = torch.tensor(loss.detach().item()).to(
                self.device
            )
            # self.metric_values["percentage_loss"] = torch.tensor(
            #     total_percentage_loss.item()
            # ).to(self.device)

            self.metric_values["running_loss"] += torch.tensor(loss.detach().item()).to(
                self.device
            )
            # self.metric_values["percentage_running_loss"] += torch.tensor(
            #     total_percentage_loss.item()
            # ).to(self.device)

            self._calc_metric_scores(
                outputs,
                target_data,
                split,
                iteration_type="batch",
                iteration_num=self.current_train_step,
            )

            all_sources.append(outputs)
            all_targets.append(target_data)

            self.current_train_step += 1
            self.training_progress_bar.update(1)

            if batch + 1 == self.max_training_steps:
                break

        self._calc_metric_scores(
            all_sources, all_targets, split, iteration_type="epoch", iteration_num=epoch
        )

    def _run_val_epoch(self, epoch):

        split = "val"
        self.model.eval()
        all_sources, all_targets = [], []

        if self.device == "cpu":
            print_on_rank_zero(
                f"[CPU] Epoch: {epoch} | Val_Dataloader: {len(self.val_dataloader)}"
            )
        else:
            print_on_rank_zero(
                f"[GPU{self.global_rank}] Epoch: {epoch} | Val_Dataloader: {len(self.val_dataloader)}"
            )
            self.val_dataloader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch, data in enumerate(self.val_dataloader):

                source_img = data["img"].to(self.device)
                target_data = data["target_data"].float().to(self.device)
                img_mask = data.get("img_mask", None)

                if img_mask is not None:
                    img_mask = img_mask.to(self.device)
                    outputs = self.model(source_img, img_mask)
                else:
                    outputs = self.model(source_img)

                loss = self.criterion(outputs, target_data)

                # absolute_loss = torch.abs(outputs - target_data)
                # percentage_loss = (absolute_loss / torch.abs(target_data)) * 100
                # total_percentage_loss = torch.sum(percentage_loss) / len(
                #     percentage_loss
                # )

                self.metric_values["loss"] = torch.tensor(loss.item()).to(self.device)
                # self.metric_values["percentage_loss"] = torch.tensor(
                #     total_percentage_loss.item()
                # ).to(self.device)

                self.metric_values["running_loss"] += torch.tensor(loss.item()).to(
                    self.device
                )
                # self.metric_values["percentage_running_loss"] += torch.tensor(
                #     total_percentage_loss.item()
                # ).to(self.device)

                self._calc_metric_scores(
                    outputs,
                    target_data,
                    split,
                    iteration_type="batch",
                    iteration_num=self.current_val_step,
                )

                all_sources.append(outputs)
                all_targets.append(target_data)

                self.current_val_step += 1
                self.validation_progress_bar.update(1)

                if batch + 1 == self.max_validation_steps:
                    break

        self._calc_metric_scores(
            all_sources, all_targets, split, iteration_type="epoch", iteration_num=epoch
        )

    def _run_test_epoch(self, epoch=0):

        split = "test"
        self.model.eval()
        all_sources, all_targets = [], []

        if self.device == "cpu":
            print_on_rank_zero(
                f"[CPU] Epoch: {epoch} | Test_Dataloader: {len(self.test_dataloader)}"
            )
        else:
            print_on_rank_zero(
                f"[GPU{self.global_rank}] Epoch: {epoch} | Test_Dataloader: {len(self.test_dataloader)}"
            )
            self.test_dataloader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):

                source_img = data["img"].to(self.device)
                target_data = data["target_data"].float().to(self.device)
                img_mask = data.get("img_mask", None)

                if img_mask is not None:
                    img_mask = img_mask.to(self.device)
                    outputs = self.best_model(source_img, img_mask)
                else:
                    outputs = self.best_model(source_img)

                loss = self.criterion(outputs, target_data)

                # absolute_loss = torch.abs(outputs - target_data)
                # percentage_loss = (absolute_loss / torch.abs(target_data)) * 100
                # total_percentage_loss = torch.sum(percentage_loss) / len(
                #     percentage_loss
                # )

                self.metric_values["loss"] = torch.tensor(loss.item()).to(self.device)
                # self.metric_values["percentage_loss"] = torch.tensor(
                #     total_percentage_loss.item()
                # ).to(self.device)

                self.metric_values["running_loss"] += torch.tensor(loss.item()).to(
                    self.device
                )
                # self.metric_values["percentage_running_loss"] += torch.tensor(
                #     total_percentage_loss.item()
                # ).to(self.device)

                self._calc_metric_scores(
                    outputs,
                    target_data,
                    split,
                    iteration_type="batch",
                    iteration_num=self.current_test_step,
                )

                all_sources.append(outputs)
                all_targets.append(target_data)

                self.current_test_step += 1
                self.testing_progress_bar.update(1)

                if batch + 1 == self.max_testing_steps:
                    break

        self._calc_metric_scores(
            all_sources, all_targets, split, iteration_type="epoch", iteration_num=epoch
        )

        # Create the grouped boxplot

    #     self.create_grouped_boxplot1(all_sources, all_targets)
    #     self.create_grouped_boxplot2(all_sources, all_targets)

    # def create_grouped_boxplot1(self, all_sources, all_targets):
    #     # Flatten the lists of arrays
    #     all_sources = [item for sublist in all_sources for item in sublist]
    #     all_targets = [item for sublist in all_targets for item in sublist]

    #     # Create box plot
    #     plt.figure(figsize=(10, 6))
    #     plt.boxplot(
    #         [all_targets, all_sources], labels=["True Values", "Predicted Values"]
    #     )
    #     plt.title("Test Set - True vs Predicted Values")
    #     plt.ylabel("Values")
    #     plt.savefig(f"box_plot_1_test_set.png")
    #     plt.close()

    # def create_grouped_boxplot2(self, all_sources, all_targets):
    #     sources = np.concatenate(all_sources)
    #     targets = np.concatenate(all_targets)

    #     df = pd.DataFrame(
    #         {
    #             "Value": np.concatenate([sources, targets]),
    #             "Type": ["Predicted"] * len(sources) + ["True"] * len(targets),
    #         }
    #     )

    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(x="Type", y="Value", data=df)
    #     plt.title("Test Set - Predicted and True Values")
    #     plt.savefig(f"box_plot_2_test_set.png")
    #     plt.close()

    def _early_stop_check(self, epoch):
        tracking_metric = self.metric_values[
            f"running_{self.early_stopping_configs['tracking_metric']}"
        ]
        if tracking_metric > self.tracking_metric_best_score:
            self.early_stop_counter = 0
            self.best_model = deepcopy(self.model)
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.early_stopping_configs["patience"]:
                print(f"Early stopping at epoch: {epoch}")
                self.early_stop = True

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": (
                self.model.module.state_dict()
                if torch.cuda.is_available()
                else self.model.state_dict()
            ),
            "EPOCHS_RUN": epoch,
            "CURRENT_TRAIN_STEP": self.current_train_step,
            "CURRENT_VAL_STEP": self.current_val_step,
            "CURRENT_TEST_STEP": self.current_test_step,
            # "CURRENT_LR": ""
        }
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(snapshot, self.snapshot_path + "/model_snapshot.pkl")
        print_on_rank_zero(
            f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}"
        )

    def _save_best_model(self):
        best_model = (
            self.best_model.module.state_dict()
            if torch.cuda.is_available()
            else self.best_model.state_dict()
        )
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(best_model, self.snapshot_path + "/best_model.pt")
        print_on_rank_zero(f"Best Model saved at {self.snapshot_path}")

    def train(self):

        best_metric_value = None

        for epoch in range(self.epochs_run, self.max_epochs):

            self.metric_values = initialize_metrics(
                EVAL_METRICS=self.eval_metrics, split="train", device=self.device
            )
            self._run_train_epoch(epoch)
            self.metric_values = initialize_metrics(
                EVAL_METRICS=self.eval_metrics, split="val", device=self.device
            )
            self._run_val_epoch(epoch)

            if self.device != "cpu":
                barrier()

            current_metric_value = (
                self.metric_values["running_loss"] / self.max_validation_steps
            ) / self.world_size

            if self.global_rank == 0 and ((epoch + 1) % self.save_every_n_epoch == 0):
                self._save_snapshot(epoch)

            if self.early_stopping_configs:
                self._early_stop_check(epoch)
            else:
                if (
                    best_metric_value is None
                    or current_metric_value < best_metric_value
                ):
                    best_metric_value = current_metric_value
                    self.best_model = deepcopy(self.model)

                    if self.global_rank == 0:
                        self._save_best_model()

                    print_on_rank_zero(
                        f"Saved Best Model at Epoch: {epoch} with best_metric_value: {best_metric_value}"
                    )

            if self.early_stop:
                break
            if self.scheduler:
                self.scheduler.step()

        self.metric_values = initialize_metrics(
            EVAL_METRICS=self.eval_metrics, split="test", device=self.device
        )
        self._run_test_epoch()


def training_pipeline(args):

    # parse yml to dict - to get configurations
    with open(args.config_file, mode="r") as f:
        config_file = yaml.safe_load(f)
        print_on_rank_zero("config_file = ", config_file, "\n")

    seed_val = config_file.get("seed", random.randint(1, 10000))
    config_file["seed"] = seed_val

    if int(os.environ.get("RANK", 0)) == 0:
        # check: if resume training then get exising run_id else None
        run_id = None
        if config_file.get("checkpointing_configs"):
            if config_file["checkpointing_configs"].get("resume"):
                run_id = config_file["checkpointing_configs"].get("run_id", None)
        mlflow.start_run(run_id=run_id)

    seed_everything(seed_val)

    # setup multi-node, multi-gpu training & testing via DDP
    if torch.cuda.is_available():
        init_process_group(**config_file["dist_params"])
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # get training & testing objects: model, optimizer, loss_function, dataloaders
    model, num_params, num_trainable_params, model_size = get_model(
        model_configs=config_file["model_configs"]
    )
    optimizer = get_optimizer(
        model_parameters=model.parameters(),
        optimizer_configs=config_file["optimizer_configs"],
    )
    criterion = get_loss_function(
        loss_function_configs=config_file["loss_function_configs"]
    )
    prepared_data_dict = prepare_dataloader(
        dataset_configs=config_file["dataset_configs"],
        dataloader_configs=config_file["dataloader_configs"],
        seed_val=seed_val,
    )

    train_dataloader = prepared_data_dict["dataloaders"]["train"]
    val_dataloader = prepared_data_dict["dataloaders"]["val"]
    test_dataloader = prepared_data_dict["dataloaders"]["test"]

    train_dataset = prepared_data_dict["datasets"]["train"]
    val_dataset = prepared_data_dict["datasets"]["val"]
    test_dataset = prepared_data_dict["datasets"]["test"]

    # Add Learning Rate Scheduler Here
    scheduler = None
    scheduler_configs = config_file.get("lr_scheduler_configs")
    if scheduler_configs:
        warmup_configs = scheduler_configs.get("warmup_configs", None)
        if warmup_configs:
            scheduler_configs.pop("warmup_configs")
        scheduler = get_lr_scheduler(
            optimizer=optimizer, scheduler_configs=scheduler_configs
        )
        if warmup_configs:
            multiplier = warmup_configs.get("multiplier", 1)
            total_epoch = warmup_configs.get("total_epoch", 1)
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=multiplier,
                total_epoch=total_epoch,
                after_scheduler=scheduler,
            )

    max_steps = config_file["train_test_configs"].get("max_steps")
    if max_steps is None:
        max_training_steps = len(train_dataloader)
        max_validation_steps = len(val_dataloader)
        max_testing_steps = len(test_dataloader)

        num_training_steps = config_file["train_test_configs"]["epochs"] * len(
            train_dataloader
        )
        num_val_steps = config_file["train_test_configs"]["epochs"] * len(
            val_dataloader
        )
        num_test_steps = config_file["train_test_configs"]["epochs"] * len(
            test_dataloader
        )
    else:
        max_training_steps = min(max_steps, len(train_dataloader))
        max_validation_steps = min(max_steps, len(val_dataloader))
        max_testing_steps = min(max_steps, len(test_dataloader))

        num_training_steps = (
            config_file["train_test_configs"]["epochs"] * max_training_steps
        )
        num_val_steps = (
            config_file["train_test_configs"]["epochs"] * max_validation_steps
        )
        num_test_steps = config_file["train_test_configs"]["epochs"] * max_testing_steps

    config_file["train_test_configs"]["max_training_steps"] = max_training_steps
    config_file["train_test_configs"]["max_validation_steps"] = max_validation_steps
    config_file["train_test_configs"]["max_testing_steps"] = max_testing_steps
    config_file["train_test_configs"]["num_training_steps"] = num_training_steps
    config_file["train_test_configs"]["num_val_steps"] = num_val_steps
    config_file["train_test_configs"]["num_test_steps"] = num_test_steps

    # log all args / hyperparameters to MLflow. log values only from 1st gpu (otherwise we will log duplicate values from all the gpus)
    if int(os.environ.get("RANK", 0)) == 0:
        config = vars(args)
        config.update(
            {
                "datasets_length": json.dumps(
                    {
                        "train": len(train_dataset),
                        "val": len(val_dataset),
                        "test": len(test_dataset),
                        "total": len(train_dataset)
                        + len(val_dataset)
                        + len(test_dataset),
                    }
                ),
                "dataloaders_length": json.dumps(
                    {
                        "train": len(train_dataloader),
                        "val": len(val_dataloader),
                        "test": len(test_dataloader),
                        "total": len(train_dataloader)
                        + len(val_dataloader)
                        + len(test_dataloader),
                    }
                ),
                "steps": json.dumps(
                    {
                        "train": num_training_steps,
                        "val": num_val_steps,
                        "test": num_test_steps,
                        "total": num_training_steps + num_val_steps + num_test_steps,
                    }
                ),
                "model_config": json.dumps(
                    {
                        "num_params": "{:.3f} K".format(num_params / 1000),
                        "num_trainable_params": "{:.3f} K".format(
                            num_trainable_params / 1000
                        ),
                        "model_size": "{:.3f} MB".format(model_size),
                    }
                ),
                "optimizer": optimizer.__class__.__name__,
                "loss_function": criterion.__class__.__name__,
            }
        )
        mlflow.log_params(config)
        mlflow.log_artifact(args.config_file)
        active_run = mlflow.active_run()
        artifact_repo = get_artifact_repository(active_run.info.artifact_uri)
        artifacts_path = artifact_repo.__getattribute__("artifact_dir")
        if config_file["checkpointing_configs"].get("checkpoint_path") is None:
            config_file["checkpointing_configs"]["checkpoint_path"] = os.path.join(
                artifacts_path, "trained_model"
            )

    # create and setup training & testing objects on multi-node, multi-gpus
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        train_dataloader=train_dataloader,
        train_test_configs=config_file["train_test_configs"],
        early_stopping_configs=config_file.get("early_stopping_configs", None),
        checkpointing_configs=config_file.get("checkpointing_configs", None),
    )

    print_on_rank_zero("Training Start!!!\n")
    trainer.train()
    print_on_rank_zero("\nTraining Done!!!")

    if int(os.environ.get("RANK", 0)) == 0:
        mlflow.end_run()

    if torch.cuda.is_available():
        destroy_process_group()


if __name__ == "__main__":

    print_on_rank_zero("root_path = ", root_path)

    parser = argparse.ArgumentParser(description="code generation training.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs/train.yml"),
        required=False,
        help="Path to config file.",
    )

    args = parser.parse_args()
    print_on_rank_zero(
        "args:\n", json.dumps(vars(args), sort_keys=True, indent=4), "\n"
    )

    training_pipeline(args)
