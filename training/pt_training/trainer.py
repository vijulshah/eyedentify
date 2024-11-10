import os
import sys
import torch
import mlflow
import evaluate
from tqdm import tqdm
from copy import deepcopy
from torch.distributed import barrier
from torch.distributed.nn import all_reduce, ReduceOp

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

from common_utils import initialize_metrics, print_on_rank_zero


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
        selected_targets,
        use_ddp=True,
        device="cpu",
        rank=0,
        world_size_or_num_gpus=1,
    ):
        self.device = device
        self.global_rank = rank
        self.world_size = world_size_or_num_gpus
        self.use_ddp = use_ddp

        self.epochs_run = 0
        self.current_train_step = 0
        self.current_val_step = 0
        self.current_test_step = 0
        self.tracking_metric_best_score = 0.0
        self.early_stop_counter = 0
        self.early_stop = False

        self.model = model
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

        # self.clip_grad_norm = train_test_configs.get("clip_grad_norm", None)
        self.eval_metrics_average = train_test_configs.get("eval_metrics_average")
        eval_metrics = train_test_configs["eval_metrics"]
        self.eval_metrics = eval_metrics
        self.metric_values = {}
        self.metrics = {}
        self.best_model = None

        # for metric_name in eval_metrics:
        #     if metric_name == "mape":
        #         self.metrics[metric_name] = evaluate.load(
        #             metric_name, "multilist", num_process=self.world_size, process_id=self.global_rank
        #         )
        #     else:
        #         self.metrics[metric_name] = evaluate.load(
        #             metric_name, num_process=self.world_size, process_id=self.global_rank
        #         )

        for metric_name in eval_metrics:
            if metric_name == "mape":
                self.metrics[metric_name] = evaluate.load(metric_name, "multilist")
            else:
                self.metrics[metric_name] = evaluate.load(metric_name)

        if checkpointing_configs.get("save_every_n_epoch") is None:
            self.save_every_n_epoch = self.max_epochs
        else:
            self.save_every_n_epoch = checkpointing_configs.get("save_every_n_epoch")

        self.checkpoint_path = checkpointing_configs.get("checkpoint_path")

        self.selected_targets = selected_targets

    def _calc_metric_scores(self, predictions, labels, split, iteration_type, iteration_num):

        if iteration_type == "batch":
            for key in self.eval_metrics:
                if key == "mse":
                    self.metric_values[key] = self.metrics[key].compute(
                        predictions=predictions.view(-1),
                        references=labels.view(-1),
                        squared=True,  # if false, then it returns RMSE
                    )[key]
                else:
                    self.metric_values[key] = self.metrics[key].compute(predictions=predictions, references=labels)[key]
                self.metric_values[key] = torch.tensor(self.metric_values[key]).to(self.device)
                self.metric_values.update(
                    {f"running_{key}": self.metric_values[key] + self.metric_values[f"running_{key}"]}
                )

        if self.device != "cpu" and self.use_ddp == True:
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
                    log_value = (self.metric_values[metric_name] / max_steps) / self.world_size

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

    # def _calc_metric_scores(self, predictions, labels, split, iteration_type, iteration_num):

    #     if iteration_type == "batch":
    #         for key in self.eval_metrics:
    #             if key == "mse":
    #                 mv = self.metrics[key].compute(
    #                     predictions=predictions.view(-1),
    #                     references=labels.view(-1),
    #                     squared=True,  # if false, then it returns RMSE
    #                 )
    #             else:
    #                 mv = self.metrics[key].compute(predictions=predictions, references=labels)

    #             if self.global_rank == 0 and mv is not None:
    #                 self.metric_values[key] = torch.tensor(mv[key]).to(self.device)
    #                 self.metric_values.update(
    #                     {f"running_{key}": self.metric_values[key] + self.metric_values[f"running_{key}"]}
    #                 )

    #     if self.global_rank == 0:
    #         for metric_name in self.metric_values.keys():
    #             log_value = None
    #             if iteration_type == "batch":
    #                 if "running" not in metric_name:
    #                     log_value = self.metric_values[metric_name]
    #             else:
    #                 if "running" in metric_name:
    #                     if split == "train":
    #                         max_steps = self.max_training_steps
    #                     elif split == "val":
    #                         max_steps = self.max_validation_steps
    #                     else:
    #                         max_steps = self.max_testing_steps
    #                     log_value = self.metric_values[metric_name] / max_steps
    #             if log_value is not None:
    #                 mlflow.log_metrics(
    #                     {f"{split}_{iteration_type}_{metric_name}": log_value.item()},
    #                     step=iteration_num,
    #                 )

    #         if split == "train":
    #             curr_lr = (
    #                 self.scheduler.optimizer.param_groups[0]["lr"]
    #                 if self.scheduler
    #                 else self.optimizer.param_groups[0]["lr"]
    #             )
    #             mlflow.log_metrics(
    #                 {f"{iteration_type}_lr": curr_lr},
    #                 step=iteration_num,
    #             )

    def _run_train_epoch(self, epoch):

        split = "train"
        self.model.train()

        if self.device == "cpu":
            print_on_rank_zero(f"[CPU] Epoch: {epoch} | Train_Dataloader: {len(self.train_dataloader)}")
        elif self.use_ddp == True:
            self.train_dataloader.sampler.set_epoch(epoch)

        for batch, data in enumerate(self.train_dataloader):

            self.optimizer.zero_grad()

            source_img = data["img"].to(self.device)
            target_data = data["target_data"].float().to(self.device)
            outputs = self.model(source_img)
            if batch == 0 and epoch == 0:
                print_on_rank_zero("\nsource_img = ", source_img.shape)
                print_on_rank_zero("target_data = ", target_data.shape)
                print_on_rank_zero("outputs = ", outputs.shape)

            loss = self.criterion(outputs, target_data)
            loss.backward()
            self.optimizer.step()

            self.metric_values["loss"] = torch.tensor(loss.detach().item()).to(self.device)
            self.metric_values["running_loss"] += torch.tensor(loss.detach().item()).to(self.device)

            self._calc_metric_scores(
                outputs,
                target_data,
                split,
                iteration_type="batch",
                iteration_num=self.current_train_step,
            )

            self.current_train_step += 1
            self.training_progress_bar.update(1)

            if batch + 1 == self.max_training_steps:
                break

        self._calc_metric_scores([], [], split, iteration_type="epoch", iteration_num=epoch)

    def _run_val_epoch(self, epoch):

        split = "val"
        self.model.eval()

        if self.device == "cpu":
            print_on_rank_zero(f"[CPU] Epoch: {epoch} | Val_Dataloader: {len(self.val_dataloader)}")
        else:
            print_on_rank_zero(f"[GPU{self.global_rank}] Epoch: {epoch} | Val_Dataloader: {len(self.val_dataloader)}")
            if self.use_ddp == True:
                self.val_dataloader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch, data in enumerate(self.val_dataloader):

                source_img = data["img"].to(self.device)
                target_data = data["target_data"].float().to(self.device)
                outputs = self.model(source_img)

                loss = self.criterion(outputs, target_data)
                self.metric_values["loss"] = torch.tensor(loss.item()).to(self.device)
                self.metric_values["running_loss"] += torch.tensor(loss.item()).to(self.device)

                self._calc_metric_scores(
                    outputs,
                    target_data,
                    split,
                    iteration_type="batch",
                    iteration_num=self.current_val_step,
                )

                self.current_val_step += 1
                self.validation_progress_bar.update(1)

                if batch + 1 == self.max_validation_steps:
                    break

        self._calc_metric_scores([], [], split, iteration_type="epoch", iteration_num=epoch)

    def _run_test_epoch(self, epoch=0):

        split = "test"
        self.model.eval()

        if self.device == "cpu":
            print_on_rank_zero(f"[CPU] Epoch: {epoch} | Test_Dataloader: {len(self.test_dataloader)}")
        else:
            print_on_rank_zero(f"[GPU{self.global_rank}] Epoch: {epoch} | Test_Dataloader: {len(self.test_dataloader)}")
            if self.use_ddp == True:
                self.test_dataloader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):

                source_img = data["img"].to(self.device)
                target_data = data["target_data"].float().to(self.device)
                outputs = self.best_model(source_img)

                loss = self.criterion(outputs, target_data)
                self.metric_values["loss"] = torch.tensor(loss.item()).to(self.device)
                self.metric_values["running_loss"] += torch.tensor(loss.item()).to(self.device)

                self._calc_metric_scores(
                    outputs,
                    target_data,
                    split,
                    iteration_type="batch",
                    iteration_num=self.current_test_step,
                )

                self.current_test_step += 1
                self.testing_progress_bar.update(1)

                if batch + 1 == self.max_testing_steps:
                    break

        self._calc_metric_scores([], [], split, iteration_type="epoch", iteration_num=epoch)

    def _early_stop_check(self, epoch):
        tracking_metric = self.metric_values[f"running_{self.early_stopping_configs['tracking_metric']}"]
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
                if torch.cuda.is_available() and self.use_ddp == True
                else self.model.state_dict()
            ),
            "EPOCHS_RUN": epoch,
            "CURRENT_TRAIN_STEP": self.current_train_step,
            "CURRENT_VAL_STEP": self.current_val_step,
            "CURRENT_TEST_STEP": self.current_test_step,
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(snapshot, self.checkpoint_path + "/model_snapshot.pkl")
        print_on_rank_zero(f"Epoch {epoch} | Training snapshot saved at {self.checkpoint_path}")

    def _save_best_model(self):
        best_model = (
            self.best_model.module.state_dict()
            if torch.cuda.is_available() and self.use_ddp == True
            else self.best_model.state_dict()
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(best_model, self.checkpoint_path + "/best_model.pt")
        print_on_rank_zero(f"Best Model saved at {self.checkpoint_path}")

    def run_training(self):
        best_metric_value = None
        for epoch in range(self.epochs_run, self.max_epochs):

            self.metric_values = initialize_metrics(EVAL_METRICS=self.eval_metrics, split="train", device=self.device)
            self._run_train_epoch(epoch)
            self.metric_values = initialize_metrics(EVAL_METRICS=self.eval_metrics, split="val", device=self.device)
            self._run_val_epoch(epoch)

            if self.device != "cpu" and self.use_ddp == True:
                barrier()

            # if self.global_rank == 0:
            #     current_metric_value = self.metric_values["running_loss"] / self.max_validation_steps

            #     if self.early_stopping_configs:
            #         self._early_stop_check(epoch)
            #     else:
            #         if best_metric_value is None or current_metric_value < best_metric_value:
            #             best_metric_value = current_metric_value
            #             self.best_model = deepcopy(self.model)

            #             if self.global_rank == 0:
            #                 self._save_best_model()

            #             print_on_rank_zero(
            #                 f"Saved Best Model at Epoch: {epoch} with best_metric_value: {best_metric_value}"
            #             )

            #     if self.early_stop:
            #         break

            current_metric_value = self.metric_values["running_loss"] / self.max_validation_steps

            # if self.global_rank == 0 and ((epoch + 1) % self.save_every_n_epoch == 0):
            #     self._save_snapshot(epoch)

            if self.early_stopping_configs:
                self._early_stop_check(epoch)
            else:
                if best_metric_value is None or current_metric_value < best_metric_value:
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

    def run_testing(self):
        self.metric_values = initialize_metrics(EVAL_METRICS=self.eval_metrics, split="test", device=self.device)
        self._run_test_epoch()
