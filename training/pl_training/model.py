import os
import sys
import torch
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

from common_utils import get_model, get_loss_function
from training.pt_training.train_utils import get_optimizer, get_scheduler


class NN(pl.LightningModule):
    def __init__(self, model_configs, loss_function_configs, optimizer_configs, lr_scheduler_configs=None):
        super().__init__()

        self.save_hyperparameters()

        self.model, _, _, _ = get_model(model_configs=model_configs.copy(), use_ddp=False)
        self.criterion = get_loss_function(loss_function_configs=loss_function_configs.copy())
        self.optimizer = get_optimizer(
            model_parameters=self.model.parameters(),
            optimizer_configs=optimizer_configs.copy(),
        )
        if lr_scheduler_configs is not None:
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_configs=lr_scheduler_configs.copy(),
            )
        else:
            self.scheduler = None

        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()

    def forward(self, x):
        return self.model(x)

    def _common_step(self, x, y=None):
        scores = self.forward(x)
        loss = self.criterion(scores, y) if y is not None else None
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["target_data"]
        loss, scores, y = self._common_step(x, y)
        metrics = {
            "loss": loss,
            "mae": self.mae(scores, y),
            "mape": self.mape(scores, y),
        }
        self.log_metrics(split="train", metrics=metrics, batch_size=len(y))
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch["img"], batch["target_data"]
        loss, scores, y = self._common_step(x, y)
        metrics = {
            "loss": loss,
            "mae": self.mae(scores, y),
            "mape": self.mape(scores, y),
        }
        self.log_metrics(split="val", metrics=metrics, batch_size=len(y))
        return {"loss": loss, "scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        x, y = batch["img"], batch["target_data"]
        loss, scores, y = self._common_step(x, y)
        metrics = {
            "loss": loss,
            "mae": self.mae(scores, y),
            "mape": self.mape(scores, y),
        }
        self.log_metrics(split="test", metrics=metrics, batch_size=len(y))
        return {"loss": loss, "scores": scores, "y": y}

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return self.optimizer

    def log_metrics(self, split, metrics, batch_size):
        for key, value in metrics.items():
            self.log(
                f"{split}_batch/{key}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{split}_epoch/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
