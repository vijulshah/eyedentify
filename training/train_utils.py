import os
import sys
import torch
import random
import numpy as np
import os.path as osp

# To join other directories with this file, append the main folder
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)
from registry import (
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    LR_SCHEDULERS_REGISTRY,
    OPTIMIZERS_REGISTRY,
)


def calc_model_size(model):
    """
    Calculates the size of the PyTorch model in terms of parameters.

    Parameters:
        model (nn.Module): PyTorch model.

    Returns:
        tuple: Number of total parameters, trainable parameters, and model size in MB.
    """
    num_total_params = sum(p.numel() for p in model.parameters())
    print("\nNumber of total parameters in the model: ", num_total_params / 1000)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters in the model: ", num_trainable_params / 1000)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(model_size))

    return num_total_params, num_trainable_params, model_size


def get_loss_function(loss_function_configs):
    return METRIC_REGISTRY.get("loss_function")(loss_function_configs)


def get_lr_scheduler(optimizer, scheduler_configs):
    return LR_SCHEDULERS_REGISTRY.get("learning_rate_scheduler")(
        optimizer, scheduler_configs
    )


def get_optimizer(model_parameters, optimizer_configs):
    return OPTIMIZERS_REGISTRY.get("optimizers")(model_parameters, optimizer_configs)


def initialize_metrics(EVAL_METRICS, split, device="cpu"):
    return METRIC_REGISTRY.get("initialize_metrics")(EVAL_METRICS, split, device=device)


def get_model(model_configs):
    registered_model = MODEL_REGISTRY.get(model_configs["registered_model_name"])
    model_configs.pop("registered_model_name")
    if len(model_configs) > 0:
        model = registered_model(model_configs)
    else:
        model = registered_model()
    num_params, num_trainable_params, model_size = calc_model_size(model)
    return model, num_params, num_trainable_params, model_size


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def print_on_rank_zero(*data):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*data)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
