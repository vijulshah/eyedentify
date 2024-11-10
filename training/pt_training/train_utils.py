import sys
import yaml
import random
import os.path as osp
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from registry import LR_SCHEDULERS_REGISTRY, OPTIMIZERS_REGISTRY


def get_lr_scheduler(optimizer, scheduler_configs):
    return LR_SCHEDULERS_REGISTRY.get("learning_rate_scheduler")(optimizer, scheduler_configs)


def get_optimizer(model_parameters, optimizer_configs):
    return OPTIMIZERS_REGISTRY.get("optimizers")(model_parameters, optimizer_configs)


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
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
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
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
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


def lr_lambda(current_step, num_training_steps, num_warmup_steps=0):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_scheduler(optimizer, lr_scheduler_configs=None, num_training_steps_for_linear_schedule_with_warmup=1):

    scheduler = None
    if lr_scheduler_configs:
        scheduler_name = lr_scheduler_configs.get("scheduler_name")
        if scheduler_name == "LinearTransformer":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps_for_linear_schedule_with_warmup,
            )
        else:
            warmup_configs = lr_scheduler_configs.get("warmup_configs", None)
            if warmup_configs:
                lr_scheduler_configs.pop("warmup_configs")
            scheduler = get_lr_scheduler(optimizer=optimizer, scheduler_configs=lr_scheduler_configs)
            if warmup_configs:
                multiplier = warmup_configs.get("multiplier", 1)
                total_epoch = warmup_configs.get("total_epoch", 1)
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=multiplier,
                    total_epoch=total_epoch,
                    after_scheduler=scheduler,
                )
    return scheduler


def set_args_and_configs(args):

    with open(args.config_file, mode="r") as f:
        config_file = yaml.safe_load(f)

    if args.exp_name:
        config_file["exp_name"] = args.exp_name

    if args.data_path:
        config_file["dataset_configs"]["data_path"] = args.data_path

    if args.selected_targets:
        config_file["dataset_configs"]["dataset_registry_params"]["selected_targets"] = [
            str(item) for item in args.selected_targets
        ]

    if args.registered_model_name:
        config_file["model_configs"]["registered_model_name"] = args.registered_model_name

    if args.split_fold:
        split_fold = args.split_fold
        config_file["dataset_configs"]["split_fold"] = split_fold
        if split_fold == "fold1":
            left_out_participants_for_val = [3, 7, 15, 44, 51]
            left_out_participants_for_test = [1, 4, 6, 25, 36]
        elif split_fold == "fold2":
            left_out_participants_for_val = [4, 8, 16, 45, 50]
            left_out_participants_for_test = [2, 5, 7, 26, 37]
        elif split_fold == "fold3":
            left_out_participants_for_val = [8, 12, 22, 34, 47]
            left_out_participants_for_test = [3, 16, 26, 38, 43]
        elif split_fold == "fold4":
            left_out_participants_for_val = [5, 13, 23, 33, 41]
            left_out_participants_for_test = [9, 19, 29, 39, 49]
        elif split_fold == "fold5":
            left_out_participants_for_val = [1, 11, 20, 32, 48]
            left_out_participants_for_test = [10, 14, 24, 28, 31]

        config_file["dataset_configs"]["left_out_participants_for_val"] = left_out_participants_for_val
        config_file["dataset_configs"]["left_out_participants_for_test"] = left_out_participants_for_test
    else:
        if args.left_out_participants_for_val:
            config_file["dataset_configs"]["left_out_participants_for_val"] = [
                int(item) for item in args.left_out_participants_for_val
            ]
        if args.left_out_participants_for_test:
            config_file["dataset_configs"]["left_out_participants_for_test"] = [
                int(item) for item in args.left_out_participants_for_test
            ]

    if args.img_size:
        config_file["dataset_configs"]["dataset_registry_params"]["img_size"] = [int(item) for item in args.img_size]

    seed_val = config_file.get("seed", random.randint(1, 10000))
    config_file["seed"] = seed_val

    print("config_file = ", config_file, "\n")
    return config_file
