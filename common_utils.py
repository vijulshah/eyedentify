import os
import sys
import torch
import random
import shutil
import numpy as np
import os.path as osp
from torch.nn.parallel import DistributedDataParallel as DDP

root_path = osp.abspath(osp.join(__file__, osp.pardir))
sys.path.append(root_path)

from registry import METRIC_REGISTRY, MODEL_REGISTRY


def print_on_rank_zero(*data):
    if int(os.getenv("RANK", 0)) == 0:
        print(*data)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def copy_file_somewhere(dest_dir, source_file_path, file_name=None):
    file_name = os.path.basename(source_file_path) if file_name is None else file_name
    destination_file_path = os.path.join(dest_dir, file_name)
    shutil.copy(source_file_path, destination_file_path)
    print_on_rank_zero("File copied at: ", destination_file_path)
    return destination_file_path


def get_model(model_configs, device="cpu", use_ddp=True):
    registered_model = MODEL_REGISTRY.get(model_configs["registered_model_name"])
    model_configs.pop("registered_model_name")
    if len(model_configs) > 0:
        model = registered_model(model_configs)
    else:
        model = registered_model()
    num_params, num_trainable_params, model_size = calc_model_size(model)
    model = model.to(device)
    if use_ddp:
        model = DDP(model)
    return model, num_params, num_trainable_params, model_size


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
    print("Model Size: {:.3f}MB".format(model_size), "\n")

    return num_total_params, num_trainable_params, model_size


def get_loss_function(loss_function_configs):
    return METRIC_REGISTRY.get("loss_function")(loss_function_configs)


def initialize_metrics(EVAL_METRICS, split, device="cpu"):
    return METRIC_REGISTRY.get("initialize_metrics")(EVAL_METRICS, split, device=device)
