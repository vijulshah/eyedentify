import torch
import torch.nn as nn
from registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def loss_function(loss_function_configs):
    """
    Load a PyTorch loss function with the specified name and arguments.

    Parameters:
        loss_function_configs (str):  Dictionary containing the los function name and its arguments.
        kwargs: Additional keyword arguments to be passed to the loss function.

    Returns:
        torch.nn.Module: PyTorch loss function instance.
    """
    loss_function_name = loss_function_configs.pop("loss_function_name", None)
    if loss_function_name is None:
        raise ValueError("Loss function name not provided")
    else:
        loss_module = getattr(nn, loss_function_name, None)

        if loss_module is None or not callable(loss_module):
            raise ValueError(f"Invalid loss function name: {loss_function_name}")

        return loss_module(**loss_function_configs)


@METRIC_REGISTRY.register()
def initialize_metrics(EVAL_METRICS, split, device="cpu"):
    """
    Initialize metrics for tracking prediction scores.

    Args:
        EVAL_METRICS (list): List of evaluation metrics to initialize.

    Returns:
        dict: Dictionary containing initialized metric values.

    """
    BATCH_metrics = ["loss"]
    EPOCH_metrics = ["running_loss"]

    # if split == "train":
    #     BATCH_metrics.append("grad_norm")
    #     EPOCH_metrics.append("running_grad_norm")

    for metric in EVAL_METRICS:
        EPOCH_metrics.append(f"running_{metric}")

    ALL_metrics = BATCH_metrics + EPOCH_metrics
    metric_values = {metric: torch.tensor(0.0).to(device) for metric in ALL_metrics}

    return metric_values


print("Registered metrics in METRIC_REGISTRY:", METRIC_REGISTRY.keys())
