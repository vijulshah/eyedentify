import torch.optim as optim
from registry import OPTIMIZERS_REGISTRY


@OPTIMIZERS_REGISTRY.register()
def optimizers(model_parameters, optimizer_configs):
    """
    Load a PyTorch optimizer with the specified name and arguments.

    Parameters:
        model_parameters: Parameters of the model for which the optimizer will be used.
        optimizer_configs (str):  Dictionary containing the optimizer name and its arguments.

    Returns:
        torch.optim.Optimizer: PyTorch optimizer instance.
    """
    optimizer_name = optimizer_configs.pop("optimizer_name", None)
    if optimizer_name is None:
        raise ValueError("Optimizer name not provided")

    optimizer_module = getattr(optim, optimizer_name, None)

    if optimizer_module is None or not callable(optimizer_module):
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    return optimizer_module(model_parameters, **optimizer_configs)


print("Registered functions in OPTIMIZERS_REGISTRY:", OPTIMIZERS_REGISTRY.keys())
