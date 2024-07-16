from registry import LR_SCHEDULERS_REGISTRY
import torch.optim.lr_scheduler as lr_scheduler


@LR_SCHEDULERS_REGISTRY.register()
def learning_rate_scheduler(optimizer, scheduler_configs={}):
    """
    Load a PyTorch scheduler with the specified name and arguments.

    Parameters:
        optimizer: Optimizer for which the scheduler will be used.
        scheduler_configs (dict): Dictionary containing the scheduler name and its arguments.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: PyTorch scheduler instance.
    """
    scheduler_name = scheduler_configs.pop("scheduler_name", None)
    if scheduler_name is None:
        raise ValueError("Scheduler name not provided")

    scheduler_module = getattr(lr_scheduler, scheduler_name, None)

    if scheduler_module is None or not callable(scheduler_module):
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")

    return scheduler_module(optimizer, **scheduler_configs)


print("Registered functions in LR_SCHEDULERS_REGISTRY:", LR_SCHEDULERS_REGISTRY.keys())
