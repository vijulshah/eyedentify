import os
import sys
import json
import torch
import mlflow
from dataloader import prepare_dataloader
from torch.distributed import init_process_group, destroy_process_group
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

from training.pt_training.trainer import Trainer
from training.pt_training.train_utils import get_optimizer, get_scheduler
from common_utils import get_model, get_loss_function, print_on_rank_zero, seed_everything


def training_pipeline(args, config_file):

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size_or_num_gpus = int(os.getenv("WORLD_SIZE", 1))
    use_ddp = config_file.get("use_ddp", True)
    if torch.cuda.is_available():
        if use_ddp == True:
            init_process_group(**config_file["dist_params"], rank=rank, world_size=world_size_or_num_gpus)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    if rank == 0:
        run_id = None
        if config_file.get("checkpointing_configs"):
            if config_file["checkpointing_configs"].get("resume"):
                run_id = config_file["checkpointing_configs"].get("run_id", None)
        mlflow.start_run(run_id=run_id)

    seed_everything(seed=config_file["seed"])

    model, num_params, num_trainable_params, model_size = get_model(
        model_configs=config_file["model_configs"].copy(), device=device, use_ddp=use_ddp
    )
    criterion = get_loss_function(loss_function_configs=config_file["loss_function_configs"].copy())
    optimizer = get_optimizer(
        model_parameters=model.parameters(),
        optimizer_configs=config_file["optimizer_configs"],
    )
    prepared_data_dict = prepare_dataloader(
        dataset_configs=config_file["dataset_configs"],
        dataloader_configs=config_file["dataloader_configs"],
        seed_val=config_file["seed"],
        use_ddp=use_ddp,
        device=device,
        rank=rank,
        world_size_or_num_gpus=world_size_or_num_gpus,
    )

    train_dataloader = prepared_data_dict["dataloaders"]["train"]
    val_dataloader = prepared_data_dict["dataloaders"]["val"]
    test_dataloader = prepared_data_dict["dataloaders"]["test"]

    train_dataset = prepared_data_dict["datasets"]["train"]
    val_dataset = prepared_data_dict["datasets"]["val"]
    test_dataset = prepared_data_dict["datasets"]["test"]

    scheduler = get_scheduler(
        lr_scheduler_configs=config_file.get("lr_scheduler_configs"),
        optimizer=optimizer,
        num_training_steps_for_linear_schedule_with_warmup=config_file["train_test_configs"]["epochs"]
        * len(train_dataloader),
    )

    max_steps = config_file["train_test_configs"].get("max_steps")
    if max_steps is None:
        max_training_steps = len(train_dataloader)
        max_validation_steps = len(val_dataloader)
        max_testing_steps = len(test_dataloader)

        num_training_steps = config_file["train_test_configs"]["epochs"] * len(train_dataloader)
        num_val_steps = config_file["train_test_configs"]["epochs"] * len(val_dataloader)
        num_test_steps = config_file["train_test_configs"]["epochs"] * len(test_dataloader)
    else:
        max_training_steps = min(max_steps, len(train_dataloader))
        max_validation_steps = min(max_steps, len(val_dataloader))
        max_testing_steps = min(max_steps, len(test_dataloader))

        num_training_steps = config_file["train_test_configs"]["epochs"] * max_training_steps
        num_val_steps = config_file["train_test_configs"]["epochs"] * max_validation_steps
        num_test_steps = config_file["train_test_configs"]["epochs"] * max_testing_steps

    config_file["train_test_configs"]["max_training_steps"] = max_training_steps
    config_file["train_test_configs"]["max_validation_steps"] = max_validation_steps
    config_file["train_test_configs"]["max_testing_steps"] = max_testing_steps
    config_file["train_test_configs"]["num_training_steps"] = num_training_steps
    config_file["train_test_configs"]["num_val_steps"] = num_val_steps
    config_file["train_test_configs"]["num_test_steps"] = num_test_steps

    if int(os.environ.get("RANK", 0)) == 0:
        config = vars(args)
        config.update(
            {
                "datasets_length": json.dumps(
                    {
                        "train": len(train_dataset),
                        "val": len(val_dataset),
                        "test": len(test_dataset),
                        "total": len(train_dataset) + len(val_dataset) + len(test_dataset),
                    }
                ),
                "dataloaders_length": json.dumps(
                    {
                        "train": len(train_dataloader),
                        "val": len(val_dataloader),
                        "test": len(test_dataloader),
                        "total": len(train_dataloader) + len(val_dataloader) + len(test_dataloader),
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
                        "num_trainable_params": "{:.3f} K".format(num_trainable_params / 1000),
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

        checkpoint_path = config_file["checkpointing_configs"].get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(artifacts_path, "trained_model")
            config_file["checkpointing_configs"]["checkpoint_path"] = checkpoint_path

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
        selected_targets=config_file["dataset_configs"]["dataset_registry_params"]["selected_targets"],
        use_ddp=config_file.get("use_ddp", True),
        device=device,
        rank=rank,
        world_size_or_num_gpus=world_size_or_num_gpus,
    )

    print_on_rank_zero("\nTraining Start!!!\n")
    trainer.run_training()
    print_on_rank_zero("\nTraining Done!!!\n")

    print_on_rank_zero("\nTesting Start!!!\n")
    trainer.run_testing()
    print_on_rank_zero("\nTesting Done!!!\n")

    if rank == 0 and config_file.get("run_inference_pipeline", True) == False:
        mlflow.end_run()

    if torch.cuda.is_available() and use_ddp == True:
        destroy_process_group()

    if rank == 0:
        config_file["model_configs"]["model_path"] = trainer.checkpoint_path + "/best_model.pt"

    return {"config_file": config_file}
