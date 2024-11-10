import os
import sys
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

from registry_utils import import_registered_modules
from inference.predict import run_prediction_pipeline
from training.pt_training.train_utils import set_args_and_configs
from training.pl_training.callbacks import get_model_checkpoint_callback

import_registered_modules()

from model import NN
from dataset import EyeDentifyDataModule

# torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore")


def train_model(config_file):
    logger = TensorBoardLogger(config_file["log_dir"], name=f"{config_file['exp_name']}_TRAIN")
    logger.log_hyperparams(config_file)

    profiler = None
    if config_file["profiler"] is not None:
        if config_file["profiler"] == "custom":
            profiler = PyTorchProfiler(
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"{config_file['log_dir']}/pytorch_profiler_TRAIN"
                ),
                schedule=torch.profiler.schedule(**config_file["profiler_args"]),
            )
        else:
            profiler = config_file["profiler"]

    model = NN(
        model_configs=config_file["model_configs"],
        loss_function_configs=config_file["loss_function_configs"],
        optimizer_configs=config_file["optimizer_configs"],
        lr_scheduler_configs=config_file.get("lr_scheduler_configs", None),
    )

    dm = EyeDentifyDataModule(
        dataset_configs=config_file["dataset_configs"],
        dataloader_configs=config_file["dataloader_configs"],
    )

    checkpoint_callback = get_model_checkpoint_callback(dirpath=logger.log_dir)

    train_val_trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        **config_file["pl_trainer_configs"],
        callbacks=[checkpoint_callback],
    )

    train_val_trainer.fit(model=model, datamodule=dm)

    # Load the best model checkpoint after training
    best_model_path = checkpoint_callback.best_model_path
    config_file["model_configs"]["model_path"] = best_model_path

    return config_file, dm


def test_model(config_file, dm=None):
    logger = TensorBoardLogger(config_file["log_dir"], name=f"{config_file['exp_name']}_TEST")
    logger.log_hyperparams(config_file)

    profiler = None
    if config_file["profiler"] is not None:
        if config_file["profiler"] == "custom":
            profiler = PyTorchProfiler(
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"{config_file['log_dir']}/pytorch_profiler_TEST"
                ),
                schedule=torch.profiler.schedule(**config_file["profiler_args"]),
            )
        else:
            profiler = config_file["profiler"]

    best_model_path = config_file["model_configs"]["model_path"]
    model = NN.load_from_checkpoint(best_model_path)

    if dm is None:
        dm = EyeDentifyDataModule(
            dataset_configs=config_file["dataset_configs"],
            dataloader_configs=config_file["dataloader_configs"],
        )

    if (
        config_file["pl_trainer_configs"].get("devices", None) is not None
        and config_file["pl_trainer_configs"].get("devices", None) != "cpu"
    ):
        config_file["pl_trainer_configs"]["devices"] = 1

    test_trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        **config_file["pl_trainer_configs"],
    )

    test_trainer.test(model=model, datamodule=dm)

    return config_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script to run training and inference pipelines.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to config_file file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to data folder.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        help="Experiment name.",
    )
    parser.add_argument(
        "--registered_model_name",
        type=str,
        required=False,
        help="Model name.",
    )
    parser.add_argument(
        "--split_fold",
        type=str,
        required=False,
        help="specify any from: fold1, fold2, fold3, fold4 or fold5.",
    )
    parser.add_argument(
        "--selected_targets",
        nargs="+",
        required=False,
        help="left or right or both pupils.",
    )
    parser.add_argument(
        "--left_out_participants_for_val",
        nargs="+",
        required=False,
        help="participant ids for validation",
    )
    parser.add_argument(
        "--left_out_participants_for_test",
        nargs="+",
        required=False,
        help="participant ids for testing",
    )
    parser.add_argument(
        "--img_size",
        required=False,
        nargs="+",
        help="image size input dimensions",
    )

    args = parser.parse_args()
    config_file = set_args_and_configs(args)
    pl.seed_everything(config_file["seed"], workers=True)

    if config_file.get("run_training_pipeline", True):

        config_file, dm = train_model(config_file)

        # NOTE: It is recommended to test with Trainer(devices=1) since distributed strategies such as DDP use DistributedSampler internally, which replicates some samples to make sure all devices have same batch size in case of uneven inputs. This is helpful to make sure benchmarking for research papers is done the right way.
        if (
            config_file["pl_trainer_configs"].get("devices", None) is not None
            and config_file["pl_trainer_configs"].get("devices", None) != "cpu"
        ):
            config_file["pl_trainer_configs"]["devices"] = 1

        if int(os.getenv("RANK", 0)) == 0 and int(os.getenv("LOCAL_RANK", 0)) == 0:
            config_file = test_model(config_file, dm)

            if config_file.get("run_inference_pipeline", True):
                if len(config_file["dataset_configs"].get("selected_participants", [])) == 0:
                    config_file["dataset_configs"]["selected_participants"] = config_file["dataset_configs"][
                        "left_out_participants_for_test"
                    ]
                config_file["log_mlflow_params"] = True
                run_prediction_pipeline(args, config_file)

# Visualize logs: tensorboard --logdir=./<path_to_log_dir>/<log_dir>/<experiment_name>
