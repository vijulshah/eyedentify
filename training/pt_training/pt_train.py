import os
import sys
import json
import argparse

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

from common_utils import print_on_rank_zero
from registry_utils import import_registered_modules
from inference.predict import run_prediction_pipeline
from training.pt_training.train_utils import set_args_and_configs
from training.pt_training.training_pipeline import training_pipeline

import_registered_modules()

num_local_gpus = int(os.getenv("SLURM_GPUS_ON_NODE", 1))
print("num_local_gpus = ", num_local_gpus)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_local_gpus)))

if __name__ == "__main__":

    print_on_rank_zero("root_path = ", root_path)

    parser = argparse.ArgumentParser(description="script to run training and inference pipelines.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to data folder.",
    )
    parser.add_argument(
        "--registered_model_name",
        type=str,
        required=False,
        help="Model name.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        help="Experiment name.",
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
    print_on_rank_zero("args:\n", json.dumps(vars(args), sort_keys=True, indent=4), "\n")

    config_file = set_args_and_configs(args)

    if config_file.get("run_training_pipeline", True):
        trained_data = training_pipeline(args, config_file)
        config_file = trained_data["config_file"]

    if (
        config_file.get("run_inference_pipeline", True)
        and int(os.getenv("RANK", 0)) == 0
        and int(os.getenv("LOCAL_RANK", 0)) == 0
    ):
        if len(config_file["dataset_configs"].get("selected_participants", [])) == 0:
            config_file["dataset_configs"]["selected_participants"] = config_file["dataset_configs"][
                "left_out_participants_for_test"
            ]
        config_file["log_mlflow_params"] = False
        run_prediction_pipeline(args, config_file)
