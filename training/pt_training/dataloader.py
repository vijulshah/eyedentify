import os
import sys
import json
import yaml
import torch
import random
import argparse
import os.path as osp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from common_utils import print_on_rank_zero, seed_everything
from registry import DATASET_REGISTRY, DATASET_PATHS_REGISTRY


def cal_means_stds(dataset_configs, datasets_paths, data_path, split):

    means = 0.0
    stds = 0.0

    print_on_rank_zero(f"\n==================== Normalizing Data : {split} ====================")
    dataset_registry_params = dataset_configs["dataset_registry_params"]
    full_dataset = DATASET_REGISTRY.get(dataset_configs["dataset_registry"])(
        data=datasets_paths,
        data_path=data_path,
        selected_targets=dataset_registry_params.get("selected_targets"),
        img_mode=dataset_registry_params.get("img_mode", "RGB"),
        img_size=dataset_registry_params.get("img_size", None),
        means=None,
        stds=None,
        augment=False,
        augmentation_configs=None,
    )
    print_on_rank_zero("full_dataset = ", len(full_dataset))

    full_dataset_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    print_on_rank_zero("full_dataloader = ", len(full_dataset_dataloader), "\n")

    total_samples = 0
    for data in full_dataset_dataloader:
        image = data["img"]
        batch_samples = image.size(0)
        image = image.view(batch_samples, image.size(1), -1)
        means += image.mean(2).sum(0)
        stds += image.std(2).sum(0)
        total_samples += batch_samples

    means /= total_samples
    stds /= total_samples

    return means, stds


def prepare_dataloader(
    dataset_configs,
    dataloader_configs,
    seed_val,
    use_ddp=True,
    device="cpu",
    rank=0,
    world_size_or_num_gpus=1,
):
    """
    Creates or loads dataloaders (if already exists in the provided data_path) for training and testing.

    Returns:
        tuple: Train and test dataloaders and datasets.
    """
    data_path = dataset_configs.get("data_path", ".")
    data_path = os.path.join(root_path, data_path)

    split_fold = dataset_configs.get("split_fold", None)
    dataset_configs["dataset_paths_registry_params"]["split_fold"] = split_fold

    dataset_configs["dataset_paths_registry_params"]["left_out_participants_for_val"] = dataset_configs.get(
        "left_out_participants_for_val", []
    )

    dataset_configs["dataset_paths_registry_params"]["left_out_participants_for_test"] = dataset_configs.get(
        "left_out_participants_for_test", []
    )

    dataset_paths, output_folder = DATASET_PATHS_REGISTRY.get(dataset_configs["dataset_paths_registry"])(
        data_path=data_path, **dataset_configs["dataset_paths_registry_params"]
    )

    print_on_rank_zero("total images = ", len(dataset_paths))
    print_on_rank_zero("output_folder = ", output_folder)

    force_creation = dataset_configs.get("dataset_paths_registry_params", False)
    if (
        os.path.exists(f"{output_folder}/train_dataloader.pth")
        and os.path.exists(f"{output_folder}/val_dataloader.pth")
        and os.path.exists(f"{output_folder}/test_dataloader.pth")
        and os.path.exists(f"{output_folder}/train_dataset.pth")
        and os.path.exists(f"{output_folder}/val_dataset.pth")
        and os.path.exists(f"{output_folder}/test_dataset.pth")
        and force_creation == False
    ):
        map_location = device
        train_dataloader = torch.load(f"{output_folder}/train_dataloader.pth", map_location=map_location)
        val_dataloader = torch.load(f"{output_folder}/val_dataloader.pth", map_location=map_location)
        test_dataloader = torch.load(f"{output_folder}/test_dataloader.pth", map_location=map_location)
        train_dataset = torch.load(f"{output_folder}/train_dataset.pth", map_location=map_location)
        val_dataset = torch.load(f"{output_folder}/val_dataset.pth", map_location=map_location)
        test_dataset = torch.load(f"{output_folder}/test_dataset.pth", map_location=map_location)
    else:
        strategy = dataset_configs.get("strategy", "random")
        if strategy == "cross_validation":
            participants = sorted([int(item[0]) for item in dataset_paths])
            unique_participants_list = list(set(participants))
            print_on_rank_zero("unique_participants_list = ", unique_participants_list)
            left_out_participants_for_val = dataset_configs.get(
                "left_out_participants_for_val", unique_participants_list[-1]
            )
            print_on_rank_zero("left_out_participants_for_val = ", left_out_participants_for_val)
            left_out_participants_for_test = dataset_configs.get(
                "left_out_participants_for_test", unique_participants_list[-2]
            )
            print_on_rank_zero("left_out_participants_for_test = ", left_out_participants_for_test)
            train_dataset_paths = [
                item
                for item in dataset_paths
                if int(item[0]) not in left_out_participants_for_val
                and int(item[0]) not in left_out_participants_for_test
            ]
            val_dataset_paths = [item for item in dataset_paths if int(item[0]) in left_out_participants_for_val]
            test_dataset_paths = [item for item in dataset_paths if int(item[0]) in left_out_participants_for_test]
        elif strategy == "stratify_participants":
            participants = [item[0] for item in dataset_paths]
            unique_participants_list = list(set(participants))
            print_on_rank_zero("unique_participants_list = ", unique_participants_list)
            train_dataset_paths, tmp_dataset_paths = train_test_split(
                dataset_paths,
                train_size=dataset_configs.get("train_size", 0.5),
                random_state=seed_val,
                stratify=participants,
            )
            val_dataset_paths, test_dataset_paths = train_test_split(
                tmp_dataset_paths,
                train_size=dataset_configs.get("val_size", 0.5),
                random_state=seed_val,
                stratify=participants,
            )
        else:
            if strategy != "random":
                Warning(
                    "Specified CV strategy not available for dataset creation. Using random split with 50% train and 25% val sets and 25% test sets."
                )
            train_dataset_paths, tmp_dataset_paths = train_test_split(
                dataset_paths,
                train_size=dataset_configs.get("train_size", 0.5),
                random_state=seed_val,
            )
            val_dataset_paths, test_dataset_paths = train_test_split(
                tmp_dataset_paths,
                train_size=dataset_configs.get("val_size", 0.5),
                random_state=seed_val,
            )

        unique_participants_list_in_train = list(set(sorted([int(item[0]) for item in train_dataset_paths])))
        print_on_rank_zero("unique_participants_list_in_train = ", unique_participants_list_in_train)
        print_on_rank_zero("train_dataset_paths length = ", len(train_dataset_paths))

        unique_participants_list_in_val = list(set(sorted([int(item[0]) for item in val_dataset_paths])))
        print_on_rank_zero("unique_participants_list_in_val = ", unique_participants_list_in_val)
        print_on_rank_zero("val_dataset_paths length = ", len(val_dataset_paths))

        unique_participants_list_in_test = list(set(sorted([int(item[0]) for item in test_dataset_paths])))
        print_on_rank_zero("unique_participants_list_in_test = ", unique_participants_list_in_test)
        print_on_rank_zero("test_dataset_paths length = ", len(test_dataset_paths))

        if dataset_configs.get("normalize", False):
            if os.path.exists(f"{output_folder}/train_mean.pth") and os.path.exists(f"{output_folder}/train_stds.pth"):
                means = torch.load(f"{output_folder}/train_mean.pth")
                stds = torch.load(f"{output_folder}/train_stds.pth")
            else:
                means, stds = cal_means_stds(
                    dataset_configs,
                    train_dataset_paths,
                    data_path=data_path,
                    split="train",
                )
                torch.save(means, f"{output_folder}/train_mean.pth")
                torch.save(stds, f"{output_folder}/train_stds.pth")
            print_on_rank_zero(f"Mean:", means)
            print_on_rank_zero(f"Std:", stds)
            print_on_rank_zero("--------------------------------------------------")
        else:
            means, stds = None, None

        train_dataset = DATASET_REGISTRY.get(dataset_configs["dataset_registry"])(
            data=train_dataset_paths,
            data_path=data_path,
            means=means,
            stds=stds,
            **dataset_configs["dataset_registry_params"],
        )

        dataset_configs["dataset_registry_params"]["augment"] = False

        val_dataset = DATASET_REGISTRY.get(dataset_configs["dataset_registry"])(
            data=val_dataset_paths,
            data_path=data_path,
            means=means,
            stds=stds,
            **dataset_configs["dataset_registry_params"],
        )

        test_dataset = DATASET_REGISTRY.get(dataset_configs["dataset_registry"])(
            data=test_dataset_paths,
            data_path=data_path,
            means=means,
            stds=stds,
            **dataset_configs["dataset_registry_params"],
        )

        print_on_rank_zero("\n------- DATASETS: -------")
        print_on_rank_zero(
            "train_dataset = ",
            len(train_dataset),
            " | val_dataset = ",
            len(val_dataset),
            " | test_dataset = ",
            len(test_dataset),
            " | total_dataset = ",
            len(train_dataset) + len(val_dataset) + len(test_dataset),
        )

        torch.save(train_dataset, f"{output_folder}/train_dataset.pth")
        torch.save(val_dataset, f"{output_folder}/val_dataset.pth")
        torch.save(test_dataset, f"{output_folder}/test_dataset.pth")

        shuffle = False
        train_sampler = None
        if use_ddp:
            train_sampler = (
                DistributedSampler(train_dataset, num_replicas=world_size_or_num_gpus, rank=rank)
                if torch.cuda.is_available()
                else None
            )
        shuffle = train_sampler is None
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=dataloader_configs.get("batch_size", 1),
            pin_memory=dataloader_configs.get("pin_memory", False),
            shuffle=shuffle,
            num_workers=dataloader_configs.get("num_workers", 0),
            sampler=train_sampler,
        )

        val_sampler = None
        if use_ddp:
            val_sampler = (
                DistributedSampler(val_dataset, num_replicas=world_size_or_num_gpus, rank=rank)
                if torch.cuda.is_available()
                else None
            )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=dataloader_configs.get("batch_size", 1),
            pin_memory=dataloader_configs.get("pin_memory", False),
            shuffle=False,
            num_workers=dataloader_configs.get("num_workers", 0),
            sampler=val_sampler,
        )

        test_sampler = None
        if use_ddp:
            test_sampler = (
                DistributedSampler(test_dataset, num_replicas=world_size_or_num_gpus, rank=rank)
                if torch.cuda.is_available()
                else None
            )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=dataloader_configs.get("batch_size", 1),
            pin_memory=dataloader_configs.get("pin_memory", False),
            shuffle=False,
            num_workers=dataloader_configs.get("num_workers", 0),
            sampler=test_sampler,
        )

        print_on_rank_zero("\n------- DATALOADERS: -------")
        print_on_rank_zero(
            "train_dataloader = ",
            len(train_dataloader),
            " | val_dataloader = ",
            len(val_dataloader),
            " | test_dataloader = ",
            len(test_dataloader),
            " | total_dataloader = ",
            len(train_dataloader) + len(val_dataloader) + len(test_dataloader),
        )
        print_on_rank_zero("\n")

        torch.save(train_dataloader, f"{output_folder}/train_dataloader.pth")
        torch.save(val_dataloader, f"{output_folder}/val_dataloader.pth")
        torch.save(test_dataloader, f"{output_folder}/test_dataloader.pth")

    return {
        "dataloaders": {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        },
        "datasets": {"train": train_dataset, "val": val_dataset, "test": test_dataset},
    }


if __name__ == "__main__":

    print_on_rank_zero("root_path = ", root_path)

    parser = argparse.ArgumentParser(description="code generation training.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "train.yml"),
        required=False,
        help="Path to config file.",
    )

    args = parser.parse_args()
    print_on_rank_zero("args:\n", json.dumps(vars(args), sort_keys=True, indent=4), "\n")

    with open(args.config_file, mode="r") as f:
        config_file = yaml.safe_load(f)
        print_on_rank_zero("config_file = ", config_file, "\n")

    seed_val = config_file.get("seed", random.randint(1, 10000))
    config_file["seed"] = seed_val

    seed_everything(seed_val)

    prepared_data_dict = prepare_dataloader(
        dataset_configs=config_file["dataset_configs"],
        dataloader_configs=config_file["dataloader_configs"],
        seed_val=seed_val,
    )

    train_dataloader = prepared_data_dict["dataloaders"]["train"]
    val_dataloader = prepared_data_dict["dataloaders"]["val"]
    test_dataloader = prepared_data_dict["dataloaders"]["test"]

    train_dataset = prepared_data_dict["datasets"]["train"]
    val_dataset = prepared_data_dict["datasets"]["val"]
    test_dataset = prepared_data_dict["datasets"]["test"]

    print("train_dataloader = ", len(train_dataloader))
    print("val_dataloader = ", len(val_dataloader))
    print("test_dataloader = ", len(test_dataloader))

    print("train_dataset = ", len(train_dataset))
    print("val_dataset = ", len(val_dataset))
    print("test_dataset = ", len(test_dataset))

    for batch, data in enumerate(train_dataloader):

        source_img = data["img"]
        print("source_img = ", source_img.shape)
        break
