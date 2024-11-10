import os
import sys
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

from registry import DATASET_PATHS_REGISTRY, DATASET_REGISTRY


class EyeDentifyDataModule(pl.LightningDataModule):
    def __init__(self, dataset_configs, dataloader_configs):
        super().__init__()
        self.dataset_configs = dataset_configs
        self.dataloader_configs = dataloader_configs

        self.data_path = self.dataset_configs.get("data_path", ".")
        self.data_path = os.path.join(root_path, self.data_path)

        split_fold = self.dataset_configs.get("split_fold", None)
        self.dataset_configs["dataset_paths_registry_params"]["split_fold"] = split_fold

        self.dataset_configs["dataset_paths_registry_params"]["left_out_participants_for_val"] = (
            self.dataset_configs.get("left_out_participants_for_val", [])
        )

        self.dataset_configs["dataset_paths_registry_params"]["left_out_participants_for_test"] = (
            self.dataset_configs.get("left_out_participants_for_test", [])
        )

        self.dataset_paths, self.output_folder = DATASET_PATHS_REGISTRY.get(
            self.dataset_configs["dataset_paths_registry"]
        )(data_path=self.data_path, **self.dataset_configs["dataset_paths_registry_params"])

    # called 1st - called from CPU only on 1st local rank
    def prepare_data(self):

        participants = sorted([int(item[0]) for item in self.dataset_paths])
        unique_participants_list = list(set(participants))

        left_out_participants_for_val = self.dataset_configs.get(
            "left_out_participants_for_val", unique_participants_list[-1]
        )
        left_out_participants_for_test = self.dataset_configs.get(
            "left_out_participants_for_test", unique_participants_list[-1]
        )

        self.train_dataset_paths = [
            item
            for item in self.dataset_paths
            if int(item[0]) not in left_out_participants_for_val and int(item[0]) not in left_out_participants_for_test
        ]
        self.val_dataset_paths = [item for item in self.dataset_paths if int(item[0]) in left_out_participants_for_val]
        self.test_dataset_paths = [
            item for item in self.dataset_paths if int(item[0]) in left_out_participants_for_test
        ]

        self.train_dataset = DATASET_REGISTRY.get(self.dataset_configs["dataset_registry"])(
            data=self.train_dataset_paths,
            data_path=self.data_path,
            **self.dataset_configs["dataset_registry_params"],
        )

        self.dataset_configs["dataset_registry_params"]["augment"] = False

        self.val_dataset = DATASET_REGISTRY.get(self.dataset_configs["dataset_registry"])(
            data=self.val_dataset_paths,
            data_path=self.data_path,
            **self.dataset_configs["dataset_registry_params"],
        )

        self.test_dataset = DATASET_REGISTRY.get(self.dataset_configs["dataset_registry"])(
            data=self.test_dataset_paths,
            data_path=self.data_path,
            **self.dataset_configs["dataset_registry_params"],
        )

        torch.save(self.train_dataset, f"{self.output_folder}/train_dataset.pth")
        torch.save(self.val_dataset, f"{self.output_folder}/val_dataset.pth")
        torch.save(self.test_dataset, f"{self.output_folder}/test_dataset.pth")

    # called 2nd
    def setup(self, stage):
        self.train_dataset = torch.load(f"{self.output_folder}/train_dataset.pth")
        self.val_dataset = torch.load(f"{self.output_folder}/val_dataset.pth")
        self.test_dataset = torch.load(f"{self.output_folder}/test_dataset.pth")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.dataloader_configs.get("batch_size", 1),
            pin_memory=self.dataloader_configs.get("pin_memory", False),
            shuffle=True,
            num_workers=self.dataloader_configs.get("num_workers", 0),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.dataloader_configs.get("batch_size", 1),
            pin_memory=self.dataloader_configs.get("pin_memory", False),
            shuffle=False,
            num_workers=self.dataloader_configs.get("num_workers", 0),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            pin_memory=self.dataloader_configs.get("pin_memory", False),
            shuffle=False,
            num_workers=self.dataloader_configs.get("num_workers", 0),
        )
