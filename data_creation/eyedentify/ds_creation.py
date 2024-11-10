import os
import sys
import cv2
import json
import yaml
import pickle
import random
import argparse
import pandas as pd
import os.path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from common_utils import seed_everything
from data_creation.features_extraction.features_extractor import FeaturesExtractor
from data_creation.eyedentify.ds_utils import (
    get_sr_method,
    set_main_output_folders,
    set_participant_folders,
    set_session_folders,
    save_session_data_csv,
    save_frames,
    plot_frame_statistics,
    plot_avg_EARs_statistics,
    save_blinked_frames,
)


class EyeDentityDatasetCreation:

    def __init__(self, dataset_configs, feature_extraction_configs, sr_configs=None):
        self.data_path = os.path.join(root_path, dataset_configs["data_path"])
        self.selected_participants = dataset_configs.get("selected_participants", [])
        self.extraction_library = feature_extraction_configs.get("extraction_library", "mediapipe")
        self.sr_configs = sr_configs
        if self.sr_configs:
            self.sr_method_name = sr_configs["method"]
            self.upscale = sr_configs["params"].get("upscale", 2)
            sr_configs["params"]["upscale"] = self.upscale
            self.sr_method = get_sr_method(self, sr_configs)
            self.output_folder = os.path.join(
                root_path,
                dataset_configs["output_folder"],
                f"{self.sr_method_name}_x{self.upscale}",
            )
        else:
            self.upscale = 1
            self.output_folder = os.path.join(root_path, dataset_configs["output_folder"])
        self.blink_detection = feature_extraction_configs.get("blink_detection", False)
        self.upscale = max(int(feature_extraction_configs.get("upscale", 1)), self.upscale)
        self.features_extractor = FeaturesExtractor(
            extraction_library=self.extraction_library,
            blink_detection=self.blink_detection,
            upscale=self.upscale,
        )
        self.save_features = feature_extraction_configs.get("save_features", ["eyes"])
        self.eye_types = ["left_eyes", "right_eyes"]
        self.iris_types = ["left_iris", "right_iris"]
        self.features_subdirs = [
            "wo_outlines",
            # "eyes_outlined",
            # "iris_outlined",
            # "eyes_n_iris_outlined",
            "depth",
        ]
        self.segmentation_polygon_subdirs = [
            # "segmented_polygon",
            # "segmented_mask_polygon",
        ]
        self.segmentation_otsu_subdirs = [
            # "segmented_otsu",
            # "segmented_mask_otsu",
        ]
        self.frame_stats = {}
        self.avg_EARs_stats = {}
        self.overall_total_frames = 0
        self.overall_skipped_frames = 0
        set_main_output_folders(self)

    def __call__(self):
        participants = sorted([int(folder_name) for folder_name in os.listdir(self.data_path) if folder_name.isdigit()])
        print("available participants = ", len(participants))
        print(
            "selected participants = ",
            (self.selected_participants if len(self.selected_participants) > 0 else len(participants)),
        )

        for pid in map(str, participants):
            if len(self.selected_participants) == 0 or (pid in self.selected_participants):
                print(f"\n==================== Processing Participant: {pid} ====================")
                pfolders = set_participant_folders(self, pid)
                participant_sessions_folders = sorted(
                    [int(folder) for folder in os.listdir(pfolders["participant_folder"]) if folder.isdigit()]
                )

                self.frame_stats[pid] = {}
                self.avg_EARs_stats[pid] = {}

                for session_folder in map(str, participant_sessions_folders):

                    print(f"---------- Session: {session_folder} ----------")

                    sfolders = set_session_folders(self, pfolders, session_folder)

                    session_path = os.path.join(pfolders["participant_folder"], session_folder)

                    session_data_csv_path = os.path.join(session_path, "session_data.csv")

                    df = pd.read_csv(session_data_csv_path)
                    session_img_paths = df["frame_path"]

                    total_frames = len(session_img_paths)
                    skipped_frames = 0

                    paths_to_remove = []
                    participant_session_avg_EARs = []

                    for img_path in session_img_paths:
                        frame_name = img_path.split("/")[-1]
                        img = cv2.imread(os.path.join(self.data_path, img_path))
                        if self.sr_configs:
                            img = self.sr_method(img)
                        result_dict = self.features_extractor(img)
                        if result_dict is not None and len(result_dict.keys()) > 0:
                            eyes_info = result_dict.get("eyes", None)
                            if eyes_info is not None and eyes_info.get("blinked", True) == False:
                                print(f"p{pid}-s{session_folder}-{frame_name} ==> All good!")
                                save_frames(self, sfolders, frame_name, result_dict)
                                avg_EAR = eyes_info["avg_EAR"]
                                participant_session_avg_EARs.append(avg_EAR)
                            else:
                                print(
                                    f"p{pid}-s{session_folder}-{frame_name} ==> Blinked Incident noted! frame will be removed"
                                )
                                paths_to_remove.append(img_path)
                                skipped_frames += 1
                                if eyes_info is None:
                                    print("No eyes available to save blinked frames")
                                    participant_session_avg_EARs.append(0)
                                else:
                                    avg_EAR = eyes_info["avg_EAR"]
                                    participant_session_avg_EARs.append(avg_EAR)
                                    if "blinks" in self.save_features:
                                        save_blinked_frames(
                                            self,
                                            pfolders,
                                            session_folder,
                                            frame_name,
                                            result_dict,
                                        )
                        else:
                            print(
                                f"p{pid}-s{session_folder}-{frame_name} ==> Other Incident noted! frame will be removed"
                            )
                            paths_to_remove.append(img_path)
                            skipped_frames += 1

                    if len(paths_to_remove) > 0:
                        df = df[~df["frame_path"].isin(paths_to_remove)]
                        print("Removed unwanted session frames")
                        save_session_data_csv(self, sfolders, df)
                    else:
                        save_session_data_csv(self, sfolders, df)

                    self.avg_EARs_stats[pid][session_folder] = participant_session_avg_EARs
                    self.frame_stats[pid][session_folder] = {
                        "total_frames": total_frames,
                        "skipped_frames": skipped_frames,
                    }

                    self.overall_total_frames += total_frames
                    self.overall_skipped_frames += skipped_frames

        p = os.path.join(self.output_folder, "data_distribution", "EARs")
        os.makedirs(p, exist_ok=True)
        with open(f"{p}/EARs_statistics.pkl", "wb") as file:
            pickle.dump(self.avg_EARs_stats, file)

        plot_frame_statistics(self)
        plot_avg_EARs_statistics(self)
        print("\nDataset Created!!!")


if __name__ == "__main__":

    print("root_path = ", root_path)

    parser = argparse.ArgumentParser(description="Features Extraction and Dataset Creation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "dataset_creation.yml"),
        required=False,
        help="Path to config file.",
    )
    parser.add_argument(
        "--selected_participants",
        nargs="+",
        default=None,
        required=False,
        help="List of selected participants for creating SR dataset.",
    )

    args = parser.parse_args()
    print("args:\n", json.dumps(vars(args), sort_keys=True, indent=4), "\n")

    # parse yml to dict - to get configurations
    with open(args.config_file, mode="r") as f:
        config_file = yaml.safe_load(f)
        print("config_file = ", config_file, "\n")

    seed_val = config_file.get("seed", random.randint(1, 10000))
    config_file["seed"] = seed_val
    seed_everything(seed_val)

    selected_participants = args.selected_participants
    if selected_participants is not None and len(selected_participants) > 0:
        print(
            "selected_participants = ",
            selected_participants,
            " | length = ",
            len(selected_participants),
            " | types = ",
            type(selected_participants),
            type(selected_participants[0]),
        )
        config_file["dataset_configs"]["selected_participants"] = selected_participants

    EyeDentityDatasetCreation(
        dataset_configs=config_file["dataset_configs"],
        feature_extraction_configs=config_file["feature_extraction_configs"],
        sr_configs=config_file.get("sr_configs"),
    )()
