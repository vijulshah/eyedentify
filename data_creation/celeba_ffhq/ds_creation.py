import os
import sys
import cv2
import json
import yaml
import random
import pickle
import argparse
import os.path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from common_utils import seed_everything
from data_creation.features_extraction.features_extractor import FeaturesExtractor
from data_creation.celeba_ffhq.ds_utils import save_frames, plot_avg_EARs_statistics, plot_frame_statistics


class SrEyesDatasetCreation:
    def __init__(self, dataset_configs, feature_extraction_configs):

        self.input_path = dataset_configs["input_path"]
        self.output_path = dataset_configs["output_path"]
        self.img_size = dataset_configs.get("img_size", None)
        self.max_imgs = dataset_configs.get("max_imgs", None)

        self.extraction_library = feature_extraction_configs.get("extraction_library", "mediapipe")
        self.blink_detection = feature_extraction_configs.get("blink_detection", False)
        self.upscale = int(feature_extraction_configs.get("upscale", 1))

        self.input_path = os.path.join(root_path, self.input_path)
        self.img_paths = os.listdir(self.input_path)
        self.features_extractor = FeaturesExtractor(
            extraction_library=self.extraction_library,
            blink_detection=self.blink_detection,
            upscale=self.upscale,
        )

        self.save_features = feature_extraction_configs.get("save_features", ["eyes"])

        self.output_path = os.path.join(root_path, self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

        self.data_distribution_folder = os.path.join(
            self.output_path, f"data_distribution_{self.input_path.split('/')[-1]}"
        )
        os.makedirs(self.data_distribution_folder, exist_ok=True)

    def __call__(self):

        total_imgs = self.max_imgs if self.max_imgs is not None else len(self.img_paths)
        paths_to_remove = []
        img_avg_EARs = []

        for idx, frame_name in enumerate(self.img_paths):

            if self.max_imgs is not None and idx == self.max_imgs:
                break

            img_path = os.path.join(self.input_path, frame_name)
            img = cv2.imread(img_path)

            frame_name = f"{img_path.split('/')[-2]}_{frame_name}"

            if img is None:
                print(f"Could not read image at {img_path}. Skipping...")
                paths_to_remove.append(idx)
                continue

            if self.img_size is not None:
                img_height, img_width = img.shape[:2]
                if (img_width, img_height) != (self.img_size[0], self.img_size[1]):
                    img = cv2.resize(
                        img,
                        (self.img_size[0], self.img_size[-1]),
                        interpolation=cv2.INTER_CUBIC,
                    )

            result_dict = self.features_extractor(img)

            if result_dict is not None and len(result_dict.keys()) > 0:
                eyes_info = result_dict.get("eyes", None)
                if eyes_info is not None and not eyes_info.get("blinked", False):
                    print(f"Frame: {frame_name} is good!")
                    save_frames(self.output_path, frame_name, result_dict, self.save_features)
                    img_avg_EARs.append(eyes_info["avg_EAR"])
                else:
                    print(f"Blinked Incident noted! Frame: {frame_name} will be removed!")
                    paths_to_remove.append(idx)
                    if eyes_info is None:
                        print("No eyes available to save blinked frames")
                        img_avg_EARs.append(0)
                    else:
                        img_avg_EARs.append(eyes_info["avg_EAR"])
                        if "blinks" in self.save_features and eyes_info.get("blinked", True):
                            save_frames(
                                self.output_path,
                                frame_name,
                                result_dict,
                                self.save_features,
                            )
            else:
                print(f"result_dict is None. Incident noted! Frame: {frame_name} will be removed!")
                paths_to_remove.append(idx)
                img_avg_EARs.append(0)

        frame_stats = {
            "total_imgs": total_imgs,
            "skipped_imgs": len(paths_to_remove),
        }
        print(f"\nEARs counted for: {len(img_avg_EARs)} Images.")
        print(f"Removed: {frame_stats['skipped_imgs']} Unwanted Images.")
        print(f"New Total: {frame_stats['total_imgs'] - frame_stats['skipped_imgs']} Images.")

        with open(f"{self.data_distribution_folder}/EARs_statistics.pkl", "wb") as file:
            pickle.dump(img_avg_EARs, file)

        with open(f"{self.data_distribution_folder}/frame_statistics.pkl", "wb") as file:
            pickle.dump(frame_stats, file)

        plot_avg_EARs_statistics(self.data_distribution_folder, img_avg_EARs)
        plot_frame_statistics(self.data_distribution_folder, frame_stats)
        print("\nDataset Created!!!")


if __name__ == "__main__":

    print("root_path = ", root_path)

    parser = argparse.ArgumentParser(description="Features Extraction and Dataset Creation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "sr_eyes_dataset_creation.yml"),
        required=False,
        help="Path to config file.",
    )

    args = parser.parse_args()
    print("args:\n", json.dumps(vars(args), sort_keys=True, indent=4), "\n")

    with open(args.config_file, mode="r") as f:
        config_file = yaml.safe_load(f)
        print("config_file = ", config_file, "\n")

    seed_val = config_file.get("seed", random.randint(1, 10000))
    config_file["seed"] = seed_val
    seed_everything(seed_val)

    sr_eyes_dataset_creation = SrEyesDatasetCreation(
        dataset_configs=config_file["dataset_configs"],
        feature_extraction_configs=config_file["feature_extraction_configs"],
    )
    print("Images to be processed = ", len(sr_eyes_dataset_creation.img_paths))
    sr_eyes_dataset_creation()
