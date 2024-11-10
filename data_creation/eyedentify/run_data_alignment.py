import os
import sys
import json
import yaml
import random
import argparse
import os.path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from common_utils import seed_everything
from data_creation.eyedentify.tobii_and_webcam_data_alignment import TobiiAndWebCamDataAlignemnt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pre-processing.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(root_path, "configs", "preprocessing.yml"),
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

    source_path = os.path.join(root_path, config_file["dataset_source_dir"])
    destination_path = os.path.join(root_path, config_file["dataset_destination_dir"])
    plots_dir_path = os.path.join(root_path, config_file["plots_dir"])
    selected_participants = config_file.get("selected_participants", [])

    items_in_source_path = os.listdir(source_path)

    participants_dirs_paths = [
        os.path.join(source_path, item)
        for item in items_in_source_path
        if os.path.isdir(os.path.join(source_path, item))
    ]

    for participant_dir_path in participants_dirs_paths:

        selected_participant = os.path.basename(participant_dir_path)

        if selected_participant.isdigit() and (
            len(selected_participants) == 0 or selected_participant in selected_participants
        ):

            # Get the list of subfolders
            subfolders = [
                f for f in os.listdir(participant_dir_path) if os.path.isdir(os.path.join(participant_dir_path, f))
            ]

            # Look for the "tobii_pro_gaze.csv" file within the subfolders
            for subfolder in subfolders:
                participant_tobii_pro_gaze_csv = os.path.join(participant_dir_path, subfolder, "tobii_pro_gaze.csv")
                if os.path.isfile(participant_tobii_pro_gaze_csv):
                    break

            if os.path.isfile(participant_tobii_pro_gaze_csv):

                participant_timestamp_csv = os.path.join(participant_dir_path, "timestamp.csv")

                if os.path.isfile(participant_timestamp_csv):

                    output_path = os.path.join(destination_path, selected_participant)

                    data_analysis = TobiiAndWebCamDataAlignemnt(
                        input_dir=participant_dir_path,
                        output_dir=output_path,
                        destination_path=destination_path,
                        plots_dir_path=plots_dir_path,
                        selected_participant=selected_participant,
                        timestamp_csv=participant_timestamp_csv,
                        tobii_pro_gaze_csv=participant_tobii_pro_gaze_csv,
                    )

                    # Align the measured timestamps with tobii pro's timestamps and extract the image frames from the videos and combine both
                    results = data_analysis.algin_data()

                    # Create a box plot for all sessions
                    data_analysis.create_box_plot(results=results)

                    # Create a histogram with rug plot for each session
                    for session_id, data in results.items():
                        data_analysis.create_histogram_with_rug_plot(data=results[session_id], session_id=session_id)
                else:
                    print(f"Not Found 'timestamp.csv' Skipping participant: {participant_dir_path}")
                    continue
            else:
                print(f"Not Found 'tobii_pro_gaze.csv' Skipping participant: {participant_dir_path}")
                continue
