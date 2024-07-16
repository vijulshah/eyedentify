import os
import pandas as pd
from registry import DATASET_PATHS_REGISTRY


@DATASET_PATHS_REGISTRY.register()
def pd_ds_paths(data_path, session_frames=None, force_creation=False, return_df=False):

    # Output CSV to store the combined data
    output_folder = data_path
    output_csv_path = os.path.join(data_path, "dataset.csv")

    # Check if the CSV file exists
    if os.path.exists(output_csv_path) and force_creation == False:
        # Load the CSV into a pandas DataFrame
        combined_df = pd.read_csv(output_csv_path)
        print("combined_df = ", len(combined_df))
        print("combined dataset loaded from: ", output_csv_path)
    else:
        # The desired column names
        desired_columns = [
            "participant_id",
            "#timestamp",
            "session_id",
            "left_pupil",
            "right_pupil",
            "pupil_diameter",
            "frame_path",
        ]

        # List to store data from all CSV files
        all_data = []

        # Get a sorted list of participants folders
        participants = sorted(
            [
                int(folder_name)
                for folder_name in os.listdir(data_path)
                if folder_name.isdigit()
            ]
        )

        # Iterate over each participant folder (converted back to string)
        for pid in map(str, participants):

            participant_folder = os.path.join(data_path, pid)

            # Get participant's all sessions folders
            participant_sessions_folders = sorted(
                [
                    int(folder)
                    for folder in os.listdir(participant_folder)
                    if folder.isdigit()
                ]
            )

            # Iterate over each participant's each session folder (converted back to string)
            for session_folder in map(str, participant_sessions_folders):

                session_path = os.path.join(participant_folder, session_folder)
                session_data_csv_path = os.path.join(session_path, "session_data.csv")

                # Check if the session_data.csv file exists
                if os.path.exists(session_data_csv_path):

                    # Read the session CSV file into a DataFrame
                    df = pd.read_csv(session_data_csv_path)

                    # Keep only the desired columns
                    subset_df = df[desired_columns[1:]].copy()

                    # Add the participant ID as a new column
                    if len(subset_df) == 0:
                        print(
                            "No data available for pid = ",
                            pid,
                            " | session_data.csv at = ",
                            session_data_csv_path,
                        )
                        continue

                    subset_df.loc[:, desired_columns[0]] = pid

                    # Reorder the columns to match the desired order
                    subset_df = subset_df[desired_columns]

                    # eg: Select rows from 31 to 40 (inclusive) i.e only 10 frames per session
                    if session_frames is not None and len(session_frames) > 0:
                        desired_subset_df = subset_df.iloc[
                            session_frames[0] : session_frames[-1]
                        ]
                    else:
                        desired_subset_df = subset_df.iloc[:]

                    # Append the data to the list
                    all_data.append(desired_subset_df)

        # Concatenate all the DataFrames into one DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)

        # Set the columns to ensure the correct header order
        combined_df.columns = desired_columns
        print("combined_df = ", len(combined_df))

        # Check if the output CSV already exists
        if os.path.exists(output_csv_path):
            # If it exists, delete it
            os.remove(output_csv_path)

        # Write the combined data to the output CSV
        combined_df.to_csv(output_csv_path, index=False)
        print("combined dataset saved to: ", output_csv_path)

    if return_df:
        return combined_df, output_folder
    else:
        # Get the data as a simple list, without column names
        dataset_paths = combined_df.to_numpy().tolist()
        return dataset_paths, output_folder


print("Registered datasets in DATASET_PATHS_REGISTRY:", DATASET_PATHS_REGISTRY.keys())
