import os
import pandas as pd
from registry import DATASET_PATHS_REGISTRY


@DATASET_PATHS_REGISTRY.register()
def pd_ds_paths_individual(
    data_path,
    session_frames=None,
    force_creation=False,
    return_df=False,
    split_fold=None,
    left_out_participants_for_val=[],
    left_out_participants_for_test=[],
):

    # Output CSV to store the combined data
    if split_fold:
        output_folder = os.path.join(data_path, "Folds", split_fold)
    else:
        output_folder = data_path
        if len(left_out_participants_for_val) > 0 and len(left_out_participants_for_test) > 0:
            output_folder = os.path.join(data_path, "LOOCV")
            left_out_participants_for_val_string_representation = "_".join(map(str, left_out_participants_for_val))
            left_out_participants_for_test_string_representation = "_".join(map(str, left_out_participants_for_test))
            output_folder = f"{output_folder}_val-pids-{left_out_participants_for_val_string_representation}_test-pids-{left_out_participants_for_test_string_representation}"

    output_csv_path = os.path.join(output_folder, "dataset.csv")
    os.makedirs(output_folder, exist_ok=True)

    # Check if the CSV file exists
    if os.path.exists(output_csv_path) and force_creation == False:
        # Load the CSV into a pandas DataFrame
        combined_df = pd.read_csv(output_csv_path)
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
        participants = sorted([int(folder_name) for folder_name in os.listdir(data_path) if folder_name.isdigit()])

        # Iterate over each participant folder (converted back to string)
        for pid in map(str, participants):

            participant_folder = os.path.join(data_path, pid)

            # Get participant's all sessions folders
            participant_sessions_folders = sorted(
                [int(folder) for folder in os.listdir(participant_folder) if folder.isdigit()]
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
                        desired_subset_df = subset_df.iloc[session_frames[0] : session_frames[-1]]
                    else:
                        desired_subset_df = subset_df.iloc[:]

                    # Append the data to the list
                    all_data.append(desired_subset_df)

        # Concatenate all the DataFrames into one DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)

        # Set the columns to ensure the correct header order
        combined_df.columns = desired_columns
        print("combined_df before droping nan = ", len(combined_df))

        combined_df.to_csv(os.path.join(output_folder, "dataset_with_nan.csv"), index=False)
        print("dataset_with_nan saved")

        # Create a mask to identify rows with NaN values
        nan_mask = combined_df.isna().any(axis=1)

        # NOTE: The NaN values found - indicate that tobii was not able to capture / record one or more of the metric values ('desired column'). So don't include that frames in our dataset.
        # ALTERNATIVELY: you can take mean of the values per session for each participant and then run this dataset / dataloader creation. We don't provide code for taking means for each session for each participant. We just drop the NaN rows and move on since they are comparatively less than the amount of the entire dataset.

        # Create a new DataFrame with rows that contain NaN values
        df_with_nan = combined_df[nan_mask]
        print("df_with_nan = ", len(df_with_nan))

        nan_indices = df_with_nan.index.tolist()
        print("indices of rows with NaN values:", nan_indices)

        # Create a DataFrame without rows containing NaN values
        combined_df = combined_df.dropna(axis=0, how="any")
        combined_df = combined_df.reset_index(drop=True)
        print("combined_df after droping nan = ", len(combined_df))

        # Check if the output CSV already exists
        if os.path.exists(output_csv_path):
            # If it exists, delete it
            os.remove(output_csv_path)

        # Write the combined data to the output CSV
        combined_df.to_csv(output_csv_path, index=False)
        print(f"cleaned and combined dataset saved to {output_csv_path}")

    if return_df:
        return combined_df, output_folder
    else:
        # Get the data as a simple list, without column names
        dataset_paths = combined_df.to_numpy().tolist()
        return dataset_paths, output_folder


@DATASET_PATHS_REGISTRY.register()
def pd_ds_paths_combinations(
    data_path,
    selected_feature,
    selected_datasets=[],
    ignored_datasets=[],
    force_creation=False,
    return_df=False,
):

    # Combine the names of selected datasets to create the combined dataset name
    combined_ds_name = "_".join(selected_datasets) if selected_datasets else "all_datasets"

    # Output CSV to store the combined data
    output_folder = os.path.join(data_path, combined_ds_name, selected_feature)
    output_csv_path = os.path.join(output_folder, "dataset.csv")

    # Check if the CSV file exists
    if os.path.exists(output_csv_path) and force_creation == False:
        # Load the CSV into a pandas DataFrame
        combined_ds_df = pd.read_csv(output_csv_path)
    else:
        selected_ds_folders = [
            str(folder)
            for folder in os.listdir(data_path)
            if len(selected_datasets) == 0 or (folder in selected_datasets and folder not in ignored_datasets)
        ]

        # Initialize an empty DataFrame
        combined_ds_df = pd.DataFrame()

        for ds_folder in selected_ds_folders:
            feature_folders = [f for f in os.listdir(os.path.join(data_path, ds_folder))]
            if selected_feature in feature_folders:
                selected_ds_paths_df, _ = pd_ds_paths_individual(
                    data_path=os.path.join(data_path, ds_folder, selected_feature),
                    force_creation=force_creation,
                    return_df=True,
                )
                selected_ds_paths_df["frame_path"] = (
                    ds_folder + "/" + selected_feature + "/" + selected_ds_paths_df["frame_path"]
                )
                # Combine the individual DataFrame into the combined DataFrame
                combined_ds_df = pd.concat([combined_ds_df, selected_ds_paths_df], ignore_index=True)
            else:
                print("Selected Feature Not Found in Datsets")

        os.makedirs(output_folder, exist_ok=True)

        # Check if the output CSV already exists
        if os.path.exists(output_csv_path):
            # If it exists, delete it
            os.remove(output_csv_path)

        # Write the combined data to the output CSV
        combined_ds_df.to_csv(output_csv_path, index=False)
        print(f"Combined datasets saved to {output_csv_path}")

    if return_df:
        return combined_ds_df, output_folder
    else:
        # Get the data as a simple list, without column names
        dataset_paths = combined_ds_df.to_numpy().tolist()
        return dataset_paths, output_folder


print("Registered datasets in DATASET_PATHS_REGISTRY:", DATASET_PATHS_REGISTRY.keys())
