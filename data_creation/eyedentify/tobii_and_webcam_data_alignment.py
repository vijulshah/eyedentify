import os
import cv2
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class TobiiAndWebCamDataAlignemnt:

    def __init__(
        self,
        input_dir,
        output_dir,
        destination_path,
        plots_dir_path,
        selected_participant,
        timestamp_csv,
        tobii_pro_gaze_csv,
    ):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.destination_path = destination_path
        self.plots_dir = plots_dir_path
        self.timestamp_csv = timestamp_csv
        self.tobii_pro_gaze_csv = tobii_pro_gaze_csv
        self.selected_participant = selected_participant
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def create_box_plot(self, results, plots_dir=None):
        # Prepare the data
        session_ids = list(results.keys())
        pupil_diameters = [
            {"session_id": session, "pupil_diameter": diameter}
            for session in session_ids
            for diameter in results[session]["pupil_diameter"]
        ]

        # Convert to DataFrame
        data = pd.DataFrame(pupil_diameters)
        pupil_diameter_list_of_lists = data.groupby("session_id")["pupil_diameter"].apply(list).tolist()

        # Define colors for each session group
        colors = []
        for i in range(1, 51):
            if 1 <= i <= 10:
                colors.append("white")
            elif 11 <= i <= 15:
                colors.append("black")
            elif 16 <= i <= 20:
                colors.append("red")
            elif 21 <= i <= 25:
                colors.append("blue")
            elif 26 <= i <= 30:
                colors.append("yellow")
            elif 31 <= i <= 35:
                colors.append("green")
            elif 36 <= i <= 40:
                colors.append("gray")
            elif 41 <= i <= 50:
                colors.append("white")

        # Plot the boxplot
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(
            data=pupil_diameter_list_of_lists,
            ax=ax,
            palette=colors,
            flierprops={"marker": "d"},
            fliersize=5,
        )
        ax.set_xticklabels(session_ids, rotation=90, fontsize=12)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlabel("Session ID", fontsize=14, labelpad=10)
        ax.set_ylabel("Pupil Diameter", fontsize=14, labelpad=10)
        ax.set_title(
            f"Pupil Diameter Distribution - Participant-{self.selected_participant}",
            fontsize=14,
        )
        # Each box plot consists of 6 lines, the median is the 5th line
        for i in range(10, 15):
            median_line = ax.lines[i * 6 + 4]
            median_line.set_color("white")
            median_line.set_linewidth(1)

        plt.tight_layout()

        # Save the plot to the specified directory
        plots_dir = os.path.join(self.plots_dir, "box_plots") if plots_dir == None else plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        box_plot_path = os.path.join(plots_dir, f"{self.selected_participant}.png")
        plt.savefig(box_plot_path)
        plt.close(fig)

    def create_histogram_with_rug_plot(self, data, session_id, plots_dir=None):

        # Create a new figure for each session
        fig, ax = plt.subplots(figsize=(6, 4))

        # Create histogram with rug plot
        sns.histplot(data["pupil_diameter"], kde=True, ax=ax)  # Histogram with KDE
        sns.rugplot(data["pupil_diameter"], ax=ax, height=0.05, linewidth=1, color="red")  # Rug plot

        # Set titles and labels
        ax.set_title(f"Histogram with Rug Plot - Participant-{self.selected_participant} - Session {session_id}")
        ax.set_xlabel("Pupil Diameter")
        ax.set_ylabel("Density")
        plt.tight_layout()

        # Save each plot in a separate file
        plots_dir = (
            os.path.join(self.plots_dir, "histogram_with_rug_plots", self.selected_participant)
            if plots_dir == None
            else plots_dir
        )
        os.makedirs(plots_dir, exist_ok=True)
        historgam_rug_plot_path = os.path.join(plots_dir, f"session_{session_id}.png")
        plt.savefig(historgam_rug_plot_path)
        plt.close(fig)

    def extract_frames(self, video_path, output_dir):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_capture = cv2.VideoCapture(video_path)

        frames_paths = []
        frame_count = 0
        success, frame = video_capture.read()

        while success:

            frame_filename = os.path.join(output_dir, f"frame_{frame_count+1:02d}.png")
            cv2.imwrite(frame_filename, frame)
            frames_paths.append(frame_filename.split(self.destination_path + "/")[-1])
            frame_count += 1
            success, frame = video_capture.read()

        video_capture.release()

        return frames_paths

    def algin_data(self):

        # Load the CSV data
        df1 = pd.read_csv(self.timestamp_csv)
        df2 = pd.read_csv(self.tobii_pro_gaze_csv)

        # Convert timestamps to datetime
        df1["start"] = df1["start"].apply(
            lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        )
        df1["end"] = df1["end"].apply(
            lambda d: datetime.datetime.fromtimestamp(int(d) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        )

        df2["#timestamp"] = df2["#timestamp"].apply(
            lambda d: (datetime.datetime.fromtimestamp(int(d) / 1000) + datetime.timedelta(hours=2)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        rows_to_include = []

        # Get data within the session time ranges
        for _, row in df1.iterrows():
            condition = (df2["#timestamp"] >= row["start"]) & (df2["#timestamp"] < row["end"])
            matching_rows = df2[condition].copy()  # Use copy to avoid SettingWithCopyWarning
            matching_rows["session_id"] = row["session_id"]
            rows_to_include.append(matching_rows)

        # Concatenate all session data to create dataframe
        new_df = pd.concat(rows_to_include)

        # Get unique session IDs
        unique_sessions = new_df["session_id"].unique().tolist()

        # Columns to average
        columns_to_average = [
            "right_pupil",
            "left_pupil",
            "pupil_diameter",
            "gaze_x",  # 2d screen coordinates (absolute)
            "gaze_y",  # 2d screen coordinates (absolute)
            "left_gaze_x",  # 2d normalized coordinates (relative)
            "left_gaze_y",  # 2d normalized coordinates (relative)
            "right_gaze_x",  # 2d normalized coordinates (relative)
            "right_gaze_y",  # 2d normalized coordinates (relative)
            "left_gaze_origin_in_user_coordinate_system",  # 3d coordinate x-axis
            "left_gaze_origin_in_user_coordinate_system.1",  # 3d coordinate y-axis
            "left_gaze_origin_in_user_coordinate_system.2",  # 3d coordinate z-axis
            "right_gaze_origin_in_user_coordinate_system",  # 3d coordinate x-axis
            "right_gaze_origin_in_user_coordinate_system.1",  # 3d coordinate y-axis
            "right_gaze_origin_in_user_coordinate_system.2",  # 3d coordinate z-axis
        ]

        # Dictionary to store results
        results = {}

        # Loop through each session and calculate averages
        for session_id in unique_sessions:

            session_data = new_df[new_df["session_id"] == session_id].reset_index(drop=True)
            unique_timestamps = session_data["#timestamp"].unique()

            # Get data for each timestamp
            data_1 = session_data[session_data["#timestamp"] == unique_timestamps[0]][columns_to_average].reset_index(
                drop=True
            )
            data_2 = session_data[session_data["#timestamp"] == unique_timestamps[1]][columns_to_average].reset_index(
                drop=True
            )
            data_3 = session_data[session_data["#timestamp"] == unique_timestamps[2]][columns_to_average].reset_index(
                drop=True
            )

            # Create a dictionary to store the row-wise mean for each specified column
            rowwise_means = {}

            # Loop through each column and compute the row-wise mean
            for column in columns_to_average:
                # Concatenate the specific column from each DataFrame
                concatenated_column = pd.concat([data_1[column], data_2[column], data_3[column]], axis=1)

                # Calculate row-wise mean, ignoring NaN values
                rowwise_means[column] = concatenated_column.mean(axis=1)

            # Create a new DataFrame from the calculated row-wise means
            average_selected_data = pd.DataFrame(rowwise_means)

            # Create a DataFrame with the common timestamp
            common_timestamp = session_data[session_data["#timestamp"] == unique_timestamps[0]]["#timestamp"].iloc[0]

            final_selected_data = pd.DataFrame(
                {
                    "#timestamp": [common_timestamp] * len(average_selected_data),
                    "session_id": [session_id] * len(average_selected_data),
                    **average_selected_data.to_dict(orient="list"),
                }
            )

            # Rename columns in the final_selected_data DataFrame
            final_selected_data = final_selected_data.rename(
                columns={
                    "gaze_x": "gaze_x_screen_abs_2d_x",
                    "gaze_y": "gaze_y_screen_abs_2d_y",
                    "left_gaze_x": "left_gaze_screen_rel_2d_x",
                    "left_gaze_y": "left_gaze_screen_rel_2d_y",
                    "right_gaze_x": "right_gaze_screen_rel_2d_x",
                    "right_gaze_y": "right_gaze_screen_rel_2d_y",
                    "left_gaze_origin_in_user_coordinate_system": "left_gaze_3d_x",
                    "left_gaze_origin_in_user_coordinate_system.1": "left_gaze_3d_y",
                    "left_gaze_origin_in_user_coordinate_system.2": "left_gaze_3d_z",
                    "right_gaze_origin_in_user_coordinate_system": "right_gaze_3d_x",
                    "right_gaze_origin_in_user_coordinate_system.1": "right_gaze_3d_y",
                    "right_gaze_origin_in_user_coordinate_system.2": "right_gaze_3d_z",
                }
            )

            out_path = os.path.join(self.output_dir, str(session_id))
            os.makedirs(out_path, exist_ok=True)

            video_file_path = os.path.join(self.input_dir, f"{str(session_id)}.webm")
            frames_paths = self.extract_frames(video_file_path, out_path)
            frames_paths_df = pd.DataFrame({"frame_path": frames_paths})

            # Trim excess rows from final_selected_data and frames_paths_df
            min_length = min(len(final_selected_data), len(frames_paths_df))
            final_selected_data = final_selected_data.iloc[:min_length].reset_index(drop=True)
            frames_paths_df = frames_paths_df.iloc[:min_length].reset_index(drop=True)

            # Combine the two DataFrames side-by-side
            combined_data = pd.concat([final_selected_data, frames_paths_df], axis=1)

            # Save the final_selected_data to a CSV file
            output_filename = f"session_data.csv"
            combined_data.to_csv(os.path.join(out_path, output_filename), index=False)

            # Store the results
            results[session_id] = combined_data

        return results
