import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_main_output_folders(self):

    os.makedirs(self.output_folder, exist_ok=True)

    for feature in self.save_features:

        feature_path = os.path.join(self.output_folder, feature)
        os.makedirs(feature_path, exist_ok=True)

        if feature == "full_imgs":
            self.full_imgs_path = feature_path

        elif feature == "faces":
            self.cropped_faces_path = feature_path

        elif feature == "eyes":
            self.eyes_path = feature_path
            self.left_eyes_path = os.path.join(self.eyes_path, "left_eyes")
            os.makedirs(self.left_eyes_path, exist_ok=True)
            self.right_eyes_path = os.path.join(self.eyes_path, "right_eyes")
            os.makedirs(self.right_eyes_path, exist_ok=True)

        elif feature == "blinks":
            self.blinked_eyes_path = feature_path
            self.left_blinked_eyes_path = os.path.join(
                self.blinked_eyes_path, "left_eyes"
            )
            os.makedirs(self.left_blinked_eyes_path, exist_ok=True)
            self.right_blinked_eyes_path = os.path.join(
                self.blinked_eyes_path, "right_eyes"
            )
            os.makedirs(self.right_blinked_eyes_path, exist_ok=True)

        elif feature == "iris":
            self.iris_path = feature_path
            self.left_iris_path = os.path.join(self.iris_path, "left_iris")
            os.makedirs(self.left_iris_path, exist_ok=True)
            self.right_iris_path = os.path.join(self.iris_path, "right_iris")
            os.makedirs(self.right_iris_path, exist_ok=True)

    self.data_distribution_plots_folder = os.path.join(
        self.output_folder, "data_distribution"
    )
    os.makedirs(self.data_distribution_plots_folder, exist_ok=True)


def set_participant_folders(self, pid):

    pfolders = {}

    participant_folder = os.path.join(self.data_path, pid)
    pfolders["participant_folder"] = participant_folder

    for feature in self.save_features:

        if feature == "full_imgs":
            participant_full_imgs_folder = os.path.join(self.full_imgs_path, pid)
            os.makedirs(participant_full_imgs_folder, exist_ok=True)
            pfolders["participant_full_imgs_folder"] = participant_full_imgs_folder

        elif feature == "faces":
            participant_cropped_faces_folder = os.path.join(
                self.cropped_faces_path, pid
            )
            os.makedirs(participant_cropped_faces_folder, exist_ok=True)
            pfolders["participant_cropped_faces_folder"] = (
                participant_cropped_faces_folder
            )

        elif feature == "eyes":
            participant_left_eyes_folder = os.path.join(self.left_eyes_path, pid)
            os.makedirs(participant_left_eyes_folder, exist_ok=True)
            pfolders["participant_left_eyes_folder"] = participant_left_eyes_folder

            participant_right_eyes_folder = os.path.join(self.right_eyes_path, pid)
            os.makedirs(participant_right_eyes_folder, exist_ok=True)
            pfolders["participant_right_eyes_folder"] = participant_right_eyes_folder

        elif feature == "blinks":
            participant_left_blinked_eyes_folder = os.path.join(
                self.left_blinked_eyes_path, pid
            )
            os.makedirs(participant_left_blinked_eyes_folder, exist_ok=True)
            pfolders["participant_left_blinked_eyes_folder"] = (
                participant_left_blinked_eyes_folder
            )

            participant_right_blinked_eyes_folder = os.path.join(
                self.right_blinked_eyes_path, pid
            )
            os.makedirs(participant_right_blinked_eyes_folder, exist_ok=True)
            pfolders["participant_right_blinked_eyes_folder"] = (
                participant_right_blinked_eyes_folder
            )

        elif feature == "iris":
            participant_left_iris_folder = os.path.join(self.left_iris_path, pid)
            os.makedirs(participant_left_iris_folder, exist_ok=True)
            pfolders["participant_left_iris_folder"] = participant_left_iris_folder

            participant_right_iris_folder = os.path.join(self.right_iris_path, pid)
            os.makedirs(participant_right_iris_folder, exist_ok=True)
            pfolders["participant_right_iris_folder"] = participant_right_iris_folder

    return pfolders


def set_session_folders(self, pfolders, session_folder):

    sfolders = {}

    for feature in self.save_features:

        if feature == "full_imgs":
            full_imgs_session_path = os.path.join(
                pfolders["participant_full_imgs_folder"], session_folder
            )
            os.makedirs(full_imgs_session_path, exist_ok=True)
            sfolders["full_imgs_session_path"] = full_imgs_session_path

        elif feature == "faces":
            cropped_faces_session_path = os.path.join(
                pfolders["participant_cropped_faces_folder"], session_folder
            )
            os.makedirs(cropped_faces_session_path, exist_ok=True)
            sfolders["cropped_faces_session_path"] = cropped_faces_session_path

        elif feature == "eyes":
            left_eye_session_path = os.path.join(
                pfolders["participant_left_eyes_folder"], session_folder
            )
            os.makedirs(left_eye_session_path, exist_ok=True)
            sfolders["left_eyes_session_path"] = left_eye_session_path

            right_eye_session_path = os.path.join(
                pfolders["participant_right_eyes_folder"], session_folder
            )
            os.makedirs(right_eye_session_path, exist_ok=True)
            sfolders["right_eyes_session_path"] = right_eye_session_path

        elif feature == "iris":
            left_iris_session_path = os.path.join(
                pfolders["participant_left_iris_folder"], session_folder
            )
            os.makedirs(left_iris_session_path, exist_ok=True)
            sfolders["left_iris_session_path"] = left_iris_session_path

            right_iris_session_path = os.path.join(
                pfolders["participant_right_iris_folder"], session_folder
            )
            os.makedirs(right_iris_session_path, exist_ok=True)
            sfolders["right_iris_session_path"] = right_iris_session_path

    return sfolders


def save_session_data_csv(self, sfolders, df):

    session_data_csv_name = "session_data.csv"

    for feature in self.save_features:

        if feature == "full_imgs":
            df.to_csv(
                os.path.join(sfolders["full_imgs_session_path"], session_data_csv_name),
                index=False,
            )

        elif feature == "faces":
            df.to_csv(
                os.path.join(
                    sfolders["cropped_faces_session_path"], session_data_csv_name
                ),
                index=False,
            )

        elif feature == "eyes":
            df.to_csv(
                os.path.join(sfolders["left_eyes_session_path"], session_data_csv_name),
                index=False,
            )
            df.to_csv(
                os.path.join(
                    sfolders["right_eyes_session_path"], session_data_csv_name
                ),
                index=False,
            )

        elif feature == "iris":
            df.to_csv(
                os.path.join(sfolders["left_iris_session_path"], session_data_csv_name),
                index=False,
            )
            df.to_csv(
                os.path.join(
                    sfolders["right_iris_session_path"], session_data_csv_name
                ),
                index=False,
            )


def save_blinked_frames(self, pfolders, session_folder, frame_name, result_dict):

    if "blinks" in self.save_features and result_dict["eyes"]["blinked"]:

        cv2.imwrite(
            f"{pfolders['participant_left_blinked_eyes_folder']}/s{session_folder}_{frame_name}",
            result_dict["eyes"]["left_eye"],
        )

        cv2.imwrite(
            f"{pfolders['participant_right_blinked_eyes_folder']}/s{session_folder}_{frame_name}",
            result_dict["eyes"]["right_eye"],
        )


def save_frames(self, sfolders, frame_name, result_dict):

    if result_dict["eyes"]["blinked"]:
        print("Results contains blinked eyes. It will not be saved")
        return

    results_list = list(result_dict.keys())

    for feature in self.save_features:

        if feature == "full_imgs" and "img" in results_list:
            cv2.imwrite(
                f"{sfolders['full_imgs_session_path']}/{frame_name}",
                result_dict["img"],
            )

        elif feature == "faces" and "face" in results_list:
            cv2.imwrite(
                f"{sfolders['cropped_faces_session_path']}/{frame_name}",
                result_dict["face"],
            )

        elif feature == "eyes" and "eyes" in results_list:
            cv2.imwrite(
                f"{sfolders['left_eyes_session_path']}/{frame_name}",
                result_dict["eyes"]["left_eye"],
            )
            cv2.imwrite(
                f"{sfolders['right_eyes_session_path']}/{frame_name}",
                result_dict["eyes"]["right_eye"],
            )

        elif feature == "iris" and "iris" in results_list:
            cv2.imwrite(
                f"{sfolders['left_iris_session_path']}/{frame_name}",
                result_dict["iris"]["left_iris"]["img"],
            )
            cv2.imwrite(
                f"{sfolders['right_iris_session_path']}/{frame_name}",
                result_dict["iris"]["right_iris"]["img"],
            )


def plot_frame_statistics(self):

    for pid, sessions in self.frame_stats.items():

        sessions_sorted = sorted(sessions.keys(), key=int)

        # Select only even session IDs
        # sessions_sorted = [
        #     session for session in sessions_sorted if int(session) % 2 == 0
        # ]

        total_frames = [
            sessions[session]["total_frames"] for session in sessions_sorted
        ]
        skipped_frames = [
            sessions[session]["skipped_frames"] for session in sessions_sorted
        ]
        processed_frames = [
            total - skipped for total, skipped in zip(total_frames, skipped_frames)
        ]

        plt.figure(figsize=(14, 6))
        plt.plot(
            sessions_sorted,
            total_frames,
            label="Before Blink Detection",
            marker="o",
        )
        plt.plot(
            sessions_sorted,
            processed_frames,
            label="After Blink Detection",
            marker="o",
        )
        plt.xlabel("Session ID", fontsize=14, labelpad=10)
        plt.ylabel("Number of Frames", fontsize=14, labelpad=10)
        plt.title(f"All Session Frames - Participant {pid}", fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.tight_layout()

        participant_plots_folder = os.path.join(
            self.data_distribution_plots_folder, "sessions_stats"
        )
        os.makedirs(participant_plots_folder, exist_ok=True)
        plt.savefig(
            os.path.join(
                participant_plots_folder,
                f"{pid}.png",
            )
        )
        plt.close()

        overall_processed_frames = sum(total_frames) - sum(skipped_frames)

        x = ["Before Blink Detection", "After Blink Detection"]
        y = [sum(total_frames), overall_processed_frames]

        fig, ax = plt.subplots()
        bars = ax.bar(x, y, color=["tab:blue", "tab:orange"])

        # Adding the value labels inside the middle of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                xytext=(0, 0),  # Center the text
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                fontsize=14,
                # fontweight="bold",
            )

        # Setting labels and title
        # ax.set_xlabel("Frame Type", fontsize=14, labelpad=10)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Number of Frames", fontsize=14, labelpad=10)
        ax.set_title(f"Overall Frames - Participant {pid}", fontsize=14)
        plt.tight_layout()

        overall_stats_folder = os.path.join(
            self.data_distribution_plots_folder, "overall_stats"
        )
        os.makedirs(overall_stats_folder, exist_ok=True)
        plt.savefig(os.path.join(overall_stats_folder, f"{pid}.png"))
        plt.close()

    overall_processed_frames = self.overall_total_frames - self.overall_skipped_frames

    x = ["Before Blink Detection", "After Blink Detection"]
    y = [self.overall_total_frames, overall_processed_frames]

    fig, ax = plt.subplots()
    bars = ax.bar(x, y, color=["#444466", "#BB5555"])

    # Adding the value labels inside the middle of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            xytext=(0, 0),  # Center the text
            textcoords="offset points",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
            # fontweight="bold",
        )

    # Setting labels and title
    # ax.set_xlabel("Frame Type", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Number of Frames", fontsize=14, labelpad=10)
    ax.set_title("Overall Frames - All Participants", fontsize=14)
    plt.tight_layout()

    overall_stats_folder = os.path.join(
        self.data_distribution_plots_folder,
        "overall_stats",
    )
    os.makedirs(
        overall_stats_folder,
        exist_ok=True,
    )
    plt.savefig(os.path.join(overall_stats_folder, f"total_frames.png"))
    plt.close()
