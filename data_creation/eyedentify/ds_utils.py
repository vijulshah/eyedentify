import os
import cv2
import sys
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from super_resolution.inference.inference_hat import HAT
from super_resolution.inference.inference_gfpgan import GFPGAN
from super_resolution.inference.inference_realesr import RealEsr
from super_resolution.inference.inference_srresnet import SRResNet
from super_resolution.inference.inference_codeformer import CodeFormer


def get_sr_method(self, sr_configs):
    sr_method_class = globals().get(self.sr_method_name)
    if sr_method_class is not None:
        return sr_method_class(**sr_configs["params"])
    else:
        raise Exception(f"No such SR method called '{self.sr_method_name}' implemented!")


def set_main_output_folders(self):
    for feature in self.save_features:

        feature_path = os.path.join(self.output_folder, feature)
        os.makedirs(feature_path, exist_ok=True)

        if feature == "full_imgs":
            self.full_imgs_path = feature_path

            for subdir in self.features_subdirs:
                full_imgs_type_path = os.path.join(self.full_imgs_path, subdir)
                os.makedirs(full_imgs_type_path, exist_ok=True)

        elif feature == "faces":
            self.faces_imgs_path = feature_path

            for subdir in self.features_subdirs:
                face_imgs_type_path = os.path.join(self.faces_imgs_path, subdir)
                os.makedirs(face_imgs_type_path, exist_ok=True)

        elif feature == "eyes":
            self.eyes_imgs_path = feature_path

            for eye_type in self.eye_types:
                eye_type_path = os.path.join(self.eyes_imgs_path, eye_type)
                os.makedirs(eye_type_path, exist_ok=True)
                for subdir in (
                    self.features_subdirs + self.segmentation_polygon_subdirs + self.segmentation_otsu_subdirs
                ):
                    subdir_path = os.path.join(eye_type_path, subdir)
                    os.makedirs(subdir_path, exist_ok=True)

        elif feature == "blinks":
            self.blinked_eyes = feature_path

            for eye_type in self.eye_types:
                blinked_eyes_path = os.path.join(self.blinked_eyes, eye_type)
                os.makedirs(blinked_eyes_path, exist_ok=True)

        elif feature == "iris":
            self.iris_path = feature_path

            for iris_type in self.iris_types:
                iris_type_path = os.path.join(self.iris_path, iris_type)
                os.makedirs(iris_type_path, exist_ok=True)
                for subdir in (
                    # [self.features_subdirs[0], self.features_subdirs[2], self.features_subdirs[-1]]
                    [self.features_subdirs[0], self.features_subdirs[-1]]
                    + self.segmentation_polygon_subdirs
                    + self.segmentation_otsu_subdirs
                ):
                    subdir_path = os.path.join(iris_type_path, subdir)
                    os.makedirs(subdir_path, exist_ok=True)

    self.data_distribution_plots_folder = os.path.join(self.output_folder, "data_distribution")
    os.makedirs(self.data_distribution_plots_folder, exist_ok=True)


def set_participant_folders(self, pid):
    pfolders = {}
    participant_folder = os.path.join(self.data_path, pid)
    pfolders["participant_folder"] = participant_folder

    for feature in self.save_features:

        if feature == "full_imgs":
            for subdir in self.features_subdirs:
                full_imgs_type_path = os.path.join(self.full_imgs_path, subdir)
                participant_full_imgs_folder = os.path.join(full_imgs_type_path, pid)
                os.makedirs(participant_full_imgs_folder, exist_ok=True)
                pfolders[f"participant_full_imgs_{subdir}_folder"] = participant_full_imgs_folder

        elif feature == "faces":
            for subdir in self.features_subdirs:
                face_imgs_type_path = os.path.join(self.faces_imgs_path, subdir)
                participant_face_imgs_folder = os.path.join(face_imgs_type_path, pid)
                os.makedirs(participant_face_imgs_folder, exist_ok=True)
                pfolders[f"participant_face_imgs_{subdir}_folder"] = participant_face_imgs_folder

        elif feature == "eyes":
            for eye_type in self.eye_types:
                eye_type_path = os.path.join(self.eyes_imgs_path, eye_type)
                for subdir in (
                    self.features_subdirs + self.segmentation_polygon_subdirs + self.segmentation_otsu_subdirs
                ):
                    subdir_path = os.path.join(eye_type_path, subdir)
                    participant_eyes_folder = os.path.join(subdir_path, pid)
                    os.makedirs(participant_eyes_folder, exist_ok=True)
                    pfolders[f"participant_{eye_type}_{subdir}_folder"] = participant_eyes_folder

        elif feature == "blinks":
            for eye_type in self.eye_types:
                blinked_eyes_path = os.path.join(self.blinked_eyes, eye_type)
                participant_blinked_eyes_folder = os.path.join(blinked_eyes_path, pid)
                os.makedirs(participant_blinked_eyes_folder, exist_ok=True)
                pfolders[f"participant_blinked_{eye_type}_folder"] = participant_blinked_eyes_folder

        elif feature == "iris":
            for iris_type in self.iris_types:
                iris_type_path = os.path.join(self.iris_path, iris_type)
                arr = (
                    # [self.features_subdirs[0], self.features_subdirs[2], self.features_subdirs[-1]]
                    [self.features_subdirs[0], self.features_subdirs[-1]]
                    + self.segmentation_polygon_subdirs
                    + self.segmentation_otsu_subdirs
                )
                for subdir in arr:
                    subdir_path = os.path.join(iris_type_path, subdir)
                    participant_iris_folder = os.path.join(subdir_path, pid)
                    os.makedirs(participant_iris_folder, exist_ok=True)
                    pfolders[f"participant_{iris_type}_{subdir}_folder"] = participant_iris_folder

    return pfolders


def set_session_folders(self, pfolders, session_folder):
    sfolders = {}
    for feature in self.save_features:

        if feature == "full_imgs":
            for subdir in self.features_subdirs:
                full_imgs_session_path = os.path.join(
                    pfolders[f"participant_full_imgs_{subdir}_folder"], session_folder
                )
                os.makedirs(full_imgs_session_path, exist_ok=True)
                sfolders[f"full_imgs_{subdir}_session_path"] = full_imgs_session_path

        elif feature == "faces":
            for subdir in self.features_subdirs:
                faces_session_path = os.path.join(pfolders[f"participant_face_imgs_{subdir}_folder"], session_folder)
                os.makedirs(faces_session_path, exist_ok=True)
                sfolders[f"faces_{subdir}_session_path"] = faces_session_path

        elif feature == "eyes":
            for eye_type in self.eye_types:
                for subdir in (
                    self.features_subdirs + self.segmentation_polygon_subdirs + self.segmentation_otsu_subdirs
                ):
                    eye_session_path = os.path.join(
                        pfolders[f"participant_{eye_type}_{subdir}_folder"],
                        session_folder,
                    )
                    os.makedirs(eye_session_path, exist_ok=True)
                    sfolders[f"{eye_type}_{subdir}_session_path"] = eye_session_path

        elif feature == "iris":
            for iris_type in self.iris_types:
                for subdir in (
                    # [self.features_subdirs[0], self.features_subdirs[2], self.features_subdirs[-1]]
                    [self.features_subdirs[0], self.features_subdirs[-1]]
                    + self.segmentation_polygon_subdirs
                    + self.segmentation_otsu_subdirs
                ):
                    iris_session_path = os.path.join(
                        pfolders[f"participant_{iris_type}_{subdir}_folder"],
                        session_folder,
                    )
                    os.makedirs(iris_session_path, exist_ok=True)
                    sfolders[f"{iris_type}_{subdir}_session_path"] = iris_session_path

    return sfolders


def save_session_data_csv(self, sfolders, df, file_name="session_data.csv"):
    for feature in self.save_features:

        if feature == "full_imgs":
            for subdir in self.features_subdirs:
                df.to_csv(
                    os.path.join(
                        sfolders[f"full_imgs_{subdir}_session_path"],
                        file_name,
                    ),
                    index=False,
                )

        elif feature == "faces":
            for subdir in self.features_subdirs:
                df.to_csv(
                    os.path.join(sfolders[f"faces_{subdir}_session_path"], file_name),
                    index=False,
                )

        elif feature == "eyes":
            for eye_type in self.eye_types:
                for subdir in (
                    self.features_subdirs + self.segmentation_polygon_subdirs + self.segmentation_otsu_subdirs
                ):
                    df.to_csv(
                        os.path.join(
                            sfolders[f"{eye_type}_{subdir}_session_path"],
                            file_name,
                        ),
                        index=False,
                    )

        elif feature == "iris":
            for iris_type in self.iris_types:
                for subdir in (
                    # [self.features_subdirs[0], self.features_subdirs[2], self.features_subdirs[-1]]
                    [self.features_subdirs[0], self.features_subdirs[-1]]
                    + self.segmentation_polygon_subdirs
                    + self.segmentation_otsu_subdirs
                ):
                    df.to_csv(
                        os.path.join(
                            sfolders[f"{iris_type}_{subdir}_session_path"],
                            file_name,
                        ),
                        index=False,
                    )


def save_blinked_frames(self, pfolders, session_folder, frame_name, result_dict):
    if "blinks" in self.save_features and result_dict["eyes"]["blinked"]:
        for eye_type in self.eye_types:
            cv2.imwrite(
                os.path.join(
                    pfolders[f"participant_blinked_{eye_type}_folder"],
                    f"s{session_folder}_{frame_name}",
                ),
                result_dict["eyes"][eye_type.replace("s", "")]["wo_outlines"],
            )


def save_frames(self, sfolders, frame_name, result_dict):

    if result_dict["eyes"]["blinked"]:
        print("Results contain blinked eyes. They will not be saved.")
        return

    results_list = list(result_dict.keys())

    for feature in self.save_features:

        if feature == "full_imgs" and "full_imgs" in result_dict["eyes"]:
            for subdir in self.features_subdirs:
                eyes_data = result_dict["eyes"]["full_imgs"][subdir]
                out_path = os.path.join(sfolders[f"full_imgs_{subdir}_session_path"], frame_name)
                out_path_np = os.path.join(sfolders[f"full_imgs_{subdir}_session_path"], frame_name).replace(".png", "")
                if eyes_data.ndim == 3:
                    cv2.imwrite(out_path, eyes_data)
                else:
                    np.save(out_path_np, eyes_data)
                if subdir == "eyes_outlined" or subdir == "eyes_n_iris_outlined":
                    continue
                iris_data = result_dict["iris"]["full_imgs"][subdir]
                if iris_data.ndim == 3:
                    cv2.imwrite(out_path, iris_data)
                else:
                    np.save(out_path_np, iris_data)
        if feature == "faces" and "face_imgs" in result_dict["eyes"]:
            for subdir in self.features_subdirs:
                eyes_data = result_dict["eyes"]["face_imgs"][subdir]
                out_path = os.path.join(sfolders[f"faces_{subdir}_session_path"], frame_name)
                out_path_np = os.path.join(sfolders[f"faces_{subdir}_session_path"], frame_name).replace(".png", "")
                if eyes_data.ndim == 3:
                    cv2.imwrite(out_path, eyes_data)
                else:
                    np.save(out_path_np, eyes_data)
                if subdir == "eyes_outlined" or subdir == "eyes_n_iris_outlined":
                    continue
                iris_data = result_dict["iris"]["face_imgs"][subdir]
                if iris_data.ndim == 3:
                    cv2.imwrite(out_path, iris_data)
                else:
                    np.save(out_path_np, iris_data)

        if feature == "eyes" and "eyes" in results_list:
            for eye_type in self.eye_types:
                for subdir in (
                    self.features_subdirs + self.segmentation_polygon_subdirs + self.segmentation_otsu_subdirs
                ):
                    out_path = os.path.join(sfolders[f"{eye_type}_{subdir}_session_path"], frame_name)
                    out_path_np = os.path.join(sfolders[f"{eye_type}_{subdir}_session_path"], frame_name).replace(
                        ".png", ""
                    )

                    eyes_data = result_dict["eyes"][eye_type.replace("s", "")][subdir]
                    if eyes_data.ndim == 3:
                        cv2.imwrite(out_path, eyes_data)
                    else:
                        np.save(out_path_np, eyes_data)

        elif feature == "iris" and "iris" in results_list:
            for iris_type in self.iris_types:
                for subdir in (
                    # [self.features_subdirs[0], self.features_subdirs[2], self.features_subdirs[-1]]
                    [self.features_subdirs[0], self.features_subdirs[-1]]
                    + self.segmentation_polygon_subdirs
                    + self.segmentation_otsu_subdirs
                ):
                    out_path = os.path.join(sfolders[f"{iris_type}_{subdir}_session_path"], frame_name)
                    out_path_np = os.path.join(sfolders[f"{iris_type}_{subdir}_session_path"], frame_name).replace(
                        ".png", ""
                    )
                    iris_data = result_dict["iris"][iris_type][subdir]
                    if iris_data.ndim == 3:
                        cv2.imwrite(out_path, iris_data)
                    else:
                        np.save(out_path_np, iris_data)


def plot_avg_EARs_statistics(self):

    for pid, sessions in self.avg_EARs_stats.items():

        sessions_sorted = sorted(sessions.keys(), key=int)

        for session in sessions_sorted:
            avg_EARs_per_session = sessions[session]

            plt.figure(figsize=(14, 6))
            plt.plot(
                list(range(0, len(avg_EARs_per_session))),
                avg_EARs_per_session,
                marker="o",
            )

            plt.xlabel("Frame Number", fontsize=14, labelpad=10)
            plt.ylabel("Eye Aspect Ratio (EAR)", fontsize=14, labelpad=10)
            plt.title(f"Eye Aspect Ratio (EAR) - Participant {pid} - Session - {session}", fontsize=14)
            # plt.legend(fontsize=14)
            plt.xticks(rotation=90, fontsize=12)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.tight_layout()

            participant_EAR_plots_folder = os.path.join(self.data_distribution_plots_folder, "EARs", pid)
            os.makedirs(participant_EAR_plots_folder, exist_ok=True)
            plt.savefig(
                os.path.join(
                    participant_EAR_plots_folder,
                    f"{session}.png",
                )
            )
            plt.close()


def plot_frame_statistics(self):

    for pid, sessions in self.frame_stats.items():

        sessions_sorted = sorted(sessions.keys(), key=int)

        total_frames = [sessions[session]["total_frames"] for session in sessions_sorted]
        skipped_frames = [sessions[session]["skipped_frames"] for session in sessions_sorted]
        processed_frames = [total - skipped for total, skipped in zip(total_frames, skipped_frames)]

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

        participant_plots_folder = os.path.join(self.data_distribution_plots_folder, "sessions_stats")
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

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                fontsize=14,
            )

        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Number of Frames", fontsize=14, labelpad=10)
        ax.set_title(f"Overall Frames - Participant {pid}", fontsize=14)
        plt.tight_layout()

        overall_stats_folder = os.path.join(self.data_distribution_plots_folder, "overall_stats")
        os.makedirs(overall_stats_folder, exist_ok=True)
        plt.savefig(os.path.join(overall_stats_folder, f"{pid}.png"))
        plt.close()

    overall_processed_frames = self.overall_total_frames - self.overall_skipped_frames

    x = ["Before Blink Detection", "After Blink Detection"]
    y = [self.overall_total_frames, overall_processed_frames]

    fig, ax = plt.subplots()
    bars = ax.bar(x, y, color=["#444466", "#BB5555"])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )

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
