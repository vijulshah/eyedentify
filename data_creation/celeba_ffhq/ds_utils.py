import os
import cv2
import matplotlib.pyplot as plt


def save_frames(output_path, frame_name, result_dict, save_features):

    eye_types = ["left_eyes", "right_eyes"]
    iris_types = ["left_iris", "right_iris"]

    results_list = list(result_dict.keys())

    blinked = result_dict.get("eyes", {}).get("blinked", False)

    if "blinks" in save_features and blinked:
        for eye_type in eye_types:
            save_path = os.path.join(output_path, "blinks", eye_type)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_path, frame_name),
                result_dict["eyes"][eye_type.replace("s", "")]["wo_outlines"],
            )
    elif not blinked:
        for feature in save_features:
            if feature == "eyes" and "eyes" in results_list:
                for eye_type in eye_types:
                    save_path = os.path.join(output_path, "eyes", eye_type)
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(save_path, frame_name),
                        result_dict["eyes"][eye_type.replace("s", "")]["wo_outlines"],
                    )
            elif feature == "iris" and "iris" in results_list:
                for iris_type in iris_types:
                    save_path = os.path.join(output_path, "iris", iris_type)
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(save_path, frame_name),
                        result_dict["iris"][iris_type]["wo_outlines"],
                    )


def plot_avg_EARs_statistics(data_distribution_folder, img_avg_EARs):

    plt.figure(figsize=(14, 6))
    plt.plot(
        list(range(0, len(img_avg_EARs))),
        img_avg_EARs,
        marker="o",
    )

    plt.xlabel("Image Number", fontsize=14, labelpad=10)
    plt.ylabel("Eye Aspect Ratio (EAR)", fontsize=14, labelpad=10)
    plt.title(
        f"Eye Aspect Ratio (EAR) of all the Images",
        fontsize=14,
    )
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(data_distribution_folder, exist_ok=True)
    plt.savefig(os.path.join(data_distribution_folder, "EARs.png"))
    plt.close()


def plot_frame_statistics(data_distribution_folder, frame_stats):

    total_frames = frame_stats["total_imgs"]
    skipped_frames = frame_stats["skipped_imgs"]
    processed_frames = total_frames - skipped_frames

    x = ["Before Blink Detection", "After Blink Detection"]
    y = [total_frames, processed_frames]

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
    ax.set_ylabel("Number of Images", fontsize=14, labelpad=10)
    ax.set_title(f"Overall Images", fontsize=14)
    plt.tight_layout()

    os.makedirs(data_distribution_folder, exist_ok=True)
    plt.savefig(os.path.join(data_distribution_folder, "overall_stats.png"))
    plt.close()
