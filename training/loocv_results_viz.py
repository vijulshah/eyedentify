import os
import sys
import matplotlib.pyplot as plt

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
sys.path.append(root_path)

metrics = ["MAE", "MAPE"]
folders = ["ResNet18_left_eyes", "ResNet18_right_eyes", "ResNet50_left_eyes", "ResNet50_right_eyes"]

data = {folder: [] for folder in folders}
participants = list(range(1, 52))

for metric in metrics:
    for folder in folders:
        for pid in participants:
            if metric == "MAE":
                file_path = os.path.join(
                    root_path, f"local_{folder}", "mlruns", "0", str(pid), "metrics", "test_epoch_running_loss"
                )
            elif metric == "MAPE":
                file_path = os.path.join(
                    root_path, f"local_{folder}", "mlruns", "0", str(pid), "metrics", f"test_epoch_running_mape"
                )

            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    line = f.readline().strip()

                    _, loss, _ = line.split()
                    if metric == "MAE":
                        loss = float(loss)
                    elif metric == "MAPE":
                        loss = float(loss) * 100

                    data[folder].append(loss)
            else:
                print(f"File not found: {file_path}")
                data[folder].append(None)

    plt.figure(figsize=(18, 6))

    for folder, losses in data.items():
        plt.plot(range(1, 52), losses, marker="o", label=folder)

    plt.xlabel("Participant ID", fontsize=14, labelpad=10)
    plt.ylabel(metric, fontsize=14, labelpad=10)
    plt.title(f"LOOCV {metric}", fontsize=14)
    plt.xticks(participants)
    plt.legend()
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=14)

    plt_file = os.path.join(root_path, "local", "publications", "imgs", f"LOOCV_Models_Comparision_{metric}.png")
    plt.savefig(plt_file)

    plt.close()
    print("Saved: ", plt_file)
