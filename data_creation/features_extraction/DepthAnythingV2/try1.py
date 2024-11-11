import cv2
import torch
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

encoder = "vits"  # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(
    torch.load(f"./depth_estimation/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu")
)
model = model.to(DEVICE).eval()

path = "./eyedentify/local/data/EyeDentify/Wo_SR/original/1/1/frame_01.png"

misc = "./eyedentify/depth_estimation/20230812_184704.jpg"

raw_img = cv2.imread(misc)
depth = model.infer_image(raw_img)  # HxW raw depth map in nump
print("depth = ", depth, depth.shape)
plt.imsave("tst2.jpg", depth)
