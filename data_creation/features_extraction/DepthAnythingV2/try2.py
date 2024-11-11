from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

path_base = (
    "./eyedentify/local/data/EyeDentify/Wo_SR"
)
path_full = path_base + "/original/1/1/frame_01.png"
path_face = path_base + "/faces/1/1/frame_01.png"
path_left_eye = path_base + "/eyes/left_eyes/1/1/frame_01.png"
path_right_eye = path_base + "/eyes/right_eyes/1/1/frame_01.png"
misc = "./eyedentify/depth_estimation/20230812_184704.jpg"
image = Image.open(path_full)

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")
print("inputs = ", inputs)
print("pixel_values = ", inputs["pixel_values"].shape)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

print("outputs = ", outputs)
print("predicted_depth = ", predicted_depth)
print("predicted_depth = ", predicted_depth.shape)

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)
print("prediction = ", prediction.shape)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
print("output = ", output)
print("output = ", output.shape)

formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
print("depth = ", depth)
depth.save("tst.jpg")
