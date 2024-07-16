import sys
import torch.nn as nn
import os.path as osp
from torchvision import models
import torch.nn.functional as F
from registry import MODEL_REGISTRY

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

# ============================= ResNets =============================


@MODEL_REGISTRY.register()
class ResNet18(nn.Module):
    def __init__(self, model_args):
        super(ResNet18, self).__init__()
        self.num_classes = model_args.get("num_classes", 1)
        self.resnet = models.resnet18(weights=None)
        self.regression_head = nn.Linear(1000, self.num_classes)

    def forward(self, x, masks=None):
        # Calculate the padding dynamically based on the input size
        height, width = x.shape[2], x.shape[3]
        pad_height = max(0, (224 - height) // 2)
        pad_width = max(0, (224 - width) // 2)

        # Apply padding
        x = F.pad(
            x, (pad_width, pad_width, pad_height, pad_height), mode="constant", value=0
        )
        x = self.resnet(x)
        x = self.regression_head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet50(nn.Module):
    def __init__(self, model_args):
        super(ResNet50, self).__init__()
        self.num_classes = model_args.get("num_classes", 1)
        self.resnet = models.resnet50(weights=None)
        self.regression_head = nn.Linear(1000, self.num_classes)

    def forward(self, x, masks=None):
        # Calculate the padding dynamically based on the input size
        height, width = x.shape[2], x.shape[3]
        pad_height = max(0, (224 - height) // 2)
        pad_width = max(0, (224 - width) // 2)

        # Apply padding
        x = F.pad(
            x, (pad_width, pad_width, pad_height, pad_height), mode="constant", value=0
        )
        x = self.resnet(x)
        x = self.regression_head(x)
        return x


print("Registered models in MODEL_REGISTRY:", MODEL_REGISTRY.keys())
