import torch
import torch.nn as nn
import torch.nn.functional as F


# Utility function for padding
def apply_padding(x, height, width, dim):
    pad_height = max(0, (dim - height) // 2) if height < dim else 0
    pad_width = max(0, (dim - width) // 2) if width < dim else 0

    if pad_height > 0 or pad_width > 0:
        x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), mode="constant", value=0)
    return x


# Base class for ResNet models
class ResNetImageNet(nn.Module):
    def __init__(self, model_args, resnet_func):
        super(ResNetImageNet, self).__init__()

        self.num_classes = model_args.get("num_classes", 1)
        self.pad_img = model_args.get("pad_img", True)
        self.use_regression_head = model_args.get("use_regression_head", True)
        self.pretrained = model_args.get("pretrained", False)
        weights = "DEFAULT" if self.pretrained else None

        # Initialize ResNet model
        self.resnet = resnet_func(weights=weights)

        # Modify for regression task
        if self.use_regression_head:
            self.regression_head = nn.Linear(1000, self.num_classes)
        else:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        if self.pad_img:
            height, width = x.shape[2], x.shape[3]
            x = apply_padding(x, height, width, dim=192)

        x = self.resnet(x)
        if self.use_regression_head:
            x = self.regression_head(x)
        return x
