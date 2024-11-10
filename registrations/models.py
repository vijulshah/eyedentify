import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from registry import MODEL_REGISTRY
from registrations.ResNetVariants.CifarBasedResNets import ResNetCifar
from registrations.ResNetVariants.ImageNetBasedResNets import ResNetImageNet


# ============================= ImageNet Based ResNet Variants =============================


@MODEL_REGISTRY.register()
class ResNet18(ResNetImageNet):
    def __init__(self, model_args):
        super(ResNet18, self).__init__(model_args, models.resnet18)


@MODEL_REGISTRY.register()
class ResNet34(ResNetImageNet):
    def __init__(self, model_args):
        super(ResNet34, self).__init__(model_args, models.resnet34)


@MODEL_REGISTRY.register()
class ResNet50(ResNetImageNet):
    def __init__(self, model_args):
        super(ResNet50, self).__init__(model_args, models.resnet50)


@MODEL_REGISTRY.register()
class ResNet101(ResNetImageNet):
    def __init__(self, model_args):
        super(ResNet101, self).__init__(model_args, models.resnet101)


@MODEL_REGISTRY.register()
class ResNet152(ResNetImageNet):
    def __init__(self, model_args):
        super(ResNet152, self).__init__(model_args, models.resnet152)


# ============================= Cifar Based ResNet Variants =============================


@MODEL_REGISTRY.register()
class ResNetCifar20(ResNetCifar):
    def __init__(self, model_args):
        super(ResNetCifar20, self).__init__(model_args, num_blocks=[3, 3, 3])


@MODEL_REGISTRY.register()
class ResNetCifar32(ResNetCifar):
    def __init__(self, model_args):
        super(ResNetCifar32, self).__init__(model_args, num_blocks=[5, 5, 5])


@MODEL_REGISTRY.register()
class ResNetCifar44(ResNetCifar):
    def __init__(self, model_args):
        super(ResNetCifar44, self).__init__(model_args, num_blocks=[7, 7, 7])


@MODEL_REGISTRY.register()
class ResNetCifar56(ResNetCifar):
    def __init__(self, model_args):
        super(ResNetCifar56, self).__init__(model_args, num_blocks=[9, 9, 9])


@MODEL_REGISTRY.register()
class ResNetCifar110(ResNetCifar):
    def __init__(self, model_args):
        super(ResNetCifar110, self).__init__(model_args, num_blocks=[18, 18, 18])


@MODEL_REGISTRY.register()
class ResNetCifar1202(ResNetCifar):
    def __init__(self, model_args):
        super(ResNetCifar1202, self).__init__(model_args, num_blocks=[200, 200, 200])


print("Registered models in MODEL_REGISTRY:", MODEL_REGISTRY.keys())
