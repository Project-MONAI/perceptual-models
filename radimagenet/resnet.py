""" Resnet implementation based on https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py """
import errno
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import monai


# Layers

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

# Blocks 

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        width = int(planes)
        self.conv1 = conv1x1(inplanes, width, stride=stride)
        self.bn1 = nn.BatchNorm2d(width, eps=1.001e-5, momentum=0.99)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.bn2 = nn.BatchNorm2d(width, eps=1.001e-5, momentum=0.99)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1.001e-5, momentum=0.99)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1.001e-5, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion, eps=1.001e-5, momentum=0.99),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def download_model(filename:str, 
                   cache_dir: Optional[str] = None,
                   ) -> str:
    """Downloads a model from the Huggingface MONAI library.

    Args:
        filename (str): name of the pth file. 
        cache_dir (Optional[str], optional): cache directory (-> torch.hub cache dir). Defaults to None.

    Returns:
        str: file path of the downloaded. model.
    """

    monai.apps.utils.download_url(
        "https://huggingface.co/MONAI/checkpoints/resolve/main/RadImageNet-ResNet50_notop.pth", 
        filepath=os.path.join(cache_dir, 'monai_checkpoints', filename)
        )
    
    return os.path.join(cache_dir, 'monai_checkpoints', filename)
    
# RadImageNet ResNet50

def radimagenet_resnet50(
    cache_dir: Optional[str] = None,
    file_name: str = "RadImageNet-ResNet50_notop.pth",
):
    if cache_dir is None:
        hub_dir = torch.hub.get_dir()
        cache_dir = os.path.join(hub_dir)

    filename = file_name

    cached_file = os.path.join(cache_dir, filename)

    if not os.path.exists(cached_file):
        cached_file = download_model(filename, cache_dir=cache_dir)
    

    model = ResNet50()
    model.load_state_dict(torch.load(cached_file))
    return model