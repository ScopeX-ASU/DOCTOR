"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:24:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:24:50
"""

from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.activation import ReLU
from torch.types import Device, _size

from core.models.devices.mrr_configs import MRRConfig_5um_LQ

# from torchonn.devices.mrr import *
from core.models.layers.utils import *

from .layers.activation import ReLUN
from .layers.mrr_conv2d import AddDropMRRBlockConv2d
from .layers.mrr_linear import AddDropMRRBlockLinear
from .sparse_bp_base import SparseBP_Base

__all__ = [
    "SparseBP_MRR_ResNet18",
    "SparseBP_MRR_ResNet20",
    "SparseBP_MRR_ResNet32",
    "SparseBP_MRR_ResNet34",
    "SparseBP_MRR_ResNet50",
    "SparseBP_MRR_ResNet101",
    "SparseBP_MRR_ResNet152",
]


def conv3x3(
    in_planes,
    out_planes,
    miniblock: int = 8,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    mode: str = "weight",
    MRRConfig=MRRConfig_5um_LQ,
    device: Device = torch.device("cuda"),
):
    conv = AddDropMRRBlockConv2d(
        in_planes,
        out_planes,
        3,
        miniblock=miniblock,
        bias=bias,
        stride=stride,
        padding=padding,
        mode=mode,
        MRRConfig=MRRConfig,
        device=device,
    )
    return conv


def conv1x1(
    in_planes,
    out_planes,
    miniblock: int = 8,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    mode: str = "weight",
    MRRConfig=MRRConfig_5um_LQ,
    device: Device = torch.device("cuda"),
):
    conv = AddDropMRRBlockConv2d(
        in_planes,
        out_planes,
        1,
        miniblock=miniblock,
        bias=bias,
        stride=stride,
        padding=padding,
        mode=mode,
        MRRConfig=MRRConfig,
        device=device,
    )
    return conv


def Linear(
    in_channel,
    out_channel,
    miniblock: int = 8,
    bias: bool = False,
    mode: str = "weight",
    MRRConfig=MRRConfig_5um_LQ,
    device: Device = torch.device("cuda"),
):
    linear = AddDropMRRBlockLinear(
        in_channel,
        out_channel,
        bias,
        miniblock=miniblock,
        mode=mode,
        MRRConfig=MRRConfig,
        device=device,
    )
    return linear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        # unique parameters
        miniblock: int = 8,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_LQ,
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(
            in_planes,
            planes,
            miniblock=miniblock,
            bias=False,
            stride=stride,
            padding=1,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.conv2 = conv3x3(
            planes,
            planes,
            miniblock=miniblock,
            bias=False,
            stride=1,
            padding=1,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.shortcut = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    miniblock=miniblock,
                    bias=False,
                    stride=stride,
                    padding=0,
                    mode=mode,
                    MRRConfig=MRRConfig,
                    device=device,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        # unique parameters
        miniblock: int = 8,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_LQ,
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(
            in_planes,
            planes,
            miniblock=miniblock,
            bias=False,
            stride=1,
            padding=0,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.conv2 = conv3x3(
            planes,
            planes,
            miniblock=miniblock,
            bias=False,
            stride=stride,
            padding=1,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.conv3 = conv1x1(
            planes,
            self.expansion * planes,
            miniblock=miniblock,
            bias=False,
            stride=1,
            padding=0,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    miniblock=miniblock,
                    bias=False,
                    stride=stride,
                    padding=0,
                    mode=mode,
                    MRRConfig=MRRConfig,
                    device=device,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(SparseBP_Base):
    """MRR ResNet. Support sparse backpropagation. Blocking matrix multiplication."""

    def __init__(
        self,
        block,
        num_blocks,
        in_planes,
        img_height: int,
        img_width: int,
        in_channel: int,
        n_class: int,
        block_list: List[int] = [8],
        in_bit: int = 32,
        w_bit: int = 32,
        mode: str = "usv",
        v_max: float = 10.8,
        v_pi: float = 4.36,
        act_thres: float = 6.0,
        photodetect: bool = True,
        MRRConfig=MRRConfig_5um_LQ,
        bias: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()

        # resnet params
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.img_height = img_height
        self.img_width = img_width

        self.in_channel = in_channel
        self.n_class = n_class

        # list of block size
        self.block_list = block_list

        self.in_bit = in_bit
        self.w_bit = w_bit
        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres
        self.photodetect = photodetect

        self.device = device

        # build layers
        blkIdx = 0
        self.conv1 = conv3x3(
            in_channel,
            self.in_planes,
            miniblock=self.block_list[0],
            bias=False,
            stride=1 if img_height <= 64 else 2,  # downsample for imagenet, dogs, cars
            padding=1,
            mode=mode,
            MRRConfig=MRRConfig,
            device=self.device,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        blkIdx += 1

        self.layer1 = self._make_layer(
            block,
            in_planes,
            num_blocks[0],
            stride=1,
            miniblock=self.block_list[0],
            mode=self.mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        blkIdx += 1

        self.layer2 = self._make_layer(
            block,
            in_planes * 2,
            num_blocks[1],
            stride=2,
            miniblock=self.block_list[0],
            mode=self.mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        blkIdx += 1

        self.layer3 = self._make_layer(
            block,
            in_planes * 4,
            num_blocks[2],
            stride=2,
            miniblock=self.block_list[0],
            mode=self.mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        blkIdx += 1

        self.layer4 = self._make_layer(
            block,
            in_planes * 8,
            num_blocks[3],
            stride=2,
            miniblock=self.block_list[0],
            mode=self.mode,
            MRRConfig=MRRConfig,
            device=device,
        )
        blkIdx += 1

        n_channel = in_planes * 8 if num_blocks[3] > 0 else in_planes * 4
        self.linear = Linear(
            n_channel * block.expansion,
            self.n_class,
            miniblock=self.block_list[0],
            bias=False,
            mode=self.mode,
            MRRConfig=MRRConfig,
            device=device,
        )

        self.drop_masks = None

        self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0
        self.set_weight_noise(0.0)
        self.backup_ideal_weights()

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        # unique parameters
        miniblock: int = 8,
        mode: str = "usv",
        MRRConfig=MRRConfig_5um_LQ,
        device: Device = torch.device("cuda"),
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    miniblock=miniblock,
                    mode=mode,
                    MRRConfig=MRRConfig,
                    device=device,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if x.size(-1) > 64:  # 224 x 224, e.g., cars, dogs, imagenet
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def SparseBP_MRR_ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, *args, **kwargs)


def SparseBP_MRR_ResNet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3, 0], 16, *args, **kwargs)


def SparseBP_MRR_ResNet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5, 0], 16, *args, **kwargs)


def SparseBP_MRR_ResNet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], 64, *args, **kwargs)


def SparseBP_MRR_ResNet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], 64, *args, **kwargs)


def SparseBP_MRR_ResNet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], 64, *args, **kwargs)


def SparseBP_MRR_ResNet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], 64, *args, **kwargs)


def test():
    device = torch.device("cuda")
    net = SparseBP_MRR_ResNet18(
        in_channel=3,
        n_class=10,
        block_list=[8, 8, 8, 8, 8, 8],
        in_bit=32,
        w_bit=32,
        mode="usv",
        v_max=10.8,
        v_pi=4.36,
        act_thres=6,
        photodetect=True,
        device=device,
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    print(net)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
