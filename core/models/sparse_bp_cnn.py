"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:50
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-11-17 03:13:01
"""

from collections import OrderedDict
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.types import Device, _size

# from remap_bf import *
from core.models.devices.mrr_configs import MRRConfig_5um_LQ, lambda_res

# from torchonn.devices.mrr import *
from core.models.layers.utils import *

from .layers.activation import ReLUN
from .layers.mrr_conv2d import AddDropMRRBlockConv2d
from .layers.mrr_linear import AddDropMRRBlockLinear
from .sparse_bp_base import SparseBP_Base

__all__ = ["SparseBP_MRR_CNN"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        miniblock: int = 8,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_LQ,
        act_thres: int = 6,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.conv = AddDropMRRBlockConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            miniblock,
            mode=mode,
            MRRConfig=MRRConfig,
            device=device,
        )

        self.bn = nn.BatchNorm2d(out_channel)

        self.activation = ReLUN(act_thres, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        miniblock: int = 8,
        bias: bool = False,
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_LQ,
        activation: bool = True,
        act_thres: int = 6,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.linear = AddDropMRRBlockLinear(
            in_channel, out_channel, bias, miniblock, mode, MRRConfig, device
        )

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SparseBP_MRR_CNN(SparseBP_Base):
    """MRR CNN."""

    _conv_linear = (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)
    _linear = (AddDropMRRBlockLinear,)
    _conv = (AddDropMRRBlockConv2d,)

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channel: int,
        n_class: int,
        kernel_list: List[int] = [32],
        kernel_size_list: List[int] = [3],
        pool_out_size: int = 5,
        stride_list=[1],
        padding_list=[1],
        dilation_list=[1],
        groups=1,
        hidden_list: List[int] = [32],
        block_list: List[int] = [8],
        mode: str = "weight",
        MRRConfig=MRRConfig_5um_LQ,
        act_thres: int = 6,
        bias: bool = False,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.groups = groups

        self.pool_out_size = pool_out_size

        self.hidden_list = hidden_list
        self.block_list = block_list
        self.mode = mode

        self.act_thres = act_thres
        self.MRRConfig = MRRConfig
        self.bias = bias

        self.lambda_res = lambda_res
        self.device = device
        self.build_layers()
        self.drop_masks = None

        self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0
        self.backup_ideal_weights()

        self.set_phase_variation(False)
        self.set_global_temp_drift(False)
        self.set_crosstalk_noise(False)
        self.set_noise_schedulers()

    def build_layers(self):
        self.features = OrderedDict()
        for idx, out_channel in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channel = self.in_channel if (idx == 0) else self.kernel_list[idx - 1]
            self.features[layer_name] = ConvBlock(
                in_channel,
                out_channel,
                self.kernel_size_list[idx],
                self.stride_list[idx],
                self.padding_list[idx],
                self.dilation_list[0],
                self.groups,
                self.bias,
                self.block_list[idx],
                self.mode,
                act_thres=self.act_thres,
                MRRConfig=self.MRRConfig,
                device=self.device,
            )

        self.features = nn.Sequential(self.features)

        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = (
                self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
            )
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, AddDropMRRBlockConv2d):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_channel = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_channel = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_channel,
                out_channel,
                miniblock=self.block_list[idx + len(self.kernel_list)],
                bias=self.bias,
                mode=self.mode,
                MRRConfig=self.MRRConfig,
                activation=True,
                act_thres=self.act_thres,
                device=self.device,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = AddDropMRRBlockLinear(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.n_class,
            miniblock=self.block_list[-1],
            bias=self.bias,
            mode=self.mode,
            MRRConfig=self.MRRConfig,
            device=self.device,
        )
        self.classifier = nn.Sequential(self.classifier)

    # def call_scheduler() -> None:

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
