# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from torch import nn


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class DeepLabStem(CNNBlockBase):
    """
    The DeepLab ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=128, stride=1):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn1=nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2=nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn3=nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
