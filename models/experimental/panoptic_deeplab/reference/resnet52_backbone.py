from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
import torch.nn as nn
from typing import Callable, List, Optional
from torch import Tensor
from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem


class Res5Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            norm_layer(planes * self.expansion),
        )

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


class ResNet52BackBone(nn.Module):
    def __init__(
        self,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        groups: int = 1,
        width_per_group: int = 64,
        dialate_layer_config: Optional[List[List[int]]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        out_features: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        if dialate_layer_config is None:
            # each element in the tuple indicates the dialtaion to be used
            # in the 3x3 conv for each layer
            dialate_layer_config = [[1] * num_layers for num_layers in layers]
        if len(dialate_layer_config) != len(layers):
            raise ValueError(
                "dialate_layer_config should be None " f"or a {len(layers)}-element tuple, got {dialate_layer_config}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.stem = DeepLabStem(in_channels=3, out_channels=self.inplanes, stride=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dialate_config=dialate_layer_config[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dialate_config=dialate_layer_config[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dialate_config=dialate_layer_config[2])
        # layer4 = [Res5Bottleneck(self.inplanes, 512, stride=1, groups=1, base_width=self.base_width, dilation=2)]
        # self.layer4 = nn.Sequential(*layer4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dialate_config=[2, 4, 8])

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dialate_config: Optional[List[int]] = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        if dialate_config is None:
            dialate_config = [1] * blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, dialate_config[0], norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for block_index in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=dialate_config[block_index],
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        res_2 = self.layer1(x)
        res_3 = self.layer2(res_2)
        res_4 = self.layer3(res_3)
        res_5 = self.layer4(res_4)

        out = {
            "res_2": res_2,
            "res_3": res_3,
            "res_4": res_4,
            "res_5": res_5,
        }

        return out
