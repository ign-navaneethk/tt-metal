# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from typing import List, Optional
from models.experimental.panoptic_deeplab.tt.bottleneck import TTBottleneck
from models.experimental.panoptic_deeplab.tt.stem import resnet52Stem


class TTBackbone:
    def __init__(self, parameters, model_config):
        layers = [3, 4, 6, 3]
        self.inplanes = 128
        # stem
        self.stem = resnet52Stem(parameters.stem, stride=1, model_config=model_config)
        # Four bottleneck stages (layer1, layer2, layer3, layer4)
        self.layer1 = self._make_layer(
            parameters=parameters.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            dialate_config=None,
            model_config=model_config,
        )
        self.layer2 = self._make_layer(
            parameters=parameters.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            dialate_config=None,
            model_config=model_config,
        )
        self.layer3 = self._make_layer(
            parameters=parameters.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            dialate_config=None,
            model_config=model_config,
        )
        self.layer4 = self._make_layer(
            parameters=parameters.layer4,
            planes=512,
            blocks=layers[3],
            stride=1,
            dialate_config=[2, 4, 8],
            model_config=model_config,
        )

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        dialate_config: Optional[List[int]] = None,
        model_config=None,
    ) -> List[TTBottleneck]:
        if dialate_config is None:
            dialate_config = [1] * blocks
        layers = []
        layers.append(
            TTBottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * TTBottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * TTBottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                TTBottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                    dilation=dialate_config[block_num],
                )
            )
        return layers

    def __call__(self, x, device):
        logger.debug(f"Running RN52_backbone Stem")
        x = self.stem(x, device)
        # res1 = x
        logger.debug(f"Running RN52_backbone Layer1")
        for block in self.layer1:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reallocate(x)
            x = block(x, device)
            res_2 = x
        logger.debug(f"Running RN52_backbone Layer2")
        for block in self.layer2:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reallocate(x)
            x = block(x, device)
            res_3 = x
        logger.debug(f"Running RN52_backbone Layer3")
        for block in self.layer3:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reallocate(x)
            x = block(x, device)
        logger.debug(f"Running RN52_backbone Layer4")
        for block in self.layer4:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reallocate(x)
            x = block(x, device)
            res_5 = x
        # return x

        # # Layer 2
        # res_3 = res_2
        # for bottleneck in self.layer2:
        #     res_3 = bottleneck(res_3, device)

        # # Layer 3
        # x = res_3
        # for bottleneck in self.layer3:
        #     x = bottleneck(x, device)

        # # Layer 4
        # res_5 = x
        # for bottleneck in self.layer4:
        #     res_5 = bottleneck(res_5, device)

        return {"res_2": res_2, "res_3": res_3, "res_5": res_5}
