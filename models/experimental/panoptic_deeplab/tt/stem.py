# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.tt.common import TTConv2D


class resnet52Stem:

    def __init__(self, parameters, stride, model_config) -> None:
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            deallocate_activation=True,
            reallocate_halo_output=True,
            is_reshape=False, # not working
            memory_config=None,
            slice_config = ttnn.Conv2dSliceConfig(  
                slice_type=ttnn.Conv2dSliceHeight,  # or ttnn.Conv2dSliceWidth  
                num_slices=8 # Adjust based on memory constraints
            ),
        )
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            deallocate_activation=True,
            reallocate_halo_output=True,
            memory_config=None,
            slice_config = ttnn.Conv2dSliceConfig(  
                slice_type=ttnn.Conv2dSliceHeight,  # or ttnn.Conv2dSliceWidth  
                num_slices=4  # Adjust based on memory constraints  
            ),
        )
        self.conv3 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            activation="relu",
            act_block_h=32,
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            deallocate_activation=True,
            reallocate_halo_output=True,
            memory_config=None,
            slice_config = ttnn.Conv2dSliceConfig(  
                slice_type=ttnn.Conv2dSliceHeight,  # or ttnn.Conv2dSliceWidth  
                num_slices=4  # Adjust based on memory constraints  
            ),
        )
    
    def __call__(
        self,
        x,
        device,
    ):

        # conv1 is stride 2 conv 3x3
        logger.debug(f"Running 3x3 conv1")
        out, shape = self.conv1(device, x, x.shape)

        # conv2 and 3 are 3x3 conv's with stride 1
        logger.debug(f"Running 3x3 conv2")
        out, shape = self.conv2(device, out, shape)
        logger.debug(f"Running 3x3 conv3")
        out, shape = self.conv3(device, out, shape)

        out = ttnn.max_pool2d(
            input_tensor=out,
            batch_size=shape[-4],
            input_h=shape[-3],
            input_w=shape[-2],
            channels=shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            in_place_halo=True,
        )
        return out