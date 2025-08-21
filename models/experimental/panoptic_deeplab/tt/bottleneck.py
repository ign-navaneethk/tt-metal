# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.tt.common import TTConv2D


class TTBottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config, dilation: int = 1) -> None:
        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            is_reshape=False,  # not working
        )
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride if downsample else 1,
            padding=dilation,
            dilation=dilation,
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )

        self.downsample = downsample
        if downsample:
            shard_layout = None
            slice_config = None
            if parameters.downsample.weight.shape[0] == 512 and parameters.downsample.weight.shape[1] == 256:
                shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4)
            self.downsample_conv = TTConv2D(
                kernel_size=1,
                stride=stride,
                padding=0,
                parameters=parameters.downsample,
                kernel_fidelity=model_config,
                deallocate_activation=True,
                reallocate_halo_output=True,
                shard_layout=shard_layout,
                slice_config=slice_config,
                memory_config=None,
            )

        self.model_config = model_config
        return

    def __call__(
        self,
        x,
        device,
        eltwise_binary_out_in_place=True,
    ):
        # conv1 is 1x1 conv
        logger.debug(f"Running conv1")
        out1, shape = self.conv1(device, x, x.shape)

        # conv2 is 3x3 conv
        logger.debug(f"Running conv2")
        out2, shape = self.conv2(device, out1, shape)
        ttnn.deallocate(out1)
        # conv3 is 1x1 conv
        logger.debug(f"Running conv3")
        out, shape = self.conv3(device, out2, shape)
        ttnn.deallocate(out2)

        # run downsample conv 1x1 if required
        if self.downsample:
            logger.debug(f"Running downsample")
            ds_out, _ = self.downsample_conv(device, x, x.shape)
        else:
            ds_out = x
        ds_out = ttnn.reallocate(ds_out)
        ttnn.deallocate(x)
        if ds_out.shape != out.shape:
            ds_out = ttnn.reshape(ds_out, (1, 1, ds_out.shape[0] * ds_out.shape[1] * ds_out.shape[2], ds_out.shape[3]))
        if ds_out.layout != out.layout:
            ds_out = ttnn.to_layout(ds_out, out.layout)
        if ds_out.memory_config() != out.memory_config():
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reallocate(out)
        if eltwise_binary_out_in_place:
            # underscore version is in_place = True
            out = ttnn.add_(
                out,
                ds_out,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
            )
        else:
            out = ttnn.add(
                out,
                ds_out,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        out = ttnn.reshape(out, shape)

        ttnn.deallocate(ds_out)
        return out
