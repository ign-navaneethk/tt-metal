# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.common import TTConv2D, TTUpsample
from dataclasses import dataclass


@dataclass
class HeadOptimizer:
    conv1: dict()
    conv2: dict()
    conv3: dict()


head_layer_optimisations = {
    "default": HeadOptimizer(
        conv1={"act_block_h": 32, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        conv2={
            "act_block_h": 32,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv3={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
        },
    ),
    "semantic_head": HeadOptimizer(
        conv1={
            "act_block_h": 256,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "reshard_if_not_optimal": True,
        },
        conv2={
            "act_block_h": 32,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        },
        conv3={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
    ),
    "instance_offset_head": HeadOptimizer(
        conv1={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv2={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv3={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
        },
    ),
    "instance_center_head": HeadOptimizer(
        conv1={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv2={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv3={
            "act_block_h": 64,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": False,
            "input_channels_alignment": 32,
        },
    ),
}


class TTHead:
    def __init__(
        self,
        parameters,
        model_config,
        layer_optimisations=head_layer_optimisations["default"],
    ) -> None:
        self.conv1 = TTConv2D(
            kernel_size=parameters.conv_args["conv1"]["0"].kernel_size,
            stride=parameters.conv_args["conv1"]["0"].stride,
            padding=parameters.conv_args["conv1"]["0"].padding,
            dilation=parameters.conv_args["conv1"]["0"].dilation,
            groups=parameters.conv_args["conv1"]["0"].groups,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation="relu",
            **layer_optimisations.conv1,
        )
        self.conv2 = TTConv2D(
            kernel_size=parameters.conv_args["conv2"]["0"].kernel_size,
            stride=parameters.conv_args["conv2"]["0"].stride,
            padding=parameters.conv_args["conv2"]["0"].padding,
            groups=parameters.conv_args["conv2"]["0"].groups,
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            activation="relu",
            **layer_optimisations.conv2,
        )
        self.conv3 = TTConv2D(
            kernel_size=parameters.conv_args["conv3"]["0"].kernel_size,
            stride=parameters.conv_args["conv3"]["0"].stride,
            padding=parameters.conv_args["conv3"]["0"].padding,
            groups=parameters.conv_args["conv3"]["0"].groups,
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            **layer_optimisations.conv3,
        )

        self.upsample = TTUpsample(
            scale_factor=(4),
            mode="bilinear",
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(
        self,
        x,
        shape,
        device,
    ):
        logger.debug("Running conv1")
        out, shape = self.conv1(device, x, shape)

        logger.debug("Running conv2")
        out, shape = self.conv2(device, out, shape)

        logger.debug("Running conv3")
        out, shape = self.conv3(device, out, shape)

        logger.debug("Running final upsample")
        out = self.upsample(device, out, shape, reshape_output=False, pad_ch_to_32=True, sent_to_dram=False)

        return out
