# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.tt.common import TTConv2D
from dataclasses import dataclass


def _safe_maxpool2d_large_tensor_device(input_tensor, shape, kernel_size, stride, padding, dilation, ceil_mode=False):
    """
    Applies 2D max pooling on a large input tensor by manually slicing along the height dimension and processing
    each chunk independently to avoid L1 memory overflows.

    This function is designed for TTNN execution environments where L1 memory constraints are critical.
    It explicitly slices the input tensor into predefined height ranges, reshapes and reconfigures each chunk
    for row-major layout and suitable datatype, and performs max pooling using TTNN with appropriate padding.

    Args:
        input_tensor (ttnn.Tensor): Input tensor of shape [B, C, H, W].
        kernel_size (list[int]): Pooling kernel size as [kh, kw].
        stride (list[int]): Pooling stride as [sh, sw].
        padding (list[int]): Padding as [ph, pw] applied to the first slice; subsequent slices get adjusted padding.
        dilation (list[int]): Dilation as [dh, dw] for pooling kernel.
        ceil_mode (bool, optional): Whether to use ceil instead of floor in output shape calculation. Defaults to False.

    Returns:
        ttnn.Tensor: Concatenated pooled tensor across height dimension.

    Raises:
        RuntimeError: If all height slices fail to pool due to memory or configuration issues.

    Notes:
        - Only the first slice uses the full padding specified; subsequent slices adjust to avoid overlapping padded regions.
        - Slices are reshaped into 4D tensors with a flat spatial dimension to align with TTNN memory constraints.
        - Uses DRAM memory config with height sharding and in-place halo optimization.
    """
    splits = []
    splits = [(0, 256), (255, 512), (511, 768), (767, 1024)]
    pooled_rows = []
    h_id = 0
    input_tensor = ttnn.reshape(input_tensor, shape)
    for index, i in enumerate(splits):  # Split width
        h_chunk = input_tensor[:, :, i[0] : i[1], :]
        batch_size = shape[0]
        input_h = shape[1]
        input_w = splits[index][1] - splits[index][0]
        channels = shape[3]
        h_chunk = ttnn.reshape(h_chunk, (1, 1, batch_size * input_h * input_w, channels))
        ttnn.reallocate(h_chunk)

        if index == 0:
            in_pad = padding
        else:
            in_pad = [1, 0]

        logger.info(f"Running Slice {index}")
        pooled_block = ttnn.max_pool2d(
            input_tensor=h_chunk,
            batch_size=batch_size,
            input_h=input_h,
            input_w=input_w,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=in_pad,
            dilation=dilation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            in_place_halo=True,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ceil_mode=ceil_mode,
        )
        pooled_block = ttnn.reshape(pooled_block, (shape[0], shape[1] // 2, (shape[2] // len(splits)) // 2, shape[3]))
        pooled_rows.append(pooled_block)
        h_id += 1

    if pooled_rows:
        ttnn.deallocate(h_chunk)
        out = ttnn.concat(pooled_rows, dim=2)
        ttnn.deallocate(pooled_block)
        return out

    else:
        raise RuntimeError("All chunks failed to pool. Reduce chunk height or try TILE memory.")


@dataclass
class NeckOptimizer:
    conv1: dict()
    conv2: dict()
    conv3: dict()


neck_optimisations = {
    "optimization_full_tensor": NeckOptimizer(
        conv1={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4),
        },
        conv2={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4),
        },
        conv3={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4),
        },
    ),
    "optimization_small_tensor": NeckOptimizer(
        conv1={
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=2),
        },
        conv2={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
}


class resnet52Stem:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        layer_optimisations=neck_optimisations["optimization_full_tensor"],
    ) -> None:
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            activation="relu",
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            **layer_optimisations.conv1,
        )
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            activation="relu",
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            **layer_optimisations.conv2,
        )
        self.conv3 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            activation="relu",
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            **layer_optimisations.conv3,
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

        if shape[-3] == 256:
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
                in_place_halo=True,
                applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ceil_mode=False,
            )
        else:
            logger.debug(f"Running  maxpool")
            out = _safe_maxpool2d_large_tensor_device(
                input_tensor=out,
                shape=shape,
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[1, 1],
                ceil_mode=False,
            )

        return out
