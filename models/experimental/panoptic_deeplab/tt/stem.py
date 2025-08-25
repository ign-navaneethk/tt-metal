# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.tt.common import TTConv2D


def safe_maxpool2d_large_tensor(input_tensor, kernel_size, stride, padding, dilation, ceil_mode=False):
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
    # splits = [(0, 256), (255, 512), (511, 768), (767, 1024)]
    splits = [(0, 256), (255, 512)]
    pooled_rows = []
    h_id = 0
    for index, i in enumerate(splits):  # Split height
        h_chunk = input_tensor[:, :, i[0] : i[1], :]

        batch_size = h_chunk.shape[-4]
        input_h = h_chunk.shape[-3]
        input_w = h_chunk.shape[-2]
        channels = h_chunk.shape[-1]
        h_chunk = ttnn.reshape(h_chunk, (1, 1, batch_size * input_h * input_w, channels))
        h_chunk = ttnn.to_dtype(h_chunk, ttnn.bfloat16)
        h_chunk = ttnn.to_layout(h_chunk, ttnn.ROW_MAJOR_LAYOUT)
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # or try TILE_MEMORY_CONFIG
            in_place_halo=True,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ceil_mode=ceil_mode,
        )
        # pooled_block = ttnn.reshape(pooled_block, (1, 256, 128, channels))
        pooled_block = ttnn.reshape(pooled_block, (1, 128, 128, channels))
        pooled_rows.append(pooled_block)
        h_id += 1

    if pooled_rows:
        ttnn.deallocate(h_chunk)
        out = ttnn.concat(pooled_rows, dim=2)
        ttnn.deallocate(pooled_block)
        return out

    else:
        raise RuntimeError("All chunks failed to pool. Reduce chunk height or try TILE memory.")


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
            is_reshape=False,  # not working
            memory_config=None,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,  # or ttnn.Conv2dSliceWidth
                num_slices=8,  # Adjust based on memory constraints
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
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,  # or ttnn.Conv2dSliceWidth
                num_slices=4,  # Adjust based on memory constraints
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
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,  # or ttnn.Conv2dSliceWidth
                num_slices=4,  # Adjust based on memory constraints
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

        out = ttnn.reshape(out, shape)
        logger.debug(f"Running  maxpool")
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reallocate(out)
        out = safe_maxpool2d_large_tensor(
            input_tensor=out,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=False,
        )

        return out
