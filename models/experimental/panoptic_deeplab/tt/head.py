# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

# from models.experimental.panoptic_deeplab.tt.common import TTConv2D
from models.experimental.panoptic_deeplab.tt.common import TTConv2D, TTUpsample
from dataclasses import dataclass


@dataclass
class HeadOptimizer:
    conv1: dict()
    conv2: dict()
    conv3: dict()
    # downsample: dict()


# class PanopticDeeplabInstanceASPP:
#     def __init__(self, parameters, model_config) -> None:
#         self.model_config = model_config

#         # Ins_Seg_ASPP_0
#         self.Ins_Seg_ASPP_0_Conv = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_ASPP_0_Conv,
#             kernel_fidelity=model_config,
#             activation="relu",
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_1_Depthwise
#         self.Ins_Seg_ASPP_1_Depthwise = TTConv2D(
#             kernel_size=3,
#             stride=1,
#             padding=6,
#             dilation=6,
#             groups=2048,
#             parameters=parameters.Ins_Seg_ASPP_1_Depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=64,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_1_pointwise
#         self.Ins_Seg_ASPP_1_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_ASPP_1_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             deallocate_activation=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_2_Depthwise
#         self.Ins_Seg_ASPP_2_Depthwise = TTConv2D(
#             kernel_size=3,
#             stride=1,
#             padding=12,
#             dilation=12,
#             groups=2048,
#             parameters=parameters.Ins_Seg_ASPP_2_Depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=1024,
#             shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_2_pointwise
#         self.Ins_Seg_ASPP_2_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_ASPP_2_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             deallocate_activation=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_3_Depthwise
#         self.Ins_Seg_ASPP_3_Depthwise = TTConv2D(
#             kernel_size=3,
#             stride=1,
#             padding=18,
#             dilation=18,
#             groups=2048,
#             parameters=parameters.Ins_Seg_ASPP_3_Depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=512,
#             shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_3_pointwise
#         self.Ins_Seg_ASPP_3_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_ASPP_3_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             deallocate_activation=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_4_Conv_1
#         self.Ins_Seg_ASPP_4_Conv_1 = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_ASPP_4_Conv_1,
#             kernel_fidelity=model_config,
#             activation="relu",
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
#             deallocate_activation=True,
#             reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_ASPP_project
#         self.Ins_Seg_ASPP_project = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_ASPP_project,
#             kernel_fidelity=model_config,
#             activation="relu",
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reshard_if_not_optimal=True,
#         )

#     def __call__(self, x, device):
#         # ASPP branch - exact copy from original
#         logger.debug("Running Ins_Seg_ASPP_0_Conv")
#         aspp0, shape = self.Ins_Seg_ASPP_0_Conv(device, x, x.shape)

#         logger.debug("Running Ins_Seg_ASPP_1_Depthwise")
#         aspp1_dw, shape = self.Ins_Seg_ASPP_1_Depthwise(device, x, x.shape)

#         logger.debug("Running Ins_Seg_ASPP_1_pointwise")
#         aspp1, shape = self.Ins_Seg_ASPP_1_pointwise(device, aspp1_dw, shape)

#         logger.debug("Running Ins_Seg_ASPP_2_Depthwise")
#         aspp2_dw, shape = self.Ins_Seg_ASPP_2_Depthwise(device, x, x.shape)

#         logger.debug("Running Ins_Seg_ASPP_2_pointwise")
#         aspp2, shape = self.Ins_Seg_ASPP_2_pointwise(device, aspp2_dw, shape)

#         logger.debug("Running Ins_Seg_ASPP_3_Depthwise")
#         aspp3_dw, shape = self.Ins_Seg_ASPP_3_Depthwise(device, x, x.shape)

#         logger.debug("Running Ins_Seg_ASPP_3_pointwise")
#         aspp3, shape = self.Ins_Seg_ASPP_3_pointwise(device, aspp3_dw, shape)

#         aspp3_dw.deallocate()

#         logger.debug("Running Ins_Seg_ASPP_4_avg_pool")
#         x = ttnn.reshape(x, [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])

#         aspp4 = ttnn.avg_pool2d(
#             input_tensor=x,
#             batch_size=1,
#             input_h=32,
#             input_w=64,
#             channels=2048,
#             kernel_size=(32, 64),
#             stride=(1, 1),
#             padding=(0, 0),
#         )
#         x.deallocate()

#         logger.debug("Running Ins_Seg_ASPP_4_Conv_1")
#         shape = (1, 1, 1, 2048)
#         aspp4_conv, shape = self.Ins_Seg_ASPP_4_Conv_1(device, aspp4, shape)
#         aspp4.deallocate()

#         logger.debug("Running Ins_Seg_ASPP_4_upsample")
#         # aspp4_conv = ttnn.sharded_to_interleaved(aspp4_conv, ttnn.DRAM_MEMORY_CONFIG)
#         print(f"{aspp4_conv=}")
#         aspp4_conv = ttnn.to_layout(aspp4_conv, ttnn.ROW_MAJOR_LAYOUT)

#         aspp4_conv_upsample = ttnn.upsample(
#             aspp4_conv,
#             scale_factor=(32, 64),
#             mode="bilinear",
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             compute_kernel_config=ttnn.WormholeComputeKernelConfig(
#                 math_fidelity=ttnn.MathFidelity.LoFi,
#                 math_approx_mode=True,
#             ),
#         )

#         aspp4_conv_upsample = ttnn.to_layout(aspp4_conv_upsample, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
#         # aspp4_conv_upsample = ttnn.typecast(aspp4_conv_upsample, dtype=ttnn.bfloat8_b)
#         print(f"{aspp4_conv_upsample=}")

#         aspp4_conv_upsample = ttnn.reshape(
#             aspp4_conv_upsample,
#             [1, 1, 1 * 32 * 64, aspp4_conv_upsample.shape[3]],
#         )
#         # ASPP project
#         logger.debug("Running Ins_Seg_ASPP_concat")
#         aspp_concat = ttnn.concat(
#             [aspp0, aspp1, aspp2, aspp3, aspp4_conv_upsample],
#             dim=3,
#         )
#         aspp0.deallocate()
#         aspp1.deallocate()
#         aspp2.deallocate()
#         aspp3.deallocate()
#         aspp4_conv.deallocate()
#         aspp4_conv_upsample.deallocate()

#         logger.debug("Running Ins_Seg_ASPP_project")
#         shape = (1, 32, 64, 1280)
#         aspp_project, shape = self.Ins_Seg_ASPP_project(device, aspp_concat, shape)
#         aspp_concat.deallocate()

#         logger.debug("finished with instance ASPP")
#         return aspp_project


# class PanopticDeeplabInstanceDecoderRes3:
#     def __init__(self, parameters, model_config) -> None:
#         # Ins_Seg_Decoder_res3_project_conv
#         self.Ins_Seg_Decoder_res3_project_conv = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Decoder_res3_project_conv,
#             kernel_fidelity=model_config,
#             activation="relu",
#             # act_block_h=32,
#             # memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             # shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             deallocate_activation=True,
#             # reallocate_halo_output=True,
#             # enable_split_reader=True,
#             # enable_act_double_buffer=True,
#             # enable_weights_double_buffer=True,
#             # reshard_if_not_optimal=True,
#         )
#         # Ins_Seg_Decoder_res3_fuse_conv_depthwise
#         self.Ins_Seg_Decoder_res3_fuse_conv_depthwise = TTConv2D(
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             dilation=1,
#             groups=320,
#             parameters=parameters.Ins_Seg_Decoder_res3_fuse_conv_depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=512,
#             # memory_config=ttnn.L1_MEMORY_CONFIG,
#             # memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             # shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#             reshard_if_not_optimal=True,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )
#         # Ins_Seg_Decoder_res3_fuse_conv_pointwise
#         self.Ins_Seg_Decoder_res3_fuse_conv_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Decoder_res3_fuse_conv_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=32,
#             # memory_config=ttnn.L1_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )

#     def __call__(self, x, res3, device):
#         # Exact copy from original implementation
#         print("res3.shape", res3.shape)
#         logger.debug("Running upsample after ASPP project")
#         aspp_project = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
#         aspp_project = ttnn.to_layout(aspp_project, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
#         # aspp_project = ttnn.reshape(aspp_project, [1, 32, 64, 256])

#         aspp_project_upsampled = ttnn.upsample(
#             aspp_project,
#             scale_factor=2,
#             mode="bilinear",
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             compute_kernel_config=ttnn.WormholeComputeKernelConfig(
#                 math_fidelity=ttnn.MathFidelity.LoFi,
#                 math_approx_mode=True,
#                 fp32_dest_acc_en=False,
#             ),
#         )
#         aspp_project.deallocate()
#         aspp_project_upsampled = ttnn.to_layout(aspp_project_upsampled, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
#         aspp_project_upsampled = ttnn.reshape(
#             aspp_project_upsampled,
#             [1, 1, 1 * 64 * 128, aspp_project_upsampled.shape[3]],
#         )

#         logger.debug("Running Ins_Seg_Decoder_res3_project_conv")
#         res3_project, shape = self.Ins_Seg_Decoder_res3_project_conv(device, res3, res3.shape)

#         logger.debug("Running concat for res3 and ASPP upsampled")
#         decoder_res3_concat = ttnn.concat([res3_project, aspp_project_upsampled], dim=3)
#         aspp_project_upsampled.deallocate()
#         res3_project.deallocate()

#         logger.debug("Running Ins_Seg_Decoder_res3_fuse_conv_depthwise")
#         shape = (1, 64, 128, 320)
#         decoder_res3_fuse_dw, shape = self.Ins_Seg_Decoder_res3_fuse_conv_depthwise(device, decoder_res3_concat, shape)
#         decoder_res3_concat.deallocate()

#         logger.debug("Running Ins_Seg_Decoder_res3_fuse_conv_pointwise")
#         decoder_res3_fuse_pw, shape = self.Ins_Seg_Decoder_res3_fuse_conv_pointwise(device, decoder_res3_fuse_dw, shape)
#         decoder_res3_fuse_dw.deallocate()

#         return decoder_res3_fuse_pw


# class PanopticDeeplabInstanceDecoderRes2:
#     def __init__(self, parameters, model_config) -> None:
#         # Ins_Seg_Decoder_res2_project_conv
#         self.Ins_Seg_Decoder_res2_project_conv = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Decoder_res2_project_conv,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=32,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )
#         # Ins_Seg_Decoder_res2_fuse_conv_depthwise
#         self.Ins_Seg_Decoder_res2_fuse_conv_depthwise = TTConv2D(
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             dilation=1,
#             # groups=1,
#             groups=160,
#             parameters=parameters.Ins_Seg_Decoder_res2_fuse_conv_depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=128,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#             # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             # shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#             reshard_if_not_optimal=True,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )
#         # Ins_Seg_Decoder_res2_fuse_conv_pointwise
#         self.Ins_Seg_Decoder_res2_fuse_conv_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Decoder_res2_fuse_conv_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=32,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )

#     def __call__(self, x, res2, device):
#         # Exact copy from original implementation
#         logger.debug("Running upsample after res3 fuse")
#         decoder_res3_fuse_pw = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
#         decoder_res3_fuse_pw = ttnn.to_layout(
#             decoder_res3_fuse_pw, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
#         )

#         decoder_res3_fuse_upsampled = ttnn.upsample(
#             decoder_res3_fuse_pw,
#             scale_factor=2,
#             mode="bilinear",
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             compute_kernel_config=ttnn.WormholeComputeKernelConfig(
#                 math_fidelity=ttnn.MathFidelity.LoFi,
#                 math_approx_mode=True,
#                 fp32_dest_acc_en=False,
#             ),
#         )
#         decoder_res3_fuse_pw.deallocate()
#         decoder_res3_fuse_upsampled = ttnn.to_layout(
#             decoder_res3_fuse_upsampled, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
#         )

#         decoder_res3_fuse_upsampled = ttnn.reshape(
#             decoder_res3_fuse_upsampled,
#             [1, 1, 1 * 128 * 256, decoder_res3_fuse_upsampled.shape[3]],
#         )

#         logger.debug("Running Ins_Seg_Decoder_res2_project_conv")
#         res2_project, shape = self.Ins_Seg_Decoder_res2_project_conv(device, res2, res2.shape)

#         # decoder_res3_fuse_upsampled = ttnn.typecast(decoder_res3_fuse_upsampled, dtype=ttnn.bfloat8_b)
#         logger.debug("Running concat for res2 and decoder upsampled")
#         decoder_res2_concat = ttnn.concat([res2_project, decoder_res3_fuse_upsampled], dim=3)
#         res2_project.deallocate()
#         decoder_res3_fuse_upsampled.deallocate()
#         decoder_res2_concat = ttnn.reallocate(decoder_res2_concat)
#         logger.debug("Running Ins_Seg_Decoder_res2_fuse_conv_depthwise")
#         shape = (1, 128, 256, 160)
#         decoder_res2_fuse_dw, shape = self.Ins_Seg_Decoder_res2_fuse_conv_depthwise(device, decoder_res2_concat, shape)
#         decoder_res2_concat.deallocate()
#         # ttnn.reallocate(decoder_res2_fuse_dw)
#         logger.debug("Running Ins_Seg_Decoder_res2_fuse_conv_pointwise")
#         decoder_res2_fuse_pw, shape = self.Ins_Seg_Decoder_res2_fuse_conv_pointwise(device, decoder_res2_fuse_dw, shape)
#         decoder_res2_fuse_dw.deallocate()

#         return decoder_res2_fuse_pw


# class PanopticDeeplabInstanceCenterHead:
#     def __init__(self, parameters, model_config) -> None:
#         # Ins_Seg_Center_Head_Conv_0
#         self.Ins_Seg_Center_Head_Conv_0 = TTConv2D(
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             dilation=1,
#             groups=1,
#             parameters=parameters.Ins_Seg_Center_Head_Conv_0,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=128,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )
#         # Ins_Seg_Center_Head_Conv_1
#         self.Ins_Seg_Center_Head_Conv_1 = TTConv2D(
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=1,
#             parameters=parameters.Ins_Seg_Center_Head_Conv_1,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=128,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#         )
#         # Ins_Seg_Center_predictor
#         self.Ins_Seg_Center_predictor = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Center_predictor,
#             kernel_fidelity=model_config,
#             act_block_h=64,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=False,
#             input_channels_alignment=32,
#         )

#     def __call__(self, x, device):
#         # # Exact copy from original - clone for offset processing
#         # logger.debug("Creating copy for offset head processing")
#         # offset_input = ttnn.clone(x, memory_config=x.memory_config())

#         shape = (1, 128, 256, 128)
#         logger.debug("Running Ins_Seg_Center_Head_Conv_0")
#         center_head_0, shape = self.Ins_Seg_Center_Head_Conv_0(device, x, shape)

#         logger.debug("Running Ins_Seg_Center_Head_Conv_1")
#         center_head_1, shape = self.Ins_Seg_Center_Head_Conv_1(device, center_head_0, shape)
#         # center_head_0.deallocate()

#         logger.debug("Running Ins_Seg_Center_predictor")
#         center_predictor, shape = self.Ins_Seg_Center_predictor(device, center_head_1, shape)
#         center_head_1.deallocate()
#         # x.deallocate()

#         logger.debug("Running center head upsample")
#         # center_predictor = ttnn.sharded_to_interleaved(center_predictor, ttnn.L1_MEMORY_CONFIG)
#         center_predictor = ttnn.to_layout(center_predictor, ttnn.ROW_MAJOR_LAYOUT)

#         center_predictor = ttnn.pad(center_predictor, [(0, 0), (0, 0), (0, 0), (0, 31)], 0)
#         center_predictor = ttnn.reshape(center_predictor, [1, 128, 256, 32])

#         center_output = ttnn.upsample(
#             center_predictor,
#             scale_factor=4,
#             mode="bilinear",
#             compute_kernel_config=ttnn.WormholeComputeKernelConfig(
#                 math_fidelity=ttnn.MathFidelity.HiFi2,
#                 math_approx_mode=True,
#                 fp32_dest_acc_en=False,
#             ),
#         )
#         center_output = ttnn.to_memory_config(center_output, ttnn.DRAM_MEMORY_CONFIG)
#         center_predictor.deallocate()
#         return center_output


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
    "segmentation_offset_head": HeadOptimizer(
        conv1={
            "act_block_h": 32,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
        # conv1
        self.conv1 = TTConv2D(
            kernel_size=parameters.conv_args["conv1"]["0"].kernel_size,
            # stride=1,
            stride=parameters.conv_args["conv1"]["0"].stride,
            padding=parameters.conv_args["conv1"]["0"].padding,
            dilation=parameters.conv_args["conv1"]["0"].dilation,
            groups=parameters.conv_args["conv1"]["0"].groups,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation="relu",
            # act_block_h=32,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # deallocate_activation=True,
            # reallocate_halo_output=True,
            # # enable_split_reader=True,
            # enable_act_double_buffer=True,
            # shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            # reshard_if_not_optimal=True,
            **layer_optimisations.conv1,
        )
        # Sem_Seg_Head_pointwise
        self.conv2 = TTConv2D(
            kernel_size=parameters.conv_args["conv2"]["0"].kernel_size,
            stride=parameters.conv_args["conv2"]["0"].stride,
            padding=parameters.conv_args["conv2"]["0"].padding,
            groups=parameters.conv_args["conv2"]["0"].groups,
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            activation="relu",
            # act_block_h=32,
            # # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # deallocate_activation=True,
            # reallocate_halo_output=True,
            # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            # enable_act_double_buffer=True,
            # reshard_if_not_optimal=True
            **layer_optimisations.conv2,
        )
        # Sem_Seg_predictor
        self.conv3 = TTConv2D(
            kernel_size=parameters.conv_args["conv3"]["0"].kernel_size,
            stride=parameters.conv_args["conv3"]["0"].stride,
            padding=parameters.conv_args["conv3"]["0"].padding,
            groups=parameters.conv_args["conv3"]["0"].groups,
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            # act_block_h=32,
            # # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            # deallocate_activation=True,
            # reallocate_halo_output=True,
            **layer_optimisations.conv3,
        )

        # Sem_Seg_predictor_upsample
        self.upsample = TTUpsample(
            scale_factor=(4),
            mode="bilinear",
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(
        self,
        x,
        device,
    ):
        logger.debug("Running conv1")
        # Use actual input tensor shape instead of hardcoded values
        input_shape = x.shape
        print(f"{input_shape=}")
        shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        out, shape = self.conv1(device, x, shape)

        logger.debug("Running conv2")
        out, shape = self.conv2(device, out, shape)

        logger.debug("Running conv3")
        out, shape = self.conv3(device, out, shape)

        logger.debug("Running final upsample")
        out_shape = (shape[0], shape[1], shape[2], shape[3])
        # out = self.upsample(device, out, out_shape, False, True, )
        out = self.upsample(device, out, out_shape, reshape_output=False, pad_ch_to_32=False, sent_to_dram=True)

        return out


# class PanopticDeeplabInstanceOffsetHead:
#     def __init__(self, parameters, model_config) -> None:
#         # Ins_Seg_Offset_Head_depthwise
#         self.Ins_Seg_Offset_Head_depthwise = TTConv2D(
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             dilation=1,
#             groups=128,
#             parameters=parameters.Ins_Seg_Offset_Head_depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=128,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#         )
#         # Ins_Seg_Offset_Head_pointwise
#         self.Ins_Seg_Offset_Head_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Offset_Head_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#         )
#         # Ins_Seg_Offset_predictor
#         self.Ins_Seg_Offset_predictor = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Offset_predictor,
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             kernel_fidelity=model_config,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#         )

#     def __call__(self, x, device):
#         # Offset head processing
#         shape = (1, 128, 256, 128)
#         x = ttnn.reallocate(x)
#         logger.debug("Running Ins_Seg_Offset_Head_depthwise")
#         offset_dw, shape = self.Ins_Seg_Offset_Head_depthwise(device, x, shape)

#         logger.debug("Running Ins_Seg_Offset_Head_pointwise")
#         offset_pw, shape = self.Ins_Seg_Offset_Head_pointwise(device, offset_dw, shape)
#         # x.deallocate()
#         offset_dw.deallocate()

#         offset_predictor, shape = self.Ins_Seg_Offset_predictor(device, offset_pw, shape)
#         offset_pw.deallocate()

#         logger.debug("Running instance upsample")
#         # print(f"{offset_predictor=}")
#         offset_predictor = ttnn.to_layout(offset_predictor, ttnn.ROW_MAJOR_LAYOUT)
#         # print(f"{offset_predictor=}")

#         offset_predictor = ttnn.pad(offset_predictor, [(0, 0), (0, 0), (0, 0), (0, 30)], 0)
#         offset_predictor = ttnn.reshape(offset_predictor, [1, 128, 256, 32])
#         # print(f"{offset_predictor=}")

#         offset_upsampled = ttnn.upsample(
#             offset_predictor,
#             scale_factor=4,
#             mode="bilinear",
#             # memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             compute_kernel_config=ttnn.WormholeComputeKernelConfig(
#                 math_fidelity=ttnn.MathFidelity.LoFi,
#                 math_approx_mode=True,
#                 fp32_dest_acc_en=False,
#             ),
#         )
#         print(f"{offset_upsampled=}")
#         offset_predictor.deallocate()
#         offset_upsampled = ttnn.to_layout(offset_upsampled, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
#         # offset_upsampled = ttnn.slice(offset_upsampled, [0, 0, 0, 0], [1, 512, 1024, 2])
#         print(f"{offset_upsampled=}")

#         logger.debug("Applying MulByConstant (x4)")
#         offset_output = ttnn.mul(offset_upsampled, 4)
#         offset_upsampled.deallocate()

#         return offset_output


# # Composite classes
# class PanopticDeeplabInstanceRes3Res2:
#     def __init__(self, parameters, model_config) -> None:
#         self.res3 = PanopticDeeplabInstanceDecoderRes3(parameters, model_config)
#         self.res2 = PanopticDeeplabInstanceDecoderRes2(parameters, model_config)

#     def __call__(self, x, res3, res2, device):
#         logger.debug("Running instance res3")
#         y = self.res3(x, res3, device)

#         logger.debug("Running instance res2")
#         output = self.res2(y, res2, device)

#         return output


# class PanopticDeeplabInstanceASPPRes3Res2:
#     def __init__(self, parameters, model_config) -> None:
#         self.aspp = PanopticDeeplabInstanceASPP(parameters, model_config)
#         self.res3 = PanopticDeeplabInstanceDecoderRes3(parameters, model_config)
#         self.res2 = PanopticDeeplabInstanceDecoderRes2(parameters, model_config)

#     def __call__(self, x, res3, res2, device):
#         logger.debug("Running instance ASPP")
#         y = self.aspp(x, device)

#         logger.debug("Running instance res3")
#         y = self.res3(y, res3, device)

#         logger.debug("Running instance res2")
#         output = self.res2(y, res2, device)

#         return output


# class PanopticDeeplabInstanceASPPRes3Res2Heads:
#     def __init__(self, parameters, model_config) -> None:
#         self.aspp = PanopticDeeplabInstanceASPP(parameters, model_config)
#         self.res3 = PanopticDeeplabInstanceDecoderRes3(parameters, model_config)
#         self.res2 = PanopticDeeplabInstanceDecoderRes2(parameters, model_config)
#         self.center_head = PanopticDeeplabInstanceCenterHead(parameters, model_config)
#         self.offset_head = PanopticDeeplabInstanceOffsetHead(parameters, model_config)

#     def __call__(self, x, res3, res2, device):
#         logger.debug("Running instance ASPP")
#         y = self.aspp(x, device)

#         logger.debug("Running instance res3")
#         y = self.res3(y, res3, device)

#         logger.debug("Running instance res2")
#         y = self.res2(y, res2, device)

#         center_output, offset_output = self.heads(y, device)

#         return center_output, offset_output


# class PanopticDeeplabInstanceSegmentation:
#     """
#     Modular instance segmentation - exact same logic as original monolithic implementation
#     """

#     def __init__(self, parameters, model_config) -> None:
#         self.model_config = model_config
#         self.aspp = PanopticDeeplabInstanceASPP(parameters, model_config)
#         self.res3 = PanopticDeeplabInstanceDecoderRes3(parameters, model_config)
#         self.res2 = PanopticDeeplabInstanceDecoderRes2(parameters, model_config)
#         # self.heads = PanopticDeeplabInstanceHeads(parameters, model_config)
#         self.center_head = PanopticDeeplabInstanceCenterHead(parameters, model_config)
#         self.offset_head = PanopticDeeplabInstanceOffsetHead(parameters, model_config)

#     def __call__(
#         self,
#         x,
#         res3,
#         res2,
#         device,
#         batch_size,
#         input_height_1,
#         input_width_1,
#         input_height_2,
#         input_width_2,
#         input_height_3,
#         input_width_3,
#         reshard_if_not_optimal=False,
#         height_sharding=None,
#         eltwise_binary_out_in_place=True,
#         packer_l1_acc=True if not is_grayskull() else False,
#         enable_act_double_buffer=False,
#         enable_split_reader=False,
#         enable_subblock_padding=False,
#         ops_parallel_config=None,
#         layer_module=None,
#     ):
#         logger.debug("Running instance ASPP")
#         y = self.aspp(x, device)

#         logger.debug("Running instance res3")
#         y = self.res3(y, res3, device)

#         logger.debug("Running instance res2")
#         y = self.res2(y, res2, device)
#         offset_input = ttnn.clone(y, memory_config=y.memory_config())
#         logger.debug("Running instance center head")
#         center_output = self.center_head(y, device)

#         logger.debug("Running instance offset head")
#         offset_output = self.offset_head(offset_input, device)

#         logger.debug("Offset instance output {}", offset_output.shape)
#         logger.debug("Center instance output {}", center_output.shape)

#         return center_output, offset_output

# class PanopticDeeplabInstanceOffsetHead:
#     def __init__(self, parameters, model_config) -> None:
#         # Ins_Seg_Offset_Head_depthwise
#         self.Ins_Seg_Offset_Head_depthwise = TTConv2D(
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             dilation=1,
#             groups=128,
#             parameters=parameters.Ins_Seg_Offset_Head_depthwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             act_block_h=128,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#             reallocate_halo_output=True,
#             enable_split_reader=True,
#             enable_act_double_buffer=True,
#             enable_weights_double_buffer=True,
#         )
#         # Ins_Seg_Offset_Head_pointwise
#         self.Ins_Seg_Offset_Head_pointwise = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Offset_Head_pointwise,
#             kernel_fidelity=model_config,
#             activation="relu",
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#         )
#         # Ins_Seg_Offset_predictor
#         self.Ins_Seg_Offset_predictor = TTConv2D(
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1,
#             parameters=parameters.Ins_Seg_Offset_predictor,
#             memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             kernel_fidelity=model_config,
#             shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#             deallocate_activation=True,
#         )

#     def __call__(self, x, device):
#         # Offset head processing
#         shape = (1, 128, 256, 128)
#         x = ttnn.reallocate(x)
#         logger.debug("Running Ins_Seg_Offset_Head_depthwise")
#         offset_dw, shape = self.Ins_Seg_Offset_Head_depthwise(device, x, shape)

#         logger.debug("Running Ins_Seg_Offset_Head_pointwise")
#         offset_pw, shape = self.Ins_Seg_Offset_Head_pointwise(device, offset_dw, shape)
#         # x.deallocate()
#         offset_dw.deallocate()

#         offset_predictor, shape = self.Ins_Seg_Offset_predictor(device, offset_pw, shape)
#         offset_pw.deallocate()

#         logger.debug("Running instance upsample")
#         # print(f"{offset_predictor=}")
#         offset_predictor = ttnn.to_layout(offset_predictor, ttnn.ROW_MAJOR_LAYOUT)
#         # print(f"{offset_predictor=}")

#         offset_predictor = ttnn.pad(offset_predictor, [(0, 0), (0, 0), (0, 0), (0, 30)], 0)
#         offset_predictor = ttnn.reshape(offset_predictor, [1, 128, 256, 32])
#         # print(f"{offset_predictor=}")

#         offset_upsampled = ttnn.upsample(
#             offset_predictor,
#             scale_factor=4,
#             mode="bilinear",
#             # memory_config=ttnn.DRAM_MEMORY_CONFIG,
#             compute_kernel_config=ttnn.WormholeComputeKernelConfig(
#                 math_fidelity=ttnn.MathFidelity.LoFi,
#                 math_approx_mode=True,
#                 fp32_dest_acc_en=False,
#             ),
#         )
#         print(f"{offset_upsampled=}")
#         offset_predictor.deallocate()
#         offset_upsampled = ttnn.to_layout(offset_upsampled, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
#         # offset_upsampled = ttnn.slice(offset_upsampled, [0, 0, 0, 0], [1, 512, 1024, 2])
#         print(f"{offset_upsampled=}")

#         logger.debug("Applying MulByConstant (x4)")
#         offset_output = ttnn.mul(offset_upsampled, 4)
#         offset_upsampled.deallocate()

#         return offset_output
