# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.utility_functions import is_grayskull
from models.experimental.instance.tt.common import TTConv2D


class PanopticDeeplabInstanceSegmentation:
    def __init__(self, parameters, model_config, test) -> None:
        self.model_config = model_config

        # Ins_Seg_ASPP_0
        self.Ins_Seg_ASPP_0 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_ASPP_0_Conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            is_reshape=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Ins_Seg_ASPP_1_Depthwise
        self.Ins_Seg_ASPP_1_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=6,
            dilation=6,
            groups=2048,
            parameters=parameters.Ins_Seg_ASPP_1_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_1_pointwise
        self.Ins_Seg_ASPP_1_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_ASPP_1_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_2_Depthwise
        self.Ins_Seg_ASPP_2_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=12,
            dilation=12,
            groups=2048,
            parameters=parameters.Ins_Seg_ASPP_2_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_2_pointwise
        self.Ins_Seg_ASPP_2_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_ASPP_2_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_3_Depthwise
        self.Ins_Seg_ASPP_3_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=18,
            dilation=18,
            groups=2048,
            parameters=parameters.Ins_Seg_ASPP_3_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_3_pointwise
        self.Ins_Seg_ASPP_3_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_ASPP_3_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_4_Conv_1
        self.Ins_Seg_ASPP_4_Conv_1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_ASPP_4_Conv_1,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            # memory_config=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_ASPP_project
        self.Ins_Seg_ASPP_project = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_ASPP_project,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Decoder_res3_project_conv
        self.Ins_Seg_Decoder_res3_project_conv = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Decoder_res3_project_conv[0]
            if test == "full"
            else parameters.Ins_Seg_Decoder_res3_project_conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Decoder_res3_fuse_conv_depthwise
        self.Ins_Seg_Decoder_res3_fuse_conv_depthwise = TTConv2D(
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            groups=320,
            parameters=parameters.Ins_Seg_Decoder_res3_fuse_conv_depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Decoder_res3_fuse_conv_pointwise
        self.Ins_Seg_Decoder_res3_fuse_conv_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Decoder_res3_fuse_conv_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Decoder_res2_project_conv
        self.Ins_Seg_Decoder_res2_project_conv = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Decoder_res2_project_conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Decoder_res2_fuse_conv_depthwise
        self.Ins_Seg_Decoder_res2_fuse_conv_depthwise = TTConv2D(
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            groups=160,
            parameters=parameters.Ins_Seg_Decoder_res2_fuse_conv_depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Decoder_res2_fuse_conv_pointwise
        self.Ins_Seg_Decoder_res2_fuse_conv_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Decoder_res2_fuse_conv_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=False,
            reallocate_halo_output=False,
        )
        # Ins_Seg_Center_Head_Conv_0
        self.Ins_Seg_Center_Head_Conv_0 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            parameters=parameters.Ins_Seg_Center_Head_Conv_0,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Center_Head_Conv_1
        self.Ins_Seg_Center_Head_Conv_1 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            parameters=parameters.Ins_Seg_Center_Head_Conv_1,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Center_predictor
        self.Ins_Seg_Center_predictor = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Center_predictor,
            kernel_fidelity=model_config,
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=False,
            reallocate_halo_output=False,
            input_channels_alignment=32,
        )

        # Ins_Seg_Offset_Head_depthwise
        self.Ins_Seg_Offset_Head_depthwise = TTConv2D(
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            groups=128,
            parameters=parameters.Ins_Seg_Offset_Head_depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Offset_Head_pointwise
        self.Ins_Seg_Offset_Head_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Offset_Head_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Ins_Seg_Offset_predictor
        self.Ins_Seg_Offset_predictor = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Ins_Seg_Offset_predictor,
            kernel_fidelity=model_config,
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=False,
            reallocate_halo_output=False,
        )

    def __call__(
        self,
        x,
        res3,
        res2,
        device,
        batch_size,
        input_height_1,
        input_width_1,
        input_height_2,
        input_width_2,
        input_height_3,
        input_width_3,
        reshard_if_not_optimal=False,
        height_sharding=None,
        eltwise_binary_out_in_place=True,
        packer_l1_acc=True if not is_grayskull() else False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        ops_parallel_config=None,
        layer_module=None,
    ):
        # ASPP branch
        logger.debug("Running Ins_Seg_ASPP_0_Conv")
        aspp0, shape = self.Ins_Seg_ASPP_0(device, x, x.shape)
        # return aspp0

        logger.debug("Running Ins_Seg_ASPP_1_Depthwise")
        aspp1_dw, shape = self.Ins_Seg_ASPP_1_Depthwise(device, x, x.shape)

        logger.debug("Running Ins_Seg_ASPP_1_pointwise")
        aspp1, shape = self.Ins_Seg_ASPP_1_pointwise(device, aspp1_dw, shape)

        logger.debug("Running Ins_Seg_ASPP_2_Depthwise")
        aspp2_dw, shape = self.Ins_Seg_ASPP_2_Depthwise(device, x, x.shape)

        logger.debug("Running Ins_Seg_ASPP_2_pointwise")
        aspp2, shape = self.Ins_Seg_ASPP_2_pointwise(device, aspp2_dw, shape)

        logger.debug("Running Ins_Seg_ASPP_3_Depthwise")
        aspp3_dw, shape = self.Ins_Seg_ASPP_3_Depthwise(device, x, x.shape)

        logger.debug("Running Ins_Seg_ASPP_3_pointwise")
        aspp3, shape = self.Ins_Seg_ASPP_3_pointwise(device, aspp3_dw, shape)

        logger.debug("Running Ins_Seg_ASPP_4_avg_pool")
        x = ttnn.reshape(x, [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])

        aspp4 = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=1,
            input_h=32,
            input_w=64,
            channels=2048,
            kernel_size=(32, 64),
            stride=(1, 1),
            padding=(0, 0),
        )

        logger.debug("Running Ins_Seg_ASPP_4_Conv_1")
        shape = (1, 1, 1, 2048)
        aspp4_conv, shape = self.Ins_Seg_ASPP_4_Conv_1(device, aspp4, shape)

        logger.debug("Running Ins_Seg_ASPP_4_upsample")
        aspp4_conv = ttnn.sharded_to_interleaved(aspp4_conv, ttnn.DRAM_MEMORY_CONFIG)
        aspp4_conv = ttnn.to_layout(aspp4_conv, ttnn.ROW_MAJOR_LAYOUT)

        aspp4_conv_upsample = ttnn.upsample(
            aspp4_conv,
            scale_factor=(32, 64),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            ),
        )

        aspp4_conv_upsample = ttnn.from_device(aspp4_conv_upsample)
        aspp4_conv_upsample = ttnn.to_dtype(aspp4_conv_upsample, ttnn.bfloat16)
        aspp4_conv_upsample = ttnn.to_device(aspp4_conv_upsample, device)
        aspp4_conv_upsample = ttnn.reshape(
            aspp4_conv_upsample,
            [
                1,
                1,
                aspp4_conv_upsample.shape[0] * aspp4_conv_upsample.shape[1] * aspp4_conv_upsample.shape[2],
                aspp4_conv_upsample.shape[3],
            ],
        )

        # ASPP project
        logger.debug("Running Ins_Seg_ASPP_concat")

        aspp_concat = ttnn.concat(
            [aspp0, aspp1, aspp2, aspp3, aspp4_conv_upsample],
            dim=3,
        )

        # return aspp_concat

        logger.debug("Running Ins_Seg_ASPP_project")
        shape = (1, 32, 64, 1280)
        aspp_project, shape = self.Ins_Seg_ASPP_project(device, aspp_concat, shape)  # change shape

        logger.debug("Running upsample after ASPP project")
        aspp_project = ttnn.sharded_to_interleaved(aspp_project, ttnn.DRAM_MEMORY_CONFIG)
        aspp_project = ttnn.to_layout(aspp_project, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print(aspp_project.shape[0], aspp_project.shape[1], aspp_project.shape[2], aspp_project.shape[-1])
        aspp_project = ttnn.reshape(aspp_project, [1, 32, 64, 256])

        aspp_project_upsampled = ttnn.upsample(
            aspp_project,
            scale_factor=2,
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            ),
        )

        aspp_project_upsampled = ttnn.from_device(aspp_project_upsampled)
        aspp_project_upsampled = ttnn.to_dtype(aspp_project_upsampled, ttnn.bfloat16)
        aspp_project_upsampled = ttnn.to_device(aspp_project_upsampled, device)
        aspp_project_upsampled = ttnn.reshape(
            aspp_project_upsampled,
            [
                1,
                1,
                aspp_project_upsampled.shape[0] * aspp_project_upsampled.shape[1] * aspp_project_upsampled.shape[2],
                aspp_project_upsampled.shape[3],
            ],
        )

        logger.debug("Running Ins_Seg_Decoder_res3_project_conv")
        res3_project, shape = self.Ins_Seg_Decoder_res3_project_conv(device, res3, res3.shape)

        logger.debug("Running concat for res3 and ASPP upsampled")
        decoder_res3_concat = ttnn.concat([res3_project, aspp_project_upsampled], dim=3)

        logger.debug("Running Ins_Seg_Decoder_res3_fuse_conv_depthwise")
        shape = (1, 64, 128, 320)
        decoder_res3_fuse_dw, shape = self.Ins_Seg_Decoder_res3_fuse_conv_depthwise(
            device, decoder_res3_concat, shape
        )  # change shape

        logger.debug("Running Ins_Seg_Decoder_res3_fuse_conv_pointwise")
        decoder_res3_fuse_pw, shape = self.Ins_Seg_Decoder_res3_fuse_conv_pointwise(device, decoder_res3_fuse_dw, shape)

        logger.debug("Running upsample after res3 fuse")
        decoder_res3_fuse_pw = ttnn.sharded_to_interleaved(decoder_res3_fuse_pw, ttnn.DRAM_MEMORY_CONFIG)
        decoder_res3_fuse_pw = ttnn.to_layout(
            decoder_res3_fuse_pw, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        decoder_res3_fuse_pw = ttnn.reshape(decoder_res3_fuse_pw, [1, 64, 128, 128])

        decoder_res3_fuse_upsampled = ttnn.upsample(
            decoder_res3_fuse_pw,
            scale_factor=2,
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            ),
        )

        decoder_res3_fuse_upsampled = ttnn.from_device(decoder_res3_fuse_upsampled)
        decoder_res3_fuse_upsampled = ttnn.to_dtype(decoder_res3_fuse_upsampled, ttnn.bfloat16)
        decoder_res3_fuse_upsampled = ttnn.to_device(decoder_res3_fuse_upsampled, device)
        decoder_res3_fuse_upsampled = ttnn.reshape(
            decoder_res3_fuse_upsampled,
            [
                1,
                1,
                decoder_res3_fuse_upsampled.shape[0]
                * decoder_res3_fuse_upsampled.shape[1]
                * decoder_res3_fuse_upsampled.shape[2],
                decoder_res3_fuse_upsampled.shape[3],
            ],
        )
        logger.debug("Decoder res3 fuse upsampled shape1: {}".format(decoder_res3_fuse_upsampled.shape))

        logger.debug("Running Ins_Seg_Decoder_res2_project_conv")
        res2_project, shape = self.Ins_Seg_Decoder_res2_project_conv(device, res2, res2.shape)

        logger.debug("Running concat for res2 and decoder upsampled")
        decoder_res2_concat = ttnn.concat([res2_project, decoder_res3_fuse_upsampled], dim=3)

        logger.debug("Running Ins_Seg_Decoder_res2_fuse_conv_depthwise")
        shape = (1, 128, 256, 160)
        decoder_res2_fuse_dw, shape = self.Ins_Seg_Decoder_res2_fuse_conv_depthwise(device, decoder_res2_concat, shape)

        logger.debug("Running Ins_Seg_Decoder_res2_fuse_conv_pointwise")
        decoder_res2_fuse_pw, shape = self.Ins_Seg_Decoder_res2_fuse_conv_pointwise(device, decoder_res2_fuse_dw, shape)

        logger.debug("Creating copy for offset head processing")
        offset_input = ttnn.clone(decoder_res2_fuse_pw, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shape = (1, 128, 256, 128)
        logger.debug("Running Ins_Seg_Center_Head_Conv_0")
        center_head_0, shape = self.Ins_Seg_Center_Head_Conv_0(device, decoder_res2_fuse_pw, shape)
        logger.debug("Running Ins_Seg_Center_Head_Conv_1")
        center_head_1, shape = self.Ins_Seg_Center_Head_Conv_1(device, center_head_0, shape)
        # # center_head_0.deallocate()
        logger.debug("Running Ins_Seg_Center_predictor")
        center_predictor, shape = self.Ins_Seg_Center_predictor(device, center_head_1, shape)

        # # center_head_1.deallocate()

        logger.debug("Center predictor shape: {}".format(center_predictor.shape))

        center_predictor = ttnn.sharded_to_interleaved(center_predictor, ttnn.DRAM_MEMORY_CONFIG)
        center_predictor = ttnn.to_layout(
            center_predictor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        logger.debug("Running Center predictor {}", center_predictor.shape)
        logger.debug("Running center head upsample")

        # print(center_predictor.shape[0],center_predictor.shape[1],center_predictor.shape[2], center_predictor.shape[-1])

        center_predictor = ttnn.reshape(center_predictor, [1, 128, 256, 1])
        center_predictor = ttnn.pad(center_predictor, [(0, 0), (0, 0), (0, 0), (0, 31)], 0)
        center_output = ttnn.upsample(
            center_predictor,
            scale_factor=4,
            # mode="nearest",
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                # packer_l1_acc=False,
            ),
        )

        center_output = ttnn.slice(center_output, [0, 0, 0, 0], [1, 512, 1024, 1])
        logger.debug("Center output shape after upsample: {}".format(center_output.shape))

        shape = (1, 128, 256, 128)
        logger.debug("Running Ins_Seg_Offset_Head_depthwise")
        offset_dw, shape = self.Ins_Seg_Offset_Head_depthwise(device, offset_input, shape)
        logger.debug("Running Ins_Seg_Offset_Head_pointwise")
        offset_pw, shape = self.Ins_Seg_Offset_Head_pointwise(device, offset_dw, shape)

        # offset_input.deallocate()

        offset_predictor, shape = self.Ins_Seg_Offset_predictor(device, offset_pw, shape)

        shape = (1, 128, 256, 32)
        logger.debug("Running instance upsample")
        offset_predictor = ttnn.sharded_to_interleaved(offset_predictor, ttnn.DRAM_MEMORY_CONFIG)
        offset_predictor = ttnn.to_layout(
            offset_predictor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        print(
            offset_predictor.shape[0], offset_predictor.shape[1], offset_predictor.shape[2], offset_predictor.shape[-1]
        )
        offset_predictor = ttnn.reshape(offset_predictor, [1, 128, 256, 2])
        offset_predictor = ttnn.pad(offset_predictor, [(0, 0), (0, 0), (0, 0), (0, 30)], 0)
        offset_upsampled = ttnn.upsample(
            offset_predictor,
            scale_factor=4,
            # mode="nearest",
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            ),
        )

        offset_upsampled = ttnn.slice(offset_upsampled, [0, 0, 0, 0], [1, 512, 1024, 2])
        logger.debug("Offset shape after upsample: {}".format(offset_upsampled.shape))

        logger.debug("Applying MulByConstant (x4)")
        offset_output = ttnn.mul(offset_upsampled, 4)

        logger.debug("Offset instance output {}", offset_output.shape)

        return center_output, offset_output
        # return center_output
