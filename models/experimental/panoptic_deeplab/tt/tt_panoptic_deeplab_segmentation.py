# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.panoptic_deeplab.tt.common import TTConv2D, TTUpsample


class PanopticDeeplabSemanticsSegmentation:
    def __init__(self, parameters, model_config) -> None:
        self.model_config = model_config

        # Sem_Seg_ASPP_0
        self.Sem_Seg_ASPP_0 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_ASPP_0_Conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            is_reshape=False,
        )
        # Sem_Seg_ASPP_1_Depthwise
        self.Sem_Seg_ASPP_1_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=6,
            dilation=6,
            groups=2048,
            parameters=parameters.Sem_Seg_ASPP_1_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_ASPP_1_pointwise
        self.Sem_Seg_ASPP_1_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_ASPP_1_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_ASPP_2_Depthwise
        self.Sem_Seg_ASPP_2_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=12,
            dilation=12,
            groups=2048,
            parameters=parameters.Sem_Seg_ASPP_2_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_ASPP_2_pointwise
        self.Sem_Seg_ASPP_2_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_ASPP_2_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_ASPP_3_Depthwise
        self.Sem_Seg_ASPP_3_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=18,
            dilation=18,
            groups=2048,
            parameters=parameters.Sem_Seg_ASPP_3_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_ASPP_3_pointwise
        self.Sem_Seg_ASPP_3_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_ASPP_3_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_ASPP_4_Conv_1
        self.Sem_Seg_ASPP_4_Conv_1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_ASPP_4_Conv_1,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            memory_config=None,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )

        # Sem_Seg_ASPP4_upsample
        self.Sem_Seg_ASPP4_upsample = TTUpsample(
            scale_factor=(32, 64),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # Sem_Seg_ASPP_project
        self.Sem_Seg_ASPP_project = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_ASPP_project,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )

        # Sem_Seg_ASPP_project_upsample
        self.Sem_Seg_ASPP_project_upsample = TTUpsample(
            scale_factor=(2),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # Sem_Seg_Decoder_res3_project_conv
        self.Sem_Seg_Decoder_res3_project_conv = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_Decoder_res3_project_conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_Decoder_res3_fuse_conv_depthwise
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise = TTConv2D(
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            groups=320,
            parameters=parameters.Sem_Seg_Decoder_res3_fuse_conv_depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_Decoder_res3_fuse_conv_pointwise
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_Decoder_res3_fuse_conv_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )

        # Sem_Seg_Decoder_res3_fuse_conv_pointwise_upsample
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_upsample = TTUpsample(
            scale_factor=(2),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # Sem_Seg_Decoder_res2_project_conv
        self.Sem_Seg_Decoder_res2_project_conv = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_Decoder_res2_project_conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_Decoder_res2_fuse_conv_depthwise
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise = TTConv2D(
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            groups=288,
            parameters=parameters.Sem_Seg_Decoder_res2_fuse_conv_depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_Decoder_res2_fuse_conv_pointwise
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_Decoder_res2_fuse_conv_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_Head_depthwise
        self.Sem_Seg_Head_depthwise = TTConv2D(
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            groups=256,
            parameters=parameters.Sem_Seg_Head_depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_Head_pointwise
        self.Sem_Seg_Head_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_Head_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        # Sem_Seg_predictor
        self.Sem_Seg_predictor = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_predictor,
            kernel_fidelity=model_config,
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )

        # Sem_Seg_predictor_upsample
        self.Sem_Seg_predictor_upsample = TTUpsample(
            scale_factor=(4),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(
        self,
        x,
        res3,
        res2,
        device,
    ):
        # ASPP branch
        logger.debug("Running Sem_Seg_ASPP_0_Conv")
        aspp0, shape = self.Sem_Seg_ASPP_0(device, x, x.shape)

        logger.debug("Running Sem_Seg_ASPP_1_Depthwise")
        aspp1_dw, shape = self.Sem_Seg_ASPP_1_Depthwise(device, x, x.shape)

        logger.debug("Running Sem_Seg_ASPP_1_pointwise")
        aspp1, shape = self.Sem_Seg_ASPP_1_pointwise(device, aspp1_dw, shape)

        logger.debug("Running Sem_Seg_ASPP_2_Depthwise")
        aspp2_dw, shape = self.Sem_Seg_ASPP_2_Depthwise(device, x, x.shape)

        logger.debug("Running Sem_Seg_ASPP_2_pointwise")
        aspp2, shape = self.Sem_Seg_ASPP_2_pointwise(device, aspp2_dw, shape)

        logger.debug("Running Sem_Seg_ASPP_3_Depthwise")
        aspp3_dw, shape = self.Sem_Seg_ASPP_3_Depthwise(device, x, x.shape)

        logger.debug("Running Sem_Seg_ASPP_3_pointwise")
        aspp3, shape = self.Sem_Seg_ASPP_3_pointwise(device, aspp3_dw, shape)

        logger.debug("Running Sem_Seg_ASPP_4_avg_pool")
        x = ttnn.reshape(x, [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])

        aspp4 = ttnn.avg_pool2d(  # change hardcoding
            input_tensor=x,
            batch_size=1,
            input_h=32,
            input_w=64,
            channels=2048,
            kernel_size=(32, 64),
            stride=(1, 1),
            padding=(0, 0),
        )

        logger.debug("Running Sem_Seg_ASPP_4_Conv_1")
        shape = (1, 1, 1, 2048)
        aspp4_conv, shape = self.Sem_Seg_ASPP_4_Conv_1(device, aspp4, shape)  # change shape

        logger.debug("Running Sem_Seg_ASPP_4_upsample")
        aspp4_conv_upsample = self.Sem_Seg_ASPP4_upsample(device, aspp4_conv, [1, 1, 1, 256])

        # ASPP project
        logger.debug("Running Sem_Seg_ASPP_concat")

        aspp_concat = ttnn.concat(
            [aspp0, aspp1, aspp2, aspp3, aspp4_conv_upsample],
            dim=3,
        )

        logger.debug("Running Sem_Seg_ASPP_project")
        shape = (1, 32, 64, 1280)
        aspp_project, shape = self.Sem_Seg_ASPP_project(device, aspp_concat, shape)  # change shape

        # Decoder: upsample and fuse with res3
        logger.debug("Running upsample after ASPP project")
        aspp_project_upsampled = self.Sem_Seg_ASPP_project_upsample(device, aspp_project, [1, 32, 64, 256])

        logger.debug("Running Sem_Seg_Decoder_res3_project_conv")
        res3_project, shape = self.Sem_Seg_Decoder_res3_project_conv(device, res3, res3.shape)

        logger.debug("Running concat for res3 and ASPP upsampled")
        decoder_res3_concat = ttnn.concat([res3_project, aspp_project_upsampled], dim=3)

        logger.debug("Running Sem_Seg_Decoder_res3_fuse_conv_depthwise")
        shape = (1, 64, 128, 320)
        decoder_res3_fuse_dw, shape = self.Sem_Seg_Decoder_res3_fuse_conv_depthwise(
            device, decoder_res3_concat, shape
        )  # change shape

        logger.debug("Running Sem_Seg_Decoder_res3_fuse_conv_pointwise")
        decoder_res3_fuse_pw, shape = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise(device, decoder_res3_fuse_dw, shape)

        logger.debug("Running upsample after res3 fuse")
        decoder_res3_fuse_upsampled = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_upsample(
            device, decoder_res3_fuse_pw, [1, 64, 128, 256]
        )

        logger.debug("Running Sem_Seg_Decoder_res2_project_conv")
        res2_project, shape = self.Sem_Seg_Decoder_res2_project_conv(device, res2, res2.shape)

        logger.debug("Running concat for res2 and decoder upsampled")
        decoder_res2_concat = ttnn.concat([res2_project, decoder_res3_fuse_upsampled], dim=3)

        logger.debug("Running Sem_Seg_Decoder_res2_fuse_conv_depthwise")
        shape = (1, 128, 256, 288)
        decoder_res2_fuse_dw, shape = self.Sem_Seg_Decoder_res2_fuse_conv_depthwise(
            device, decoder_res2_concat, shape
        )  # change shape

        logger.debug("Running Sem_Seg_Decoder_res2_fuse_conv_pointwise")
        decoder_res2_fuse_pw, shape = self.Sem_Seg_Decoder_res2_fuse_conv_pointwise(device, decoder_res2_fuse_dw, shape)

        logger.debug("Running Sem_Seg_Head_depthwise")
        head_dw, shape = self.Sem_Seg_Head_depthwise(device, decoder_res2_fuse_pw, shape)
        logger.debug("Running Sem_Seg_Head_pointwise")
        head_pw, shape = self.Sem_Seg_Head_pointwise(device, head_dw, shape)

        logger.debug("Running Sem_Seg_predictor")
        predictor, shape = self.Sem_Seg_predictor(device, head_pw, shape)

        logger.debug("Running final upsample")
        output = self.Sem_Seg_predictor_upsample(device, predictor, [1, 128, 256, 19], False, True)

        logger.debug("Running final upsample {}", output.shape)
        logger.debug("finished with ttnn imp")

        return output
