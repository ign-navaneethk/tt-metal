# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.panoptic_deeplab.tt.common import TTConv2D, TTUpsample
from models.experimental.panoptic_deeplab.tt.aspp import PanopticDeeplabASPP


class PanopticDeeplabDecoderRes3:
    def __init__(self, parameters, model_config) -> None:
        # Sem_Seg_ASPP_project_upsample
        self.Sem_Seg_ASPP_project_upsample = TTUpsample(
            scale_factor=(2),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.HiFi2,
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

    def __call__(
        self,
        x,
        res3,
        device,
    ):
        # Decoder: upsample and fuse with res3
        logger.debug("Running upsample after ASPP project")
        aspp_project_upsampled = self.Sem_Seg_ASPP_project_upsample(device, x, [1, 32, 64, 256])

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
        output, shape = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise(device, decoder_res3_fuse_dw, shape)

        return output


class PanopticDeeplabDecoderRes2:
    def __init__(self, parameters, model_config) -> None:
        # Sem_Seg_Decoder_res3_fuse_conv_pointwise_upsample
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_upsample = TTUpsample(
            scale_factor=(2),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.HiFi2,
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

    def __call__(
        self,
        x,
        res2,
        device,
    ):
        logger.debug("Running upsample after res3 fuse")
        decoder_res3_fuse_upsampled = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_upsample(
            device, x, [1, 64, 128, 256]
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
        output, shape = self.Sem_Seg_Decoder_res2_fuse_conv_pointwise(device, decoder_res2_fuse_dw, shape)

        return output


class PanopticDeeplabHead:
    def __init__(self, parameters, model_config) -> None:
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
        self.Sem_Seg_Head_predictor = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.Sem_Seg_Head_predictor,
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
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(
        self,
        x,
        device,
    ):
        logger.debug("Running Sem_Seg_Head_depthwise")
        shape = (1, 128, 256, 256)
        head_dw, shape = self.Sem_Seg_Head_depthwise(device, x, shape)

        logger.debug("Running Sem_Seg_Head_pointwise")
        head_pw, shape = self.Sem_Seg_Head_pointwise(device, head_dw, shape)

        logger.debug("Running Sem_Seg_predictor")
        predictor, shape = self.Sem_Seg_Head_predictor(device, head_pw, shape)

        logger.debug("Running final upsample")
        output = self.Sem_Seg_predictor_upsample(device, predictor, [1, 128, 256, 19], False, True)

        return output


class PanopticDeeplabRes3Res2:
    def __init__(self, parameters, model_config) -> None:
        self.res3 = PanopticDeeplabDecoderRes3(parameters, model_config)
        self.res2 = PanopticDeeplabDecoderRes2(parameters, model_config)

    def __call__(
        self,
        x,
        res3,
        res2,
        device,
    ):
        logger.debug("Running res3")
        y = self.res3(x, res3, device)

        logger.debug("Running res2")
        output = self.res2(y, res2, device)

        return output


class PanopticDeeplabASPPRes3Res2:
    def __init__(self, parameters, model_config) -> None:
        self.aspp = PanopticDeeplabASPP(parameters, model_config)
        self.res3 = PanopticDeeplabDecoderRes3(parameters, model_config)
        self.res2 = PanopticDeeplabDecoderRes2(parameters, model_config)

    def __call__(
        self,
        x,
        res3,
        res2,
        device,
    ):
        logger.debug("Running ASPP")
        y = self.aspp(x, device)

        logger.debug("Running res3")
        y = self.res3(y, res3, device)

        logger.debug("Running res2")
        output = self.res2(y, res2, device)

        return output


class TTPanopticDeeplabSemanticSegmentationModel:
    def __init__(self, parameters, model_config) -> None:
        self.aspp = PanopticDeeplabASPP(parameters, model_config)
        self.res3 = PanopticDeeplabDecoderRes3(parameters, model_config)
        self.res2 = PanopticDeeplabDecoderRes2(parameters, model_config)
        self.head = PanopticDeeplabHead(parameters, model_config)

    def __call__(
        self,
        x,
        res3,
        res2,
        device,
    ):
        logger.debug("Running ASPP")
        y = self.aspp(x, device)

        logger.debug("Running res3")
        y = self.res3(y, res3, device)

        logger.debug("Running res2")
        y = self.res2(y, res2, device)

        output = self.head(y, device)

        return output
