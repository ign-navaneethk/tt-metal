# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.utility_functions import is_grayskull


class PanopticDeeplabSemanticsSegmentation:
    def __init__(self, parameters, model_config) -> None:
        self.model_config = model_config
        # init is just to pre-process pytorch weights and bias tensors

        self.Sem_Seg_ASPP_0_Conv_weight_tensor = parameters.conv1.weight
        self.Sem_Seg_ASPP_0_Conv_bias_tensor = parameters.conv1.bias
        self.Sem_Seg_ASPP_0_Conv_input_channels = self.Sem_Seg_ASPP_0_Conv_weight_tensor.shape[1]
        self.Sem_Seg_ASPP_0_Conv_output_channels = self.Sem_Seg_ASPP_0_Conv_weight_tensor.shape[0]
        assert self.Sem_Seg_ASPP_0_Conv_weight_tensor.shape[2] == 1

        # Sem_Seg_ASPP_1_Depthwise
        self.Sem_Seg_ASPP_1_Depthwise_weight_tensor = parameters.aspp.convs[1].conv.weight
        self.Sem_Seg_ASPP_1_Depthwise_bias_tensor = parameters.aspp.convs[1].conv.bias
        self.Sem_Seg_ASPP_1_Depthwise_input_channels = self.Sem_Seg_ASPP_1_Depthwise_weight_tensor.shape[1]
        self.Sem_Seg_ASPP_1_Depthwise_output_channels = self.Sem_Seg_ASPP_1_Depthwise_weight_tensor.shape[0]
        # Sem_Seg_ASPP_1_pointwise
        self.Sem_Seg_ASPP_1_pointwise_weight_tensor = parameters.aspp.convs[1].bn.weight
        self.Sem_Seg_ASPP_1_pointwise_bias_tensor = parameters.aspp.convs[1].bn.bias
        self.Sem_Seg_ASPP_1_pointwise_input_channels = self.Sem_Seg_ASPP_1_pointwise_weight_tensor.shape[0]
        self.Sem_Seg_ASPP_1_pointwise_output_channels = self.Sem_Seg_ASPP_1_pointwise_weight_tensor.shape[0]

        # Sem_Seg_ASPP_2_Depthwise
        self.Sem_Seg_ASPP_2_Depthwise_weight_tensor = parameters.aspp.convs[2].conv.weight
        self.Sem_Seg_ASPP_2_Depthwise_bias_tensor = parameters.aspp.convs[2].conv.bias
        self.Sem_Seg_ASPP_2_Depthwise_input_channels = self.Sem_Seg_ASPP_2_Depthwise_weight_tensor.shape[1]
        self.Sem_Seg_ASPP_2_Depthwise_output_channels = self.Sem_Seg_ASPP_2_Depthwise_weight_tensor.shape[0]
        # Sem_Seg_ASPP_2_pointwise
        self.Sem_Seg_ASPP_2_pointwise_weight_tensor = parameters.aspp.convs[2].bn.weight
        self.Sem_Seg_ASPP_2_pointwise_bias_tensor = parameters.aspp.convs[2].bn.bias
        self.Sem_Seg_ASPP_2_pointwise_input_channels = self.Sem_Seg_ASPP_2_pointwise_weight_tensor.shape[0]
        self.Sem_Seg_ASPP_2_pointwise_output_channels = self.Sem_Seg_ASPP_2_pointwise_weight_tensor.shape[0]

        # Sem_Seg_ASPP_3_Depthwise
        self.Sem_Seg_ASPP_3_Depthwise_weight_tensor = parameters.aspp.convs[3].conv.weight
        self.Sem_Seg_ASPP_3_Depthwise_bias_tensor = parameters.aspp.convs[3].conv.bias
        self.Sem_Seg_ASPP_3_Depthwise_input_channels = self.Sem_Seg_ASPP_3_Depthwise_weight_tensor.shape[1]
        self.Sem_Seg_ASPP_3_Depthwise_output_channels = self.Sem_Seg_ASPP_3_Depthwise_weight_tensor.shape[0]
        # Sem_Seg_ASPP_3_pointwise
        self.Sem_Seg_ASPP_3_pointwise_weight_tensor = parameters.aspp.convs[3].bn.weight
        self.Sem_Seg_ASPP_3_pointwise_bias_tensor = parameters.aspp.convs[3].bn.bias
        self.Sem_Seg_ASPP_3_pointwise_input_channels = self.Sem_Seg_ASPP_3_pointwise_weight_tensor.shape[0]
        self.Sem_Seg_ASPP_3_pointwise_output_channels = self.Sem_Seg_ASPP_3_pointwise_weight_tensor.shape[0]

        # Sem_Seg_ASPP_4_avg_pool
        self.Sem_Seg_ASPP_4_avg_pool_weight_tensor = parameters.aspp.global_avg_pool[1].conv.weight
        self.Sem_Seg_ASPP_4_avg_pool_bias_tensor = parameters.aspp.global_avg_pool[1].conv.bias
        self.Sem_Seg_ASPP_4_avg_pool_input_channels = self.Sem_Seg_ASPP_4_avg_pool_weight_tensor.shape[1]
        self.Sem_Seg_ASPP_4_avg_pool_output_channels = self.Sem_Seg_ASPP_4_avg_pool_weight_tensor.shape[0]
        # Sem_Seg_ASPP_4_Conv_1
        self.Sem_Seg_ASPP_4_Conv_1_weight_tensor = parameters.aspp.global_avg_pool[1].bn.weight
        self.Sem_Seg_ASPP_4_Conv_1_bias_tensor = parameters.aspp.global_avg_pool[1].bn.bias
        self.Sem_Seg_ASPP_4_Conv_1_input_channels = self.Sem_Seg_ASPP_4_Conv_1_weight_tensor.shape[0]
        self.Sem_Seg_ASPP_4_Conv_1_output_channels = self.Sem_Seg_ASPP_4_Conv_1_weight_tensor.shape[0]

        # Sem_Seg_ASPP_project
        self.Sem_Seg_ASPP_project_weight_tensor = parameters.aspp.conv_out.conv.weight
        self.Sem_Seg_ASPP_project_bias_tensor = parameters.aspp.conv_out.conv.bias
        self.Sem_Seg_ASPP_project_input_channels = self.Sem_Seg_ASPP_project_weight_tensor.shape[1]
        self.Sem_Seg_ASPP_project_output_channels = self.Sem_Seg_ASPP_project_weight_tensor.shape[0]

        # Sem_Seg_Decoder_res3_project_conv
        self.Sem_Seg_Decoder_res3_project_conv_weight_tensor = parameters.decoder.project_conv.conv.weight
        self.Sem_Seg_Decoder_res3_project_conv_bias_tensor = parameters.decoder.project_conv.conv.bias
        self.Sem_Seg_Decoder_res3_project_conv_input_channels = self.Sem_Seg_Decoder_res3_project_conv_weight_tensor.shape[1]
        self.Sem_Seg_Decoder_res3_project_conv_output_channels = self.Sem_Seg_Decoder_res3_project_conv_weight_tensor.shape[0]
        # Sem_Seg_Decoder_res3_fuse_conv_depthwise
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_weight_tensor = parameters.decoder.fuse_conv[0].conv.weight
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_bias_tensor = parameters.decoder.fuse_conv[0].conv.bias
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_input_channels = self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_weight_tensor.shape[1]
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_output_channels = self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_weight_tensor.shape[0]
        # Sem_Seg_Decoder_res3_fuse_conv_pointwise
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_weight_tensor = parameters.decoder.fuse_conv[0].bn.weight
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_bias_tensor = parameters.decoder.fuse_conv[0].bn.bias
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_input_channels = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_weight_tensor.shape[0]
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_output_channels = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_weight_tensor.shape[0]

        # Sem_Seg_Decoder_res2_project_conv
        self.Sem_Seg_Decoder_res2_project_conv_weight_tensor = parameters.decoder.project_conv2.conv.weight
        self.Sem_Seg_Decoder_res2_project_conv_bias_tensor = parameters.decoder.project_conv2.conv.bias
        self.Sem_Seg_Decoder_res2_project_conv_input_channels = self.Sem_Seg_Decoder_res2_project_conv_weight_tensor.shape[1]
        self.Sem_Seg_Decoder_res2_project_conv_output_channels = self.Sem_Seg_Decoder_res2_project_conv_weight_tensor.shape[0]
        # Sem_Seg_Decoder_res2_fuse_conv_depthwise
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_weight_tensor = parameters.decoder.fuse_conv[1].conv.weight
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_bias_tensor = parameters.decoder.fuse_conv[1].conv.bias
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_input_channels = self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_weight_tensor.shape[1]
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_output_channels = self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_weight_tensor.shape[0]
        # Sem_Seg_Decoder_res2_fuse_conv_pointwise
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_weight_tensor = parameters.decoder.fuse_conv[1].bn.weight
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_bias_tensor = parameters.decoder.fuse_conv[1].bn.bias
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_input_channels = self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_weight_tensor.shape[0]
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_output_channels = self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_weight_tensor.shape[0]

        # Sem_Seg_Head_depthwise
        self.Sem_Seg_Head_depthwise_weight_tensor = parameters.semantic_head.conv[0].conv.weight
        self.Sem_Seg_Head_depthwise_bias_tensor = parameters.semantic_head.conv[0].conv.bias
        self.Sem_Seg_Head_depthwise_input_channels = self.Sem_Seg_Head_depthwise_weight_tensor.shape[1]
        self.Sem_Seg_Head_depthwise_output_channels = self.Sem_Seg_Head_depthwise_weight_tensor.shape[0]
        # Sem_Seg_Head_pointwise
        self.Sem_Seg_Head_pointwise_weight_tensor = parameters.semantic_head.conv[0].bn.weight
        self.Sem_Seg_Head_pointwise_bias_tensor = parameters.semantic_head.conv[0].bn.bias
        self.Sem_Seg_Head_pointwise_input_channels = self.Sem_Seg_Head_pointwise_weight_tensor.shape[0]
        self.Sem_Seg_Head_pointwise_output_channels = self.Sem_Seg_Head_pointwise_weight_tensor.shape[0]

        # Sem_Seg_predictor
        self.Sem_Seg_predictor_weight_tensor = parameters.semantic_head.predictor.weight
        self.Sem_Seg_predictor_bias_tensor = parameters.semantic_head.predictor.bias
        self.Sem_Seg_predictor_input_channels = self.Sem_Seg_predictor_weight_tensor.shape[1]
        self.Sem_Seg_predictor_output_channels = self.Sem_Seg_predictor_weight_tensor.shape[0]




    


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

        logger.debug(
            f"==== Running {batch_size}, {input_height_1}, {input_width_1}, {self.Sem_Seg_ASPP_0_Conv_input_channels}, {self.Sem_Seg_ASPP_0_Conv_output_channels}"
        )

            # ASPP branch
        logger.debug("Running Sem_Seg_ASPP_0_Conv")
        aspp0, [aspp0_h, aspp0_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.Sem_Seg_ASPP_0_Conv_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_0_Conv_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_0_Conv_input_channels,
            out_channels=self.Sem_Seg_ASPP_0_Conv_output_channels,
            batch_size=batch_size,
            input_height=input_height_1,
            input_width=input_width_1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_1_Depthwise")
        aspp1_dw, [aspp1_dw_h, aspp1_dw_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.Sem_Seg_ASPP_1_Depthwise_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_1_Depthwise_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_1_Depthwise_input_channels,
            out_channels=self.Sem_Seg_ASPP_1_Depthwise_output_channels,
            batch_size=batch_size,
            input_height=input_height_1,
            input_width=input_width_1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(6, 6),
            dilation=(6, 6),
            groups=self.Sem_Seg_ASPP_1_Depthwise_input_channels,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_1_pointwise")
        aspp1, [aspp1_h, aspp1_w] = ttnn.conv2d(
            input_tensor=aspp1_dw,
            weight_tensor=self.Sem_Seg_ASPP_1_pointwise_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_1_pointwise_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_1_pointwise_input_channels,
            out_channels=self.Sem_Seg_ASPP_1_pointwise_output_channels,
            batch_size=batch_size,
            input_height=aspp1_dw_h,
            input_width=aspp1_dw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_2_Depthwise")
        aspp2_dw, [aspp2_dw_h, aspp2_dw_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.Sem_Seg_ASPP_2_Depthwise_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_2_Depthwise_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_2_Depthwise_input_channels,
            out_channels=self.Sem_Seg_ASPP_2_Depthwise_output_channels,
            batch_size=batch_size,
            input_height=input_height_1,
            input_width=input_width_1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(12, 12),
            dilation=(12, 12),
            groups=self.Sem_Seg_ASPP_2_Depthwise_input_channels,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_2_pointwise")
        aspp2, [aspp2_h, aspp2_w] = ttnn.conv2d(
            input_tensor=aspp2_dw,
            weight_tensor=self.Sem_Seg_ASPP_2_pointwise_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_2_pointwise_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_2_pointwise_input_channels,
            out_channels=self.Sem_Seg_ASPP_2_pointwise_output_channels,
            batch_size=batch_size,
            input_height=aspp2_dw_h,
            input_width=aspp2_dw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_3_Depthwise")
        aspp3_dw, [aspp3_dw_h, aspp3_dw_w] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.Sem_Seg_ASPP_3_Depthwise_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_3_Depthwise_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_3_Depthwise_input_channels,
            out_channels=self.Sem_Seg_ASPP_3_Depthwise_output_channels,
            batch_size=batch_size,
            input_height=input_height_1,
            input_width=input_width_1,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(18, 18),
            dilation=(18, 18),
            groups=self.Sem_Seg_ASPP_3_Depthwise_input_channels,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_3_pointwise")
        aspp3, [aspp3_h, aspp3_w] = ttnn.conv2d(
            input_tensor=aspp3_dw,
            weight_tensor=self.Sem_Seg_ASPP_3_pointwise_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_3_pointwise_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_3_pointwise_input_channels,
            out_channels=self.Sem_Seg_ASPP_3_pointwise_output_channels,
            batch_size=batch_size,
            input_height=aspp3_dw_h,
            input_width=aspp3_dw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_ASPP_4_avg_pool")
        aspp4 = ttnn.avg_pool2d(
            input_tensor=x,
            kernel_size=(32, 64),
            stride=(1, 1),
            padding=(0, 0),
        )

        logger.debug("Running Sem_Seg_ASPP_4_Conv_1")
        aspp4_conv, [aspp4_h, aspp4_w] = ttnn.conv2d(
            input_tensor=aspp4,
            weight_tensor=self.Sem_Seg_ASPP_4_Conv_1_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_4_Conv_1_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_4_Conv_1_input_channels,
            out_channels=self.Sem_Seg_ASPP_4_Conv_1_output_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        # ASPP project
        logger.debug("Running Sem_Seg_ASPP_project")
        aspp_concat = ttnn.concat([aspp0, aspp1, aspp2, aspp3, aspp4_conv], dim=1)
        aspp_project, [aspp_project_h, aspp_project_w] = ttnn.conv2d(
            input_tensor=aspp_concat,
            weight_tensor=self.Sem_Seg_ASPP_project_weight_tensor,
            bias_tensor=self.Sem_Seg_ASPP_project_bias_tensor,
            in_channels=self.Sem_Seg_ASPP_project_input_channels,
            out_channels=self.Sem_Seg_ASPP_project_output_channels,
            batch_size=batch_size,
            input_height=aspp0_h,
            input_width=aspp0_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        # Decoder: upsample and fuse with res3
        logger.debug("Running upsample after ASPP project")
        aspp_project_upsampled = ttnn.interpolate(
            input_tensor=aspp_project,
            scale_factor=2,
            mode="bilinear"
        )

        logger.debug("Running Sem_Seg_Decoder_res3_project_conv")
        res3_project, [res3_project_h, res3_project_w] = ttnn.conv2d(
            input_tensor=res3,
            weight_tensor=self.Sem_Seg_Decoder_res3_project_conv_weight_tensor,
            bias_tensor=self.Sem_Seg_Decoder_res3_project_conv_bias_tensor,
            in_channels=self.Sem_Seg_Decoder_res3_project_conv_input_channels,
            out_channels=self.Sem_Seg_Decoder_res3_project_conv_output_channels,
            batch_size=batch_size,
            input_height=aspp_project_upsampled.shape[2],
            input_width=aspp_project_upsampled.shape[3],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running concat for res3 and ASPP upsampled")
        decoder_res3_concat = ttnn.concat([res3_project, aspp_project_upsampled], dim=1)

        logger.debug("Running Sem_Seg_Decoder_res3_fuse_conv_depthwise")
        decoder_res3_fuse_dw, [decoder_res3_fuse_dw_h, decoder_res3_fuse_dw_w] = ttnn.conv2d(
            input_tensor=decoder_res3_concat,
            weight_tensor=self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_weight_tensor,
            bias_tensor=self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_bias_tensor,
            in_channels=self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_input_channels,
            out_channels=self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_output_channels,
            batch_size=batch_size,
            input_height=decoder_res3_concat.shape[2],
            input_width=decoder_res3_concat.shape[3],
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(1, 1),
            groups=self.Sem_Seg_Decoder_res3_fuse_conv_depthwise_input_channels,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_Decoder_res3_fuse_conv_pointwise")
        decoder_res3_fuse_pw, [decoder_res3_fuse_pw_h, decoder_res3_fuse_pw_w] = ttnn.conv2d(
            input_tensor=decoder_res3_fuse_dw,
            weight_tensor=self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_weight_tensor,
            bias_tensor=self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_bias_tensor,
            in_channels=self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_input_channels,
            out_channels=self.Sem_Seg_Decoder_res3_fuse_conv_pointwise_output_channels,
            batch_size=batch_size,
            input_height=decoder_res3_fuse_dw_h,
            input_width=decoder_res3_fuse_dw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running upsample after res3 fuse")
        decoder_res3_fuse_upsampled = ttnn.interpolate(
            input_tensor=decoder_res3_fuse_pw,
            scale_factor=2,
            mode="bilinear"
        )

        logger.debug("Running Sem_Seg_Decoder_res2_project_conv")
        res2_project, [res2_project_h, res2_project_w] = ttnn.conv2d(
            input_tensor=res2,
            weight_tensor=self.Sem_Seg_Decoder_res2_project_conv_weight_tensor,
            bias_tensor=self.Sem_Seg_Decoder_res2_project_conv_bias_tensor,
            in_channels=self.Sem_Seg_Decoder_res2_project_conv_input_channels,
            out_channels=self.Sem_Seg_Decoder_res2_project_conv_output_channels,
            batch_size=batch_size,
            input_height=decoder_res3_fuse_upsampled.shape[2],
            input_width=decoder_res3_fuse_upsampled.shape[3],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running concat for res2 and decoder upsampled")
        decoder_res2_concat = ttnn.concat([res2_project, decoder_res3_fuse_upsampled], dim=1)

        logger.debug("Running Sem_Seg_Decoder_res2_fuse_conv_depthwise")
        decoder_res2_fuse_dw, [decoder_res2_fuse_dw_h, decoder_res2_fuse_dw_w] = ttnn.conv2d(
            input_tensor=decoder_res2_concat,
            weight_tensor=self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_weight_tensor,
            bias_tensor=self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_bias_tensor,
            in_channels=self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_input_channels,
            out_channels=self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_output_channels,
            batch_size=batch_size,
            input_height=decoder_res2_concat.shape[2],
            input_width=decoder_res2_concat.shape[3],
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(1, 1),
            groups=self.Sem_Seg_Decoder_res2_fuse_conv_depthwise_input_channels,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_Decoder_res2_fuse_conv_pointwise")
        decoder_res2_fuse_pw, [decoder_res2_fuse_pw_h, decoder_res2_fuse_pw_w] = ttnn.conv2d(
            input_tensor=decoder_res2_fuse_dw,
            weight_tensor=self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_weight_tensor,
            bias_tensor=self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_bias_tensor,
            in_channels=self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_input_channels,
            out_channels=self.Sem_Seg_Decoder_res2_fuse_conv_pointwise_output_channels,
            batch_size=batch_size,
            input_height=decoder_res2_fuse_dw_h,
            input_width=decoder_res2_fuse_dw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_Head_depthwise")
        head_dw, [head_dw_h, head_dw_w] = ttnn.conv2d(
            input_tensor=decoder_res2_fuse_pw,
            weight_tensor=self.Sem_Seg_Head_depthwise_weight_tensor,
            bias_tensor=self.Sem_Seg_Head_depthwise_bias_tensor,
            in_channels=self.Sem_Seg_Head_depthwise_input_channels,
            out_channels=self.Sem_Seg_Head_depthwise_output_channels,
            batch_size=batch_size,
            input_height=decoder_res2_fuse_pw_h,
            input_width=decoder_res2_fuse_pw_w,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(1, 1),
            groups=self.Sem_Seg_Head_depthwise_input_channels,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_Head_pointwise")
        head_pw, [head_pw_h, head_pw_w] = ttnn.conv2d(
            input_tensor=head_dw,
            weight_tensor=self.Sem_Seg_Head_pointwise_weight_tensor,
            bias_tensor=self.Sem_Seg_Head_pointwise_bias_tensor,
            in_channels=self.Sem_Seg_Head_pointwise_input_channels,
            out_channels=self.Sem_Seg_Head_pointwise_output_channels,
            batch_size=batch_size,
            input_height=head_dw_h,
            input_width=head_dw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running Sem_Seg_predictor")
        predictor, [predictor_h, predictor_w] = ttnn.conv2d(
            input_tensor=head_pw,
            weight_tensor=self.Sem_Seg_predictor_weight_tensor,
            bias_tensor=self.Sem_Seg_predictor_bias_tensor,
            in_channels=self.Sem_Seg_predictor_input_channels,
            out_channels=self.Sem_Seg_predictor_output_channels,
            batch_size=batch_size,
            input_height=head_pw_h,
            input_width=head_pw_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=None,
                act_block_h_override=32,
                in_place=True,
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug("Running final upsample")
        output = ttnn.interpolate(
            input_tensor=predictor,
            scale_factor=4,
            mode="bilinear"
        )

        return output


