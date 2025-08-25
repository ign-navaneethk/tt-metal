# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TTConv2D:
    def __init__(
        self,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        parameters: dict | None = None,
        kernel_fidelity: dict | None = None,
        *,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        act_block_h=None,
        act_block_w=None,
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=None,
        activation="",
        groups=1,
        num_cores_nhw=None,
        is_reshape=False,
        enable_split_reader=False,
        enable_act_double_buffer=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        reshard_if_not_optimal=False,
        slice_config=None,
    ) -> None:
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            ValueError("Invalid config")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            ValueError("Invalid config")
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            ValueError("Invalid config")
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            ValueError("Invalid config")

        self.kernel_fidelity = kernel_fidelity
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.deallocate_activation = deallocate_activation
        self.reallocate_halo_output = reallocate_halo_output
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.act_block_w = act_block_w
        self.groups = groups
        self.activation = activation
        self.memory_config = memory_config
        # self.shard_layout = (
        #     ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        # )
        self.shard_layout = shard_layout
        self.slice_config = slice_config
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_split_reader = enable_split_reader
        self.enable_act_double_buffer = enable_act_double_buffer

    def __call__(self, device, input_tensor, input_shape):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.kernel_fidelity["WEIGHTS_DTYPE"],
            activation=self.activation,
            in_place=True,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            enable_split_reader=self.enable_split_reader,
            enable_act_double_buffer=self.enable_act_double_buffer,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=self.reallocate_halo_output,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.kernel_fidelity["MATH_FIDELITY"],
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        # [output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_shape[-1],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=input_shape[-4],
            input_height=input_shape[-3],
            input_width=input_shape[-2],
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=self.slice_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.kernel_fidelity["ACTIVATIONS_DTYPE"],
            memory_config=self.memory_config,
        )
        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.reshape(
                output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])
            )
            output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "bilinear",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.memory_config = memory_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

    def __call__(self, device, input_tensor, input_shape=None, reshape_output=True, pad_ch_to_32=False):
        input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 13)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        if pad_ch_to_32:
            output_tensor = ttnn.slice(output_tensor, [0, 0, 0, 0], [1, 512, 1024, 19])

        if reshape_output:
            output_tensor = ttnn.from_device(output_tensor)
            output_tensor = ttnn.to_dtype(output_tensor, ttnn.bfloat8_b)
            output_tensor = ttnn.to_device(output_tensor, device)

            output_tensor = ttnn.reshape(
                output_tensor,
                [
                    1,
                    1,
                    output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
                    output_tensor.shape[3],
                ],
            )

        return output_tensor


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}

    conv_names = [
        ("Sem_Seg_ASPP_0_Conv"),
        ("Sem_Seg_ASPP_1_Depthwise"),
        ("Sem_Seg_ASPP_1_pointwise"),
        ("Sem_Seg_ASPP_2_Depthwise"),
        ("Sem_Seg_ASPP_2_pointwise"),
        ("Sem_Seg_ASPP_3_Depthwise"),
        ("Sem_Seg_ASPP_3_pointwise"),
        ("Sem_Seg_ASPP_4_Conv_1"),
        ("Sem_Seg_ASPP_project"),
        ("Sem_Seg_Decoder_res3_project_conv"),
        ("Sem_Seg_Decoder_res3_fuse_conv_depthwise"),
        ("Sem_Seg_Decoder_res3_fuse_conv_pointwise"),
        ("Sem_Seg_Decoder_res2_project_conv"),
        ("Sem_Seg_Decoder_res2_fuse_conv_depthwise"),
        ("Sem_Seg_Decoder_res2_fuse_conv_pointwise"),
        ("Sem_Seg_Head_depthwise"),
        ("Sem_Seg_Head_pointwise"),
        ("Ins_Seg_ASPP_0_Conv"),
        ("Ins_Seg_ASPP_1_Depthwise"),
        ("Ins_Seg_ASPP_1_pointwise"),
        ("Ins_Seg_ASPP_2_Depthwise"),
        ("Ins_Seg_ASPP_2_pointwise"),
        ("Ins_Seg_ASPP_3_Depthwise"),
        ("Ins_Seg_ASPP_3_pointwise"),
        ("Ins_Seg_ASPP_4_Conv_1"),
        ("Ins_Seg_ASPP_project"),
        ("Ins_Seg_Decoder_res3_project_conv"),
        ("Ins_Seg_Decoder_res3_fuse_conv_depthwise"),
        ("Ins_Seg_Decoder_res3_fuse_conv_pointwise"),
        ("Ins_Seg_Decoder_res2_project_conv"),
        ("Ins_Seg_Decoder_res2_fuse_conv_depthwise"),
        ("Ins_Seg_Decoder_res2_fuse_conv_pointwise"),
        ("Ins_Seg_Center_Head_Conv_0"),
        ("Ins_Seg_Center_Head_Conv_1"),
        ("Ins_Seg_Offset_Head_depthwise"),
        ("Ins_Seg_Offset_Head_pointwise"),
        ("Ins_Seg_Center_predictor"),
        ("Ins_Seg_Offset_predictor"),
    ]

    for conv_name in conv_names:
        try:
            for conv_name in conv_names:
                parameters[conv_name] = {}
                conv = getattr(model, conv_name)

                if hasattr(conv, "__getitem__"):
                    conv_layer = conv[0]
                else:
                    conv_layer = conv

                weight_clean = conv_layer.weight.clone().detach().contiguous()
                bias_clean = conv_layer.bias.clone().detach().contiguous()

                weight_clean = torch.clamp(weight_clean, -10.0, 10.0)
                bias_clean = torch.clamp(bias_clean, -10.0, 10.0)

                parameters[conv_name]["weight"] = ttnn.from_torch(weight_clean, mesh_mapper=mesh_mapper)

                bias_reshaped = torch.reshape(bias_clean, (1, 1, 1, -1))
                parameters[conv_name]["bias"] = ttnn.from_torch(bias_reshaped, mesh_mapper=mesh_mapper)
        except:
            continue
    try:
        conv_name = "Sem_Seg_predictor"
        parameters[conv_name] = {}
        conv = getattr(model, conv_name)
        parameters[conv_name]["weight"] = ttnn.from_torch(conv.weight, mesh_mapper=mesh_mapper)
        parameters[conv_name]["bias"] = ttnn.from_torch(
            torch.reshape(conv.bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
        )
    except:
        pass

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
