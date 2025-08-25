# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

import ttnn
from models.utility_functions import pad_and_fold_conv_filters_for_unity_stride
from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab_instance_segmentation import (
    PanopticDeeplabInstanceSegmentationModel,
)
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab_segmentation_head import (
    PanopticDeeplabSemanticsSegmentationModel,
)


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    if isinstance(model, torchvision.models.resnet.Bottleneck):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)
        parameters["conv1"] = {}
        parameters["conv2"] = {}
        parameters["conv3"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight, mesh_mapper=mesh_mapper)
        parameters["conv2"]["weight"] = ttnn.from_torch(conv2_weight, mesh_mapper=mesh_mapper)
        parameters["conv3"]["weight"] = ttnn.from_torch(conv3_weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        parameters["conv2"]["bias"] = ttnn.from_torch(torch.reshape(conv2_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        parameters["conv3"]["bias"] = ttnn.from_torch(torch.reshape(conv3_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        if model.downsample is not None:
            downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(model.downsample[0], model.downsample[1])
            parameters["downsample"] = {}
            parameters["downsample"]["weight"] = ttnn.from_torch(downsample_weight, mesh_mapper=mesh_mapper)
            parameters["downsample"]["bias"] = ttnn.from_torch(
                torch.reshape(downsample_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
            )
    elif isinstance(model, torchvision.models.resnet.ResNet):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv1_weight = pad_and_fold_conv_filters_for_unity_stride(conv1_weight, 2, 2)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        named_parameters = tuple((name, parameter) for name, parameter in model.named_parameters() if "." not in name)
        for child_name, child in tuple(model.named_children()) + named_parameters:
            if child_name in {"conv1", "bn1"}:
                continue
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=name,
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )
    elif isinstance(model, DeepLabStem):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)
        parameters["conv1"] = {}
        parameters["conv2"] = {}
        parameters["conv3"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(conv1_weight, mesh_mapper=mesh_mapper)
        parameters["conv2"]["weight"] = ttnn.from_torch(conv2_weight, mesh_mapper=mesh_mapper)
        parameters["conv3"]["weight"] = ttnn.from_torch(conv3_weight, mesh_mapper=mesh_mapper)
        parameters["conv1"]["bias"] = ttnn.from_torch(torch.reshape(conv1_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        parameters["conv2"]["bias"] = ttnn.from_torch(torch.reshape(conv2_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
        parameters["conv3"]["bias"] = ttnn.from_torch(torch.reshape(conv3_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)

    elif isinstance(model, PanopticDeeplabInstanceSegmentationModel):
        parameters = {}
        conv_names = [
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
            ("Ins_Seg_Center_predictor"),
            ("Ins_Seg_Offset_Head_depthwise"),
            ("Ins_Seg_Offset_Head_pointwise"),
            ("Ins_Seg_Offset_predictor"),
        ]

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

    elif isinstance(model, PanopticDeeplabSemanticsSegmentationModel):
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
        ]

        for conv_name in conv_names:
            try:
                conv = getattr(model, conv_name)
                parameters[conv_name] = {}
                parameters[conv_name]["weight"] = ttnn.from_torch(conv[0].weight, mesh_mapper=mesh_mapper)
                parameters[conv_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv[0].bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
                )
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
