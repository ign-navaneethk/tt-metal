# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import (
    PanopticDeepLabUnified as TorchPanopticDeepLab,
)
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLabUnified
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


class PanopticDeepLabUnifiedTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    ):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Initialize torch model
        torch_model = TorchPanopticDeepLab(
            num_classes=19,
            thing_classes=[11, 12, 13, 14, 15, 16, 17, 18],  # Common thing classes in cityscapes
            stuff_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Common stuff classes
        ).eval()

        # Create input tensor
        input_shape = (batch_size * self.num_devices, in_channels, height, width)
        self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
        torch.onnx.export(
            torch_model,
            self.torch_input_tensor,
            "panoptic_deeplab_ful_net.onnx",
            input_names=["input"],
            output_names=["output"],
        )

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = {}
        random_x = torch.randn(1, 2048, 32, 64)
        random_res3 = torch.randn(1, 512, 64, 128)
        random_res2 = torch.randn(1, 256, 128, 256)

        # For semantic decoder
        if hasattr(parameters, "semantic_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.aspp, run_model=lambda model: model(random_x), device=None
            )
            if hasattr(parameters.semantic_decoder, "aspp"):
                parameters.semantic_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.semantic_decoder.aspp(random_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res3,
                run_model=lambda model: model(aspp_out, random_res3),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res3"):
                parameters.semantic_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.semantic_decoder.res3(aspp_out, random_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res2,
                run_model=lambda model: model(res3_out, random_res2),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res2"):
                parameters.semantic_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.semantic_decoder.res2(res3_out, random_res2)
            head_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.head, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.semantic_decoder, "head"):
                parameters.semantic_decoder.head.conv_args = head_args

        # For instance center decoder
        if hasattr(parameters, "instance_center_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.instance_center_decoder.aspp, run_model=lambda model: model(random_x), device=None
            )
            if hasattr(parameters.instance_center_decoder, "aspp"):
                parameters.instance_center_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.instance_center_decoder.aspp(random_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.instance_center_decoder.res3,
                run_model=lambda model: model(aspp_out, random_res3),
                device=None,
            )
            if hasattr(parameters.instance_center_decoder, "res3"):
                parameters.instance_center_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.instance_center_decoder.res3(aspp_out, random_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.instance_center_decoder.res2,
                run_model=lambda model: model(res3_out, random_res2),
                device=None,
            )
            if hasattr(parameters.instance_center_decoder, "res2"):
                parameters.instance_center_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.instance_center_decoder.res2(res3_out, random_res2)
            head_args = infer_ttnn_module_args(
                model=torch_model.instance_center_decoder.head, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.instance_center_decoder, "head"):
                parameters.instance_center_decoder.head.conv_args = head_args

        # For instance offset decoder
        if hasattr(parameters, "instance_offset_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.instance_offset_decoder.aspp, run_model=lambda model: model(random_x), device=None
            )
            if hasattr(parameters.instance_offset_decoder, "aspp"):
                parameters.instance_offset_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.instance_offset_decoder.aspp(random_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.instance_offset_decoder.res3,
                run_model=lambda model: model(aspp_out, random_res3),
                device=None,
            )
            if hasattr(parameters.instance_offset_decoder, "res3"):
                parameters.instance_offset_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.instance_offset_decoder.res3(aspp_out, random_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.instance_offset_decoder.res2,
                run_model=lambda model: model(res3_out, random_res2),
                device=None,
            )
            if hasattr(parameters.instance_offset_decoder, "res2"):
                parameters.instance_offset_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.instance_offset_decoder.res2(res3_out, random_res2)
            head_args = infer_ttnn_module_args(
                model=torch_model.instance_offset_decoder.head, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.instance_offset_decoder, "head"):
                parameters.instance_offset_decoder.head.conv_args = head_args

        # Run torch model with bfloat16
        torch_model.to(torch.bfloat16)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)

        # Convert input to TTNN format (NHWC)
        logger.info("Converting input to TTNN format...")
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # Initialize TTNN model
        logger.info("Initializing TTNN model...")
        self.ttnn_model = TTPanopticDeepLabUnified(
            parameters=parameters,
            model_config=model_config,
            num_classes=19,
            thing_classes=[11, 12, 13, 14, 15, 16, 17, 18],
            stuff_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        # run and validate
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        self.output_tensor = self.ttnn_model(
            self.input_tensor,
            self.device,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        if output_tensor is None:
            if self.output_tensor is None:
                raise ValueError("self.output_tensor is None.")
            output_tensor = self.output_tensor

        # Convert outputs to torch tensors
        outputs_torch = {}
        outputs_torch["semantic_logits"] = ttnn.to_torch(
            output_tensor["semantic_logits"], device=self.device, mesh_composer=self.output_mesh_composer
        )
        outputs_torch["center_heatmap"] = ttnn.to_torch(
            output_tensor["center_heatmap"], device=self.device, mesh_composer=self.output_mesh_composer
        )
        outputs_torch["offset_map"] = ttnn.to_torch(
            output_tensor["offset_map"], device=self.device, mesh_composer=self.output_mesh_composer
        )
        outputs_torch["panoptic_pred"] = ttnn.to_torch(
            output_tensor["panoptic_pred"], device=self.device, mesh_composer=self.output_mesh_composer
        )

        # Get expected shapes
        expected_shapes = {
            "semantic_logits": self.torch_output_tensor["semantic_logits"].shape,
            "center_heatmap": self.torch_output_tensor["center_heatmap"].shape,
            "offset_map": self.torch_output_tensor["offset_map"].shape,
            "panoptic_pred": self.torch_output_tensor["panoptic_pred"].shape,
        }

        # Reshape and permute outputs
        # Semantic logits
        outputs_torch["semantic_logits"] = torch.reshape(
            outputs_torch["semantic_logits"],
            (
                expected_shapes["semantic_logits"][0],
                expected_shapes["semantic_logits"][2],
                expected_shapes["semantic_logits"][3],
                expected_shapes["semantic_logits"][1],
            ),
        )
        outputs_torch["semantic_logits"] = torch.permute(outputs_torch["semantic_logits"], (0, 3, 1, 2))

        # Center heatmap
        outputs_torch["center_heatmap"] = torch.reshape(
            outputs_torch["center_heatmap"],
            (
                expected_shapes["center_heatmap"][0],
                expected_shapes["center_heatmap"][2],
                expected_shapes["center_heatmap"][3],
                expected_shapes["center_heatmap"][1],
            ),
        )
        outputs_torch["center_heatmap"] = torch.permute(outputs_torch["center_heatmap"], (0, 3, 1, 2))

        # Offset map
        outputs_torch["offset_map"] = torch.reshape(
            outputs_torch["offset_map"],
            (
                expected_shapes["offset_map"][0],
                expected_shapes["offset_map"][2],
                expected_shapes["offset_map"][3],
                expected_shapes["offset_map"][1],
            ),
        )
        outputs_torch["offset_map"] = torch.permute(outputs_torch["offset_map"], (0, 3, 1, 2))

        # Panoptic prediction
        outputs_torch["panoptic_pred"] = torch.reshape(
            outputs_torch["panoptic_pred"],
            (
                expected_shapes["panoptic_pred"][0],
                expected_shapes["panoptic_pred"][1],
                expected_shapes["panoptic_pred"][2],
            ),
        )

        # Validate each output with PCC
        valid_pcc = 0.97

        # Semantic logits validation
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor["semantic_logits"], outputs_torch["semantic_logits"], pcc=valid_pcc
        )
        assert self.pcc_passed, logger.error(f"Semantic logits PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Semantic Logits: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        # Center heatmap validation
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor["center_heatmap"], outputs_torch["center_heatmap"], pcc=valid_pcc
        )
        assert self.pcc_passed, logger.error(f"Center heatmap PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Center Heatmap: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        # Offset map validation
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor["offset_map"], outputs_torch["offset_map"], pcc=valid_pcc
        )
        assert self.pcc_passed, logger.error(f"Offset map PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Offset Map: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        # Panoptic prediction validation
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor["panoptic_pred"], outputs_torch["panoptic_pred"], pcc=valid_pcc
        )
        assert self.pcc_passed, logger.error(f"Panoptic prediction PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Panoptic Prediction: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (1, 3, 512, 1024),
    ],
)
def test_panoptic_deeplab_unified(
    device,
    batch_size,
    in_channels,
    height,
    width,
):
    PanopticDeepLabUnifiedTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    )
