# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters
import ttnn
from models.experimental.panoptic_deeplab.tt.common import create_custom_mesh_preprocessor
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.head import (
    # # PanopticDeeplabInstanceASPP,
    # # PanopticDeeplabInstanceDecoderRes3,
    # # PanopticDeeplabInstanceDecoderRes2,
    # PanopticDeeplabInstanceCenterHead,
    # PanopticDeeplabInstanceOffsetHead,
    # # PanopticDeeplabInstanceRes3Res2,
    # # PanopticDeeplabInstanceASPPRes3Res2,
    # # PanopticDeeplabInstanceASPPRes3Res2Heads,
    # PanopticDeeplabInstanceSegmentation,
    TTHead,
)

# from models.experimental.panoptic_deeplab.reference.panoptic_deeplab_instance_segmentation import (
from models.experimental.panoptic_deeplab.reference.head import (
    HeadModel,
)
from models.utility_functions import (
    comp_pcc,
)


class HeadTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
        in_channels,
        intermediate_channels,
        out_channels,
        height,
        width,
    ):
        super().__init__()
        # torch.manual_seed(0)
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)  # Only seed once
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
        # torch.manual_seed(42)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        self.torch_input_tensor = torch.randn(
            (self.batch_size, self.in_channels, self.height, self.width), dtype=torch.float32
        )
        torch_model = HeadModel(self.in_channels, self.intermediate_channels, self.out_channels).eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters.conv_args = {}
        parameters.conv_args = infer_ttnn_module_args(
            model=torch_model, run_model=lambda model: model(self.torch_input_tensor), device=None
        )

        # Generate fake input tensors for different model blocks

        torch_model.to(torch.bfloat16)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Convert torch tensors to TTNN host tensors (NHWC, bfloat8_b)
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat16,
                device=device,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)

        # Move TTNN host tensors to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)

        self.ttnn_model = TTHead(parameters, model_config)

        self.run()
        if 0:
            self.validate()
            # ttnn.deallocate(self.output_tensor)

        ttnn.deallocate(self.output_tensor)

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

    # Compute golden output for the selected block using a helper function

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor, self.device)

        return self.output_tensor

    def validate(self, output_tensor=None, output_tensor1=None):
        """Validate outputs"""
        # if self.run_block not in ["full", "aspp_res3_res2_heads", "heads"]:
        #     logger.info(f"Skipping validation for {self.run_block} test - component test passed")
        #     return True, "Component test completed"

        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        print(f"expected_shape: {expected_shape}")

        # else:
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        # print(output_tensor.shape)
        batch_size = self.batch_size
        print(f"output_tensor.shape: {output_tensor.shape}")
        # batch_size = output_tensor.shape[0]

        valid_pcc = 0.999

        _, pcc_out = comp_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        # self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        print("self.pcc_OUT", pcc_out)
        # print("self.pcc_message", self.pcc_message)
        # assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Modular Panoptic DeepLab Instance Segmentation Center batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        if self.run_block in ["aspp", "res3", "res2", "center_head", "offset_head"]:
            return self.pcc_passed, self.pcc_message
        else:
            output_tensor1 = self.output_tensor1 if output_tensor1 is None else output_tensor1
            # print(output_tensor1.shape)
            output_tensor1 = ttnn.to_torch(output_tensor1, device=self.device, mesh_composer=self.output_mesh_composer)
            expected_shape1 = self.torch_output_tensor1.shape
            # print(expected_shape1)

            output_tensor1 = torch.reshape(
                output_tensor1, (expected_shape1[0], expected_shape1[2], expected_shape1[3], expected_shape1[1])
            )
            output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))
            # print(output_tensor1.shape)

            batch_size = self.batch_size
            # batch_size = output_tensor1.shape[0]

            self.pcc_passed1, self.pcc_message1 = check_with_pcc(
                self.torch_output_tensor1, output_tensor1, pcc=valid_pcc
            )

            # assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
            logger.info(
                f"Modular Panoptic DeepLab Instance Segmentation Offset batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message1}"
            )

            return self.pcc_passed, self.pcc_message, self.pcc_passed1, self.pcc_message1


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)


@pytest.mark.parametrize(
    "batch_size, in_channels, intermediate_channels, out_channels, height, width",
    [
        # (1, 3, 1024, 2048),
        (1, 256, 256, 19, 128, 256),  # segmentation offset head
        # (1, 128, 32,  2, 128, 256), #instance offset head
        # (1, 128, 32,  1, 128, 256), # instance center head
    ],
)
def test_modular_panoptic_deeplab_instance_segmentation(
    device,
    batch_size,
    in_channels,
    intermediate_channels,
    out_channels,
    height,
    width,
):
    HeadTestInfra(
        device,
        batch_size,
        model_config,
        in_channels,
        intermediate_channels,
        out_channels,
        height,
        width,
    )

    # # Calculate and log FPS for performance comparison
    # fps, avg_inference_time = test_infra.calculate_fps(num_iterations=5)

    # logger.info(f"Test completed for {run_block} - FPS: {fps:.2f}")

    # # add assertions for performance if needed
    # assert fps >= expected_min_fps, f"FPS {fps:.2f} is below expected minimum {expected_min_fps}"
