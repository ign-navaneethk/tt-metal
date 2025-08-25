# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.experimental.panoptic_deeplab.tt.common import create_custom_mesh_preprocessor
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.tt_panoptic_deeplab_segmentation import (
    PanopticDeeplabSemanticsSegmentation,
)
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab_segmentation_head import (
    PanopticDeeplabSemanticsSegmentationModel,
)


class PanopticDeeplabSemanticsSegmentationTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        torch_model = PanopticDeeplabSemanticsSegmentationModel()

        self.fake_tensor_1 = torch.randn((1, 2048, 32, 64), dtype=torch.float32)
        self.fake_tensor_2 = torch.randn((1, 512, 64, 128), dtype=torch.float32)
        self.fake_tensor_3 = torch.randn((1, 256, 128, 256), dtype=torch.float32)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        torch_model.to(torch.bfloat16)
        self.fake_tensor_1 = self.fake_tensor_1.to(torch.bfloat16)
        self.fake_tensor_2 = self.fake_tensor_2.to(torch.bfloat16)
        self.fake_tensor_3 = self.fake_tensor_3.to(torch.bfloat16)

        ## golden
        self.torch_output_tensor = torch_model(self.fake_tensor_1, self.fake_tensor_2, self.fake_tensor_3)

        ## ttnn
        tt_host_tensor_1 = ttnn.from_torch(
            self.fake_tensor_1.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        tt_host_tensor_2 = ttnn.from_torch(
            self.fake_tensor_2.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        tt_host_tensor_3 = ttnn.from_torch(
            self.fake_tensor_3.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # Initialize TTNN model with preprocessed parameters
        self.ttnn_model = PanopticDeeplabSemanticsSegmentation(parameters, model_config)

        # First run configures convs JIT
        self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device)
        self.input_tensor_2 = ttnn.to_device(tt_host_tensor_2, device)
        self.input_tensor_3 = ttnn.to_device(tt_host_tensor_3, device)
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None  # ttnn.ReplicateTensorToMesh(device) causes unnecessary replication/takes more time on the first pass
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        self.output_tensor = self.ttnn_model(
            self.input_tensor_1,
            self.input_tensor_2,
            self.input_tensor_3,
            self.device,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.999
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab Semantics Segmentation batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_panoptic_deeplab_semantics_segmentation(
    device,
    batch_size,
):
    PanopticDeeplabSemanticsSegmentationTestInfra(
        device,
        batch_size,
        model_config,
    )
