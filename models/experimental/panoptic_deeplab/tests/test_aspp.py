# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc

# from models.experimental.panoptic_deeplab.tt.common import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor

# from models.experimental.panoptic_deeplab.reference.instance_seg_head import (
#     PanopticDeeplabASPPModel,
# )

from models.experimental.panoptic_deeplab.reference.aspp import (
    PanopticDeeplabASPPModel,
)
from models.experimental.panoptic_deeplab.tt.aspp import PanopticDeeplabASPP


class PanopticDeeplabASPPTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
        run_block="full",
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
        self.run_block = run_block
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        torch_model = PanopticDeeplabASPPModel().eval()
        # print(torch_model)
        ##############################
        my_state_dict = torch_model.state_dict()
        for key, value in my_state_dict.items():
            print(key)
        print("--------------------------------")
        #################################

        self.fake_tensor_1 = torch.randn((1, 2048, 32, 64), dtype=torch.float16)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        print(parameters)
        torch_model.to(torch.bfloat16)
        self.fake_tensor_1 = self.fake_tensor_1.to(torch.bfloat16)

        ## golden
        self.torch_output_tensor = torch_model(self.fake_tensor_1)

        ## ttnn
        tt_host_tensor_1 = ttnn.from_torch(
            self.fake_tensor_1.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        self.ttnn_model = PanopticDeeplabASPP(parameters, model_config)

        self.input_tensor_1 = ttnn.to_layout(tt_host_tensor_1, ttnn.TILE_LAYOUT)
        self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.run()
        self.validate()
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

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor_1, self.device)
        return self.output_tensor

    def validate(self, output_tensor=None, output_tensor1=None):
        """Validate outputs"""

        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        batch_size = self.batch_size

        valid_pcc = 0.999
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        # assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Modular Panoptic DeepLab Instance Segmentation Center batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, run_block",
    [
        (1, "aspp"),
    ],
)
def test_aspp(
    device,
    batch_size,
    run_block,
):
    test_infra = PanopticDeeplabASPPTestInfra(
        device,
        batch_size,
        model_config,
        run_block,
    )
