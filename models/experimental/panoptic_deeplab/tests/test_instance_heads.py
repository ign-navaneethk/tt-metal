# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.experimental.panoptic_deeplab.tt.common import create_custom_mesh_preprocessor
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.instance_seg_head import (
    PanopticDeeplabInstanceDecoderRes3,
    PanopticDeeplabInstanceDecoderRes2,
    PanopticDeeplabInstanceCenterHead,
    PanopticDeeplabInstanceOffsetHead,
    TTPanopticDeeplabInstanceSegmentationModel,
)
from models.experimental.panoptic_deeplab.reference.instance_seg_head import (
    PanopticDeeplabInstanceSegmentationModel,
)
from models.experimental.panoptic_deeplab.tt.aspp import PanopticDeeplabASPP


class PanopticDeeplabInstanceSegmentationTestInfra:
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

        torch_model = PanopticDeeplabInstanceSegmentationModel().eval()
        # print(torch_model)
        ##############################
        my_state_dict = torch_model.res2.state_dict()
        for key, value in my_state_dict.items():
            print(key)
        print("--------------------------------")
        ##############################

        self.fake_tensor_1 = torch.randn((1, 2048, 32, 64), dtype=torch.float16)
        self.fake_tensor_2 = torch.randn((1, 512, 64, 128), dtype=torch.float16)
        self.fake_tensor_3 = torch.randn((1, 256, 128, 256), dtype=torch.float16)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        print(parameters)
        torch_model.to(torch.bfloat16)
        self.fake_tensor_1 = self.fake_tensor_1.to(torch.bfloat16)
        self.fake_tensor_2 = self.fake_tensor_2.to(torch.bfloat16)
        self.fake_tensor_3 = self.fake_tensor_3.to(torch.bfloat16)

        ## golden
        # if self.run_block in ["aspp", "res3", "res2"]:
        #     self.torch_output_tensor = torch_model(self.fake_tensor_1, self.fake_tensor_2, self.fake_tensor_3)
        # else:
        self.torch_output_tensor, self.torch_output_tensor1 = torch_model(
            self.fake_tensor_1, self.fake_tensor_2, self.fake_tensor_3
        )

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

        # Initialize TTNN model with preprocessed parameters based on run_block
        if run_block == "aspp":
            self.ttnn_model = PanopticDeeplabASPP(parameters, model_config)
        elif run_block == "res3":
            self.ttnn_model = PanopticDeeplabInstanceDecoderRes3(parameters, model_config)
        elif run_block == "res2":
            self.ttnn_model = PanopticDeeplabInstanceDecoderRes2(parameters, model_config)
        elif run_block == "center_head":
            self.ttnn_model = PanopticDeeplabInstanceCenterHead(parameters, model_config)
        elif run_block == "offset_head":
            self.ttnn_model = PanopticDeeplabInstanceOffsetHead(parameters, model_config)
        # elif run_block == "res3_res2":
        #     self.ttnn_model = PanopticDeeplabInstanceRes3Res2(parameters, model_config)
        # elif run_block == "aspp_res3_res2":
        #     self.ttnn_model = PanopticDeeplabInstanceASPPRes3Res2(parameters, model_config)
        # elif run_block == "aspp_res3_res2_heads":
        #     self.ttnn_model = PanopticDeeplabInstanceASPPRes3Res2Heads(parameters, model_config)
        else:  # "full"
            self.ttnn_model = TTPanopticDeeplabInstanceSegmentationModel(parameters, model_config)

        # First run configures convs JIT
        # tracy.signpost(f"Instance_Segmentation_1x2048x32x64_compile")
        self.input_tensor_1 = ttnn.to_layout(tt_host_tensor_1, ttnn.TILE_LAYOUT)
        self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        # self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device)
        self.input_tensor_2 = ttnn.to_device(tt_host_tensor_2, device)
        self.input_tensor_3 = ttnn.to_device(tt_host_tensor_3, device)
        self.run()
        self.validate()
        if self.run_block in ["full", "heads"]:
            ttnn.deallocate(self.output_tensor1)
        ttnn.deallocate(self.output_tensor)

        # # Optimized run
        # tracy.signpost(f"Instance_Segmentation_1x2048x32x64_perf")
        # # self.input_tensor_1 = ttnn.to_layout(tt_host_tensor_1, ttnn.TILE_LAYOUT)
        # # self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        # self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device)
        # self.input_tensor_2 = ttnn.to_device(tt_host_tensor_2, device)
        # self.input_tensor_3 = ttnn.to_device(tt_host_tensor_3, device)
        # self.run()
        # self.validate()

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
        if self.run_block == "aspp":
            self.output_tensor = self.ttnn_model(self.input_tensor_1, self.device)
        elif self.run_block == "res3":
            # For res3 test, we need ASPP output - create a mock tensor with appropriate shape
            mock_aspp_output = ttnn.zeros(
                (1, 32, 64, 256), device=self.device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.output_tensor = self.ttnn_model(mock_aspp_output, self.input_tensor_2, self.device)
        elif self.run_block == "res2":
            # For res2 test, we need res3 output - create a mock tensor
            mock_res3_output = ttnn.zeros(
                (1, 64, 128, 128), device=self.device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.output_tensor = self.ttnn_model(mock_res3_output, self.input_tensor_3, self.device)
        elif self.run_block == "center_head":
            # For heads test, we need res2 output - create a mock tensor
            mock_res2_output = ttnn.zeros(
                (1, 128, 256, 128), device=self.device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.output_tensor = self.ttnn_model(mock_res2_output, self.device)

        elif self.run_block == "offset_head":
            # For heads test, we need res2 output - create a mock tensor
            mock_res2_output = ttnn.zeros(
                (1, 128, 256, 128), device=self.device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.output_tensor = self.ttnn_model(mock_res2_output, self.device)

            # return self.output_tensor, self.output_tensor1

        # elif self.run_block == "res3_res2":
        #     # For res3_res2 test, we need ASPP output
        #     mock_aspp_output = ttnn.zeros(
        #         (1, 32, 64, 256), device=self.device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
        #     )
        #     self.output_tensor = self.ttnn_model(
        #         mock_aspp_output, self.input_tensor_2, self.input_tensor_3, self.device
        #     )
        # elif self.run_block == "aspp_res3_res2":
        #     self.output_tensor = self.ttnn_model(
        #         self.input_tensor_1, self.input_tensor_2, self.input_tensor_3, self.device
        #     )
        # elif self.run_block == "aspp_res3_res2_heads":
        #     self.output_tensor, self.output_tensor1 = self.ttnn_model(
        #         self.input_tensor_1, self.input_tensor_2, self.input_tensor_3, self.device
        #     )
        else:  # "full"
            self.output_tensor, self.output_tensor1 = self.ttnn_model(
                # self.output_tensor = self.ttnn_model(
                self.input_tensor_1,
                self.input_tensor_2,
                self.input_tensor_3,
                self.device,
                self.batch_size,
                self.fake_tensor_1.shape[-2],
                self.fake_tensor_1.shape[-1],
                self.fake_tensor_2.shape[-2],
                self.fake_tensor_2.shape[-1],
                self.fake_tensor_3.shape[-2],
                self.fake_tensor_3.shape[-1],
                height_sharding=True,
                enable_act_double_buffer=True,
                enable_split_reader=True,
                enable_subblock_padding=False,
                eltwise_binary_out_in_place=True,
            )
            return self.output_tensor, self.output_tensor1

        return self.output_tensor

    def validate(self, output_tensor=None, output_tensor1=None):
        """Validate outputs"""
        # if self.run_block not in ["full", "aspp_res3_res2_heads", "heads"]:
        #     logger.info(f"Skipping validation for {self.run_block} test - component test passed")
        #     return True, "Component test completed"

        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        # print(output_tensor.shape)
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        # print(expected_shape)
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        # print(output_tensor.shape)
        batch_size = self.batch_size
        # batch_size = output_tensor.shape[0]

        valid_pcc = 0.999
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

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
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, run_block",
    [
        (1, "aspp"),  # Test ASPP component only
        # (1, "res3"),                    # Test Res3 decoder only
        # (1, "res2"),  # Test Res2 decoder only
        # (1, "center_head"),  # Test center head
        # (1, "offset_head"),                   # Test offset head
        # (1, "full"),  # Test full instance segmentation block
    ],
)
def test_instance_segmentation_heads(
    device,
    batch_size,
    run_block,
):
    test_infra = PanopticDeeplabInstanceSegmentationTestInfra(
        device,
        batch_size,
        model_config,
        run_block,
    )
