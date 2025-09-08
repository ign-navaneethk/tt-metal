# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.experimental.panoptic_deeplab.tt.common import create_custom_mesh_preprocessor
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.semantic_seg_head import (
    PanopticDeeplabDecoderRes2,
    PanopticDeeplabDecoderRes3,
    PanopticDeeplabHead,
    TTPanopticDeeplabSemanticSegmentationModel,
)
from models.experimental.panoptic_deeplab.tt.aspp import PanopticDeeplabASPP
from models.experimental.panoptic_deeplab.reference.aspp import PanopticDeeplabASPPModel
from models.experimental.panoptic_deeplab.reference.semantic_seg_head import (
    PanopticDeeplabSemanticDecoderRes3Model,
    PanopticDeeplabSemanticDecoderRes2Model,
    PanopticDeeplabSemanticHeadModel,
    PanopticDeeplabSemanticSegmentationModel,
)


class PanopticDeeplabSemanticsSegmentationTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
        run_block,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.run_block = run_block

        if run_block == "ASPP":
            torch_model = PanopticDeeplabASPPModel()
        elif run_block == "Decoder_Res3":
            torch_model = PanopticDeeplabSemanticDecoderRes3Model()
        elif run_block == "Decoder_Res2":
            torch_model = PanopticDeeplabSemanticDecoderRes2Model()
        elif run_block == "Semantics_Head":
            torch_model = PanopticDeeplabSemanticHeadModel()
        # elif run_block == "res3_res2":
        #     torch_model = PanopticDeeplabRes3Res2Model()
        # elif run_block == "ASPP_res3_res2":
        #     torch_model = PanopticDeeplabASPPRes3Res2Model()
        elif run_block == "ASPP_res3_res2_head":
            torch_model = PanopticDeeplabSemanticSegmentationModel()

        if run_block == "ASPP":
            self.fake_tensor_1 = torch.randn((1, 2048, 32, 64), dtype=torch.float32)
        elif run_block == "Decoder_Res3":
            self.fake_tensor_1 = torch.randn((1, 256, 32, 64), dtype=torch.float32)
            self.fake_tensor_2 = torch.randn((1, 512, 64, 128), dtype=torch.float32)
        elif run_block == "Decoder_Res2":
            self.fake_tensor_1 = torch.randn((1, 256, 64, 128), dtype=torch.float32)
            self.fake_tensor_2 = torch.randn((1, 256, 128, 256), dtype=torch.float32)
        elif run_block == "Semantics_Head":
            self.fake_tensor_1 = torch.randn((1, 256, 128, 256), dtype=torch.float32)
        # elif run_block == "res3_res2":
        #     self.fake_tensor_1 = torch.randn((1, 256, 32, 64), dtype=torch.float32)
        #     self.fake_tensor_2 = torch.randn((1, 512, 64, 128), dtype=torch.float32)
        #     self.fake_tensor_3 = torch.randn((1, 256, 128, 256), dtype=torch.float32)
        elif run_block == "ASPP_res3_res2" or run_block == "ASPP_res3_res2_head":
            self.fake_tensor_1 = torch.randn((1, 2048, 32, 64), dtype=torch.float32)
            self.fake_tensor_2 = torch.randn((1, 512, 64, 128), dtype=torch.float32)
            self.fake_tensor_3 = torch.randn((1, 256, 128, 256), dtype=torch.float32)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        # print(parameters)

        torch_model.to(torch.bfloat16)
        if run_block == "ASPP" or run_block == "Semantics_Head":
            self.fake_tensor_1 = self.fake_tensor_1.to(torch.bfloat16)
        elif run_block == "Decoder_Res3" or run_block == "Decoder_Res2":
            self.fake_tensor_1 = self.fake_tensor_1.to(torch.bfloat16)
            self.fake_tensor_2 = self.fake_tensor_2.to(torch.bfloat16)
        elif run_block == "res3_res2" or run_block == "ASPP_res3_res2" or run_block == "ASPP_res3_res2_head":
            self.fake_tensor_1 = self.fake_tensor_1.to(torch.bfloat16)
            self.fake_tensor_2 = self.fake_tensor_2.to(torch.bfloat16)
            self.fake_tensor_3 = self.fake_tensor_3.to(torch.bfloat16)

        ## golden
        if run_block == "ASPP" or run_block == "Semantics_Head":
            self.torch_output_tensor = torch_model(self.fake_tensor_1)
        elif run_block == "Decoder_Res3" or run_block == "Decoder_Res2":
            self.torch_output_tensor = torch_model(self.fake_tensor_1, self.fake_tensor_2)
        elif run_block == "res3_res2" or run_block == "ASPP_res3_res2" or run_block == "ASPP_res3_res2_head":
            self.torch_output_tensor = torch_model(self.fake_tensor_1, self.fake_tensor_2, self.fake_tensor_3)

            onnx_program = torch.onnx.export(
                torch_model, (self.fake_tensor_1, self.fake_tensor_2, self.fake_tensor_3), dynamo=True
            )
            onnx_program.save("semantics_res3_res2.onnx")
            print("onnx_generated")

        ## ttnn
        if run_block == "ASPP" or run_block == "Semantics_Head":
            tt_host_tensor_1 = ttnn.from_torch(
                self.fake_tensor_1.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat16,
                device=device,
                mesh_mapper=self.inputs_mesh_mapper,
            )
        elif run_block == "Decoder_Res3" or run_block == "Decoder_Res2":
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
        elif run_block == "res3_res2" or run_block == "ASPP_res3_res2" or run_block == "ASPP_res3_res2_head":
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
        if run_block == "ASPP":
            self.ttnn_model = PanopticDeeplabASPP(parameters, model_config)
        elif run_block == "Decoder_Res3":
            self.ttnn_model = PanopticDeeplabDecoderRes3(parameters, model_config)
        elif run_block == "Decoder_Res2":
            self.ttnn_model = PanopticDeeplabDecoderRes2(parameters, model_config)
        elif run_block == "Semantics_Head":
            self.ttnn_model = PanopticDeeplabHead(parameters, model_config)
        # elif run_block == "res3_res2":
        #     self.ttnn_model = PanopticDeeplabRes3Res2(parameters, model_config)
        # elif run_block == "ASPP_res3_res2":
        #     self.ttnn_model = PanopticDeeplabASPPRes3Res2(parameters, model_config)
        elif run_block == "ASPP_res3_res2_head":
            self.ttnn_model = TTPanopticDeeplabSemanticSegmentationModel(parameters, model_config)

        # First run configures convs JIT
        if run_block == "ASPP" or run_block == "Semantics_Head":
            self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device)
        elif run_block == "Decoder_Res3" or run_block == "Decoder_Res2":
            self.input_tensor_1 = ttnn.to_device(tt_host_tensor_1, device)
            self.input_tensor_2 = ttnn.to_device(tt_host_tensor_2, device)
        elif run_block == "res3_res2" or run_block == "ASPP_res3_res2" or run_block == "ASPP_res3_res2_head":
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
        if self.run_block == "ASPP" or self.run_block == "Semantics_Head":
            self.output_tensor = self.ttnn_model(
                self.input_tensor_1,
                self.device,
            )
            return self.output_tensor
        elif self.run_block == "Decoder_Res3" or self.run_block == "Decoder_Res2":
            self.output_tensor = self.ttnn_model(
                self.input_tensor_1,
                self.input_tensor_2,
                self.device,
            )
        elif (
            self.run_block == "res3_res2"
            or self.run_block == "ASPP_res3_res2"
            or self.run_block == "ASPP_res3_res2_head"
        ):
            self.output_tensor = self.ttnn_model(
                self.input_tensor_1,
                self.input_tensor_2,
                self.input_tensor_3,
                self.device,
            )

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.90
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab Semantics Segmentation batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, run_block",
    (
        # (1,"ASPP"), #ASPP
        # (1,"Decoder_Res3"), #Decoder Res3
        # (1,"Decoder_Res2"), #Decoder Res2
        # (1,"Semantics_Head"), #Semantics Head
        # (1,"res3_res2"), #res3_res2
        # (1,"ASPP_res3_res2"), #ASPP_res3_res2
        (1, "ASPP_res3_res2_head"),  # ASPP_res3_res2_head
    ),
)
def test_panoptic_deeplab_Semantics_segmentation(
    device,
    batch_size,
    run_block,
):
    PanopticDeeplabSemanticsSegmentationTestInfra(
        device,
        batch_size,
        model_config,
        run_block,
    )
