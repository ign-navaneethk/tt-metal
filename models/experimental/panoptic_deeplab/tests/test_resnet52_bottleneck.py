# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from torchvision.models.resnet import Bottleneck
from models.experimental.panoptic_deeplab.tt.bottleneck import TTBottleneck, bottleneck_layer_optimisations
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor


def map_single_key(checkpoint_key, layer_name):
    """
    Map checkpoint keys to model keys.
    """

    if not checkpoint_key:
        return ""

    key = checkpoint_key

    # BACKBONE MAPPINGS (REVERSE)
    if key.startswith("backbone."):
        # Layer mapping: res2/3/4/5 -> layer1/2/3/4
        if layer_name == "layer_1_d":
            key = key.replace("backbone.res2.0.", "")
        if layer_name == "layer_1_nd":
            key = key.replace("backbone.res2.2.", "")
        if layer_name == "layer_2_d":
            key = key.replace("backbone.res3.0.", "")
        if layer_name == "layer_2_nd":
            key = key.replace("backbone.res3.3.", "")
        if layer_name == "layer_3_d":
            key = key.replace("backbone.res4.0.", "")
        if layer_name == "layer_3_nd":
            key = key.replace("backbone.res4.5.", "")
        if layer_name == "layer_4_d":
            key = key.replace("backbone.res5.0.", "")
        if layer_name == "layer_4_nd_1":
            key = key.replace("backbone.res5.1.", "")
        if layer_name == "layer_4_nd_2":
            key = key.replace("backbone.res5.2.", "")

        # Batch norm mapping: conv1/2/3.norm -> bn1/2/3
        key = key.replace("conv1.norm", "bn1")
        key = key.replace("conv2.norm", "bn2")
        key = key.replace("conv3.norm", "bn3")

        # Downsample mapping: shortcut -> downsample
        key = key.replace("shortcut.norm", "downsample.1")
        # Handle shortcut.weight specifically to avoid matching shortcut.norm
        if "shortcut" in key and "shortcut.norm" not in checkpoint_key:
            key = key.replace("shortcut", "downsample.0")

        return key


class BottleneckTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        dilation,
        downsample,
        name,
        model_config,
    ):
        super().__init__()
        torch.manual_seed(42)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        downsample_conv = None
        if downsample:
            downsample_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, padding=0, bias=False
                ),
                torch.nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        self.torch_model = Bottleneck(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample_conv
        ).eval()

        try:
            import pickle
            import numpy as np

            # Load checkpoint
            with open("models/experimental/panoptic_deeplab/reference/panoptic_deeplab.pkl", "rb") as f:
                checkpoint = pickle.load(f, encoding="latin1")
            state_dict = checkpoint["model"]
            converted_count = 0
            for k, v in state_dict.items():
                if isinstance(v, np.ndarray):
                    state_dict[k] = torch.from_numpy(v)
                    converted_count += 1
            logger.debug(f"Converted {converted_count} numpy arrays to torch tensors")

            # Get model keys
            model_dict = self.torch_model.state_dict()
            model_keys = set(model_dict.keys())
            checkpoint_keys = set(state_dict.keys())

            # Get key mappings
            logger.info("Mapping keys...")
            key_mapping = {}
            for checkpoint_key in checkpoint_keys:  # pickle key
                mapped_key = map_single_key(checkpoint_key, name)
                if mapped_key in model_keys:  # torch keys
                    key_mapping[checkpoint_key] = mapped_key

            # Apply mappings
            mapped_state_dict = {}
            for checkpoint_key, model_key in key_mapping.items():
                mapped_state_dict[model_key] = state_dict[checkpoint_key]

            self.torch_model.load_state_dict(mapped_state_dict, strict=True)
            logger.info(f"Successfully loaded all {len(mapped_state_dict)} mapped weights with strict=True")

        except Exception as e:
            logger.error(f"Failed to load weights file: {str(e)}")
            logger.warning("Falling back to random initialization")

        input_shape = (batch_size * self.num_devices, inplanes, height, width)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        ## golden
        self.torch_model.to(torch.bfloat16)
        try:
            self.torch_input_tensor = torch.load(f"{name}_{input_shape}_input_tensor.pt")
            self.torch_output_tensor = torch.load(f"{name}_{input_shape}_output_tensor.pt")
        except:
            self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
            self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
            self.torch_output_tensor = self.torch_model(self.torch_input_tensor)
            torch.save(self.torch_input_tensor, f"{name}_{input_shape}_input_tensor.pt")
            torch.save(self.torch_output_tensor, f"{name}_{input_shape}_output_tensor.pt")

        ## ttnn
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        self.ttnn_model = TTBottleneck(
            parameters=parameters,
            downsample=downsample,
            stride=stride,
            model_config=model_config,
            dilation=dilation,
            name=name,
            layer_optimisations=bottleneck_layer_optimisations[name[:7]],
        )

        # First run configures convs JIT
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        # Optimized run
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
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
        self.output_tensor, _ = self.ttnn_model(
            self.input_tensor,
            self.device,
            self.input_tensor.shape,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        tt_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor, device=self.device, mesh_composer=self.output_mesh_composer
        )

        # Deallocate output tesnors
        ttnn.deallocate(tt_output_tensor)

        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        batch_size = tt_output_tensor_torch.shape[0]

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor, tt_output_tensor_torch, pcc=valid_pcc
        )

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"ResNet50 Bottleneck Block batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, inplanes, planes, height, width, stride, dilation, downsample, name",
    (
        # Layer 1
        (1, 128, 64, 128, 256, 1, 1, True, "layer_1_d"),
        (1, 256, 64, 128, 256, 1, 1, False, "layer_1_nd"),
        # Layer 2
        (1, 256, 128, 128, 256, 2, 1, True, "layer_2_d"),
        (1, 512, 128, 64, 128, 1, 1, False, "layer_2_nd"),
        # Layer 3
        (1, 512, 256, 64, 128, 2, 1, True, "layer_3_d"),
        (1, 1024, 256, 32, 64, 1, 1, False, "layer_3_nd"),
        # Layer 4
        (1, 1024, 512, 32, 64, 1, 2, True, "layer_4_d"),
        (1, 2048, 512, 32, 64, 1, 4, False, "layer_4_nd_1"),
        (1, 2048, 512, 32, 64, 1, 8, False, "layer_4_nd_2"),
    ),
)
def test_bottleneck(device, batch_size, inplanes, planes, height, width, stride, dilation, downsample, name):
    BottleneckTestInfra(
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        dilation,
        downsample,
        name,
        model_config,
    )
