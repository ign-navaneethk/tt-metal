# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
from models.experimental.panoptic_deeplab.tt.backbone import TTBackbone
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor


def map_single_key(checkpoint_key):
    """
    Map checkpoint keys to model keys.
    """

    if not checkpoint_key:
        return ""

    key = checkpoint_key

    # BACKBONE MAPPINGS (REVERSE)
    if key.startswith("backbone."):
        # Stem batch norm mappings (do this first to avoid conflicts)
        key = key.replace("backbone.stem", "stem")

        # Layer mapping: res2/3/4/5 -> layer1/2/3/4
        key = key.replace("backbone.res2.", "layer1.")
        key = key.replace("backbone.res3.", "layer2.")
        key = key.replace("backbone.res4.", "layer3.")
        key = key.replace("backbone.res5.", "layer4.")

        # Batch norm mapping: conv1/2/3.norm -> bn1/2/3
        key = key.replace(".conv1.norm.", ".bn1.")
        key = key.replace(".conv2.norm.", ".bn2.")
        key = key.replace(".conv3.norm.", ".bn3.")

        # Downsample mapping: shortcut -> downsample
        key = key.replace(".shortcut.norm.", ".downsample.1.")
        # Handle shortcut.weight specifically to avoid matching shortcut.norm
        if ".shortcut." in key and ".shortcut.norm." not in checkpoint_key:
            key = key.replace(".shortcut.", ".downsample.0.")

        return key


class BackboneTestInfra:
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
        torch.manual_seed(42)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        self.torch_model = TorchBackbone().eval()

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
                mapped_key = map_single_key(checkpoint_key)
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

        input_shape = (batch_size * self.num_devices, in_channels, height, width)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        ## golden
        self.torch_model.to(torch.bfloat16)
        try:
            self.torch_output = {}
            self.torch_input_tensor = torch.load(f"backbone_{input_shape}_input_tensor.pt")
            self.torch_output["res_2"] = torch.load(f"backbone_{input_shape}_res_2_output_tensor.pt")
            self.torch_output["res_3"] = torch.load(f"backbone_{input_shape}_res_3_output_tensor.pt")
            self.torch_output["res_4"] = torch.load(f"backbone_{input_shape}_res_4_output_tensor.pt")
            self.torch_output["res_5"] = torch.load(f"backbone_{input_shape}_res_5_full_output_tensor.pt")
            # self.torch_output["res_5"] = torch.load(f"backbone_{input_shape}_res_5_output_tensor.pt")
        except:
            import numpy as np

            numpy_array = np.load("models/experimental/panoptic_deeplab/reference/normalise_512_1024.npy")
            self.torch_input_tensor = torch.from_numpy(numpy_array)
            # self.torch_input_tensor = torch.randint(low=0,high=255, size=input_shape)
            self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
            self.torch_output = self.torch_model(self.torch_input_tensor)
            torch.save(self.torch_input_tensor, f"backbone_{input_shape}_input_tensor.pt")
            torch.save(self.torch_output["res_2"], f"backbone_{input_shape}_res_2_output_tensor.pt")
            torch.save(self.torch_output["res_3"], f"backbone_{input_shape}_res_3_output_tensor.pt")
            torch.save(self.torch_output["res_4"], f"backbone_{input_shape}_res_4_output_tensor.pt")
            torch.save(self.torch_output["res_5"], f"backbone_{input_shape}_res_5_full_output_tensor.pt")
            # torch.save(self.torch_output["res_5"], f"backbone_{input_shape}_res_5_output_tensor.pt")

        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat8_b,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        self.ttnn_model = TTBackbone(
            parameters=parameters,
            model_config=model_config,
        )

        # First run configures convs JIT
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        # # Optimized run
        # self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        # self.run()
        # self.validate()

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
            self.input_tensor,
            self.device,
        )

    def validate(self, output_tensor=None):
        tt_output = self.output_tensor if output_tensor is None else output_tensor
        valid_pcc = {
            "res_2": 0.99,
            "res_3": 0.99,
            "res_4": 0.99,
            "res_5": 0.80,
        }
        self.pcc_passed_all = []
        self.pcc_message_all = []

        for key in tt_output:
            tt_output_tensor_torch = ttnn.to_torch(
                tt_output[key],
                dtype=self.torch_output[key].dtype,
                device=self.device,
                mesh_composer=self.output_mesh_composer,
            )

            # Deallocate output tesnors
            ttnn.deallocate(tt_output[key])

            expected_shape = self.torch_output[key].shape
            tt_output_tensor_torch = torch.reshape(
                tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
            )
            tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

            pcc_passed, pcc_message = check_with_pcc(self.torch_output[key], tt_output_tensor_torch, pcc=valid_pcc[key])
            self.pcc_passed_all.append(pcc_passed)
            self.pcc_message_all.append(pcc_message)

        assert all(self.pcc_passed_all), logger.error(f"PCC check failed: {self.pcc_message_all}")
        logger.info(f"ResNet52 Backbone with PCC={self.pcc_message_all}")


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (1, 3, 512, 1024),
    ],
)
def test_backbone(
    device,
    batch_size,
    in_channels,
    height,
    width,
):
    BackboneTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    )
