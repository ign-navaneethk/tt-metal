# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import pickle
import numpy as np


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


def load_mapped_model_state(model_location_generator=None):
    # TODO: Cleanup properly and weight download

    # if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
    #     model_path = "models"
    # else:
    #     model_path = model_location_generator("vision-models/yolov4", model_subdir="", download_if_ci_v2=True)
    # if model_path == "models":
    #     if not os.path.exists("models/demos/yolov4/tests/pcc/yolov4.pth"):  # check if yolov4.th is availble
    #         os.system(
    #             "models/demos/yolov4/tests/pcc/yolov4_weights_download.sh"
    #         )  # execute the yolov4_weights_download.sh file
    #     weights_pth = "models/demos/yolov4/tests/pcc/yolov4.pth"
    # else:
    #     weights_pth = os.path.join(model_path, "yolov4.pth")

    # torch_dict = torch.load(weights_pth)
    # state_dict = torch_dict

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

    # Get keys
    checkpoint_keys = set(state_dict.keys())

    # Get key mappings
    logger.info("Mapping keys...")
    key_mapping = {}
    for checkpoint_key in checkpoint_keys:  # pickle key
        mapped_key = map_single_key(checkpoint_key)
        key_mapping[checkpoint_key] = mapped_key

    # Apply mappings
    mapped_state_dict = {}
    for checkpoint_key, model_key in key_mapping.items():
        mapped_state_dict[model_key] = state_dict[checkpoint_key]

    return mapped_state_dict
