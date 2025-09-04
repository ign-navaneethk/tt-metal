# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent Panoptic DeepLab Demo

Usage:
    python enhanced_tt_panoptic_demo.py --config configs/demo_config.yaml --input image.jpg --output results/
"""

import os
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from loguru import logger
import json
from dataclasses import dataclass, asdict
import pickle
import torch
import torchvision.transforms as transforms
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

# Import TT Panoptic DeepLab modules
from models.experimental.panoptic_deeplab.tt.tt_panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import PanopticDeepLab as TorchPanopticDeepLab


@dataclass
class DemoConfig:
    """Configuration class for demo parameters"""

    # Model configuration
    model_type: str = "PanopticDeepLab"
    backbone: str = "ResNet-52"
    num_classes: int = 19
    weights_path: Optional[str] = None

    # Input configuration
    input_height: int = 512
    input_width: int = 1024
    crop_enabled: bool = False
    normalize_enabled: bool = True
    mean: List[float] = None
    std: List[float] = None

    # Inference configuration
    center_threshold: float = 0.1
    nms_kernel: int = 7
    top_k_instances: int = 200
    stuff_area_threshold: int = 2048

    # Device configuration
    device_id: int = 0
    math_fidelity: str = "LoFi"
    weights_dtype: str = "bfloat16"
    activations_dtype: str = "bfloat16"

    # Output configuration
    save_semantic: bool = True
    save_instance: bool = True
    save_panoptic: bool = True
    save_visualization: bool = True
    save_comparison: bool = True

    # Pipeline configuration
    run_torch_pipeline: bool = True
    run_ttnn_pipeline: bool = True
    compare_outputs: bool = True
    pcc_threshold: float = 0.95

    # Dataset configuration (Cityscapes default)
    thing_classes: List[int] = None
    stuff_classes: List[int] = None
    class_names: List[str] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]
        if self.thing_classes is None:
            self.thing_classes = [11, 12, 13, 14, 15, 16, 17, 18]  # Cityscapes things
        if self.stuff_classes is None:
            self.stuff_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Cityscapes stuff
        if self.class_names is None:
            self.class_names = [
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic_light",
                "traffic_sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DemoConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested config structure to match dataclass fields
        flattened = {}

        # Model section
        if "MODEL" in config_dict:
            model_cfg = config_dict["MODEL"]
            flattened.update(
                {
                    "model_type": model_cfg.get("TYPE", "PanopticDeepLab"),
                    "backbone": model_cfg.get("BACKBONE", "ResNet-52"),
                    "num_classes": model_cfg.get("NUM_CLASSES", 19),
                    "weights_path": model_cfg.get("WEIGHTS", None),
                }
            )

        # Input section
        if "INPUT" in config_dict:
            input_cfg = config_dict["INPUT"]
            if "SIZE_TRAIN" in input_cfg or "SIZE_TEST" in input_cfg:
                size = input_cfg.get("SIZE_TEST", input_cfg.get("SIZE_TRAIN", [1024, 512]))
                flattened.update(
                    {
                        "input_height": size[0],
                        "input_width": size[1],
                    }
                )
            flattened.update(
                {
                    "crop_enabled": input_cfg.get("CROP", {}).get("ENABLED", False),
                    "normalize_enabled": input_cfg.get("NORMALIZE", True),
                    "mean": input_cfg.get("PIXEL_MEAN", [0.485, 0.456, 0.406]),
                    "std": input_cfg.get("PIXEL_STD", [0.229, 0.224, 0.225]),
                }
            )

        # Postprocessing section
        if "POST_PROCESSING" in config_dict:
            pp_cfg = config_dict["POST_PROCESSING"]
            flattened.update(
                {
                    "center_threshold": pp_cfg.get("CENTER_THRESHOLD", 0.1),
                    "nms_kernel": pp_cfg.get("NMS_KERNEL", 7),
                    "top_k_instances": pp_cfg.get("TOP_K_INSTANCES", 200),
                    "stuff_area_threshold": pp_cfg.get("STUFF_AREA_THRESHOLD", 2048),
                }
            )

        # Device section
        if "DEVICE" in config_dict:
            device_cfg = config_dict["DEVICE"]
            flattened.update(
                {
                    "device_id": device_cfg.get("ID", 0),
                    "math_fidelity": device_cfg.get("MATH_FIDELITY", "LoFi"),
                    "weights_dtype": device_cfg.get("WEIGHTS_DTYPE", "bfloat16"),
                    "activations_dtype": device_cfg.get("ACTIVATIONS_DTYPE", "bfloat16"),
                }
            )

        # Demo-specific sections
        if "DEMO" in config_dict:
            demo_cfg = config_dict["DEMO"]
            flattened.update(
                {
                    "run_torch_pipeline": demo_cfg.get("RUN_TORCH", True),
                    "run_ttnn_pipeline": demo_cfg.get("RUN_TTNN", True),
                    "compare_outputs": demo_cfg.get("COMPARE_OUTPUTS", True),
                    "pcc_threshold": demo_cfg.get("PCC_THRESHOLD", 0.95),
                }
            )

        # Classes section
        if "CLASSES" in config_dict:
            classes_cfg = config_dict["CLASSES"]
            flattened.update(
                {
                    "thing_classes": classes_cfg.get("THING", [11, 12, 13, 14, 15, 16, 17, 18]),
                    "stuff_classes": classes_cfg.get("STUFF", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "class_names": classes_cfg.get("NAMES", None),
                }
            )

        return cls(**flattened)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            "MODEL": {
                "TYPE": self.model_type,
                "BACKBONE": self.backbone,
                "NUM_CLASSES": self.num_classes,
                "WEIGHTS": self.weights_path,
            },
            "INPUT": {
                "SIZE_TEST": [self.input_height, self.input_width],
                "CROP": {"ENABLED": self.crop_enabled},
                "NORMALIZE": self.normalize_enabled,
                "PIXEL_MEAN": self.mean,
                "PIXEL_STD": self.std,
            },
            "POST_PROCESSING": {
                "CENTER_THRESHOLD": self.center_threshold,
                "NMS_KERNEL": self.nms_kernel,
                "TOP_K_INSTANCES": self.top_k_instances,
                "STUFF_AREA_THRESHOLD": self.stuff_area_threshold,
            },
            "DEVICE": {
                "ID": self.device_id,
                "MATH_FIDELITY": self.math_fidelity,
                "WEIGHTS_DTYPE": self.weights_dtype,
                "ACTIVATIONS_DTYPE": self.activations_dtype,
            },
            "DEMO": {
                "RUN_TORCH": self.run_torch_pipeline,
                "RUN_TTNN": self.run_ttnn_pipeline,
                "COMPARE_OUTPUTS": self.compare_outputs,
                "PCC_THRESHOLD": self.pcc_threshold,
            },
            "CLASSES": {
                "THING": self.thing_classes,
                "STUFF": self.stuff_classes,
                "NAMES": self.class_names,
            },
        }

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to: {yaml_path}")


class DualPipelineDemo:
    """Enhanced demo supporting both PyTorch and TTNN pipelines with comparison"""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.torch_model = None
        self.ttnn_model = None
        self.ttnn_device = None

        # Initialize preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std)
                if config.normalize_enabled
                else transforms.Lambda(lambda x: x),
            ]
        )

        # Color palette for visualization
        self.colors = self._get_cityscapes_colors()

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    def _get_cityscapes_colors(self) -> np.ndarray:
        """Get Cityscapes color palette"""
        return np.array(
            [
                [128, 64, 128],  # road
                [244, 35, 232],  # sidewalk
                [70, 70, 70],  # building
                [102, 102, 156],  # wall
                [190, 153, 153],  # fence
                [153, 153, 153],  # pole
                [250, 170, 30],  # traffic light
                [220, 220, 0],  # traffic sign
                [107, 142, 35],  # vegetation
                [152, 251, 152],  # terrain
                [70, 130, 180],  # sky
                [220, 20, 60],  # person
                [255, 0, 0],  # rider
                [0, 0, 142],  # car
                [0, 0, 70],  # truck
                [0, 60, 100],  # bus
                [0, 80, 100],  # train
                [0, 0, 230],  # motorcycle
                [119, 11, 32],  # bicycle
            ]
        )

    def initialize_torch_model(self):
        """Initialize PyTorch model"""
        if not self.config.run_torch_pipeline:
            return

        logger.info("Initializing PyTorch Panoptic DeepLab model...")

        self.torch_model = TorchPanopticDeepLab(
            num_classes=self.config.num_classes,
            thing_classes=self.config.thing_classes,
            stuff_classes=self.config.stuff_classes,
            center_threshold=self.config.center_threshold,
            nms_kernel=self.config.nms_kernel,
            top_k_instance=self.config.top_k_instances,
        ).eval()

        # Load weights if provided
        if self.config.weights_path and os.path.exists(self.config.weights_path):
            logger.info(f"Loading PyTorch weights from: {self.config.weights_path}")
            # checkpoint = torch.load(self.config.weights_path, map_location='cpu', weights_only=False)
            with open(self.config.weights_path, "rb") as f:
                checkpoint = pickle.load(f, encoding="latin1")

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            for k, v in state_dict.items():
                if isinstance(v, np.ndarray):
                    state_dict[k] = torch.from_numpy(v)

            self.torch_model.load_state_dict(state_dict, strict=False)
            logger.info("PyTorch weights loaded successfully")
        else:
            logger.warning("No weights provided - using random initialization")

        logger.info("PyTorch model initialized")

    def initialize_ttnn_model(self):
        """Initialize TTNN model"""
        if not self.config.run_ttnn_pipeline:
            return

        logger.info("Initializing TTNN Panoptic DeepLab model...")

        # Initialize TT device
        self.ttnn_device = ttnn.open_device(device_id=self.config.device_id, l1_small_size=65536)

        # Setup mesh mappers
        self._setup_mesh_mappers()

        # Create reference model for parameter extraction
        if self.torch_model is not None:
            reference_model = self.torch_model
        else:
            reference_model = TorchPanopticDeepLab(
                num_classes=self.config.num_classes,
                thing_classes=self.config.thing_classes,
                stuff_classes=self.config.stuff_classes,
            ).eval()

            if self.config.weights_path and os.path.exists(self.config.weights_path):
                checkpoint = torch.load(self.config.weights_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
                reference_model.load_state_dict(state_dict, strict=False)

        # Preprocess model parameters
        logger.info("Preprocessing model parameters for TTNN...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Model configuration for TTNN
        model_config = {
            "MATH_FIDELITY": getattr(ttnn.MathFidelity, self.config.math_fidelity),
            "WEIGHTS_DTYPE": getattr(ttnn, self.config.weights_dtype),
            "ACTIVATIONS_DTYPE": getattr(ttnn, self.config.activations_dtype),
        }

        # Create TTNN model
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
            num_classes=self.config.num_classes,
            thing_classes=self.config.thing_classes,
            stuff_classes=self.config.stuff_classes,
        )

        logger.info("TTNN model initialized")

    def _setup_mesh_mappers(self):
        """Setup mesh mappers for multi-device support"""
        if self.ttnn_device.get_num_devices() != 1:
            self.inputs_mesh_mapper = ttnn.ShardTensorToMesh(self.ttnn_device, dim=0)
            self.weights_mesh_mapper = None
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.ttnn_device, dim=0)
        else:
            self.inputs_mesh_mapper = None
            self.weights_mesh_mapper = None
            self.output_mesh_composer = None

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, ttnn.Tensor, np.ndarray, Tuple[int, int]]:
        """Preprocess image for both PyTorch and TTNN"""
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        original_array = np.array(image)

        # Resize to model input size
        target_size = (self.config.input_width, self.config.input_height)  # PIL expects (width, height)
        image_resized = image.resize(target_size)

        # PyTorch preprocessing
        torch_tensor = self.preprocess(image_resized).unsqueeze(0)  # Add batch dimension
        torch_tensor = torch_tensor.to(torch.float32)

        # TTNN preprocessing
        ttnn_tensor = None
        if self.config.run_ttnn_pipeline:
            ttnn_tensor = ttnn.from_torch(
                torch_tensor.permute(0, 2, 3, 1),  # BCHW -> BHWC
                dtype=ttnn.bfloat16,
                device=self.ttnn_device,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        return torch_tensor, ttnn_tensor, original_array, original_size

    def run_torch_inference(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run PyTorch inference"""
        if not self.config.run_torch_pipeline:
            return {}

        logger.info("Running PyTorch inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs = self.torch_model(input_tensor)

        inference_time = time.time() - start_time
        logger.info(f"PyTorch inference completed in {inference_time:.4f}s")

        return outputs

    def run_ttnn_inference(self, input_tensor: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        """Run TTNN inference"""
        if not self.config.run_ttnn_pipeline:
            return {}

        logger.info("Running TTNN inference...")
        start_time = time.time()

        outputs = self.ttnn_model(
            input_tensor,
            self.ttnn_device,
            batch_size=1,
            input_height_1=self.config.input_height,
            input_width_1=self.config.input_width,
            input_height_2=self.config.input_height // 2,
            input_width_2=self.config.input_width // 2,
            input_height_3=self.config.input_height // 4,
            input_width_3=self.config.input_width // 4,
        )

        inference_time = time.time() - start_time
        logger.info(f"TTNN inference completed in {inference_time:.4f}s")

        return outputs

    def postprocess_outputs(
        self, torch_outputs: Dict, ttnn_outputs: Dict, original_size: Tuple[int, int]
    ) -> Dict[str, Dict]:
        """Postprocess outputs from both pipelines - IMPROVED VERSION"""
        results = {"torch": {}, "ttnn": {}}

        # Process PyTorch outputs
        if torch_outputs:
            logger.info("Processing PyTorch outputs...")
            for key, tensor in torch_outputs.items():
                if isinstance(tensor, torch.Tensor):
                    try:
                        np_array = tensor.squeeze(0).cpu().numpy()

                        if key == "semantic_logits":
                            semantic_pred = np.argmax(np_array, axis=0)
                            results["torch"]["semantic_pred"] = cv2.resize(
                                semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                            )
                            logger.debug(f"PyTorch semantic_pred shape: {results['torch']['semantic_pred'].shape}")

                        elif key == "center_heatmap":
                            if len(np_array.shape) == 3 and np_array.shape[0] == 1:
                                center_data = np_array[0]  # Remove channel dim
                            elif len(np_array.shape) == 2:
                                center_data = np_array
                            else:
                                center_data = np_array  # Use as is

                            results["torch"]["center_heatmap"] = cv2.resize(
                                center_data, original_size, interpolation=cv2.INTER_LINEAR
                            )
                            logger.debug(f"PyTorch center_heatmap shape: {results['torch']['center_heatmap'].shape}")

                        elif key == "offset_map":
                            if len(np_array.shape) == 3 and np_array.shape[0] == 2:
                                results["torch"]["offset_map"] = np.stack(
                                    [
                                        cv2.resize(np_array[0], original_size, interpolation=cv2.INTER_LINEAR),
                                        cv2.resize(np_array[1], original_size, interpolation=cv2.INTER_LINEAR),
                                    ]
                                )
                            else:
                                logger.warning(f"Unexpected offset_map shape: {np_array.shape}")

                        elif key == "panoptic_pred":
                            if len(np_array.shape) == 3:
                                np_array = np_array.squeeze()  # Remove single dimensions
                            results["torch"]["panoptic_pred"] = cv2.resize(
                                np_array.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                            )
                            logger.debug(f"PyTorch panoptic_pred shape: {results['torch']['panoptic_pred'].shape}")

                    except Exception as e:
                        logger.error(f"Error processing PyTorch output {key}: {e}")

        # Process TTNN outputs
        if ttnn_outputs:
            logger.info("Processing TTNN outputs...")
            for key, tensor in ttnn_outputs.items():
                if hasattr(tensor, "shape"):
                    try:
                        # Convert TTNN to torch tensor
                        torch_tensor = ttnn.to_torch(
                            tensor, device=self.ttnn_device, mesh_composer=self.output_mesh_composer
                        )

                        # Reshape to proper format
                        reshaped_tensor = self._reshape_ttnn_output(torch_tensor, key)
                        np_array = reshaped_tensor.squeeze(0).cpu().float().numpy()

                        if key == "semantic_logits":
                            semantic_pred = np.argmax(np_array, axis=0)
                            results["ttnn"]["semantic_pred"] = cv2.resize(
                                semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                            )
                            logger.debug(f"TTNN semantic_pred shape: {results['ttnn']['semantic_pred'].shape}")

                        elif key == "center_heatmap":
                            if len(np_array.shape) == 3 and np_array.shape[0] == 1:
                                center_data = np_array[0]  # Remove channel dim
                            elif len(np_array.shape) == 2:
                                center_data = np_array
                            else:
                                center_data = np_array

                            results["ttnn"]["center_heatmap"] = cv2.resize(
                                center_data, original_size, interpolation=cv2.INTER_LINEAR
                            )
                            logger.debug(f"TTNN center_heatmap shape: {results['ttnn']['center_heatmap'].shape}")

                        elif key == "offset_map":
                            if len(np_array.shape) == 3 and np_array.shape[0] == 2:
                                results["ttnn"]["offset_map"] = np.stack(
                                    [
                                        cv2.resize(np_array[0], original_size, interpolation=cv2.INTER_LINEAR),
                                        cv2.resize(np_array[1], original_size, interpolation=cv2.INTER_LINEAR),
                                    ]
                                )
                            else:
                                logger.warning(f"Unexpected TTNN offset_map shape: {np_array.shape}")

                        elif key in ["panoptic_pred_ttnn", "panoptic_pred"]:
                            if len(np_array.shape) > 2:
                                np_array = np_array.squeeze()  # Remove single dimensions
                            results["ttnn"]["panoptic_pred"] = cv2.resize(
                                np_array.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
                            )
                            logger.debug(f"TTNN panoptic_pred shape: {results['ttnn']['panoptic_pred'].shape}")

                    except Exception as e:
                        logger.error(f"Error processing TTNN output {key}: {e}")

        return results

    def _reshape_ttnn_output(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Reshape TTNN output tensor to proper format - IMPROVED VERSION"""

        logger.debug(f"Reshaping TTNN output for {key}: input shape = {tensor.shape}")

        if len(tensor.shape) == 4:  # BHWC format from TTNN
            B, H, W, C = tensor.shape

            if key == "semantic_logits":
                # Should have num_classes channels
                expected_c = self.config.num_classes
                if C == expected_c:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Try to reshape if flattened
                    total_elements = B * H * W * C
                    expected_h = self.config.input_height // 4  # Typical output stride
                    expected_w = self.config.input_width // 4
                    if total_elements == B * expected_h * expected_w * expected_c:
                        tensor = tensor.reshape(B, expected_h, expected_w, expected_c)
                        result = tensor.permute(0, 3, 1, 2)
                    else:
                        logger.warning(f"Unexpected semantic_logits shape: {tensor.shape}")
                        result = tensor

            elif key == "center_heatmap":
                # Should have 1 channel
                if C == 1:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Take first channel or reshape
                    if C > 1:
                        tensor = tensor[:, :, :, :1]  # Take first channel
                    result = tensor.permute(0, 3, 1, 2)

            elif key == "offset_map":
                # Should have 2 channels (x, y offsets)
                if C == 2:
                    result = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    logger.warning(f"Unexpected offset_map channels: {C}, expected 2")
                    result = tensor

            elif key == "panoptic_pred_ttnn" or key == "panoptic_pred":
                # Single channel prediction map
                if C == 1:
                    result = tensor.permute(0, 3, 1, 2).squeeze(1)  # BHWC -> BHW
                else:
                    # Take first channel
                    result = tensor[:, :, :, 0]  # BHW
            else:
                result = tensor

        elif len(tensor.shape) == 3:  # BHW format
            result = tensor

        else:
            logger.warning(f"Unexpected tensor shape for {key}: {tensor.shape}")
            result = tensor

        logger.debug(f"Reshaped TTNN output for {key}: output shape = {result.shape}")
        return result

    def compare_outputs(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Compare PyTorch and TTNN outputs using PCC - ENHANCED VERSION"""
        if not (self.config.compare_outputs and "torch" in results and "ttnn" in results):
            return {}

        logger.info("Comparing PyTorch and TTNN outputs...")
        pcc_scores = {}

        for key in ["semantic_pred", "center_heatmap", "offset_map", "panoptic_pred"]:
            if key in results["torch"] and key in results["ttnn"]:
                torch_output = results["torch"][key]
                ttnn_output = results["ttnn"][key]

                # Debug shape information
                logger.debug(f"Comparing {key}:")
                logger.debug(f"  PyTorch shape: {torch_output.shape}")
                logger.debug(f"  TTNN shape: {ttnn_output.shape}")

                # Handle different shaped arrays
                if torch_output.shape != ttnn_output.shape:
                    logger.warning(f"  Shape mismatch for {key}: {torch_output.shape} vs {ttnn_output.shape}")
                    # Try to make compatible
                    if key == "offset_map":
                        if len(torch_output.shape) == 3 and len(ttnn_output.shape) == 2:
                            # Flatten both to same shape
                            torch_flat = torch_output.flatten()
                            ttnn_flat = ttnn_output.flatten()
                        else:
                            continue
                    else:
                        # Flatten both arrays
                        torch_flat = torch_output.flatten()
                        ttnn_flat = ttnn_output.flatten()

                        # Truncate to same length if needed
                        min_len = min(len(torch_flat), len(ttnn_flat))
                        torch_flat = torch_flat[:min_len]
                        ttnn_flat = ttnn_flat[:min_len]
                else:
                    torch_flat = torch_output.flatten()
                    ttnn_flat = ttnn_output.flatten()

                # Calculate statistics
                logger.debug(f"  PyTorch stats: mean={torch_flat.mean():.4f}, std={torch_flat.std():.4f}")
                logger.debug(f"  TTNN stats: mean={ttnn_flat.mean():.4f}, std={ttnn_flat.std():.4f}")

                # Calculate PCC
                if len(torch_flat) == len(ttnn_flat) and len(torch_flat) > 1:
                    # Remove any NaN or inf values
                    valid_mask = np.isfinite(torch_flat) & np.isfinite(ttnn_flat)
                    if valid_mask.sum() > 1:
                        torch_clean = torch_flat[valid_mask]
                        ttnn_clean = ttnn_flat[valid_mask]

                        correlation_matrix = np.corrcoef(torch_clean, ttnn_clean)
                        pcc = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
                        pcc_scores[key] = pcc

                        status = "PASS" if pcc >= self.config.pcc_threshold else "FAIL"
                        logger.info(f"  {key}: PCC = {pcc:.4f} ({status})")
                    else:
                        logger.warning(f"  {key}: No valid values for comparison")
                else:
                    logger.warning(f"  {key}: Cannot compare - length mismatch or insufficient data")

        return pcc_scores

    def visualize_results(self, original_image: np.ndarray, results: Dict, save_path: str):
        """Create comprehensive visualization comparing both pipelines"""
        logger.info("Creating visualization...")

        # Determine subplot layout based on available results
        has_torch = "torch" in results and results["torch"]
        has_ttnn = "ttnn" in results and results["ttnn"]

        if has_torch and has_ttnn:
            # 3 rows, 3 columns layout for dual pipeline
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            pipelines = ["torch", "ttnn"]
        elif has_torch or has_ttnn:
            # 2 rows, 3 columns for single pipeline
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            pipelines = ["torch"] if has_torch else ["ttnn"]
        else:
            logger.warning("No results to visualize")
            return

        # Ensure axes is 2D array for consistent indexing
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        # Original image (top-middle)
        axes[0, 1].imshow(original_image)
        axes[0, 1].set_title("Original Image", fontsize=12)
        axes[0, 1].axis("off")
        axes[0, 0].axis("off")  # Top-left empty
        axes[0, 2].axis("off")  # Top-right empty

        # For each pipeline, fill the visualization
        for i, pipeline in enumerate(pipelines):
            if pipeline not in results:
                continue

            pipeline_results = results[pipeline]

            if len(pipelines) == 2:
                # Dual pipeline layout
                row_base = i + 1  # torch=1, ttnn=2

                # Semantic segmentation
                if "semantic_pred" in pipeline_results:
                    semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                    axes[row_base, 0].imshow(semantic_colored)
                    axes[row_base, 0].set_title(f"{pipeline.upper()} Semantic", fontsize=10)
                    axes[row_base, 0].axis("off")

                # Instance centers
                if "center_heatmap" in pipeline_results:
                    axes[row_base, 1].imshow(pipeline_results["center_heatmap"], cmap="hot", alpha=0.7)
                    axes[row_base, 1].imshow(original_image, alpha=0.3)
                    axes[row_base, 1].set_title(f"{pipeline.upper()} Centers", fontsize=10)
                    axes[row_base, 1].axis("off")

                # Panoptic segmentation
                if "panoptic_pred" in pipeline_results:
                    panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                    axes[row_base, 2].imshow(panoptic_colored)
                    axes[row_base, 2].set_title(f"{pipeline.upper()} Panoptic", fontsize=10)
                    axes[row_base, 2].axis("off")

            else:
                # Single pipeline layout
                pipeline_results = results[pipeline]

                # Semantic segmentation (top-middle)
                if "semantic_pred" in pipeline_results:
                    semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                    axes[0, 1].imshow(semantic_colored)
                    axes[0, 1].set_title(f"{pipeline.upper()} Semantic", fontsize=12)
                    axes[0, 1].axis("off")

                # Instance centers (top-right)
                if "center_heatmap" in pipeline_results:
                    axes[0, 2].imshow(pipeline_results["center_heatmap"], cmap="hot", alpha=0.7)
                    axes[0, 2].imshow(original_image, alpha=0.3)
                    axes[0, 2].set_title(f"{pipeline.upper()} Centers", fontsize=12)
                    axes[0, 2].axis("off")

                # Panoptic segmentation (bottom-left)
                if "panoptic_pred" in pipeline_results:
                    panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                    axes[1, 0].imshow(panoptic_colored)
                    axes[1, 0].set_title(f"{pipeline.upper()} Panoptic", fontsize=12)
                    axes[1, 0].axis("off")

                # Overlay panoptic on original (bottom-middle)
                if "panoptic_pred" in pipeline_results:
                    axes[1, 1].imshow(original_image)
                    axes[1, 1].imshow(panoptic_colored, alpha=0.6)
                    axes[1, 1].set_title("Panoptic Overlay", fontsize=12)
                    axes[1, 1].axis("off")

                # Hide unused subplot (bottom-right)
                axes[1, 2].axis("off")

        # Hide any remaining unused subplots
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                if not axes[i, j].get_images() and not axes[i, j].get_children():
                    axes[i, j].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved to: {save_path}")

    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation map to colored image"""
        colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for class_id in range(self.config.num_classes):
            mask = segmentation == class_id
            if class_id < len(self.colors):
                colored[mask] = self.colors[class_id]
        return colored

    def _colorize_panoptic(self, panoptic: np.ndarray) -> np.ndarray:
        """Convert panoptic prediction to colored image"""
        colored = np.zeros((*panoptic.shape, 3), dtype=np.uint8)

        # Color stuff classes
        for stuff_class in self.config.stuff_classes:
            mask = panoptic == stuff_class
            if stuff_class < len(self.colors):
                colored[mask] = self.colors[stuff_class]

        # Color thing instances with unique colors
        instance_ids = np.unique(panoptic)
        for instance_id in instance_ids:
            if instance_id >= 1000:  # Instance IDs start from 1000
                mask = panoptic == instance_id
                # Generate unique color for instance
                np.random.seed(instance_id)
                instance_color = np.random.randint(0, 255, 3)
                colored[mask] = instance_color

        return colored

    def save_results(self, results: Dict, original_image: np.ndarray, output_dir: str, filename: str):
        """Save all results to output directory"""
        os.makedirs(output_dir, exist_ok=True)

        # Save original image
        original_path = os.path.join(output_dir, f"{filename}_original.png")
        Image.fromarray(original_image).save(original_path)

        # Save results for each pipeline
        for pipeline, pipeline_results in results.items():
            pipeline_dir = os.path.join(output_dir, pipeline)
            os.makedirs(pipeline_dir, exist_ok=True)

            # Save semantic segmentation
            if "semantic_pred" in pipeline_results:
                semantic_colored = self._colorize_segmentation(pipeline_results["semantic_pred"])
                semantic_path = os.path.join(pipeline_dir, f"{filename}_semantic.png")
                Image.fromarray(semantic_colored).save(semantic_path)

                # Save raw semantic prediction
                raw_semantic_path = os.path.join(pipeline_dir, f"{filename}_semantic_raw.npy")
                np.save(raw_semantic_path, pipeline_results["semantic_pred"])

            # Save center heatmap
            if "center_heatmap" in pipeline_results:
                center_path = os.path.join(pipeline_dir, f"{filename}_centers.png")
                center_normalized = (pipeline_results["center_heatmap"] * 255).astype(np.uint8)
                Image.fromarray(center_normalized, mode="L").save(center_path)

                # Save raw center heatmap
                raw_center_path = os.path.join(pipeline_dir, f"{filename}_centers_raw.npy")
                np.save(raw_center_path, pipeline_results["center_heatmap"])

            # Save offset map
            if "offset_map" in pipeline_results:
                offset_path = os.path.join(pipeline_dir, f"{filename}_offset_raw.npy")
                np.save(offset_path, pipeline_results["offset_map"])

            # Save panoptic segmentation
            if "panoptic_pred" in pipeline_results:
                panoptic_colored = self._colorize_panoptic(pipeline_results["panoptic_pred"])
                panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic.png")
                Image.fromarray(panoptic_colored).save(panoptic_path)

                # Save raw panoptic prediction
                raw_panoptic_path = os.path.join(pipeline_dir, f"{filename}_panoptic_raw.npy")
                np.save(raw_panoptic_path, pipeline_results["panoptic_pred"])

        logger.info(f"Results saved to: {output_dir}")

    def run_demo(self, image_path: str, output_dir: str):
        """Run complete demo pipeline"""
        logger.info(f"Starting demo for image: {image_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize models
        if self.config.run_torch_pipeline:
            self.initialize_torch_model()

        if self.config.run_ttnn_pipeline:
            self.initialize_ttnn_model()

        # Preprocess image
        torch_input, ttnn_input, original_image, original_size = self.preprocess_image(image_path)

        # Run inference
        torch_outputs = self.run_torch_inference(torch_input) if self.config.run_torch_pipeline else {}
        ttnn_outputs = self.run_ttnn_inference(ttnn_input) if self.config.run_ttnn_pipeline else {}

        # Postprocess results
        results = self.postprocess_outputs(torch_outputs, ttnn_outputs, original_size)

        # Compare outputs if both pipelines ran
        pcc_scores = self.compare_outputs(results)

        # Generate filename
        base_name = Path(image_path).stem

        # Save individual results
        self.save_results(results, original_image, output_dir, base_name)

        # Create visualization
        viz_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        self.visualize_results(original_image, results, viz_path)

        # Save metadata and results summary
        self._save_metadata(image_path, results, pcc_scores, output_dir, base_name)

        logger.info(f"Demo completed! Results saved to: {output_dir}")

        # Cleanup
        if ttnn_input is not None:
            ttnn.deallocate(ttnn_input)

        for tensor in ttnn_outputs.values():
            if hasattr(tensor, "deallocate"):
                ttnn.deallocate(tensor)

    def _save_metadata(self, image_path: str, results: Dict, pcc_scores: Dict, output_dir: str, filename: str):
        """Save metadata and comparison results"""
        metadata = {
            "image_path": image_path,
            "config": asdict(self.config),
            "results": {
                "pipelines_run": list(results.keys()),
                "pcc_scores": pcc_scores,
            },
            "output_files": {
                "visualization": f"{filename}_comparison.png",
                "original": f"{filename}_original.png",
            },
        }

        # Add pipeline-specific metadata
        for pipeline in results.keys():
            metadata["output_files"][pipeline] = {
                "semantic": f"{pipeline}/{filename}_semantic.png",
                "centers": f"{pipeline}/{filename}_centers.png",
                "panoptic": f"{pipeline}/{filename}_panoptic.png",
            }

        metadata_path = os.path.join(output_dir, f"{filename}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create summary report
        summary_path = os.path.join(output_dir, f"{filename}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Panoptic DeepLab Demo Results\n")
            f.write(f"============================\n\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Input Size: {self.config.input_height}x{self.config.input_width}\n")
            f.write(f"Pipelines Run: {', '.join(results.keys())}\n\n")

            if pcc_scores:
                f.write(f"PCC Comparison Results:\n")
                for key, score in pcc_scores.items():
                    status = "PASS" if score >= self.config.pcc_threshold else "FAIL"
                    f.write(f"  {key}: {score:.4f} ({status})\n")

                avg_pcc = np.mean(list(pcc_scores.values()))
                overall_status = "PASS" if avg_pcc >= self.config.pcc_threshold else "FAIL"
                f.write(f"\nOverall Average PCC: {avg_pcc:.4f} ({overall_status})\n")

    def cleanup(self):
        """Cleanup resources"""
        if self.ttnn_device is not None:
            ttnn.close_device(self.ttnn_device)
            logger.info("TTNN device closed")


def create_sample_configs():
    """Create sample configuration files"""

    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Basic Cityscapes config
    basic_config = DemoConfig(
        weights_path=None,  # No weights for basic demo
        input_height=512,
        input_width=1024,
        run_torch_pipeline=True,
        run_ttnn_pipeline=True,
        compare_outputs=True,
    )
    basic_config.to_yaml("configs/demo_basic.yaml")

    # High resolution config
    hr_config = DemoConfig(
        weights_path="models/panoptic_deeplab_r52_cityscapes.pkl",
        input_height=1024,
        input_width=2048,
        center_threshold=0.15,
        nms_kernel=9,
        math_fidelity="HiFi2",
    )
    hr_config.to_yaml("configs/demo_high_res.yaml")

    # Fast inference config
    fast_config = DemoConfig(
        input_height=256,
        input_width=512,
        center_threshold=0.2,
        top_k_instances=100,
        math_fidelity="LoFi",
        run_torch_pipeline=False,  # Only TTNN for speed
        compare_outputs=False,
    )
    fast_config.to_yaml("configs/demo_fast.yaml")

    # Comparison config (for validation)
    comparison_config = DemoConfig(
        weights_path="models/panoptic_deeplab_r52_cityscapes.pkl",
        input_height=512,
        input_width=1024,
        run_torch_pipeline=True,
        run_ttnn_pipeline=True,
        compare_outputs=True,
        pcc_threshold=0.95,
        save_semantic=True,
        save_instance=True,
        save_panoptic=True,
        save_visualization=True,
        save_comparison=True,
    )
    comparison_config.to_yaml("configs/demo_comparison.yaml")

    logger.info(f"Sample configurations created in: {configs_dir}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Enhanced TT Panoptic DeepLab Demo with YAML Config")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/demo_basic.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for results")
    parser.add_argument("--create-configs", action="store_true", help="Create sample configuration files and exit")

    # Override options
    parser.add_argument("--weights", type=str, help="Override model weights path")
    parser.add_argument(
        "--input-size", nargs=2, type=int, metavar=("H", "W"), help="Override input size (height width)"
    )
    parser.add_argument("--torch-only", action="store_true", help="Run only PyTorch pipeline")
    parser.add_argument("--ttnn-only", action="store_true", help="Run only TTNN pipeline")
    parser.add_argument("--device-id", type=int, help="Override TT device ID")

    args = parser.parse_args()

    # Create sample configs if requested
    if args.create_configs:
        create_sample_configs()
        return 0

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        return 1

    # Load configuration
    if os.path.exists(args.config):
        config = DemoConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using default configuration")
        config = DemoConfig()

    # Apply command line overrides
    if args.weights:
        config.weights_path = args.weights
    if args.input_size:
        config.input_height, config.input_width = args.input_size
    if args.torch_only:
        config.run_torch_pipeline = True
        config.run_ttnn_pipeline = False
        config.compare_outputs = False
    if args.ttnn_only:
        config.run_torch_pipeline = False
        config.run_ttnn_pipeline = True
        config.compare_outputs = False
    if args.device_id is not None:
        config.device_id = args.device_id

    # Validate configuration
    if config.weights_path and not os.path.exists(config.weights_path):
        logger.warning(f"Weights file not found: {config.weights_path}")
        logger.info("Proceeding with random initialization")

    # Initialize demo
    logger.info("=== Enhanced Panoptic DeepLab Demo ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Input Size: {config.input_height}x{config.input_width}")
    logger.info(f"PyTorch: {'ON' if config.run_torch_pipeline else 'OFF'}")
    logger.info(f"TTNN: {'ON' if config.run_ttnn_pipeline else 'OFF'}")

    try:
        demo = DualPipelineDemo(config)
        demo.run_demo(args.input, args.output)
        logger.info("Demo completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        try:
            demo.cleanup()
        except:
            pass


if __name__ == "__main__":
    exit(main())
