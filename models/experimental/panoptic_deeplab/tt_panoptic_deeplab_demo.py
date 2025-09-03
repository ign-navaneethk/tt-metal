# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent Panoptic DeepLab Demo Script

This demo script provides a complete inference pipeline for Panoptic DeepLab
on Tenstorrent hardware, similar to the original PyTorch implementation.

Usage:
    python tt_panoptic_deeplab_demo.py --input <image_path> --output <output_dir>
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger
import json

import torch
import torchvision.transforms as transforms
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

# Import TT Panoptic DeepLab modules
from models.experimental.panoptic_deeplab.tt.tt_panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import PanopticDeepLab as TorchPanopticDeepLab


class TTNNPanopticDeepLabDemo:
    """
    Demo class for TT Panoptic DeepLab inference with visualization
    """

    def __init__(
        self,
        device: ttnn.Device,
        model_config: Dict,
        input_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 19,
        thing_classes: list = None,
        stuff_classes: list = None,
    ):
        self.device = device
        self.model_config = model_config
        self.input_size = input_size
        self.num_classes = num_classes

        # Cityscapes class definitions (default)
        if thing_classes is None:
            # Thing classes (have instances)
            self.thing_classes = [
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
            ]  # person, rider, car, truck, bus, train, motorcycle, bicycle
        else:
            self.thing_classes = thing_classes

        if stuff_classes is None:
            # Stuff classes (no instances)
            self.stuff_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # road, sidewalk, building, etc.
        else:
            self.stuff_classes = stuff_classes

        # Cityscapes color palette for visualization
        self.colors = self._get_cityscapes_colors()

        # Image preprocessing
        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Initialize model (will be loaded later)
        self.model = None
        self.model_initialized = False

    def _get_cityscapes_colors(self) -> np.ndarray:
        """Get Cityscapes color palette for visualization"""
        colors = np.array(
            [
                [128, 64, 128],  # 0: road
                [244, 35, 232],  # 1: sidewalk
                [70, 70, 70],  # 2: building
                [102, 102, 156],  # 3: wall
                [190, 153, 153],  # 4: fence
                [153, 153, 153],  # 5: pole
                [250, 170, 30],  # 6: traffic light
                [220, 220, 0],  # 7: traffic sign
                [107, 142, 35],  # 8: vegetation
                [152, 251, 152],  # 9: terrain
                [70, 130, 180],  # 10: sky
                [220, 20, 60],  # 11: person
                [255, 0, 0],  # 12: rider
                [0, 0, 142],  # 13: car
                [0, 0, 70],  # 14: truck
                [0, 60, 100],  # 15: bus
                [0, 80, 100],  # 16: train
                [0, 0, 230],  # 17: motorcycle
                [119, 11, 32],  # 18: bicycle
            ]
        )
        return colors

    def initialize_model(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the TT Panoptic DeepLab model

        Args:
            checkpoint_path: Path to model checkpoint (optional)
        """
        logger.info("Initializing TT Panoptic DeepLab model...")

        # Create reference torch model for parameter extraction
        torch_model = TorchPanopticDeepLab(
            num_classes=self.num_classes, thing_classes=self.thing_classes, stuff_classes=self.stuff_classes
        ).eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            torch_model.load_state_dict(checkpoint["model_state_dict"])

        # Preprocess model parameters for TTNN
        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = self._get_mesh_mappers()

        logger.info("Preprocessing model parameters...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=None,
        )

        # Initialize TTNN model
        self.model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=self.model_config,
            num_classes=self.num_classes,
            thing_classes=self.thing_classes,
            stuff_classes=self.stuff_classes,
        )

        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer
        self.model_initialized = True

        logger.info("Model initialization completed!")

    def _get_mesh_mappers(self):
        """Get mesh mappers for multi-device support"""
        if self.device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def preprocess_image(self, image_path: str) -> Tuple[ttnn.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Preprocess input image for TT model inference

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (ttnn_tensor, original_image_array, original_size)
        """
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        original_array = np.array(image)

        # Resize to model input size
        image_resized = image.resize(self.input_size[::-1])  # PIL expects (width, height)

        # Apply preprocessing transforms
        input_tensor = self.preprocess(image_resized).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(torch.bfloat16)

        # Convert to TTNN format (BHWC)
        ttnn_tensor = ttnn.from_torch(
            input_tensor.permute(0, 2, 3, 1),  # BCHW -> BHWC
            dtype=ttnn.bfloat16,
            device=self.device,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        logger.info(f"Preprocessed image: {image_path}")
        logger.info(f"Original size: {original_size}, Model input size: {self.input_size}")

        return ttnn_tensor, original_array, original_size

    def run_inference(self, ttnn_input: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        """
        Run TT Panoptic DeepLab inference

        Args:
            ttnn_input: Preprocessed input tensor

        Returns:
            Dictionary containing model outputs
        """
        if not self.model_initialized:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        logger.info("Running TT Panoptic DeepLab inference...")

        start_time = time.time()

        # Run inference
        outputs = self.model(
            ttnn_input,
            self.device,
            batch_size=1,
            input_height_1=self.input_size[0] // 4,  # Backbone output size
            input_width_1=self.input_size[1] // 4,
            input_height_2=self.input_size[0] // 2,  # Intermediate feature size
            input_width_2=self.input_size[1] // 2,
            input_height_3=self.input_size[0],  # Final output size
            input_width_3=self.input_size[1],
        )

        inference_time = time.time() - start_time
        fps = 1.0 / inference_time

        logger.info(f"Inference completed in {inference_time:.4f}s ({fps:.2f} FPS)")

        return outputs

    def postprocess_outputs(
        self, outputs: Dict[str, ttnn.Tensor], original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess model outputs to numpy arrays and resize to original image size

        Args:
            outputs: Raw model outputs from TT inference
            original_size: Original image size (width, height)

        Returns:
            Dictionary containing postprocessed outputs as numpy arrays
        """
        logger.info("Postprocessing model outputs...")

        processed_outputs = {}

        # Convert TTNN tensors to torch tensors
        semantic_logits = ttnn.to_torch(
            outputs["semantic_logits"], device=self.device, mesh_composer=self.output_mesh_composer
        )

        center_heatmap = ttnn.to_torch(
            outputs["center_heatmap"], device=self.device, mesh_composer=self.output_mesh_composer
        )

        offset_map = ttnn.to_torch(outputs["offset_map"], device=self.device, mesh_composer=self.output_mesh_composer)

        panoptic_pred = ttnn.to_torch(
            outputs["panoptic_pred_ttnn"], device=self.device, mesh_composer=self.output_mesh_composer
        )

        # Reshape tensors from BHWC to BCHW format
        semantic_logits = self._reshape_to_bchw(semantic_logits, target_channels=self.num_classes)
        center_heatmap = self._reshape_to_bchw(center_heatmap, target_channels=1)
        offset_map = self._reshape_to_bchw(offset_map, target_channels=2)

        # Remove batch dimension and convert to numpy
        semantic_logits = semantic_logits.squeeze(0).cpu().float().numpy()  # (C, H, W)
        center_heatmap = center_heatmap.squeeze(0).cpu().float().numpy()  # (1, H, W)
        offset_map = offset_map.squeeze(0).cpu().float().numpy()  # (2, H, W)
        panoptic_pred = panoptic_pred.squeeze(0).cpu().float().numpy()  # (H, W)

        # Get semantic predictions
        semantic_pred = np.argmax(semantic_logits, axis=0)  # (H, W)

        # Resize outputs to original image size
        semantic_pred_resized = cv2.resize(
            semantic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
        )

        center_heatmap_resized = cv2.resize(center_heatmap[0], original_size, interpolation=cv2.INTER_LINEAR)

        offset_map_resized = np.stack(
            [
                cv2.resize(offset_map[0], original_size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(offset_map[1], original_size, interpolation=cv2.INTER_LINEAR),
            ],
            axis=0,
        )

        panoptic_pred_resized = cv2.resize(
            panoptic_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
        )

        processed_outputs = {
            "semantic_pred": semantic_pred_resized,
            "center_heatmap": center_heatmap_resized,
            "offset_map": offset_map_resized,
            "panoptic_pred": panoptic_pred_resized,
        }

        logger.info("Postprocessing completed!")
        return processed_outputs

    def _reshape_to_bchw(self, tensor: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Helper function to reshape BHWC tensor to BCHW format"""
        if len(tensor.shape) == 4:  # BHWC format
            B, H, W, C = tensor.shape
            if C == target_channels:
                return tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
            else:
                # Reshape flat tensor back to proper dimensions
                total_pixels = H * W
                expected_H = int(np.sqrt(total_pixels * target_channels / C))
                expected_W = total_pixels * target_channels // (expected_H * C)
                tensor = tensor.reshape(B, expected_H, expected_W, target_channels)
                return tensor.permute(0, 3, 1, 2)
        else:
            # Handle other tensor shapes
            return tensor

    def visualize_results(self, original_image: np.ndarray, outputs: Dict[str, np.ndarray], save_path: str):
        """
        Create visualization of panoptic segmentation results

        Args:
            original_image: Original input image
            outputs: Postprocessed model outputs
            save_path: Path to save visualization
        """
        logger.info("Creating visualizations...")

        semantic_pred = outputs["semantic_pred"]
        center_heatmap = outputs["center_heatmap"]
        offset_map = outputs["offset_map"]
        panoptic_pred = outputs["panoptic_pred"]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Semantic segmentation
        semantic_colored = self._colorize_segmentation(semantic_pred)
        axes[0, 1].imshow(semantic_colored)
        axes[0, 1].set_title("Semantic Segmentation")
        axes[0, 1].axis("off")

        # Instance center heatmap
        axes[0, 2].imshow(center_heatmap, cmap="hot", alpha=0.7)
        axes[0, 2].imshow(original_image, alpha=0.3)
        axes[0, 2].set_title("Instance Centers")
        axes[0, 2].axis("off")

        # Offset visualization (magnitude)
        offset_magnitude = np.sqrt(offset_map[0] ** 2 + offset_map[1] ** 2)
        axes[1, 0].imshow(offset_magnitude, cmap="viridis")
        axes[1, 0].set_title("Offset Magnitude")
        axes[1, 0].axis("off")

        # Panoptic segmentation
        panoptic_colored = self._colorize_panoptic(panoptic_pred)
        axes[1, 1].imshow(panoptic_colored)
        axes[1, 1].set_title("Panoptic Segmentation")
        axes[1, 1].axis("off")

        # Overlay panoptic on original
        axes[1, 2].imshow(original_image)
        axes[1, 2].imshow(panoptic_colored, alpha=0.6)
        axes[1, 2].set_title("Panoptic Overlay")
        axes[1, 2].axis("off")

        # Add legend for semantic classes
        legend_elements = [
            mpatches.Patch(color=self.colors[i] / 255.0, label=f"Class {i}")
            for i in range(min(len(self.colors), self.num_classes))
        ]
        fig.legend(handles=legend_elements, loc="right", bbox_to_anchor=(1.0, 0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved to: {save_path}")

    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation map to colored image"""
        colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for class_id in range(self.num_classes):
            mask = segmentation == class_id
            if class_id < len(self.colors):
                colored[mask] = self.colors[class_id]
        return colored

    def _colorize_panoptic(self, panoptic: np.ndarray) -> np.ndarray:
        """Convert panoptic prediction to colored image"""
        colored = np.zeros((*panoptic.shape, 3), dtype=np.uint8)

        # Color stuff classes with semantic colors
        for stuff_class in self.stuff_classes:
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

    def save_outputs(self, outputs: Dict[str, np.ndarray], output_dir: str, filename: str):
        """
        Save individual output files

        Args:
            outputs: Postprocessed outputs
            output_dir: Output directory
            filename: Base filename (without extension)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save semantic segmentation
        semantic_path = os.path.join(output_dir, f"{filename}_semantic.png")
        semantic_colored = self._colorize_segmentation(outputs["semantic_pred"])
        Image.fromarray(semantic_colored).save(semantic_path)

        # Save center heatmap
        center_path = os.path.join(output_dir, f"{filename}_centers.png")
        center_normalized = (outputs["center_heatmap"] * 255).astype(np.uint8)
        Image.fromarray(center_normalized, mode="L").save(center_path)

        # Save panoptic prediction
        panoptic_path = os.path.join(output_dir, f"{filename}_panoptic.png")
        panoptic_colored = self._colorize_panoptic(outputs["panoptic_pred"])
        Image.fromarray(panoptic_colored).save(panoptic_path)

        logger.info(f"Individual outputs saved to: {output_dir}")

    def run_demo(self, image_path: str, output_dir: str):
        """
        Run complete demo pipeline on a single image

        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
        """
        if not self.model_initialized:
            logger.error("Model not initialized. Call initialize_model() first.")
            return

        logger.info(f"Running demo on: {image_path}")

        # Preprocess image
        ttnn_input, original_image, original_size = self.preprocess_image(image_path)

        # Run inference
        raw_outputs = self.run_inference(ttnn_input)

        # Postprocess outputs
        processed_outputs = self.postprocess_outputs(raw_outputs, original_size)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        base_name = Path(image_path).stem

        # Save individual outputs
        self.save_outputs(processed_outputs, output_dir, base_name)

        # Create and save visualization
        viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        self.visualize_results(original_image, processed_outputs, viz_path)

        # Save metadata
        metadata = {
            "input_image": image_path,
            "original_size": original_size,
            "model_input_size": self.input_size,
            "num_classes": self.num_classes,
            "thing_classes": self.thing_classes,
            "stuff_classes": self.stuff_classes,
        }

        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Demo completed! Results saved to: {output_dir}")

        # Clean up tensors
        ttnn.deallocate(ttnn_input)
        for output_tensor in raw_outputs.values():
            if hasattr(output_tensor, "deallocate"):
                ttnn.deallocate(output_tensor)


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="TT Panoptic DeepLab Demo")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for results")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument(
        "--input_size", type=str, default="512,1024", help="Input size as height,width (default: 512,1024)"
    )
    parser.add_argument("--device_id", type=int, default=0, help="TT device ID (default: 0)")

    args = parser.parse_args()

    # Parse input size
    input_size = tuple(map(int, args.input_size.split(",")))

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        return

    # TT device setup
    logger.info("Initializing TT device...")
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=65536)

    # Model configuration
    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    try:
        # Initialize demo
        logger.info("Initializing TT Panoptic DeepLab Demo...")
        demo = TTNNPanopticDeepLabDemo(device=device, model_config=model_config, input_size=input_size)

        # Initialize model
        demo.initialize_model(checkpoint_path=args.checkpoint)

        # Run demo
        demo.run_demo(args.input, args.output)

        logger.info("Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    finally:
        # Clean up device
        ttnn.close_device(device)
        logger.info("TT device closed.")


if __name__ == "__main__":
    main()
