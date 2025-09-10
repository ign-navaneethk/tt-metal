# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import torch
from typing import Dict, List, Tuple

from models.experimental.panoptic_deeplab.tt.backbone import TTBackbone
from models.experimental.panoptic_deeplab.tt.decoder import TTDecoder, decoder_layer_optimisations


class TTPanopticDeepLabUnified:
    """
    Unified TTNN implementation of Panoptic DeepLab using modular decoder architecture.
    Combines backbone, semantic segmentation, and instance segmentation with panoptic fusion.
    """

    def __init__(
        self,
        parameters,
        model_config,
        num_classes: int = 19,
        thing_classes: List[int] = None,
        stuff_classes: List[int] = None,
        center_threshold: float = 0.1,
        nms_kernel: int = 7,
        top_k_instance: int = 200,
    ):
        self.model_config = model_config
        self.num_classes = num_classes
        self.thing_classes = thing_classes or []
        self.stuff_classes = stuff_classes or []

        # Panoptic fusion parameters
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k_instance = top_k_instance

        # Initialize backbone
        self.backbone = TTBackbone(parameters.backbone, model_config)

        # Initialize semantic segmentation decoder
        self.semantic_decoder = TTDecoder(
            parameters.semantic_decoder, model_config, layer_optimisations=decoder_layer_optimisations["sem_seg_head"]
        )

        # Initialize instance segmentation decoders
        self.instance_center_decoder = TTDecoder(
            parameters.instance_center_decoder,
            model_config,
            layer_optimisations=decoder_layer_optimisations["ins_embed_head_center"],
        )

        self.instance_offset_decoder = TTDecoder(
            parameters.instance_offset_decoder,
            model_config,
            layer_optimisations=decoder_layer_optimisations["ins_embed_head_offset"],
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        device,
    ) -> Dict[str, ttnn.Tensor]:
        """
        Forward pass of TTNN Panoptic DeepLab.

        Args:
            x: Input tensor of shape [B, H, W, C] in TTNN format
            device: TTNN device

        Returns:
            Dictionary containing:
            - semantic_logits: Semantic segmentation logits
            - center_heatmap: Instance center heatmap
            - offset_map: Instance offset map
            - panoptic_pred: Panoptic prediction
        """

        logger.debug("Running TT Panoptic DeepLab Unified forward pass")
        print("Running TT Panoptic DeepLab Unified forward pass")

        # Extract multi-scale features from backbone
        logger.debug("Running TTBackbone")
        features = self.backbone(x, device)
        print("Backbone features: ", features)
        # Extract the specific feature maps the decoders expect
        backbone_features = features["res_5"]  # 2048 channels for ASPP
        res3_features = features["res_3"]  # 512 channels for decoder
        res2_features = features["res_2"]  # 256 channels for decoder

        logger.debug(
            f"Backbone features shapes - res_5: {backbone_features.shape}, "
            f"res_3: {res3_features.shape}, res_2: {res2_features.shape}"
        )
        print("Backbone features shapes: ", backbone_features.shape, res3_features.shape, res2_features.shape)
        # Semantic segmentation branch
        logger.debug("Running semantic segmentation decoder")
        print("Running semantic segmentation decoder")
        semantic_logits = self.semantic_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,  # ASPP output channels
            device=device,
        )
        print("Semantic logits: ", semantic_logits)
        # Instance center prediction branch
        logger.debug("Running instance center decoder")
        print("Running instance center decoder")
        center_heatmap = self.instance_center_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,  # ASPP output channels
            device=device,
        )
        print("Center heatmap: ", center_heatmap)
        # Apply sigmoid to center heatmap
        center_heatmap = ttnn.sigmoid(center_heatmap)

        # Instance offset prediction branch
        logger.debug("Running instance offset decoder")
        print("Running instance offset decoder")
        offset_map = self.instance_offset_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,  # ASPP output channels
            device=device,
        )
        print("Offset map: ", offset_map)
        # Perform panoptic fusion
        logger.debug("Running panoptic fusion")
        panoptic_pred = self.panoptic_fusion_ttnn(semantic_logits, center_heatmap, offset_map, device)

        outputs = {
            "semantic_logits": semantic_logits,
            "center_heatmap": center_heatmap,
            "offset_map": offset_map,
            "panoptic_pred": panoptic_pred,
        }

        logger.debug("TT Panoptic DeepLab Unified forward pass completed")

        return outputs

    def panoptic_fusion_ttnn(
        self, semantic_logits: ttnn.Tensor, center_heatmap: ttnn.Tensor, offset_map: ttnn.Tensor, device: ttnn.Device
    ) -> ttnn.Tensor:
        """
        TTNN-based panoptic fusion (simplified version).
        For full panoptic fusion, post-processing on CPU is recommended.

        Args:
            semantic_logits: [B, H, W, num_classes] semantic predictions
            center_heatmap: [B, H, W, 1] instance center heatmap
            offset_map: [B, H, W, 2] instance offset map
            device: TTNN device

        Returns:
            panoptic_pred: Basic panoptic prediction tensor
        """

        logger.debug("Running TTNN panoptic fusion")

        # Get semantic predictions using argmax
        semantic_pred = ttnn.argmax(semantic_logits, dim=-1)

        # Apply threshold to center heatmap
        center_mask = ttnn.gt(center_heatmap, self.center_threshold)

        # For now, return semantic predictions as base panoptic output
        # Full panoptic fusion with instance grouping should be done on CPU
        panoptic_pred = semantic_pred

        logger.debug("TTNN panoptic fusion completed")

        return panoptic_pred

    def postprocess_panoptic_cpu(
        self, semantic_logits: ttnn.Tensor, center_heatmap: ttnn.Tensor, offset_map: ttnn.Tensor
    ) -> torch.Tensor:
        """
        CPU-based panoptic fusion for full functionality.
        This should be called after moving tensors from device to host.

        Args:
            semantic_logits: Semantic segmentation logits
            center_heatmap: Instance center heatmap
            offset_map: Instance offset map

        Returns:
            panoptic_pred: Full panoptic segmentation result
        """

        logger.debug("Running CPU panoptic fusion")

        # Convert TTNN tensors to torch tensors
        semantic_torch = ttnn.to_torch(semantic_logits)
        center_torch = ttnn.to_torch(center_heatmap)
        offset_torch = ttnn.to_torch(offset_map)

        # Get semantic predictions
        semantic_pred = torch.argmax(semantic_torch, dim=-1)  # [B, H, W]

        batch_size, height, width = semantic_pred.shape
        panoptic_pred = semantic_pred.clone()

        for b in range(batch_size):
            # Process each image in the batch
            sem_pred = semantic_pred[b]  # [H, W]
            center_heat = center_torch[b, :, :, 0]  # [H, W]
            offset = offset_torch[b].permute(2, 0, 1)  # [2, H, W]

            # Find instance centers using NMS
            centers = self._find_instance_centers_torch(center_heat)

            # Generate instance masks
            instance_masks = self._generate_instance_masks_torch(centers, offset, height, width)

            # Fuse semantic and instance predictions
            panoptic_img = self._fuse_semantic_instance_torch(sem_pred, instance_masks, centers)

            panoptic_pred[b] = panoptic_img

        logger.debug("CPU panoptic fusion completed")

        return panoptic_pred

    def _find_instance_centers_torch(self, center_heatmap: torch.Tensor) -> List[Tuple[int, int]]:
        """Find instance centers from center heatmap using NMS."""
        import torch.nn.functional as F

        # Apply threshold
        center_mask = center_heatmap > self.center_threshold

        # Apply NMS
        nms_heatmap = F.max_pool2d(
            center_heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=self.nms_kernel,
            stride=1,
            padding=self.nms_kernel // 2,
        ).squeeze()

        # Find local maxima
        center_mask = center_mask & (center_heatmap == nms_heatmap)

        # Get top-k centers
        center_coords = torch.nonzero(center_mask, as_tuple=False)
        center_scores = center_heatmap[center_mask]

        if len(center_coords) > self.top_k_instance:
            top_k_indices = torch.topk(center_scores, self.top_k_instance)[1]
            center_coords = center_coords[top_k_indices]

        return [(coord[0].item(), coord[1].item()) for coord in center_coords]

    def _generate_instance_masks_torch(
        self, centers: List[Tuple[int, int]], offset_map: torch.Tensor, height: int, width: int
    ) -> List[torch.Tensor]:
        """Generate instance masks from centers and offset map."""

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

        instance_masks = []

        for center_y, center_x in centers:
            # Calculate shifted coordinates using offset map
            shifted_y = y_coords + offset_map[0]  # [H, W]
            shifted_x = x_coords + offset_map[1]  # [H, W]

            # Calculate distance to center
            dist_y = shifted_y - center_y
            dist_x = shifted_x - center_x
            distance = torch.sqrt(dist_y**2 + dist_x**2)

            # Create instance mask (pixels that point to this center)
            mask = distance < 1.0  # Threshold for belonging to instance
            instance_masks.append(mask)

        return instance_masks

    def _fuse_semantic_instance_torch(
        self, semantic_pred: torch.Tensor, instance_masks: List[torch.Tensor], centers: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Fuse semantic and instance predictions."""
        panoptic_pred = semantic_pred.clone()

        instance_id = 1000  # Start instance IDs from 1000

        for mask, (center_y, center_x) in zip(instance_masks, centers):
            if mask.sum() < 32:  # Skip very small instances
                continue

            # Get semantic class at center
            center_class = semantic_pred[center_y, center_x].item()

            # Only process thing classes for instances
            if center_class in self.thing_classes:
                # Assign instance ID to mask region
                panoptic_pred[mask] = instance_id
                instance_id += 1

        return panoptic_pred
