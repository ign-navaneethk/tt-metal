# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab_segmentation_head import (
    PanopticDeeplabSemanticsSegmentationModel as SemSegTorch,
)
from models.experimental.panoptic_deeplab.reference.panoptic_deeplab_instance_segmentation import (
    PanopticDeeplabInstanceSegmentationModel as InsSegTorch,
)
from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone


class PanopticDeepLab(nn.Module):
    """
    Panoptic DeepLab model for panoptic segmentation (inference only).
    Combines semantic segmentation and instance segmentation with panoptic fusion.
    """

    def __init__(
        self,
        num_classes: int = 19,
        thing_classes: List[int] = None,
        stuff_classes: List[int] = None,
        center_threshold: float = 0.1,
        nms_kernel: int = 7,
        top_k_instance: int = 200,
        stuff_area_limit: int = 4096,
    ):
        super().__init__()

        self.backbone = ResNet52BackBone()
        self.semantic_head = SemSegTorch()
        self.instance_head = InsSegTorch()

        self.num_classes = num_classes
        self.thing_classes = thing_classes or []
        self.stuff_classes = stuff_classes or []

        # Panoptic fusion parameters
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k_instance = top_k_instance
        self.stuff_area_limit = stuff_area_limit

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Panoptic DeepLab.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Dictionary containing:
            - semantic_logits: Semantic segmentation logits [B, num_classes, H, W]
            - center_heatmap: Instance center heatmap [B, 1, H, W]
            - offset_map: Instance offset map [B, 2, H, W]
            - panoptic_pred: Panoptic prediction [B, H, W]
        """

        features = self.backbone(x)

        # Extract the specific feature maps your heads expect
        backbone_features = features["res_5"]  # 2048 channels for ASPP
        res3_features = features["res_3"]  # 512 channels for decoder
        res2_features = features["res_2"]  # 256 channels for decoder

        # Call semantic head with all required arguments
        semantic_logits = self.semantic_head(
            backbone_features, res3_features, res2_features  # x parameter  # res3 parameter  # res2 parameter
        )

        # Call instance head with all required arguments
        center_heatmap, offset_map = self.instance_head(
            backbone_features, res3_features, res2_features  # x parameter  # res3 parameter  # res2 parameter
        )

        # Perform panoptic fusion
        panoptic_pred = self.panoptic_fusion(semantic_logits, center_heatmap, offset_map)

        return {
            "semantic_logits": semantic_logits,
            "center_heatmap": center_heatmap,
            "offset_map": offset_map,
            "panoptic_pred": panoptic_pred,
        }

    def panoptic_fusion(
        self, semantic_logits: torch.Tensor, center_heatmap: torch.Tensor, offset_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse semantic and instance predictions to generate panoptic segmentation.

        Args:
            semantic_logits: [B, num_classes, H, W]
            center_heatmap: [B, 1, H, W]
            offset_map: [B, 2, H, W]

        Returns:
            panoptic_pred: [B, H, W] with instance IDs and semantic labels
        """
        batch_size, _, height, width = semantic_logits.shape
        device = semantic_logits.device

        # Get semantic predictions
        semantic_pred = torch.argmax(semantic_logits, dim=1)  # [B, H, W]

        panoptic_pred = torch.zeros_like(semantic_pred)

        for b in range(batch_size):
            # Process each image in the batch
            sem_pred = semantic_pred[b]  # [H, W]
            center_heat = center_heatmap[b, 0]  # [H, W]
            offset = offset_map[b]  # [2, H, W]

            # Find instance centers
            centers = self.find_instance_centers(center_heat)

            # Generate instance masks
            instance_masks = self.generate_instance_masks(centers, offset, height, width)

            # Fuse semantic and instance predictions
            panoptic_img = self.fuse_semantic_instance(sem_pred, instance_masks, centers)

            panoptic_pred[b] = panoptic_img

        return panoptic_pred

    def find_instance_centers(self, center_heatmap: torch.Tensor) -> List[Tuple[int, int]]:
        """Find instance centers from center heatmap using NMS."""
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

    def generate_instance_masks(
        self, centers: List[Tuple[int, int]], offset_map: torch.Tensor, height: int, width: int
    ) -> List[torch.Tensor]:
        """Generate instance masks from centers and offset map."""
        device = offset_map.device

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )

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

    def fuse_semantic_instance(
        self, semantic_pred: torch.Tensor, instance_masks: List[torch.Tensor], centers: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Fuse semantic and instance predictions."""
        height, width = semantic_pred.shape
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
