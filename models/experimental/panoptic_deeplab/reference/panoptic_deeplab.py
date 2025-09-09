# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from models.experimental.panoptic_deeplab.reference.semantic_seg_head import (
    PanopticDeeplabSemanticSegmentationModel as SemSegTorch,
)
from models.experimental.panoptic_deeplab.reference.instance_seg_head import (
    PanopticDeeplabInstanceSegmentationModel as InsSegTorch,
)
from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone

from typing import Dict, List, Tuple


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
        # panoptic_pred = improved_panoptic_fusion(semantic_logits, center_heatmap, offset_map)
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


class PanopticPostProcessor:
    """
    Improved postprocessing for Panoptic DeepLab based on detectron2 implementation.
    """

    def __init__(
        self,
        num_classes: int = 19,
        thing_classes: List[int] = None,
        stuff_classes: List[int] = None,
        center_threshold: float = 0.1,
        nms_kernel: int = 3,
        top_k_instance: int = 200,
        stuff_area_limit: int = 4096,
        instance_score_threshold: float = 0.5,
    ):
        self.num_classes = num_classes
        self.thing_classes = thing_classes or [11, 12, 13, 14, 15, 16, 17, 18]
        self.stuff_classes = stuff_classes or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k_instance = top_k_instance
        self.stuff_area_limit = stuff_area_limit
        self.instance_score_threshold = instance_score_threshold

        # Create thing mask for efficient lookup
        self.is_thing = torch.zeros(num_classes, dtype=torch.bool)
        for cls in self.thing_classes:
            self.is_thing[cls] = True

    def process(
        self,
        semantic_logits: torch.Tensor,
        center_heatmap: torch.Tensor,
        offset_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process network outputs to generate panoptic segmentation.

        Args:
            semantic_logits: [B, C, H, W] semantic logits
            center_heatmap: [B, 1, H, W] center predictions
            offset_map: [B, 2, H, W] offset predictions

        Returns:
            Dictionary with processed outputs
        """
        batch_size = semantic_logits.shape[0]
        device = semantic_logits.device

        # Get semantic predictions with softmax
        semantic_probs = F.softmax(semantic_logits, dim=1)
        semantic_pred = torch.argmax(semantic_probs, dim=1)  # [B, H, W]

        # Process each image
        panoptic_preds = []
        instance_centers = []

        for b in range(batch_size):
            panoptic, centers = self.process_single_image(
                semantic_probs[b], semantic_pred[b], center_heatmap[b, 0], offset_map[b]
            )
            panoptic_preds.append(panoptic)
            instance_centers.append(centers)

        panoptic_pred = torch.stack(panoptic_preds, dim=0)

        return {
            "semantic_pred": semantic_pred,
            "panoptic_pred": panoptic_pred,
            "instance_centers": instance_centers,
        }

    def process_single_image(
        self,
        semantic_probs: torch.Tensor,  # [C, H, W]
        semantic_pred: torch.Tensor,  # [H, W]
        center_heatmap: torch.Tensor,  # [H, W]
        offset_map: torch.Tensor,  # [2, H, W]
    ) -> Tuple[torch.Tensor, List]:
        """Process a single image."""

        height, width = semantic_pred.shape
        device = semantic_pred.device

        # Find instance centers with NMS
        centers = self.find_instance_centers_nms(center_heatmap)

        if len(centers) == 0:
            # No instances found, return semantic segmentation
            return self.merge_stuff_regions(semantic_pred), []

        # Generate instance segmentation
        instance_seg = self.generate_instance_segmentation(centers, offset_map, semantic_probs, height, width)

        # Merge semantic and instance predictions
        panoptic = self.merge_semantic_and_instance(semantic_pred, semantic_probs, instance_seg, centers)

        # Post-process stuff classes
        panoptic = self.merge_stuff_regions(panoptic)

        return panoptic, centers

    def find_instance_centers_nms(self, center_heatmap: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Find instance centers using NMS.
        Returns list of (y, x, score) tuples.
        """
        # Apply threshold
        # print(f"center_heatmap.shape: {center_heatmap.shape}")
        # print(f"self.center_threshold: {self.center_threshold}")
        center_mask = center_heatmap > self.center_threshold[0]

        if not center_mask.any():
            return []

        # Max pooling for NMS
        pooled = F.max_pool2d(
            center_heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=self.nms_kernel[0],
            stride=1,
            padding=(self.nms_kernel[0] - 1) // 2,
        )[0, 0]

        # Keep only local maxima
        center_mask = center_mask & (center_heatmap == pooled)

        # Get center coordinates and scores
        coords = torch.nonzero(center_mask, as_tuple=False)
        scores = center_heatmap[center_mask]

        # Sort by score and keep top-k
        if len(coords) > self.top_k_instance:
            top_k_idx = torch.topk(scores, min(self.top_k_instance, len(scores)))[1]
            coords = coords[top_k_idx]
            scores = scores[top_k_idx]

        centers = [(c[0].item(), c[1].item(), s.item()) for c, s in zip(coords, scores)]

        return centers

    def generate_instance_segmentation(
        self,
        centers: List[Tuple[int, int, float]],
        offset_map: torch.Tensor,
        semantic_probs: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Generate instance segmentation from centers and offsets.
        """
        device = offset_map.device
        instance_seg = torch.zeros(height, width, dtype=torch.long, device=device)

        if len(centers) == 0:
            return instance_seg

        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )

        # Process each center
        for idx, (center_y, center_x, score) in enumerate(centers, start=1):
            # Get semantic class at center
            center_class = torch.argmax(semantic_probs[:, center_y, center_x]).item()

            # Skip if not a thing class
            if center_class not in self.thing_classes:
                continue

            # Calculate pixel coordinates after offset
            pixel_y = y_grid + offset_map[0]
            pixel_x = x_grid + offset_map[1]

            # Distance to center
            dist_y = (pixel_y - center_y) ** 2
            dist_x = (pixel_x - center_x) ** 2
            distance = torch.sqrt(dist_y + dist_x)

            # Create instance mask based on distance and semantic consistency
            semantic_mask = semantic_probs[center_class] > self.instance_score_threshold
            distance_mask = distance < 2.0  # Distance threshold

            instance_mask = semantic_mask & distance_mask

            # Assign instance ID
            instance_seg[instance_mask] = idx + 1000  # Instance IDs start from 1001

        return instance_seg

    def merge_semantic_and_instance(
        self,
        semantic_pred: torch.Tensor,
        semantic_probs: torch.Tensor,
        instance_seg: torch.Tensor,
        centers: List[Tuple[int, int, float]],
    ) -> torch.Tensor:
        """
        Merge semantic and instance predictions to create panoptic segmentation.
        """
        panoptic = semantic_pred.clone()

        # Override with instance predictions where available
        instance_mask = instance_seg > 0
        panoptic[instance_mask] = instance_seg[instance_mask]

        return panoptic

    def merge_stuff_regions(self, panoptic: torch.Tensor) -> torch.Tensor:
        """
        Merge small stuff regions and handle area thresholds.
        """
        device = panoptic.device

        for stuff_class in self.stuff_classes:
            mask = panoptic == stuff_class
            if not mask.any():
                continue

            # Find connected components
            from scipy import ndimage

            mask_np = mask.cpu().numpy()
            labeled, num_features = ndimage.label(mask_np)

            # Remove small regions
            for i in range(1, num_features + 1):
                component_mask = labeled == i
                if component_mask.sum() < self.stuff_area_limit:
                    # Replace with nearest neighbor class
                    panoptic[torch.from_numpy(component_mask).to(device)] = 255  # Void label

        return panoptic


# Integration with your existing code
def improved_panoptic_fusion(
    semantic_logits: torch.Tensor,
    center_heatmap: torch.Tensor,
    offset_map: torch.Tensor,
    # config: dict
) -> torch.Tensor:
    """
    Improved panoptic fusion for TTNN outputs.
    """
    num_classes = 19
    thing_classes = ([11, 12, 13, 14, 15, 16, 17, 18],)
    stuff_classes = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],)
    center_threshold = (0.1,)
    nms_kernel = (3,)
    top_k_instance = (200,)
    stuff_area_limit = (4096,)
    processor = PanopticPostProcessor(
        num_classes=num_classes,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        center_threshold=center_threshold,
        nms_kernel=nms_kernel,
        top_k_instance=top_k_instance,
        stuff_area_limit=stuff_area_limit,
    )

    results = processor.process(semantic_logits, center_heatmap, offset_map)
    return results["panoptic_pred"]
