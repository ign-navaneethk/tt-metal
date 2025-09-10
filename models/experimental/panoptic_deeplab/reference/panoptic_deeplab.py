# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Dict

from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel


class TorchPanopticDeepLab(nn.Module):
    """
    Panoptic DeepLab model using modular decoder architecture.
    Combines semantic segmentation and instance segmentation with panoptic fusion.
    """

    def __init__(
        self,
    ):
        super().__init__()

        # Backbone
        self.backbone = ResNet52BackBone()

        # Semantic segmentation decoder
        self.semantic_decoder = DecoderModel(
            in_channels=2048,
            res3_intermediate_channels=320,
            res2_intermediate_channels=288,
            out_channels=19,
        )

        # Instance segmentation decoders
        # Center prediction branch
        self.instance_center_decoder = DecoderModel(
            in_channels=2048,
            res3_intermediate_channels=320,
            res2_intermediate_channels=160,
            out_channels=1,
        )

        # Offset prediction branch
        self.instance_offset_decoder = DecoderModel(
            in_channels=2048,
            res3_intermediate_channels=320,
            res2_intermediate_channels=160,
            out_channels=2,
        )

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
        """

        # Extract features from backbone
        features = self.backbone(x)

        # Extract specific feature maps
        backbone_features = features["res_5"]
        res3_features = features["res_3"]
        res2_features = features["res_2"]

        # Semantic segmentation branch
        semantic_logits = self.semantic_decoder(backbone_features, res3_features, res2_features)

        # Instance segmentation branches
        center_heatmap = self.instance_center_decoder(backbone_features, res3_features, res2_features)
        offset_map = self.instance_offset_decoder(backbone_features, res3_features, res2_features)

        return {
            "semantic_logits": semantic_logits,
            "center_heatmap": center_heatmap,
            "offset_map": offset_map,
        }
