# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from typing import Dict

from models.experimental.panoptic_deeplab.tt.backbone import TTBackbone
from models.experimental.panoptic_deeplab.tt.decoder import TTDecoder, decoder_layer_optimisations


class TTPanopticDeepLab:
    """
    TTNN implementation of Panoptic DeepLab using backbone and decoder architecture.
    Combines backbone, semantic segmentation, and instance segmentation.
    """

    def __init__(
        self,
        parameters,
        model_config,
    ):
        self.model_config = model_config

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
        """

        logger.debug("Running TT Panoptic DeepLab forward pass")

        # Extract features from backbone
        logger.debug("Running TTBackbone")
        features = self.backbone(x, device)

        # Extract the specific feature maps the decoders expect
        backbone_features = features["res_5"]
        res3_features = features["res_3"]
        res2_features = features["res_2"]

        logger.debug(
            f"Backbone features shapes - res_5: {backbone_features.shape}, "
            f"res_3: {res3_features.shape}, res_2: {res2_features.shape}"
        )
        # Semantic segmentation branch
        logger.debug("Running semantic segmentation decoder")
        semantic_logits = self.semantic_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,
            device=device,
        )

        # Instance center prediction branch
        logger.debug("Running instance center decoder")
        center_heatmap = self.instance_center_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,
            device=device,
        )

        # Instance offset prediction branch
        logger.debug("Running instance offset decoder")
        offset_map = self.instance_offset_decoder(
            backbone_features,
            res3_features,
            res2_features,
            upsample_channels=256,
            device=device,
        )

        outputs = {
            "semantic_logits": semantic_logits,
            "center_heatmap": center_heatmap,
            "offset_map": offset_map,
        }

        logger.debug("TT Panoptic DeepLab forward pass completed")

        return outputs
