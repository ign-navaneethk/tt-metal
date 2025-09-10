# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.panoptic_deeplab.tt.aspp import PanopticDeeplabASPP as TTASPP
from models.experimental.panoptic_deeplab.tt.head import TTHead
from models.experimental.panoptic_deeplab.tt.res_block import TTRes
from models.experimental.panoptic_deeplab.tt.res_block import res_layer_optimisations
from models.experimental.panoptic_deeplab.tt.head import head_layer_optimisations
from dataclasses import dataclass


@dataclass
class DecoderOptimizer:
    res_layer_optimisations: dict
    head_layer_optimisations: dict
    shape: tuple


decoder_layer_optimisations = {
    "default": DecoderOptimizer(
        res_layer_optimisations=res_layer_optimisations["default"],
        head_layer_optimisations=head_layer_optimisations["default"],
        shape=(0, 0, 0, 0),
    ),
    "sem_seg_head": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["semantics_Res3"],
            "res2": res_layer_optimisations["semantics_Res2"],
        },
        head_layer_optimisations={
            "head": head_layer_optimisations["segmentation_offset_head"],
        },
        shape=(1, 128, 256, 256),
    ),
    "ins_embed_head_offset": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["instance_Res3"],
            "res2": res_layer_optimisations["instance_Res2"],
        },
        head_layer_optimisations={
            "head": head_layer_optimisations["instance_offset_head"],
        },
        shape=(1, 128, 256, 128),
    ),
    "ins_embed_head_center": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["instance_Res3"],
            "res2": res_layer_optimisations["instance_Res2"],
        },
        head_layer_optimisations={
            "head": head_layer_optimisations["instance_center_head"],
        },
        shape=(1, 128, 256, 128),
    ),
}


class TTDecoder:
    def __init__(self, parameters, model_config, layer_optimisations=decoder_layer_optimisations["default"]) -> None:
        super().__init__()
        self.shape = layer_optimisations.shape
        self.aspp = TTASPP(parameters.aspp, model_config, layer_optimisations=None)
        self.res3 = TTRes(
            parameters.res3,
            model_config,
            layer_optimisations=layer_optimisations.res_layer_optimisations["res3"],
        )
        self.res2 = TTRes(
            parameters.res2,
            model_config,
            layer_optimisations=layer_optimisations.res_layer_optimisations["res2"],
        )
        self.head = TTHead(
            parameters.head,
            model_config,
            layer_optimisations=layer_optimisations.head_layer_optimisations["head"],
        )

    def __call__(self, x, res3, res2, upsample_channels, device):
        y = self.aspp(x, device)
        y = self.res3(y, res3, upsample_channels, device)
        # print(f"DEBUG: Decoder layer optimisations: {decoder_layer_optimisations}")

        if self.shape[-1] == 128:
            # print(f"DEBUG: Upsample channels: {upsample_channels}")
            upsample_channels = upsample_channels // 2

        # print(f"DEBUG: Upsample channels: {upsample_channels}")
        y = self.res2(y, res2, upsample_channels, device)

        y = self.head(y, self.shape, device)

        return y
