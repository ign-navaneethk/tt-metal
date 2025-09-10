# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.panoptic_deeplab.reference.aspp import PanopticDeeplabASPPModel as ASPPModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel


class DecoderModel(torch.nn.Module):
    def __init__(self, in_channels, res3_intermediate_channels, res2_intermediate_channels, out_channels):
        super().__init__()
        self.aspp = ASPPModel()
        if res2_intermediate_channels == 288:
            out_channels = in_channels // 8
        elif res2_intermediate_channels == 160:
            out_channels = in_channels // 16
        self.res3 = ResModel(in_channels // 4, res3_intermediate_channels, out_channels)
        self.res2 = ResModel(in_channels // 8, res2_intermediate_channels, out_channels)
        if res2_intermediate_channels == 288:
            in_channels = in_channels // 8
            res2_intermediate_channels = in_channels // 2
            out_channels = 19
        elif res2_intermediate_channels == 160:
            in_channels = in_channels // 16
            res2_intermediate_channels = in_channels // 4
            out_channels = in_channels // 64

        self.head = HeadModel(in_channels, res2_intermediate_channels, out_channels)

    def forward(self, x, res3, res2):
        y = self.aspp(x)
        y = self.res3(y, res3)
        y = self.res2(y, res2)
        y = self.head(y)
        return y
