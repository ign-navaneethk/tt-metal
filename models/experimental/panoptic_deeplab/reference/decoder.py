# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.panoptic_deeplab.reference.aspp import PanopticDeeplabASPPModel as ASPPModel
from models.experimental.panoptic_deeplab.reference.head import HeadModel
from models.experimental.panoptic_deeplab.reference.res_block import ResModel


class DecoderModel(torch.nn.Module):
    def __init__(self, in_channels, res3_intermediate_channels, res2_intermediate_channels, out_channels):
        super().__init__()
        res_out_ch = in_channels // 8 if res2_intermediate_channels == 288 else in_channels // 16
        inter_ch = 32 if res2_intermediate_channels == 160 else 256
        in_ch = 128 if res2_intermediate_channels == 160 else 256
        out_ch = 128 if res2_intermediate_channels == 160 else 256
        self.num_out = 2 if res2_intermediate_channels == 160 else 1

        self.aspp = ASPPModel()
        self.res3 = ResModel(in_channels // 4, res3_intermediate_channels, out_ch)
        self.res2 = ResModel(in_channels // 8, res2_intermediate_channels, res_out_ch)
        self.head_1 = HeadModel(in_ch, inter_ch, out_channels[0])
        if self.num_out == 2:
            self.head_2 = HeadModel(in_ch, inter_ch, out_channels[1])

    def forward(self, x, res3, res2):
        y = self.aspp(x)
        y = self.res3(y, res3)
        y = self.res2(y, res2)
        out = self.head_1(y)

        if self.num_out == 2:
            y_2 = self.head_2(y)
            return out, y_2
        return out, None
