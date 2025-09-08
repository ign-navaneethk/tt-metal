# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from models.experimental.panoptic_deeplab.reference.aspp import PanopticDeeplabASPPModel


class PanopticDeeplabSemanticDecoderRes3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem_Seg_Decoder_res3_project_conv = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise = nn.Sequential(
            nn.Conv2d(320, 320, 5, 1, 2, 1, 320, bias=False), nn.BatchNorm2d(320), nn.ReLU()
        )
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise = nn.Sequential(
            nn.Conv2d(320, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU()
        )

    def forward(self, x, res3):
        y = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        res3 = self.Sem_Seg_Decoder_res3_project_conv(res3)
        y = torch.cat((res3, y), dim=1)
        y = self.Sem_Seg_Decoder_res3_fuse_conv_depthwise(y)
        y = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise(y)
        return y


class PanopticDeeplabSemanticDecoderRes2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem_Seg_Decoder_res2_project_conv = nn.Sequential(
            nn.Conv2d(256, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise = nn.Sequential(
            nn.Conv2d(288, 288, 5, 1, 2, 1, 288, bias=False), nn.BatchNorm2d(288), nn.ReLU()
        )
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise = nn.Sequential(
            nn.Conv2d(288, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU()
        )

    def forward(self, x, res2):
        y = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        res2 = self.Sem_Seg_Decoder_res2_project_conv(res2)
        y = torch.cat((res2, y), dim=1)
        y = self.Sem_Seg_Decoder_res2_fuse_conv_depthwise(y)
        y = self.Sem_Seg_Decoder_res2_fuse_conv_pointwise(y)
        return y


class PanopticDeeplabSemanticHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem_Seg_Head_depthwise = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1, 2, 1, 256, bias=False), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.Sem_Seg_Head_pointwise = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.Sem_Seg_Head_predictor = nn.Conv2d(256, 19, 1, 1)

    def forward(self, x):
        y = self.Sem_Seg_Head_depthwise(x)
        y = self.Sem_Seg_Head_pointwise(y)
        y = self.Sem_Seg_Head_predictor(y)
        y = nn.functional.interpolate(y, scale_factor=4, mode="bilinear")

        return y


class PanopticDeeplabSemanticSegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aspp = PanopticDeeplabASPPModel()
        self.res3 = PanopticDeeplabSemanticDecoderRes3Model()
        self.res2 = PanopticDeeplabSemanticDecoderRes2Model()
        self.head = PanopticDeeplabSemanticHeadModel()

    def forward(self, x, res3, res2):
        y = self.aspp(x)
        y = self.res3(y, res3)
        y = self.res2(y, res2)
        y = self.head(y)

        return y
