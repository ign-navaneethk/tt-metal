# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from models.experimental.panoptic_deeplab.reference.aspp import PanopticDeeplabASPPModel


class MulByConstant(nn.Module):
    def __init__(self, value=4):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x * self.value


class PanopticDeeplabInstanceDecoderRes3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Ins_Seg_Decoder_res3_project_conv = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.Ins_Seg_Decoder_res3_fuse_conv_depthwise = nn.Sequential(
            nn.Conv2d(320, 320, 5, 1, 2, 1, 320, bias=False), nn.BatchNorm2d(320), nn.ReLU()
        )
        self.Ins_Seg_Decoder_res3_fuse_conv_pointwise = nn.Sequential(
            nn.Conv2d(320, 128, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU()
        )

    def forward(self, x, res3):
        y = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        res3 = self.Ins_Seg_Decoder_res3_project_conv(res3)
        y = torch.cat((res3, y), dim=1)
        y = self.Ins_Seg_Decoder_res3_fuse_conv_depthwise(y)
        y = self.Ins_Seg_Decoder_res3_fuse_conv_pointwise(y)
        return y


class PanopticDeeplabInstanceDecoderRes2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Ins_Seg_Decoder_res2_project_conv = nn.Sequential(
            nn.Conv2d(256, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.Ins_Seg_Decoder_res2_fuse_conv_depthwise = nn.Sequential(
            nn.Conv2d(160, 160, 5, 1, 2, 1, 160, bias=False), nn.BatchNorm2d(160), nn.ReLU()
        )
        self.Ins_Seg_Decoder_res2_fuse_conv_pointwise = nn.Sequential(
            nn.Conv2d(160, 128, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU()
        )

    def forward(self, x, res2):
        y = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        res2 = self.Ins_Seg_Decoder_res2_project_conv(res2)
        y = torch.cat((res2, y), dim=1)
        y = self.Ins_Seg_Decoder_res2_fuse_conv_depthwise(y)
        y = self.Ins_Seg_Decoder_res2_fuse_conv_pointwise(y)
        return y


class PanopticDeeplabInstanceCenterHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Ins_Seg_Center_Head_Conv_0 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.Ins_Seg_Center_Head_Conv_1 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.Ins_Seg_Center_predictor = nn.Conv2d(32, 1, 1, 1)

    def forward(self, x):
        y = self.Ins_Seg_Center_Head_Conv_0(x)
        y = self.Ins_Seg_Center_Head_Conv_1(y)
        y = self.Ins_Seg_Center_predictor(y)
        y = nn.functional.interpolate(y, scale_factor=4, mode="bilinear")
        return y


class PanopticDeeplabInstanceOffsetHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Ins_Seg_Offset_Head_depthwise = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2, 1, 128, bias=False), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.Ins_Seg_Offset_Head_pointwise = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.Ins_Seg_Offset_predictor = nn.Conv2d(32, 2, 1, 1)
        self.Ins_Seg_Mul = MulByConstant(4)

    def forward(self, x):
        y = self.Ins_Seg_Offset_Head_depthwise(x)
        y = self.Ins_Seg_Offset_Head_pointwise(y)
        y = self.Ins_Seg_Offset_predictor(y)
        y = nn.functional.interpolate(y, scale_factor=4, mode="bilinear")
        y = self.Ins_Seg_Mul(y)
        return y


class PanopticDeeplabInstanceSegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aspp = PanopticDeeplabASPPModel()
        self.res3 = PanopticDeeplabInstanceDecoderRes3Model()
        self.res2 = PanopticDeeplabInstanceDecoderRes2Model()
        self.center_head = PanopticDeeplabInstanceCenterHeadModel()
        self.offset_head = PanopticDeeplabInstanceOffsetHeadModel()

    def forward(self, x, res3, res2):
        y = self.aspp(x)
        y = self.res3(y, res3)
        y = self.res2(y, res2)
        y1 = self.center_head(y)
        y2 = self.offset_head(y)
        return y1, y2
