# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch


class MulByConstant(nn.Module):
    def __init__(self, value=4):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x * self.value


class PanopticDeeplabInstanceSegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.Ins_Seg_ASPP_0_Conv = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Ins_Seg_ASPP_1_Depthwise = nn.Sequential(nn.Conv2d(2048, 2048, 3, 1, 6, 6, 2048), nn.ReLU())
        self.Ins_Seg_ASPP_1_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Ins_Seg_ASPP_2_Depthwise = nn.Sequential(nn.Conv2d(2048, 2048, 3, 1, 12, 12, 2048), nn.ReLU())
        self.Ins_Seg_ASPP_2_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Ins_Seg_ASPP_3_Depthwise = nn.Sequential(nn.Conv2d(2048, 2048, 3, 1, 18, 18, 2048), nn.ReLU())
        self.Ins_Seg_ASPP_3_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Ins_Seg_ASPP_4_avg_pool = torch.nn.AvgPool2d((32, 64), stride=1, count_include_pad=True)
        self.Ins_Seg_ASPP_4_Conv_1 = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Ins_Seg_ASPP_project = nn.Sequential(nn.Conv2d(1280, 256, 1, 1), nn.ReLU())

        self.Ins_Seg_Decoder_res3_project_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1), nn.ReLU())
        self.Ins_Seg_Decoder_res3_fuse_conv_depthwise = nn.Sequential(nn.Conv2d(320, 320, 5, 1, 2, 1, 320), nn.ReLU())
        self.Ins_Seg_Decoder_res3_fuse_conv_pointwise = nn.Sequential(nn.Conv2d(320, 128, 1, 1), nn.ReLU())

        self.Ins_Seg_Decoder_res2_project_conv = nn.Sequential(nn.Conv2d(256, 32, 1, 1), nn.ReLU())
        self.Ins_Seg_Decoder_res2_fuse_conv_depthwise = nn.Sequential(nn.Conv2d(160, 160, 5, 1, 2, 1, 160), nn.ReLU())
        self.Ins_Seg_Decoder_res2_fuse_conv_pointwise = nn.Sequential(nn.Conv2d(160, 128, 1, 1), nn.ReLU())

        # Instance Center Head
        self.Ins_Seg_Center_Head_Conv_0 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.Ins_Seg_Center_Head_Conv_1 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1), nn.ReLU())
        self.Ins_Seg_Center_predictor = nn.Conv2d(32, 1, 1, 1)

        # Instance Offset Head
        self.Ins_Seg_Offset_Head_depthwise = nn.Sequential(nn.Conv2d(128, 128, 5, 1, 2, 1, 128), nn.ReLU())
        self.Ins_Seg_Offset_Head_pointwise = nn.Sequential(nn.Conv2d(128, 32, 1, 1), nn.ReLU())
        self.Ins_Seg_Offset_predictor = nn.Conv2d(32, 2, 1, 1)
        self.Ins_Seg_Mul = MulByConstant(4)

    def forward(self, x, res3, res2):
        t0 = self.Ins_Seg_ASPP_0_Conv(x)
        t1 = self.Ins_Seg_ASPP_1_Depthwise(x)
        t2 = self.Ins_Seg_ASPP_2_Depthwise(x)
        t3 = self.Ins_Seg_ASPP_3_Depthwise(x)
        t4 = self.Ins_Seg_ASPP_4_avg_pool(x)

        t4 = self.Ins_Seg_ASPP_4_Conv_1(t4)

        t4 = nn.functional.interpolate(t4, (32, 64), mode="bilinear")

        t1 = self.Ins_Seg_ASPP_1_pointwise(t1)
        t2 = self.Ins_Seg_ASPP_2_pointwise(t2)
        t3 = self.Ins_Seg_ASPP_3_pointwise(t3)

        y = torch.cat((t0, t1, t2, t3, t4), dim=1)

        y = self.Ins_Seg_ASPP_project(y)
        y = nn.functional.interpolate(y, scale_factor=2, mode="bilinear")
        res3 = self.Ins_Seg_Decoder_res3_project_conv(res3)

        y = torch.cat((res3, y), dim=1)

        y = self.Ins_Seg_Decoder_res3_fuse_conv_depthwise(y)
        y = self.Ins_Seg_Decoder_res3_fuse_conv_pointwise(y)
        y = nn.functional.interpolate(y, scale_factor=2, mode="bilinear")

        res2 = self.Ins_Seg_Decoder_res2_project_conv(res2)

        y = torch.cat((res2, y), dim=1)
        y = self.Ins_Seg_Decoder_res2_fuse_conv_depthwise(y)
        y = self.Ins_Seg_Decoder_res2_fuse_conv_pointwise(y)

        y1 = self.Ins_Seg_Center_Head_Conv_0(y)
        y1 = self.Ins_Seg_Center_Head_Conv_1(y1)
        y1 = self.Ins_Seg_Center_predictor(y1)
        y1 = nn.functional.interpolate(y1, scale_factor=4, mode="bilinear")

        y2 = self.Ins_Seg_Offset_Head_depthwise(y)
        y2 = self.Ins_Seg_Offset_Head_pointwise(y2)
        y2 = self.Ins_Seg_Offset_predictor(y2)
        y2 = nn.functional.interpolate(y2, scale_factor=4, mode="bilinear")

        y2 = self.Ins_Seg_Mul(y2)

        return y1, y2
