# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch


class PanopticDeeplabSemanticsSegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.Sem_Seg_ASPP_0_Conv = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_ASPP_1_Depthwise = nn.Sequential(nn.Conv2d(2048, 2048, 3, 1, 6, 6, 2048), nn.ReLU())
        self.Sem_Seg_ASPP_1_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_ASPP_2_Depthwise = nn.Sequential(nn.Conv2d(2048, 2048, 3, 1, 12, 12, 2048), nn.ReLU())
        self.Sem_Seg_ASPP_2_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_ASPP_3_Depthwise = nn.Sequential(nn.Conv2d(2048, 2048, 3, 1, 18, 18, 2048), nn.ReLU())
        self.Sem_Seg_ASPP_3_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_ASPP_4_avg_pool = torch.nn.AvgPool2d((32, 64), stride=1, count_include_pad=True)
        self.Sem_Seg_ASPP_4_Conv_1 = nn.Sequential(nn.Conv2d(2048, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_ASPP_project = nn.Sequential(nn.Conv2d(1280, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_Decoder_res3_project_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1), nn.ReLU())
        self.Sem_Seg_Decoder_res3_fuse_conv_depthwise = nn.Sequential(nn.Conv2d(320, 320, 5, 1, 2, 1, 320), nn.ReLU())
        self.Sem_Seg_Decoder_res3_fuse_conv_pointwise = nn.Sequential(nn.Conv2d(320, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_Decoder_res2_project_conv = nn.Sequential(nn.Conv2d(256, 32, 1, 1), nn.ReLU())
        self.Sem_Seg_Decoder_res2_fuse_conv_depthwise = nn.Sequential(nn.Conv2d(288, 288, 5, 1, 2, 1, 288), nn.ReLU())
        self.Sem_Seg_Decoder_res2_fuse_conv_pointwise = nn.Sequential(nn.Conv2d(288, 256, 1, 1), nn.ReLU())

        self.Sem_Seg_Head_depthwise = nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2, 1, 256), nn.ReLU())
        self.Sem_Seg_Head_pointwise = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.ReLU())
        self.Sem_Seg_predictor = nn.Conv2d(256, 19, 1, 1)

    def forward(self, x, res3, res2):
        t0 = self.Sem_Seg_ASPP_0_Conv(x)
        t1 = self.Sem_Seg_ASPP_1_Depthwise(x)
        t2 = self.Sem_Seg_ASPP_2_Depthwise(x)
        t3 = self.Sem_Seg_ASPP_3_Depthwise(x)
        t4 = self.Sem_Seg_ASPP_4_avg_pool(x)

        t4 = self.Sem_Seg_ASPP_4_Conv_1(t4)
        t4 = nn.functional.interpolate(t4, (32, 64), mode="bilinear")

        t1 = self.Sem_Seg_ASPP_1_pointwise(t1)
        t2 = self.Sem_Seg_ASPP_2_pointwise(t2)
        t3 = self.Sem_Seg_ASPP_3_pointwise(t3)

        y = torch.cat((t0, t1, t2, t3, t4), dim=1)

        y = self.Sem_Seg_ASPP_project(y)
        y = nn.functional.interpolate(y, scale_factor=2, mode="bilinear")
        res3 = self.Sem_Seg_Decoder_res3_project_conv(res3)

        y = torch.cat((res3, y), dim=1)

        y = self.Sem_Seg_Decoder_res3_fuse_conv_depthwise(y)
        y = self.Sem_Seg_Decoder_res3_fuse_conv_pointwise(y)
        y = nn.functional.interpolate(y, scale_factor=2, mode="bilinear")

        res2 = self.Sem_Seg_Decoder_res2_project_conv(res2)

        y = torch.cat((res2, y), dim=1)
        y = self.Sem_Seg_Decoder_res2_fuse_conv_depthwise(y)
        y = self.Sem_Seg_Decoder_res2_fuse_conv_pointwise(y)
        y = self.Sem_Seg_Head_depthwise(y)
        y = self.Sem_Seg_Head_pointwise(y)
        y = self.Sem_Seg_predictor(y)
        y = nn.functional.interpolate(y, scale_factor=4, mode="bilinear")

        return y
