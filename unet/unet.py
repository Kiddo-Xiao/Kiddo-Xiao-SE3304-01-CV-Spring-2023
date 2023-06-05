"""
UNet
来自：https://blog.csdn.net/weixin_45074568/article/details/114901600
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class DoubleConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = DoubleConv(in_ch, 64)
        self.D1 = DownSampling(64)
        self.C2 = DoubleConv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = DoubleConv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = DoubleConv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = DoubleConv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = DoubleConv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = DoubleConv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = DoubleConv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = DoubleConv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, out_ch, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测
        return self.Th(self.pred(O4))
