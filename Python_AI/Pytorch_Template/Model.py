import sys
import torch
import torch.nn as nn
from torch.nn.modules import flatten
import torch.optim as optim
import numpy as np
from torch.optim import adam  # type: ignore
import torch.nn.functional as F


# Tasks
# 1. Build Conv Structure (1d for frequency)
# 2. Make skip connections
# 3. Implement this as variational auto encoder
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, num_features):
#         super(ConvBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
#             nn.BatchNorm2d(num_features=num_features),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
#
# class ConvTransposeBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, num_features):
#         super(ConvTransposeBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
#             nn.BatchNorm2d(num_features=num_features),
#             nn.LeakyReLU(),
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
#
# # z=u+(sigma) element-wise multiplication (epsilon)
# class ConvAutoEncoder_V2(nn.Module):
#     def __init__(self, ConvBlock, ConvTransposeBlock):
#         super(ConvAutoEncoder_V2, self).__init__()
#         self.conv1 = ConvBlock(in_channels=1, out_channels=8, kernel_size=(1, 8), num_features=8)
#         self.conv2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=(1, 1), num_features=16)  # skip
#         self.conv3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=(1, 1), num_features=32)
#         self.conv4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(1, 1), num_features=64)  # skip
#         self.conv5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(1, 1), num_features=128)
#         self.conv6 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(1, 1), num_features=256)
#
#         # self.z_mean=nn.Linear(327680,1)
#         # self.z_variance=nn.Linear(327680,1)
#         # self.linear=nn.Linear(1, 65536)
#
#         self.conv_transpose6 = ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=(1, 1), num_features=128)
#         self.conv_transpose5 = ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=(1, 1), num_features=64)
#         self.conv_transpose4 = ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=(1, 1), num_features=32)  # skip
#         self.conv_transpose3 = ConvTransposeBlock(in_channels=32, out_channels=16, kernel_size=(1, 1), num_features=16)
#         self.conv_transpose2 = ConvTransposeBlock(in_channels=16, out_channels=8, kernel_size=(1, 1), num_features=8)  # skip
#         self.conv_transpose1 = ConvTransposeBlock(in_channels=8, out_channels=1, kernel_size=(1, 1), num_features=1)
#
#     def encoded(self, x):
#         self.x1 = self.conv1(x)
#         self.x2 = self.conv2(self.x1)  # skip
#         self.x3 = self.conv3(self.x2)
#         self.x4 = self.conv4(self.x3)  # skip
#         self.x5 = self.conv5(self.x4)
#         self.x6 = self.conv6(self.x5)  # skip
#         return self.x6
#
#     def decoded(self, encoded):
#         x7 = self.conv_transpose6(encoded)  # skip
#         #skip1 = self.x5 + x7
#         x8 = self.conv_transpose5(x7)
#         x9 = self.conv_transpose4(x8)  # skip
#         #skip2 = self.x3 + x9
#         x10 = self.conv_transpose3(x9)
#         x11 = self.conv_transpose2(x10)  # skip
#         #skip3 = self.x1 + x11
#         x12 = self.conv_transpose1(x11)
#         return x12
#
#     # def reparameterize(self, z_mean, z_variance):
#     #     eps=torch.rand(z_mean.size(0), 1)
#     #     z=z_mean + eps * torch.exp(z_variance/2.)
#     #     return z
#
#     def forward(self, x):
#         encoded = self.encoded(x)
#         # z_mean, z_variance=self.z_mean(x), self.z_variance(x)
#         # encoded=self.reparameterize(z_mean, z_variance)
#         # encoded=self.linear(encoded)
#         # encoded=encoded.view(2, 256, 128, 1)
#
#         decoded = self.decoded(encoded)
#         return decoded


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 8))
        self.conv1_bn = nn.BatchNorm2d(num_features=8)  # 1
        self.conv2 = nn.Conv2d(8, 16, (1, 1))
        self.conv2_bn = nn.BatchNorm2d(16)  # 2
        self.conv3 = nn.Conv2d(16, 32, (1, 1))
        self.conv3_bn = nn.BatchNorm2d(32)  # 3
        self.conv4 = nn.Conv2d(32, 64, (1, 1))
        self.conv4_bn = nn.BatchNorm2d(64)  # 4
        self.conv5 = nn.Conv2d(64, 128, (1, 1))
        self.conv5_bn = nn.BatchNorm2d(128)  # 5
        self.conv6 = nn.Conv2d(128, 256, (1, 1))
        self.conv6_bn = nn.BatchNorm2d(256)  # 6
        self.conv7 = nn.Conv2d(256, 512, (1, 1))
        self.conv7_bn = nn.BatchNorm2d(512)  # 6
        self.conv8 = nn.Conv2d(512, 4096, (1, 1))
        self.conv8_bn = nn.BatchNorm2d(4096)  # 6

        self.deconv8 = nn.ConvTranspose2d(4096, 512, (1, 1))
        self.deconv8_bn = nn.BatchNorm2d(512)
        self.deconv7 = nn.ConvTranspose2d(512, 256, (1, 1))
        self.deconv7_bn = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(256, 128, (1, 1))
        self.deconv6_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, (1, 1))
        self.deconv5_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, (1, 1))
        self.deconv4_bn = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, (1, 1))
        self.deconv3_bn = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, (1, 1))
        self.deconv2_bn = nn.BatchNorm2d(8)
        self.deconv1 = nn.ConvTranspose2d(8, 1, (1, 1))
        self.deconv1_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        encoder_1 = self.relu(self.conv1_bn(self.conv1(x)))#
        encoder_2 = self.relu(self.conv2_bn(self.conv2(encoder_1)))
        encoder_3 = self.relu(self.conv3_bn(self.conv3(encoder_2)))#
        encoder_4 = self.relu(self.conv4_bn(self.conv4(encoder_3)))
        encoder_5 = self.relu(self.conv5_bn(self.conv5(encoder_4)))#
        encoder_6 = self.relu(self.conv6_bn(self.conv6(encoder_5)))
        encoder_7 = self.relu(self.conv7_bn(self.conv7(encoder_6)))#
        encoder_8 = self.relu(self.conv8_bn(self.conv8(encoder_7)))

        decoder_8 = self.relu(self.deconv8_bn(self.deconv8(encoder_8)))
        decoder_7 = self.relu(self.deconv7_bn(self.deconv7(decoder_8+encoder_7)))#
        decoder_6 = self.relu(self.deconv6_bn(self.deconv6(decoder_7)))
        decoder_5 = self.relu(self.deconv5_bn(self.deconv5(decoder_6+encoder_5)))#
        decoder_4 = self.relu(self.deconv4_bn(self.deconv4(decoder_5)))
        decoder_3 = self.relu(self.deconv3_bn(self.deconv3(decoder_4+encoder_3)))#
        decoder_2 = self.relu(self.deconv2_bn(self.deconv2(decoder_3)))
        decoder_1 = self.relu(self.deconv1_bn(self.deconv1(decoder_2+encoder_1)))#
        return decoder_1


class ConvAutoEncoder_1st(nn.Module):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder_1st, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 8))
        self.conv1_bn = nn.BatchNorm2d(num_features=64)  # 1
        self.conv2 = nn.Conv2d(64, 2056, (1, 1))
        self.conv2_bn = nn.BatchNorm2d(2056)  # 2

        self.deconv2 = nn.ConvTranspose2d(2056, 64, (1, 1))
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 1, (1, 1))
        self.deconv1_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        layer1 = self.relu(self.conv1_bn(self.conv1(x)))
        layer2 = self.relu(self.conv2_bn(self.conv2(layer1)))

        layer3 = self.relu(self.deconv2_bn(self.deconv2(layer2)))
        layer4 = self.relu(self.deconv1_bn(self.deconv1(layer3)))
        return layer4


class ConvAutoEncoder_Dense(nn.Module):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder_Dense, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 8))
        self.conv1_bn = nn.BatchNorm2d(num_features=8)  # 1
        self.conv2 = nn.Conv2d(8, 16, (1, 1))
        self.conv2_bn = nn.BatchNorm2d(16)  # 2
        self.conv3 = nn.Conv2d(16, 32, (1, 1))
        self.conv3_bn = nn.BatchNorm2d(32)  # 3
        self.conv4 = nn.Conv2d(32, 64, (1, 1))
        self.conv4_bn = nn.BatchNorm2d(64)  # 4
        self.conv5 = nn.Conv2d(64, 128, (1, 1))
        self.conv5_bn = nn.BatchNorm2d(128)  # 5
        self.conv6 = nn.Conv2d(128, 256, (1, 1))
        self.conv6_bn = nn.BatchNorm2d(256)  # 6
        self.linear1 = nn.Linear(4194304, 2)

        self.linear2 = nn.Linear(2, 4194304)
        self.deconv6 = nn.ConvTranspose2d(256, 128, (1, 1))
        self.deconv6_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, (1, 1))
        self.deconv5_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, (1, 1))
        self.deconv4_bn = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, (1, 1))
        self.deconv3_bn = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, (1, 1))
        self.deconv2_bn = nn.BatchNorm2d(8)
        self.deconv1 = nn.ConvTranspose2d(8, 1, (1, 1))
        self.deconv1_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x1 = self.relu(self.conv1_bn(self.conv1(x)))
        x2 = self.relu(self.conv2_bn(self.conv2(x1)))
        x3 = self.relu(self.conv3_bn(self.conv3(x2)))
        x4 = self.relu(self.conv4_bn(self.conv4(x3)))
        x5 = self.relu(self.conv5_bn(self.conv5(x4)))
        x6 = self.relu(self.conv6_bn(self.conv6(x5)))
        flatten = torch.flatten(x6)
        linear1 = self.linear1(flatten)

        linear2 = self.linear2(linear1)
        unflatten = linear2.view(x6.shape)
        x7 = self.relu(self.deconv6_bn(self.deconv6(unflatten)))
        skip1 = x5 + x7
        x8 = self.relu(self.deconv5_bn(self.deconv5(skip1)))
        x9 = self.relu(self.deconv4_bn(self.deconv4(x8)))
        skip2 = x3 + x9
        x10 = self.relu(self.deconv3_bn(self.deconv3(skip2)))
        x11 = self.relu(self.deconv2_bn(self.deconv2(x10)))
        skip3 = x1 + x11
        x12 = self.relu(self.deconv1_bn(self.deconv1(skip3)))
        return x12
    # flatten=4194304
    # unflatten=2, 256, 128, 1
    # x5=128, 128, 128, 1
    # x7=2, 128, 128, 1
