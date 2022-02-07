import sys
import torch
import torch.nn as nn
from torch.nn.modules import flatten
import torch.optim as optim
import numpy as np
from torch.optim import adam  # type: ignore
import torch.nn.functional as F
import pytorch_lightning as pl


class ConvAutoEncoder8VAE_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder8VAE_lightning, self).__init__()
        self.learning_rate = 0.001
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 8)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, (1, 1),),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, (1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        # mean, var=
        # self.reparameterize(mean, var)
        decoder = self.decoder(encoder)
        return decoder

    def reparameterize(self, mean, var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def add_channels(self, combined, clean):
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def training_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        # loss=F.mse_loss(output, clean)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6, amsgrad=False)#weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.1),
            },
        }

class ConvAutoEncoder8_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder8_lightning, self).__init__()
        self.learning_rate = 0.001
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 8)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 2048, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, (1, 1),),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, (1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

    def add_channels(self, combined, clean):
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def training_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        # loss=F.mse_loss(output, clean)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6, amsgrad=True)#weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', cooldown=2),
                "monitor": "train_loss"
            },
        }

    # return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.1),
    #     },
    # }

    # return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', cooldown=2),
    #         "monitor": "train_loss"
    #     },
    # }

class ConvAutoEncoder1_lightning(pl.LightningModule):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder1_lightning, self).__init__()
        self.learning_rate = 0.001
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

    def add_channels(self, combined, clean):
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def training_step(self, batch, batch_idx):
        combined, clean = batch
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        # loss = F.l1_loss(output, clean)
        loss=F.mse_loss(output, clean)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined, clean = batch
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        combined, clean = batch
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.mse_loss(output, clean)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.1),
            },
        }

class ConvAutoEncoder_1st(nn.Module):
    def __init__(self, h_dim=1024, z_dim=32):
        # N, 1, 64, 87
        super(ConvAutoEncoder_1st, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 8)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, (1, 1), ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, (1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

class ConvAutoEncoder_1d(nn.Module):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder_1d, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.conv1_bn = nn.BatchNorm1d(num_features=64)  # 1
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv2_bn = nn.BatchNorm1d(128)  # 2

        self.deconv2 = nn.ConvTranspose1d(128, 64, 1)
        self.deconv2_bn = nn.BatchNorm1d(64)
        self.deconv1 = nn.ConvTranspose1d(64, 1, 1)
        self.deconv1_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        layer1 = self.relu(self.conv1_bn(self.conv1(x)))
        layer2 = self.relu(self.conv2_bn(self.conv2(layer1)))

        layer3 = self.relu(self.deconv2_bn(self.deconv2(layer2)))
        layer4 = self.relu(self.deconv1_bn(self.deconv1(layer3+layer1)))
        return layer4

class ConvAutoEncoder_1d_stride(nn.Module):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder_1d_stride, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=2)
        self.conv1_bn = nn.BatchNorm1d(num_features=64)  # 1
        self.conv2 = nn.Conv1d(64, 128, 1, 2)
        self.conv2_bn = nn.BatchNorm1d(128)  # 2
        self.conv3=nn.Conv1d(128, 256, 1, 2)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.deconv3=nn.ConvTranspose1d(256, 128, 1, stride=2)
        self.deconv3_bn = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, 64, 1, 2)
        self.deconv2_bn = nn.BatchNorm1d(64)
        self.deconv1 = nn.ConvTranspose1d(64, 1, 1, 2)
        self.deconv1_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        layer1 = self.relu(self.conv1_bn(self.conv1(x)))
        layer2 = self.relu(self.conv2_bn(self.conv2(layer1)))
        layer3 = self.relu(self.conv3_bn(self.conv3(layer2)))

        layer3 = self.relu(self.deconv3_bn(self.deconv3(layer3)))
        layer4 = self.relu(self.deconv2_bn(self.deconv2(layer3)))
        layer5 = self.deconv1_bn(self.deconv1(layer4))
        return layer5

class ConvAutoEncoder_1d_latent(nn.Module):
    def __init__(self):
        # N, 1, 64, 87
        super(ConvAutoEncoder_1d_latent, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=2)
        self.conv1_bn = nn.BatchNorm1d(num_features=64)  # 1
        self.conv2 = nn.Conv1d(64, 128, 1, 2)
        self.conv2_bn = nn.BatchNorm1d(128)  # 2
        self.conv3=nn.Conv1d(128, 256, 1, 2)
        self.conv3_bn = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1, 2)
        self.conv4_bn = nn.BatchNorm1d(512)

        self.deconv4 = nn.ConvTranspose1d(512, 256, 1, stride=2)
        self.deconv4_bn = nn.BatchNorm1d(256)
        self.deconv3=nn.ConvTranspose1d(256, 128, 1, 2)
        self.deconv3_bn = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, 64, 1, 2)
        self.deconv2_bn = nn.BatchNorm1d(64)
        self.deconv1 = nn.ConvTranspose1d(64, 1, 1, 2)
        self.deconv1_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        layer1 = self.relu(self.conv1_bn(self.conv1(x)))
        layer2 = self.relu(self.conv2_bn(self.conv2(layer1)))
        layer3 = self.relu(self.conv3_bn(self.conv3(layer2)))
        layer4 = self.relu(self.conv4_bn(self.conv4(layer3)))

        layer5 = self.relu(self.deconv4_bn(self.deconv4(layer4)))
        layer6 = self.relu(self.deconv3_bn(self.deconv3(layer5)))
        layer7 = self.relu(self.deconv2_bn(self.deconv2(layer6)))
        layer8 = self.deconv1(layer7)
        return layer8


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
