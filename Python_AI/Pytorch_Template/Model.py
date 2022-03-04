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
            nn.ConvTranspose2d(512, 256, (1, 1)),
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

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to('cuda')
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = h, h
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        encoder = self.encoder(x)
        z, mu, logvar = self.bottleneck(encoder)
        decoder = self.decoder(z)
        return decoder, mu, logvar

    def add_channels(self, combined, clean):
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def loss_vae(self, recon_x, x, mu, logvar):
        BCE=F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return BCE + KLD

    def training_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output, mu, logvar = self(combined)
        loss = self.loss_vae(output, clean, mu, logvar)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        combined, clean = batch
        combined = combined[:, 1:, :]
        combined, clean = self.add_channels(combined, clean)
        output = self(combined)
        loss = F.l1_loss(output, clean)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
                                     amsgrad=False)  # weight_decay=1e-6
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
        # self.block1=nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 8)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(num_features=64),
        # )
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(64, 128, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        # )
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(128, 256, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        # )
        # self.block4 = nn.Sequential(
        #     nn.Conv2d(256, 512, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(512),
        # )
        # self.block5 = nn.Sequential(
        #     nn.Conv2d(512, 1024, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1024),
        # )
        # self.unblock5 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(512),
        # )
        # self.unblock4 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        # )
        # self.unblock3 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        # )
        # self.unblock2 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, (1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        # )
        # self.unblock1 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 1, (1, 1)),
        #     nn.Tanh()
        # )

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
            nn.ConvTranspose2d(512, 256, (1, 1)),
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
        # block1 = self.block1(x)
        # block2 = self.block2(block1)
        # block3 = self.block3(block2)
        # block4 = self.block4(block3)
        # block5 = self.block5(block4)
        #
        # unblock5 = self.unblock5(block5)
        # unblock4 = self.unblock4(unblock5 + block4)
        # unblock3 = self.unblock3(unblock4)
        # unblock2 = self.unblock2(unblock3 + block2)
        # unblock1 = self.unblock1(unblock2)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6,
                                     amsgrad=True)  # weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2),
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
        loss = F.mse_loss(output, clean)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # weight_decay=1e-6
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.1),
            },
        }


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class ConvAutoEncoder_Dense(nn.Module):
    def __init__(self, h_dim=1024, z_dim=32):
        # N, 1, 64, 87
        super(ConvAutoEncoder_Dense, self).__init__()
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
            nn.ConvTranspose2d(512, 256, (1, 1)),
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

        # self.linear1 = nn.Linear(66048, 2)
        #
        # self.linear2 = nn.Linear(2, 66048)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to('cuda')
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = h, h
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        encoder = self.encoder(x)
        z, mu, logvar = self.bottleneck(encoder)
        decoder=self.decoder(z)
        return decoder, mu, logvar

        # flatten = torch.flatten(x6)
        # linear1 = self.linear1(flatten)
        #
        # linear2 = self.linear2(linear1)
        # unflatten = linear2.view(x6.shape)
