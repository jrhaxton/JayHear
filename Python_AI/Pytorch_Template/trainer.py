import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
from natsort import natsorted
import os
from pydoc import locate
import torch.optim.lr_scheduler
import soundfile as sf
import pandas as pd
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning import Trainer

import Model
from JsonParser import JsonParser
from DataLoader import AudioDenoiserDataset


class Py_Trainer:
    def __init__(self, config, model, loss, optimizer, lr_scheduler, data_paths, device):
        self.config: JsonParser = config
        self.model: Model = model
        self.loss: nn.Loss = loss
        self.optimizer: torch.optim = optimizer
        self.lr_scheduler: torch.optim.lr_scheduler = lr_scheduler
        self.data_paths: Dict[str, List[str]] = data_paths
        self.device = device

    def save_checkpoint(self, model_to_save, path_to_save):
        torch.save(model_to_save['state'], path_to_save + 'state.pt')
        torch.save(model_to_save['optimizer'], path_to_save + 'optimizer.pt')

    def add_channels(self, combined, clean):
        """
        ==========
        Adds channel:
            [1, 128, 8] -> [1, 1, 128, 8]
        ==========
        return (batch_size, channels, height, length)
        """
        return combined.unsqueeze(dim=1).float(), clean.unsqueeze(dim=1).float()

    def sdr_loss(self, pred, label):
        return -(torch.sum(label ** 2) / torch.sum((pred - label) ** 2))

    def train_single_frames(self):
        torch.manual_seed(2)
        for epoch in range(self.config.get_epochs()):
            epoch_loss = 0
            model_to_save = {'state': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            spectrograms: List[str] = natsorted(os.listdir(self.data_paths['base']))
            loop = tqdm(spectrograms, leave=False)
            loop.set_description(f'Epoch [{epoch + 1}/{self.config.get_epochs()}]')
            for spectrogram in loop:
                data: AudioDenoiserDataset = AudioDenoiserDataset(self.data_paths['base'] + spectrogram)
                dataloader: DataLoader = self.config.get_dataloader(data)
                for i, (combined, clean) in enumerate(dataloader):
                    # Data augmentation
                    combined, clean = combined.to(self.device), clean.to(self.device)
                    combined, clean = self.add_channels(combined, clean)

                    # forward
                    decoded = self.model(combined)
                    _loss = self.loss(decoded, clean)

                    # backward
                    self.optimizer.zero_grad()
                    _loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # loss
                    epoch_loss += decoded.shape[0] * _loss.item()

                    # tqdm
                    loop.set_postfix(loss=_loss.item())
            print(f'Epoch: {epoch + 1} Loss: {epoch_loss / len(data)}')
            self.save_checkpoint(model_to_save, self.config.get_model_save_path())

    def loss_vae(self, recon_x, x, mu, logvar):
        BCE=F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return BCE + KLD

    def train_multi_frame(self):
        torch.manual_seed(2)
        for epoch in range(self.config.get_epochs()):
            # epoch_loss = 0
            # model_to_save = {'state': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            # spectrograms: List[str] = natsorted(os.listdir(self.data_paths['base']))
            # loop = tqdm(spectrograms, leave=False)
            # loop.set_description(f'Epoch [{epoch + 1}/{self.config.get_epochs()}]')
            # for spectrogram in loop:
            data: AudioDenoiserDataset = AudioDenoiserDataset("/media/braden/Rage Pro/Spectrogram/noisy4_SNRdb_0.0_clnsp4")
            dataloader: DataLoader = self.config.get_dataloader(data)
            for i, (combined, clean) in enumerate(dataloader):
                # Data augmentation
                combined, clean = combined.to(self.device), clean.to(self.device)
                combined = combined[:, 1:, :]  # Had to add a frame to save with numpy so removing the added frame
                combined, clean = self.add_channels(combined, clean)

                # forward
                recon_images, mu, logvar = self.model(combined)
                # _loss=self.sdr_loss(decoded, clean)
                _loss = self.loss_vae(recon_images, clean, mu, logvar)

                # backward
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # loss
                epoch_loss += decoded.shape[0] * _loss.item()

                # tqdm
                loop.set_postfix(loss=epoch_loss)
            print(f'Epoch: {epoch + 1} Loss: {epoch_loss / len(data)}')
        self.save_checkpoint(model_to_save, self.config.get_model_save_path())

    def lightning(self):
        dataset = AudioDenoiserDataset("/media/braden/Rage Pro/Spectrogram/noisy4_SNRdb_0.0_clnsp4")
        train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=16)
        # val_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False, drop_last=False)
        dir = "/media/braden/Rage Pro"

        #1. Initialize Model
        model = Model.ConvAutoEncoder8VAE_lightning()

        #2. Initialize Trainer
        trainer = Trainer(gpus=1, default_root_dir=dir, max_epochs=self.config.get_epochs())
        trainer.fit(model, train_loader)


def main(arguments):
    """Main func."""
    data_paths: Dict[str, List[str]] = {
        'base': "/media/braden/Rage Pro/Spectrogram/",
        'folders': natsorted(
            os.listdir("/home/braden/Environments/Research/Audio/Research(Refactored)/Data/training/"))}

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')
    config: JsonParser = JsonParser('config_files/experiment23.json')
    model: Model = config.get_model(device)
    loss: nn.Loss = config.get_loss()
    optimizer: torch.optim = config.get_optimizer(model)
    lr_scheduler: torch.optim.lr_scheduler = config.get_lr_scheduler(optimizer)

    trainer = Py_Trainer(config, model, loss, optimizer, lr_scheduler, data_paths, device)
    # trainer.train_multi_frame()
    trainer.lightning()
    # summary(model, (1, 129, 8))#(channels, H, W)
    print('=> Training Finished')

    # decoded=decoded_temp[:, 0, :, :]
    # decoded=decoded.cpu().detach().numpy()
    # normalize = MinMaxScaler(feature_range=(-1, 1))
    # decoded_normalized=[]
    # decoded_normalized=torch.tensor(np.array([normalize.fit_transform(x) for x in decoded]), requires_grad=True).unsqueeze(dim=1).float().to(self.device)


if __name__ == "__main__":
    main(sys.argv[1:])
