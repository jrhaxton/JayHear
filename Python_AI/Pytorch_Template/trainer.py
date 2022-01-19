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

import Model
from JsonParser import JsonParser
from DataLoader import AudioDenoiserDataset



class Trainer:
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

    def train_single_frames(self):
        torch.manual_seed(2)
        for epoch in range(self.config.get_epochs()):
            epoch_loss = 0
            model_to_save = {'state': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            spectrograms: List[str] = natsorted(os.listdir(self.data_paths['base']))
            loop = tqdm(spectrograms, leave=False)
            loop.set_description(f'Epoch [{epoch + 1}/{self.config.get_epochs()}]')
            for spectrogram in loop:
                data: AudioDenoiserDataset = AudioDenoiserDataset(self.data_paths['base']+spectrogram)
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

                    #loss
                    epoch_loss += decoded.shape[0] * _loss.item()

                    #tqdm
                    loop.set_postfix(loss=_loss.item())
            print(f'Epoch: {epoch + 1} Loss: {epoch_loss / len(data)}')
            self.save_checkpoint(model_to_save, self.config.get_model_save_path())

    def train_multi_frame(self):
        torch.manual_seed(2)
        for epoch in range(self.config.get_epochs()):
            model_to_save = {'state': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            loop = tqdm(self.data_paths['folders'], leave=False)
            loop.set_description(f'Epoch [{epoch + 1}/{self.config.get_epochs()}]')
            for folder in loop:
                spectrograms: List[str] = natsorted(os.listdir(self.data_paths['base'] + folder))
                for frames in spectrograms:
                    data: AudioDenoiserDataset = AudioDenoiserDataset(self.data_paths['base'] + folder + '/' + frames)
                    dataloader: DataLoader = self.config.get_dataloader(data)
                    for i, (combined, clean) in enumerate(dataloader):
                        # Data augmentation
                        combined, clean = combined.to(self.device), clean.to(self.device)
                        combined = combined[:, 1:, :]  # Had to add a frame to save with numpy so removing the added frame
                        combined, clean = self.add_channels(combined, clean)

                        # forward
                        decoded = self.model(combined)
                        _loss = self.loss(decoded, clean)

                        # backward
                        self.optimizer.zero_grad()
                        _loss.backward()
                        self.optimizer.step()
                        self.lr_scheduler.step()

                        # tqdm
                        loop.set_postfix(loss=_loss.item())
            self.save_checkpoint(model_to_save, self.config.get_model_save_path())


def main(arguments):
    """Main func."""
    data_paths: Dict[str, List[str]] = {
        'base': "/media/braden/Rage Pro/Spectrogram/",
        'folders': natsorted(os.listdir("/home/braden/Environments/Research/Audio/Research(Refactored)/Data/training/"))}

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')
    config: JsonParser = JsonParser('config_files/experiment6.json')
    model: Model = config.get_model(device)
    loss: nn.Loss = config.get_loss()
    optimizer: torch.optim = config.get_optimizer(model)
    lr_scheduler: torch.optim.lr_scheduler = config.get_lr_scheduler(optimizer)

    trainer = Trainer(config, model, loss, optimizer, lr_scheduler, data_paths, device)
    trainer.train_single_frames()
    # summary(model, (1, 129))#(channels, H, W)
    print('=> Training Finished')

if __name__ == "__main__":
    main(sys.argv[1:])
