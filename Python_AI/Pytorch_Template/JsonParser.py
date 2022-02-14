import json
import Model
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class JsonParser:
    def __init__(self, path):
        self.data: Dict = json.load(open(path))

    def get_loss_name(self):
        return self.data['loss']

    def get_model_name(self):
        return self.data['arch']['type']

    def get_optimizer_name(self):
        return self.data['optimizer']['type']

    def get_epochs(self):
        return self.data['trainer']['epochs']

    def get_model_save_path(self):
        return self.data['trainer']['save_dir']

    def get_saved_model(self):
        return self.data['paths']['model']

    def get_combined_audio(self):
        return self.data['paths']['combined_audio']

    def get_clean_audio(self):
        return self.data['paths']['clean_audio']

    def get_denoised_audio(self):
        return self.data['paths']['denoised_audio']

    def get_spectrograms(self):
        return self.data['spectrograms']

    def get_experiment_name(self):
        return self.data['file_name']

    def get_batch_size(self):
        return self.data['DataLoader']['batch_size']

    def get_model(self, device, gpu=True):
        class_ = getattr(Model, self.data['arch']['type'])
        if gpu:
            instance = class_().to(device)
        else:
            instance = class_()
        return instance

    def get_loss(self):
        class_ = getattr(nn, self.data['loss'])
        instance = class_()
        return instance

    def get_optimizer(self, model):
        class_ = getattr(torch.optim, self.data['optimizer']['type'])
        instance = class_(model.parameters(), lr=self.data['optimizer']['args']['lr'],
                          weight_decay=self.data['optimizer']['args']['weight_decay'])
        return instance

    def get_lr_scheduler(self, optimizer):
        class_ = getattr(torch.optim.lr_scheduler, self.data['lr_scheduler']['type'])
        instance = class_(optimizer=optimizer, step_size=self.data['lr_scheduler']['args']['step_size'],
                          gamma=self.data['lr_scheduler']['args']['gamma'])
        return instance

    def get_dataloader(self, data):
        return DataLoader(dataset=data, batch_size=self.data['DataLoader']['batch_size'],
                          shuffle=bool(self.data['DataLoader']['shuffle']),
                          num_workers=self.data['DataLoader']['num_workers'],
                          drop_last=bool(self.data['DataLoader']['drop_last']))
