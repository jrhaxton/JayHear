import os
import sys
from natsort.natsort import natsorted# type: ignore
from torch.utils.data import Dataset
import numpy as np# type: ignore

class AudioDenoiserDataset(Dataset):

    def __init__(self, audio_dir):
        self.audio_dir=audio_dir
        self.audio_files=natsorted(os.listdir(self.audio_dir))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_sample_path=self.get_audio_sample_path(index)
        combined, clean=self.get_labels(audio_sample_path)
        return combined, clean


    def get_audio_sample_path(self, index):
        path=os.path.join(self.audio_dir, self.audio_files[index])
        return path

    def get_labels(self, audio_sample_path):
        audio=np.load(audio_sample_path, allow_pickle=True)
        combined=audio[0]
        #clean=self.add_column(clean)
        clean=audio[1]
        #combined=self.add_column(combined)
        return combined, clean

    def add_column(self, audio):
        z = np.zeros((256,3), dtype=np.int64)
        audio=np.append(audio, z, axis=1)
        return audio