from pydub import AudioSegment
import os
from itertools import cycle
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import sys
import time

class CombineAudio:
    def __init__(self, Clean_folder: str, Background_folder: str, Combined_folder: str, Pad, SNR, FolderSize) -> None:
        self.Clean_folder: str = Clean_folder
        self.Background_folder: str = Background_folder
        self.Combined_folder: str = Combined_folder
        self.files_clean: list = natsorted(os.listdir(self.Clean_folder))
        self.files_background: list = natsorted(os.listdir(self.Background_folder))
        self.Pad=Pad
        self.SNR=SNR
        self.FolderSize=FolderSize
        self.folders=natsorted(os.listdir(self.Clean_folder))

    def audio_features(self, audio):
        audio = audio.set_frame_rate(11025)
        audio = audio.set_channels(1)
        return audio

    def equivalent_parameters(self, framerate, channels):
        if(framerate==11025 and channels==1):
            return True
        else:
            return False

    def combine_audio(self) -> None:
        for folder in self.folders:
            data=self.Clean_folder+'/'+folder
            files=natsorted(os.listdir(data))
            os.mkdir(self.Combined_folder+'/'+folder)
            for i, (clean, background) in enumerate(zip(files, self.files_background)):
                print(f'Folder: {folder} File: {i} of {len(files)}')
                try:
                    clean_audio = AudioSegment.from_mp3(data+'/'+clean)
                    background_audio = AudioSegment.from_wav(self.Background_folder+'/'+background)
                    clean_audio, background_audio=self.SNR.make_soundDB_equal(clean_audio, background_audio)
                    overlapped = clean_audio.overlay(background_audio, loop=True)
                    overlapped_audio=self.audio_features(overlapped)
                    if(self.equivalent_parameters(overlapped_audio.frame_rate, overlapped_audio.channels)):
                        print(f'Folder: {folder} File: {i} of {len(files)}')
                        file_handle = overlapped_audio.export(self.Combined_folder+'/'+folder+'/'+clean[:-3]+'wav', format="wav")
                    else:
                        print("Features not matching")
                        os.remove(self.Clean_folder+'/'+folder+'/'+clean)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(e, exc_tb.tb_lineno)
                    os.remove(self.Clean_folder+'/'+folder+'/'+clean)

    def run(self) -> None:
        self.combine_audio()
