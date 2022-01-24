import os
import sys
import librosa
import math
import numpy as np
from pesq import pesq
from numpy import linalg as LA
import mir_eval
from pystoi import stoi
import matplotlib.pyplot as plt
import pandas as pd

from JsonParser import JsonParser


class EvaluationMetrics:
    """
        Evaluation class to measure sound quality between clean and denoise audio
        The higher the metrics the better
    """

    def __init__(self, clean_path, denoised_path, sr):
        self.clean, _ = librosa.load(clean_path, sr=sr)
        self.denoised, _ = librosa.load(denoised_path, sr=sr)
        self.clean_path = clean_path
        self.denoised_path = denoised_path
        self.sr = sr

    def SNR(self):
        if self.clean.shape[0] > self.denoised.shape[0]:
            zeros=self.clean.shape[0]-self.denoised.shape[0]
            self.denoised = np.pad(self.denoised, (0, zeros))
        else:
            zeros = self.denoised.shape[0] - self.clean.shape[0]
            self.denoised = np.pad(self.denoised, (0, zeros))
        top = LA.norm(self.clean, 2) ** 2
        difference = self.clean - self.denoised
        bottom = LA.norm(difference, 2) ** 2
        return 10 * math.log10(top / bottom)

    def SDR(self):
        (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(self.clean, self.denoised)
        return float(sdr)

    def SI_SDR(self):
        pass

    def PESQ(self):
        if librosa.get_samplerate(self.clean_path) != 8000 or librosa.get_samplerate(self.denoised_path) != 8000:
            clean, _ = librosa.load(self.clean_path, sr=8000)
            denoised, _ = librosa.load(self.denoised_path, sr=8000)
            return pesq(8000, clean, denoised, 'nb')

    def STOI(self):
        return stoi(self.clean, self.denoised, self.sr, extended=False)


def save_to_dataframe(path: str, config: JsonParser, snr: float, sdr: float, pesq: float, stoi: float):
    df_results = pd.read_csv(path)
    df_new = pd.DataFrame({'Experiment': [config.get_experiment_name()], 'Model': [config.get_model_name()],
                           'Optimizer': [config.get_optimizer_name()], 'Loss': [config.get_loss_name()],
                           'Epochs': [config.get_epochs()], 'Spectrograms': [config.get_spectrograms()],
                           'SNR': [snr], 'SDR': [sdr],
                           'PESQ': [pesq], 'STOI': [stoi]})
    df_save = df_results.append(df_new, ignore_index=True)
    df_save.to_csv(path, index=False)
    print('=> Saving test statistics')


def test_sound_quality(config: JsonParser):
    print('=> Creating test statistics')
    evaluate: EvaluationMetrics = EvaluationMetrics(clean_path=config.get_clean_audio(),
                                                    denoised_path=config.get_denoised_audio(),
                                                    sr=11025)
    snr, sdr, pesq, stoi = evaluate.SNR(), evaluate.SDR(), evaluate.PESQ(), evaluate.STOI()
    save_to_dataframe('/home/braden/Environments/Research/Audio/Research(Refactored)/Test_Results/test.csv', config,
                      snr, sdr, pesq, stoi)


def main(arguments):
    # """Main func."""
    config: JsonParser = JsonParser(
        '/home/braden/Environments/JayHear/Python_AI/Pytorch_Template/config_files/experiment7.json')
    test_sound_quality(config)


if __name__ == "__main__":
    main(sys.argv[1:])
