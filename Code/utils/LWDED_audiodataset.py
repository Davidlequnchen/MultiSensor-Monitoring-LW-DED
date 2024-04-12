import json
import os
import sys
from typing import Callable, Dict

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import PIL.Image
import torch
import torch.utils.data
import pandas as pd
import torchaudio
import torch.nn as nn
import numpy as np


class LDEDAudioDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_df, audio_directory, mel_spectrogram, target_sample_rate, device):
        self.annotations = annotation_df
        self.audio_directory = audio_directory
        self.device = device
        self.mel_spectrogram = mel_spectrogram.to(self.device)
        self.target_sample_rate = target_sample_rate
        
        self.class_to_idx = {
            'Balling': 0,
            'Non-defective': 1,
            'Laser-off': 2,
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_sample_label(index)
        audio_signal, sr = torchaudio.load(audio_sample_path)
        audio_signal = audio_signal.to(self.device)

        # Conduct the transformation
        signal_mel_spectrogram = self.mel_spectrogram(audio_signal)
        
        return signal_mel_spectrogram, label
    
    def _get_audio_sample_path(self, index):
        audio_file_name = self.annotations.iloc[index, 0]  # Get the .wav file name from the first column
        path = os.path.join(self.audio_directory, audio_file_name)
        return path

    def _get_sample_label(self, index):
        class_name = self.annotations.iloc[index, 3]  # Get the class label from the fourth column
        return self.class_to_idx[class_name]



if __name__ == "__main__":


    SAMPLE_RATE = 44100

    Multimodal_dataset_PATH = "/home/chenlequn/pan1/Dataset/Laser-Wire-DED-ThermalAudio-Dataset"
    Annotation_file_path = os.path.join(Multimodal_dataset_PATH, "Annotation")
    Dataset_path = os.path.join(Multimodal_dataset_PATH, 'Dataset')
    final_audio_directory = os.path.join(Multimodal_dataset_PATH, 'Dataset', "audio")
    final_image_directory = os.path.join(Multimodal_dataset_PATH, 'Dataset', "thermal_images")


 
    df_audio_dataset = pd.read_hdf(os.path.join(Dataset_path, 'df_audio_dataset_with_annotations(raw_audio).h5'), key='df')
    df_audio_dataset = df_audio_dataset.dropna(subset=['label_1'])


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                            n_fft=512,
                                                            hop_length=256,
                                                            n_mels=32)

    MFCCs = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=20)
    # spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=SAMPLE_RATE, hop_length=256)

    audiodataset = LDEDAudioDataset(df_audio_dataset,
                                    final_audio_directory,
                                    mel_spectrogram,
                                    SAMPLE_RATE,
                                    device)
    
    # random check
    print(f"There are {len(audiodataset)} samples in the audiodataset dataset.")
    audio_inputs, label = audiodataset[100]
 


    print (audio_inputs.shape, label)
    # (1, 32, 18) for 100 ms, (1, 32, 7) for 40 ms, 25 Hz
    # print ("all target enoded: " + str(all_targets))
    # print ("one-hot target enoded: " + str(all_targets_ohe))


