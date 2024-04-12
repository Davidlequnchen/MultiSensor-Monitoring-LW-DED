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
    def __init__(self,
                 annotation_df,
                 audio_directories,
                 samples,
                 mel_spectrogram,
                 #  MFCCs,
                 #  spectral_centroid,
                 target_sample_rate,
                 device):
        self.annotations = annotation_df
        self.audio_directories = audio_directories
        self.samples = samples
        self.device = device
        self.mel_spectrogram = mel_spectrogram.to(self.device)
        # self.MFCCs = MFCCs.to(self.device)
        # self.spectral_centroid = spectral_centroid.to(self.device)
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
        # signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)

        # conduct the transformations
        signal_mel_spectrogram = self.mel_spectrogram(audio_signal)
        # signal_mfcc = self.MFCCs(audio_signal)
        # signal_spectral_centroid = self.spectral_centroid(audio_signal)
        
        return signal_mel_spectrogram, label
    
    def _get_audio_sample_path(self, index):
        sample_number = int(self.annotations.iloc[index, 7])  # Get the sample number from the 8th column
        audio_file_name = self.annotations.iloc[index, 2]
        audio_dir = self.audio_directories[self.samples.index(sample_number)]  # Find the correct audio directory
        path = os.path.join(audio_dir, audio_file_name)
        return path

    def _get_sample_label(self, index):
        class_name = self.annotations.iloc[index, 8]
        return self.class_to_idx[class_name]



def get_sample_directories_from_df(df, Dataset_path):
    # Extract unique sample numbers from the DataFrame
    unique_sample_numbers = df['Sample number'].unique()
    
    # Generate the full paths for image and audio directories
    image_directories = [os.path.join(Dataset_path, str(sample_number), 'images') for sample_number in unique_sample_numbers]
    audio_directories = [os.path.join(Dataset_path, str(sample_number), 'raw_audio') for sample_number in unique_sample_numbers]
    
    return image_directories, audio_directories



if __name__ == "__main__":


    SAMPLE_RATE = 44100

    # def get_sample_directories(base_path, sample_numbers):
    #     sample_directories = []
    #     for sample_number in sample_numbers:
    #         sample_directories.append(os.path.join(base_path, f'25Hz/{sample_number}'))
    #     return sample_directories

    Multimodal_dataset_PATH = "/home/chenlequn/Dataset/LDED_acoustic_visual_monitoring_dataset"
    Dataset_path = os.path.join(Multimodal_dataset_PATH, f'25Hz')
    
    samples = [21, 22, 23, 24, 26, 32]
    # sample_directories = get_sample_directories(Multimodal_dataset_PATH, samples)

    # # Get lists of image and audio directories for each sample
    # image_directories = [os.path.join(sample_dir, 'images') for sample_dir in sample_directories]
    # audio_directories = [os.path.join(sample_dir, 'raw_audio') for sample_dir in sample_directories]

    df_multimodal = pd.read_hdf(os.path.join(Dataset_path, 'spatiotemporal_fused_multimodal.h5'), key='df')
    df_multimodal = df_multimodal.dropna(subset=['class_name'])

    samples = [21, 22, 23, 24, 26, 32]
    image_directories, audio_directories = get_sample_directories_from_df(df_multimodal, Dataset_path)


    # # Combine all annotation files into one DataFrame
    # all_annotation_dfs = []
    # for sample_dir, sample_number in zip(sample_directories, samples):
    #     annotation_file = os.path.join(sample_dir, f'annotations_{sample_number}.csv')  # Update the file name
    #     annotation_df = pd.read_csv(annotation_file)
    #     all_annotation_dfs.append(annotation_df)
    # combined_annotation_df = pd.concat(all_annotation_dfs)



    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")



    img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32,32)), # original image size: (640,480)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[47.5905037932074], std=[61.13815627108221]), 
            # note that if resize is before normalization, need to re-calculate the mean and std; if resize is after normalize, could induce distortions
        ]) # calculation of mean and std is shown in jupyter notebook 1.
 
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                            n_fft=512,
                                            hop_length=256,
                                            n_mels=32)

    MFCCs = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=20)
    # spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=SAMPLE_RATE, hop_length=256)
    
                
    mmd = MultimodalDataset(df_multimodal,
                            image_directories,
                            audio_directories,
                            samples,
                            img_transform,
                            mel_spectrogram,
                            # MFCCs,
                            # spectral_centroid,
                            SAMPLE_RATE,
                            # NUM_SAMPLES,
                            device)

    visiondataset = LDEDVisionDataset(df_multimodal,
                                      image_directories,
                                      samples,
                                      img_transform,
                                      device)

    audiodataset = LDEDAudioDataset(df_multimodal,
                                    audio_directories,
                                    samples,
                                    mel_spectrogram,
                                    # MFCCs,
                                    # spectral_centroid,
                                    SAMPLE_RATE,
                                    device)
    
    # random check
    print(f"There are {len(mmd)} samples in the multimodal dataset.")
    print(f"There are {len(visiondataset)} samples in the visiondataset dataset.")
    print(f"There are {len(audiodataset)} samples in the audiodataset dataset.")
    multimodal_inputs, label = mmd[21]
    # image_input_vision, label_vision = visiondataset[22]
    # audio_input_audioset, label_audio = audiodataset[22]

    # test_dataloader = DataLoader(visiondataset, batch_size=32, shuffle=False, num_workers=2)
    # all_targets_ohe = []
    # all_targets = []
    # with torch.no_grad():
    #     for inputs, targets in test_dataloader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         label_vision_ohe = torch.nn.functional.one_hot(targets, num_classes = 4)
    #         all_targets.append(targets.cpu().numpy())
    #         all_targets_ohe.append(label_vision_ohe.cpu().numpy())
    # all_targets = np.concatenate(all_targets, axis=0)
    # all_targets_ohe = np.concatenate(all_targets_ohe, axis=0)

    print (multimodal_inputs[0].shape, multimodal_inputs[1].shape, label)
    # print (image_input_vision.shape, label_vision)
    # print (audio_input_audioset.shape, label_audio)
    # (1, 32, 18) for 100 ms, (1, 32, 7) for 40 ms, 25 Hz
    # print ("all target enoded: " + str(all_targets))
    # print ("one-hot target enoded: " + str(all_targets_ohe))

