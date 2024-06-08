from signal_processors.textsynth_env import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio

def feature_extractor(signal, sample_rate, N_filter_bank, target_sampling_rate=11025):
    size = signal.shape[0]
    sp_centroid = torchaudio.functional.spectral_centroid(signal, sample_rate, 0, torch.hamming_window(size), size, size, size) 

    low_lim = 20  # Low limit of filter
    high_lim = sample_rate / 2  # Centre freq. of highest filter

     # Initialize filter bank
    erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
    
    # Generate subbands for noise
    erb_bank.generate_subbands(signal)
    
    # Extract subbands
    erb_subbands_signal = erb_bank.subbands[:, 1: -1]
    loudness = torch.norm(erb_subbands_signal, dim=0)

    downsampler = torchaudio.transforms.Resample(sample_rate, target_sampling_rate)
    downsample_signal = downsampler(signal)

    return [sp_centroid[0], loudness, downsample_signal]

class SoundDataset(Dataset):
    def __init__(self, audio_path, frame_size, hop_size, sampling_rate, N_filter_bank):
        self.audio_path = audio_path
        self.frame_size = frame_size
        self.hop_size   = hop_size
        self.sampling_rate = sampling_rate
        self.N_filter_bank = N_filter_bank
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        self.content = []

    def compute_dataset(self):
        audio_tensor = torch.tensor(self.audio)
        size = audio_tensor.shape[0]
        dataset_size = (size - self.frame_size) // self.hop_size
        for i in range(dataset_size):
            segment = audio_tensor[i * self.hop_size: i * self.hop_size + self.frame_size]
            features = feature_extractor(segment, self.sampling_rate, self.N_filter_bank)
            self.content.append([features, segment])

