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
    
    ## Generate subbands for noise
    #erb_bank.generate_subbands(signal)
    #
    ## Extract subbands
    #erb_subbands_signal = erb_bank.subbands[:, 1: -1]
    
    erb_subbands_signal = erb_bank.generate_subbands(signal)[:, 1:-1]
    loudness = torch.norm(erb_subbands_signal, dim=0)

    downsampler = torchaudio.transforms.Resample(sample_rate, target_sampling_rate)
    downsample_signal = downsampler(signal)

    return [sp_centroid[0], loudness, downsample_signal]

class SoundDataset(Dataset):
    def __init__(self, audio_path, frame_size, hop_size, sampling_rate, N_filter_bank, normalize):
        self.normalization = normalize
        self.audio_path = audio_path
        self.frame_size = frame_size
        self.hop_size   = hop_size
        self.sampling_rate = sampling_rate
        self.N_filter_bank = N_filter_bank
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        self.content = []

    def compute_dataset(self):
        size = len(self.audio)
        dataset_size_pre_dataaug = (size - self.frame_size) // self.hop_size
        dataset_size_pre_dataaug = min(dataset_size_pre_dataaug, 60)
        for i in range(dataset_size_pre_dataaug):
            segment = self.audio[i * self.hop_size: i * self.hop_size + self.frame_size]
            for j in range(9):
                pitch_shift = 3*j - 12
                segment_shifted = librosa.effects.pitch_shift(segment, self.sampling_rate, pitch_shift)
                segment_shifted = torch.tensor(segment_shifted)
                if self.normalization == True:
                    segment_shifted = (segment_shifted - torch.mean(segment_shifted)) / torch.std(segment_shifted)
                features = feature_extractor(segment_shifted, self.sampling_rate, self.N_filter_bank)
                self.content.append([features, segment_shifted])
        return self.content