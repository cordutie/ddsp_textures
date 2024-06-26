from signal_processors.textsynth_env import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio

# def feature_extractor(signal, sample_rate, N_filter_bank, target_sampling_rate=11025):
#     size = signal.shape[0]
#     sp_centroid = torchaudio.functional.spectral_centroid(signal, sample_rate, 0, torch.hamming_window(size), size, size, size) 

#     low_lim = 20  # Low limit of filter
#     high_lim = sample_rate / 2  # Centre freq. of highest filter

#      # Initialize filter bank
#     erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
    
#     # Generate subbands for noise
#     erb_bank.generate_subbands(signal)
    
#     # Extract subbands
#     erb_subbands_signal = erb_bank.subbands[:, 1: -1]
#     loudness = torch.norm(erb_subbands_signal, dim=0)

#     downsampler = torchaudio.transforms.Resample(sample_rate, target_sampling_rate)
#     downsample_signal = downsampler(signal)

#     return [sp_centroid[0], loudness, downsample_signal]

# for audios of more than 2 minute and long windows, recommended hop size = 2*frame_size. If less than 2 minutes try hop size = frame_size (this will generate overlaps between the data).
class SoundDataset(Dataset):
    def __init__(self, audio_path, frame_size, hop_size, sampling_rate, normalize):
        self.normalization = normalize
        self.audio_path = audio_path
        self.frame_size = frame_size
        self.hop_size   = hop_size
        self.sampling_rate = sampling_rate
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        self.content = []

    def compute_dataset(self):
        size = len(self.audio)
        pre_dataset_size = (size - 4 * self.frame_size) // self.hop_size
        pre_dataset_size = min(pre_dataset_size, 5) #Changeeeeeeeeeeeeeeeeeeeeeeee

        print("Final dataset size will be: ",pre_dataset_size*9*5)
        
        pre_dataset=[]

        # Segments are added to the dataset. The segments are bigger than necessary to be able to apply time stretching
        for i in range(pre_dataset_size):
            segment = self.audio[i * self.hop_size: i * self.hop_size + 4*self.frame_size]
            pre_dataset.append(segment)
        
        # Data augmentation: pitch shifting
        for i in range(len(pre_dataset)):
            for j in range(4):
                pitch_shift_left  = -1.5*(j+1) + (2*np.random.uniform(0, 1)-1)*(3-j)/3
                pitch_shift_right =  1.5*(j+1) + (2*np.random.uniform(0, 1)-1)*(3-j)/3
                segment_shifted_left  = librosa.effects.pitch_shift(segment, sr=self.sampling_rate, n_steps=pitch_shift_left)
                segment_shifted_right = librosa.effects.pitch_shift(segment, sr=self.sampling_rate, n_steps=pitch_shift_right)
                pre_dataset.append(segment_shifted_left)
                pre_dataset.append(segment_shifted_right)
        
        # Data augmentation: rate change + labeling
        for i in range(len(pre_dataset)):
            segment = pre_dataset[i]
            for j in range(5):
                # Audio is rate shifted
                rate = 2**(np.random.normal(0, 1)*(4-j)/4)
                segment_rate_shifted = librosa.effects.time_stretch(segment, rate=rate)
                segment_rate_shifted = segment_rate_shifted[int(0.5*self.frame_size):int(0.5*self.frame_size)+self.frame_size] # audio is cropped to be as long as the frame size
                segment_rate_shifted_tensor = torch.tensor(segment_rate_shifted)   
                # Audio is normalized
                if self.normalization == True:
                    segment_rate_shifted_tensor = (segment_rate_shifted_tensor - torch.mean(segment_rate_shifted_tensor)) / torch.std(segment_rate_shifted_tensor)
                #Features are computed
                feature_0 = torchaudio.functional.spectral_centroid(segment_rate_shifted_tensor, self.sampling_rate, 0, torch.hamming_window(self.frame_size), self.frame_size, self.frame_size, self.frame_size)[0]
                feature_1 = torch.tensor([rate])
                features = [feature_0, feature_1]
                self.content.append([features, segment_rate_shifted_tensor])
                
        return self.content