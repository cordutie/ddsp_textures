from ddsp_textures.signal_processors.synthesizers import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio
from ddsp_textures.auxiliar.features import *

# Audio improver for segments -----------------------------------------------

def audio_improver(signal_tensor, sampling_rate, level):
    signal_filtered  = torchaudio.functional.bandpass_biquad(signal_tensor, sample_rate = sampling_rate, central_freq = 10000, Q = 0.1)
    freq_mean    = computer_freq_avg(signal_filtered, sampling_rate)
    # filtered audio is centered around the spectral centroid
    segment_centered = torchaudio.functional.bandpass_biquad(signal_filtered, sample_rate = sampling_rate, central_freq = freq_mean, Q = 1)
    # improved audio is the sum of the centered audio and the original audio
    segment_improved = level*segment_centered + signal_tensor
    # Normalization
    segment_improved = (segment_improved - torch.mean(segment_improved)) / torch.std(segment_improved)
    return segment_improved

# Dataset maker -------------------------------------------------------------
def segment_annotator(signal_tensor, sampling_rate, level, features_annotator):
    # Audio is improved
    segment_improved = audio_improver(signal_tensor, sampling_rate, level)
    # Features are computed on the improved audio
    features         = features_annotator(segment_improved, sampling_rate)
    return [features, segment_improved]

class DDSP_Dataset(Dataset):
    def __init__(self, audio_path, frame_size, hop_size, sampling_rate, features_annotator, freq_avg_level):
        self.level = freq_avg_level
        self.features_annotator = features_annotator
        self.audio_path = audio_path
        self.sampling_rate = sampling_rate
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        print("Audio loaded from ", audio_path)
        size = len(self.audio)
        dataset_size = (size - frame_size) // hop_size
        self.segments = []
        for i in range(dataset_size):
            segment = self.audio[i * hop_size : i * hop_size + frame_size]
            self.segments.append(segment)
        print("Final dataset size will be: ",dataset_size)

    def compute_dataset(self):
        actual_dataset = []
        print("Computing dataset\n...")
        for i in range(len(self.segments)):
            segment = self.segments[i]
            segment = (segment - np.mean(segment)) / np.std(segment) # normalization
            segment_tensor = torch.tensor(segment) # make a tensor from the segment
            segment_annotated = segment_annotator(segment_tensor, self.sampling_rate, self.level, self.features_annotator)
            actual_dataset.append(segment_annotated)
        print("Dataset computed!")
        return actual_dataset