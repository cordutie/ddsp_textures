from ddsp_textures.signal_processors.synthesizers import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio
from ddsp_textures.auxiliar.features    import *
from ddsp_textures.auxiliar.filterbanks import *
import os

def read_wavs_from_folder(folder_path, sampling_rate):
    audio_list = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            # Load the audio file using librosa
            audio, _ = librosa.load(file_path, sr=sampling_rate, mono=True)  # sr=None to preserve original sampling rate
            # audio to torch
            audio = torch.tensor(audio)
            audio_list.append(audio)      # Store the audio, sample rate, and filename
            
    return audio_list

# features_annotators_list = list of features annotators
class DDSP_Dataset(Dataset):
    def __init__(self, audio_path, frame_size, hop_size, sampling_rate, N_filter_bank, features_annotators_list):
        self.features_annotators_list = features_annotators_list
        self.audio_path               = audio_path
        self.sampling_rate            = sampling_rate
        self.audios_list              = read_wavs_from_folder(audio_path, sampling_rate)
        print("Audio loaded from ", audio_path)
        self.segments_list = []
        for audio in self.audios_list:
            size = len(audio)
            number_of_segments = (size - frame_size) // hop_size
            for i in range(number_of_segments):
                segment = audio[i * hop_size : i * hop_size + frame_size]
                self.segments_list.append(segment)
        self.erb_bank_just_in_case_lol = EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, 20, sampling_rate//2)

    def compute_dataset(self):
        actual_dataset = []
        print("Computing dataset\n...")
        for segment in self.segments_list:
            # dataset element
            segment_annotated = []
            # prepocessing
            # segment = audio_improver(segment, self.sampling_rate, 4) # look at thiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiis
            segment = signal_normalizer(segment)
            # adding the segment to the element
            segment_annotated.append(segment)
            segment_stems = features_envelopes_stems(segment, 0, self.erb_bank_just_in_case_lol)
            segment_annotated.append(segment_stems)
            # features computation
            for feature_annotator in self.features_annotators_list:
                feature_loc = feature_annotator(segment, self.sampling_rate, self.erb_bank_just_in_case_lol)
                # adding features to the element
                segment_annotated.append(feature_loc)
            actual_dataset.append(segment_annotated)
        print("Dataset computed!")
        return actual_dataset
    
    # the output of this function is a list made of lists. Each sub list has this shape [segment, stems, feature1, feature2, ...]
    # note that each feature is a list of tensors