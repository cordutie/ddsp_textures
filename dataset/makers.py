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
import pickle
from tqdm import tqdm

def read_wavs_from_folder(folder_path, sampling_rate, torch_type=True):
    audio_list = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            # Load the audio file using librosa
            audio, _ = librosa.load(file_path, sr=sampling_rate, mono=True)  # sr=None to preserve original sampling rate
            # audio to torch
            if torch_type:
                audio = torch.tensor(audio)
            audio_list.append(audio)      # Store the audio, sample rate, and filename
            
    return audio_list

def precompute_dataset(audio_path, output_path, frame_size, hop_size, sampling_rate, N_filter_bank, features_annotators, data_augmentation=False):
    os.makedirs(output_path, exist_ok=True)  # Create output directory if not exists
    index = []  # Stores metadata (paths to saved segments)
    erb_bank_just_in_case_lol = EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, 20, sampling_rate//2)

    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    for file in audio_files:
        file_path = os.path.join(audio_path, file)
        audio, _ = librosa.load(file_path, sr=sampling_rate, mono=True)
        
        size = len(audio)
        number_of_segments = (size - frame_size) // hop_size

        for i in range(number_of_segments):
            segment = audio[i * hop_size : i * hop_size + frame_size]

            if data_augmentation:
                for j in range(9):
                    pitch_shift = 3*j - 12
                    segment_pitched = librosa.effects.pitch_shift(y=segment, sr=sampling_rate, n_steps=pitch_shift)
                    segment_pitched = torch.tensor(segment_pitched, dtype=torch.float32)
                    segment_pitched = signal_normalizer(segment_pitched)

                    # Compute features
                    features = [feature_annotator(segment_pitched, sampling_rate, erb_bank_just_in_case_lol) for feature_annotator in features_annotators]

                    # Save to disk as a list: [segment, feature1, feature2, ...]
                    segment_data = [segment_pitched] + features  

                    segment_filename = f"{file}_{i}_shift{j}.pt"  
                    segment_path = os.path.join(output_path, segment_filename)
                    torch.save(segment_data, segment_path)
                    index.append(segment_path)  
            else:
                segment = torch.tensor(segment, dtype=torch.float32)
                segment = signal_normalizer(segment)  # Ensure normalization for non-augmented case

                # Compute features
                features = [feature_annotator(segment, sampling_rate, erb_bank_just_in_case_lol) for feature_annotator in features_annotators]

                # Save to disk as a list: [segment, feature1, feature2, ...]
                segment_data = [segment] + features  

                segment_filename = f"{file}_{i}.pt"
                segment_path = os.path.join(output_path, segment_filename)
                torch.save(segment_data, segment_path)
                index.append(segment_path)  

    # Save the index file
    with open(os.path.join(output_path, "dataset_index.pkl"), "wb") as f:
        pickle.dump(index, f)

    print(f"Dataset precomputed! {len(index)} segments saved.")


class DDSP_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        # Load the index file
        index_file = os.path.join(dataset_path, "dataset_index.pkl")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Dataset index file not found: {index_file}")

        with open(index_file, "rb") as f:
            self.index = pickle.load(f)

        print(f"Dataset loaded! {len(self.index)} segments available.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        segment_path = self.index[idx]
        
        # Load the segment and features
        data = torch.load(segment_path, weights_only=True)  
        
        # # Debugging prints
        # print(f"Loaded data from {segment_path}:")
        # print(f"Type: {type(data)}")  # Should be a list or tuple
        # print(f"Length: {len(data)}")  # Should match expected number of features
        # print(f"Segment shape: {data[0].shape}")  # Check if it's a single segment
        # for i, feature in enumerate(data[1:]):
        #     print(f"Feature {i+1} shape: {feature.shape}")

        return data