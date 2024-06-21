from architectures.DDSP import *
from auxiliar.auxiliar import *
from auxiliar.filterbanks import *
from dataset.dataset_maker import *
from loss.loss_functions import *
from signal_processors.textsynth_env import *
from training.initializer import *
from training.trainer import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def short_long_decoder(name):
    if name=='short':
        input_size = 1024
        hidden_size = 128  # Example hidden size
        N_filter_bank = 16  # Example filter bank size
        frame_size = 4096  # Example frame size
        hop_size = 2048  # Example hop size
        sampling_rate = 44100  # Example sampling rate
        compression = 8  # Placeholder for compression
        batch_size = 32
        
    elif name=='long':
        input_size = 2**13
        hidden_size = 256  # Example hidden size
        N_filter_bank = 16  # Example filter bank size
        frame_size = 2**15  # Example frame size
        hop_size = 2**14  # Example hop size
        sampling_rate = 44100  # Example sampling rate
        compression = 8  # Placeholder for compression
        batch_size = 32
        
    else:
        raise NameError(f"{name} is not a valid frame type")
    
    return input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size

def model_loader(frame_type, model_type, loss_type, audio_path, model_name, best):    
    input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size = short_long_decoder(frame_type)

    # Construct the directory and file path
    directory = os.path.join("trained_models", model_name)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the seed
    seed_path = os.path.join(directory, "seed.pkl")
    if os.path.exists(seed_path):
        with open(seed_path, 'rb') as file:
            seed = pickle.load(file)
    else:
        raise NameError(f"{seed_path} not found. Please ensure the dataset is created and saved correctly.")
    seed = seed.to(device)

    # Model initialization
    if model_type == 'DDSP_textenv_gru':
        model = DDSP_textenv_gru(                       hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=3, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)
    elif model_type == 'DDSP_textenv_mlp':
        model = DDSP_textenv_mlp(input_size=input_size, hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=3, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # loss initialization
    if loss_type == 'multispectrogram_loss':
        loss_function = multispectrogram_loss
    elif loss_type == 'statistics_loss':
        loss_function = batch_statistics_loss

    # Checkpoint path
    checkpoint_path = os.path.join(directory, "checkpoint.pkl")
    checkpoint_best_path = os.path.join(directory, "checkpoint_best_local.pkl")

    if best==True:
        checkpoint = torch.load(checkpoint_best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def model_tester(frame_type, model_type, loss_type, audio_path, model_name, best):
    model = model_loader(frame_type, model_type, loss_type, audio_path, model_name, best)
    input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size = short_long_decoder(frame_type)
    
    audio_og, _ = librosa.load(audio_path, sr=sampling_rate)
    
    audio_tensor = torch.tensor(audio_og)
    size = audio_tensor.shape[0]
    N_segments = (size - frame_size) // hop_size
    
    content = []
    for i in range(N_segments):
        segment = audio_tensor[i * hop_size: i * hop_size + frame_size]
        target_loudness = torch.std(segment)
        segment = (segment - torch.mean(segment)) / torch.std(segment)
        features = feature_extractor(segment, sampling_rate, N_filter_bank)
        content.append([features, segment, target_loudness])
    
    audio_final = torch.zeros(frame_size + (N_segments-1)*hop_size)
    window = torch.hann_window(frame_size)

    for i in range(N_segments):
        [features, segment, target_loudness] = content[i]
        [sp_centroid, loudness, downsample_signal] = features
        sp_centroid = sp_centroid.unsqueeze(0)
        synthesized_segment = model.synthesizer(downsample_signal, sp_centroid, loudness, target_loudness)
        synthesized_segment = synthesized_segment * window
        audio_final[i * hop_size: i * hop_size + frame_size] += synthesized_segment
    
    audio_final = audio_final.detach().cpu().numpy()
    
    return audio_og, audio_final