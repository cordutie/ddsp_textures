from architectures.DDSP import *
from auxiliar.auxiliar import *
from auxiliar.filterbanks import *
from dataset.dataset_maker import *
from loss.loss_functions import *
from signal_processors.textsynth_env import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os

# PARAMETERS
input_size = 1024
hidden_size = 128  # Example hidden size
N_filter_bank = 16  # Example filter bank size
frame_size = 4096  # Example frame size
hop_size = 2048  # Example hop size
sampling_rate = 44100  # Example sampling rate
compression = 8  # Placeholder for compression
batch_size = 32

# Type of model
model_type = 'DDSP_textenv_gru'

# type of loss
loss_type = 'multispectrogram_loss'

# Sound path to create dataset
audio_path = 'sounds/fire.wav'

# model name
model_name = 'fire_short_gru_multispec'

####################### Standard code #######################

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
    raise FileNotFoundError(f"{seed_path} not found. Please ensure the dataset is created and saved correctly.")
seed = seed.to(device)

# Load the dataset
dataset_path = os.path.join(directory, "dataset.pkl")
if os.path.exists(dataset_path):
    with open(dataset_path, 'rb') as file:
        actual_dataset = pickle.load(file)
else:
    raise FileNotFoundError(f"{dataset_path} not found. Please ensure the dataset is created and saved correctly.")

dataloader = DataLoader(actual_dataset, batch_size=batch_size, shuffle=True)

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

# Load checkpoints if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    print(f"Loaded checkpoint from epoch {start_epoch} with best loss {best_loss:.4f}")
else:
    print("No checkpoint found, starting training from scratch.")

# Training loop
num_epochs = 100000  # Total number of epochs to train

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}"):
        # Unpack batch data
        features, segments = batch
        spectral_centroid = features[0].unsqueeze(1).to(device)
        loudness          = features[1].to(device)
        ds_signal         = features[2].to(device)
        segments          = segments.to(device)

        # Ensure the correct shape for spectral_centroid and loudness
        # if spectral_centroid.dim() == 1:
        #     spectral_centroid = spectral_centroid.unsqueeze(1)
        # if loudness.dim() == 1:
        #     loudness = loudness.unsqueeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed_signal = model(spectral_centroid, loudness, ds_signal)

        # Compute loss
        loss = loss_function(segments, reconstructed_signal)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}], Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

    # Update best loss if necessary
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), checkpoint_best_path)
        print("Best model saved with loss {:.4f}".format(best_loss))

print("Training complete.")