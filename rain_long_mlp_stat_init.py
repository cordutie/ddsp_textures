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
input_size = 2**13
hidden_size = 256  # Example hidden size
N_filter_bank = 16  # Example filter bank size
frame_size = 2**15  # Example frame size
hop_size = 2**14  # Example hop size
sampling_rate = 44100  # Example sampling rate
compression = 8  # Placeholder for compression
batch_size = 32

# Type of model
model_type = 'DDSP_textenv_mlp'

# type of loss
loss_type = 'statistics_loss'

# Sound path to create dataset
audio_path = 'sounds/rain.wav'

# model name
model_name = 'rain_long_mlp_stat'

####################### Standard code #######################

# Construct the directory and file path
directory = os.path.join("trained_models", model_name)

# Create the directory if it does not exist
os.makedirs(directory, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed creation
seed_path = os.path.join(directory, "seed.pkl")
seed = seed_maker(frame_size, sampling_rate, N_filter_bank)
# Save the dataset to a file
with open(seed_path, 'wb') as file:
    pickle.dump(seed, file)
seed = seed.to(device)

# Dataset maker
dataset = SoundDataset(audio_path=audio_path, frame_size=frame_size, hop_size=hop_size, sampling_rate=sampling_rate, N_filter_bank=N_filter_bank, normalize=False)
print("Generating dataset from ", audio_path)
dataset.compute_dataset()
actual_dataset = dataset.content

# Save the dataset to a file
dataset_path = os.path.join(directory, "dataset.pkl")
with open(dataset_path, 'wb') as file:
    pickle.dump(actual_dataset, file)

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

# Training loop
num_epochs = 5  # Define the number of epochs
best_loss = float('inf')  # Initialize best_loss to a high value

print("Training starting!")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Unpack batch data
        features, segments = batch
        spectral_centroid = features[0].unsqueeze(1).to(device)
        loudness          = features[1].to(device)
        ds_signal         = features[2].to(device)
        segments          = segments.to(device)

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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

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