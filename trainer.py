from models.model_1 import DDSP_textenv, SoundDataset, multispectrogram_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move it to the appropriate device
hidden_size = 128  # Example hidden size
N_filter_bank = 16  # Example filter bank size
frame_size = 2**15  # Example frame size
sampling_rate = 44100  # Example sampling rate
compression = 8  # Placeholder for compression

# Load the seed
seed_path = 'seed.pkl'
if os.path.exists(seed_path):
    with open(seed_path, 'rb') as file:
        seed = pickle.load(file)
else:
    raise FileNotFoundError(f"{seed_path} not found. Please ensure the dataset is created and saved correctly.")

# Model initialization
model = DDSP_textenv(hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=2, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)

# Load the dataset
dataset_path = 'dataset.pkl'
if os.path.exists(dataset_path):
    with open(dataset_path, 'rb') as file:
        actual_dataset = pickle.load(file)
else:
    raise FileNotFoundError(f"{dataset_path} not found. Please ensure the dataset is created and saved correctly.")

dataloader = DataLoader(actual_dataset, batch_size=32, shuffle=True)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Hyperparameters for multiscale FFT (loss function)
scales = [2048, 1024, 512, 256]  # Example scales
overlap = 0.5  # Example overlap

# Load previous state if available
checkpoint_path = 'local_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    print(f"Loaded checkpoint from epoch {start_epoch} with best loss {best_loss:.4f}")
else:
    print("No checkpoint found, starting training from scratch.")
    start_epoch = 0
    best_loss = float('inf')

num_epochs = 5  # Total number of epochs to train

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}"):
        # Unpack batch data
        features, segments = batch
        spectral_centroid = features[0].to(device)
        loudness = features[1].to(device)
        segments = segments.to(device)

        # Debug prints for shapes
        # print(f"spectral_centroid shape: {spectral_centroid.shape}")
        # print(f"loudness shape: {loudness.shape}")
        # print(f"segments shape: {segments.shape}")

        # Ensure the correct shape for spectral_centroid and loudness
        if spectral_centroid.dim() == 1:
            spectral_centroid = spectral_centroid.unsqueeze(1)
        if loudness.dim() == 1:
            loudness = loudness.unsqueeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed_signal, _ = model(spectral_centroid, loudness)

        # Compute loss
        loss = multispectrogram_loss(segments, reconstructed_signal, scales, overlap)

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
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved with loss {:.4f}".format(best_loss))

print("Training complete.")