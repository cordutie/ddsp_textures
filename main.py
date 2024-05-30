from models.model_1 import DDSP_textenv, SoundDataset, multispectrogram_loss
from auxiliar.auxiliar import seed_maker
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move it to the appropriate device
hidden_size = 128  # Example hidden size
N_filter_bank = 16  # Example filter bank size
frame_size = 2**15  # Example frame size
sampling_rate = 44100  # Example sampling rate
compression = 8  # Placeholder for compression

# Seed creation
seed = seed_maker(frame_size, sampling_rate, N_filter_bank)
# Save the dataset to a file
with open("seed.pkl", 'wb') as file:
    pickle.dump(seed, file)
    
# Dataset maker
audio_path = 'sounds/fire_long.wav'
dataset = SoundDataset(audio_path=audio_path, frame_size=frame_size, hop_size=2**13, sampling_rate=sampling_rate, N_filter_bank=N_filter_bank)
print("Generating dataset from ", audio_path)
dataset.compute_dataset()
actual_dataset = dataset.content

# Save the dataset to a file
with open("dataset.pkl", 'wb') as file:
    pickle.dump(actual_dataset, file)

dataloader = DataLoader(actual_dataset, batch_size=32, shuffle=True)

# Model initialization
model = DDSP_textenv(hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=2, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Hyperparameters for multiscale FFT (loss function)
scales = [2048, 1024, 512, 256]  # Example scales
overlap = 0.5  # Example overlap

# Load previous state if available
checkpoint_path = 'local_checkpoint.pth'
best_loss = float('inf')  # Initialize best_loss to a high value

# Training loop
num_epochs = 3  # Define the number of epochs

print("Training starting!")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Unpack batch data
        features, segments = batch
        spectral_centroid = features[0].unsqueeze(1).to(device)
        loudness = features[1].to(device)
        segments = segments.to(device)

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
        torch.save(model.state_dict(), 'best_local_model.pth')
        print("Best model saved with loss {:.4f}".format(best_loss))

print("Training complete.")