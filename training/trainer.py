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
    
    elif name=='medium':
        input_size = 2**13
        hidden_size = 256  # Example hidden size
        N_filter_bank = 16  # Example filter bank size
        frame_size = 2**15  # Example frame size
        hop_size = 2**15  # Example hop size
        sampling_rate = 44100  # Example sampling rate
        compression = 8  # Placeholder for compression
        batch_size = 32
    
    elif name=='long':
        input_size = 2**13
        hidden_size = 256  # Example hidden size
        N_filter_bank = 16  # Example filter bank size
        frame_size = 2**16  # Example frame size
        hop_size = 2**16  # Example hop size
        sampling_rate = 44100  # Example sampling rate
        compression = 8  # Placeholder for compression
        batch_size = 32
        
    else:
        raise NameError(f"{name} is not a valid frame type")
    
    return input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size


def trainer(frame_type, model_type, loss_type, audio_path, model_name):
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
    if model_type   == 'DDSP_textenv_gru':
        model = DDSP_textenv_gru(                       hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=3, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)
    # elif model_type == 'DDSP_textenv_mlp':
    #     model = DDSP_textenv_mlp(input_size=input_size, hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=3, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)
    # elif model_type == 'DDSP_textenv_stems_gru':
    #     model = DDSP_textenv_stems_gru(                  hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=3, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)
    # elif model_type == 'DDSP_textenv_stems_mlp':
    #     model = DDSP_textenv_stems_mlp(input_size=input_size, hidden_size=hidden_size, N_filter_bank=N_filter_bank, deepness=3, compression=compression, frame_size=frame_size, sampling_rate=sampling_rate, seed=seed).to(device)
    else:
        raise NameError("Invalid model type")
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # loss initialization
    if loss_type == 'multispectrogram_loss':
        loss_function = multispectrogram_loss
    elif loss_type == 'statistics_loss':
        loss_function = batch_statistics_loss
        # Create the filter banks for this loss
        erb_bank = fb.EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, 20, sampling_rate // 2) # the entire frame has to run over this
        log_bank = fb.Logarithmic(frame_size // 4, 11025, 6, 10, 11025 // 4)                                      # a downsampled version has to run over this (only works if sampling_rate = 44100 whch is highly recommended anyway)
        # batch_size = 24
    elif loss_type == 'sub_statistics_loss':
        loss_function = batch_sub_statistics_loss
        # Create the filter banks for this loss
        erb_bank = fb.EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, 20, sampling_rate // 2) # the entire frame has to run over this
    elif loss_type == 'stems_loss':
        loss_function = stems_loss
    else:
        raise NameError("Invalid loss type")

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
    num_epochs = 9999  # Total number of epochs to train
    avg_loss_list = []  # List to store average loss for each epoch
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}"):
            # Unpack batch data
            features, segments = batch
            spectral_centroid = features[0].unsqueeze(1).to(device)
            rate              = features[1].to(device)
            segments          = segments.to(device)

            # Ensure the correct shape for spectral_centroid and loudness
            # if spectral_centroid.dim() == 1:
            #     spectral_centroid = spectral_centroid.unsqueeze(1)
            # if loudness.dim() == 1:
            #     loudness = loudness.unsqueeze(1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed_signal = model(spectral_centroid, rate)

            # Compute loss (if stats loss is used the filter bank has to be added)
            if loss_type == 'statistics_loss':
                loss = loss_function(segments, reconstructed_signal, erb_bank, log_bank)
            if loss_type == 'sub_statistics_loss':
                loss = loss_function(segments, reconstructed_signal, erb_bank)
            else:
                loss = loss_function(segments, reconstructed_signal)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        avg_loss_list.append(avg_loss)
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
    
    
    # Plot the average loss for each epoch
    plt.figure()
    plt.plot(avg_loss_list, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Model: {model_name}, Best Loss: {best_loss:.4f}')
    plt.legend()
    plt.savefig(os.path.join(directory, f'{model_name}_training_loss.png'))
    plt.close()

    print("Training complete.")