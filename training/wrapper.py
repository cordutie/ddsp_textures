from architectures.DDSP import *
from auxiliar.seeds import *
from auxiliar.filterbanks import *
from auxiliar.configuration import *
from dataset.makers import *
from loss.functions import *
from signal_processors.synthesizers import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import signal

def save_checkpoint(model, optimizer, epoch, loss_history, main_loss_history, regularizer_history, directory, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'main_loss_history': main_loss_history,
        'regularizer_history': regularizer_history
    }
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(directory, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)

def train(json_path):
    # Get the parameters from the json file
    with open(json_path, 'r') as file:
        parameters = json.load(file)
    architecture = parameters['architecture']
    if architecture == 'DDSP_TextEnv':
        trainer_SubEnv(json_path)
    elif architecture == 'DDSP_PVAE':
        trainer_PVAE(json_path)
    else:
        raise ValueError(f"Architecture {architecture} not recognized.")

def trainer_SubEnv(json_path):
    # Get the parameters from the json file
    parameters_dict = model_name_to_parameters(json_path)
    
    # Print parameters in bold
    print("\033[1mModel Parameters:\033[0m\n")
    for key, value in parameters_dict.items():
        print(f"{key}: {value}")
    
    # Unpack parameters
    sound_path       = parameters_dict['audio_path']
    frame_size       = parameters_dict['frame_size']
    hop_size         = parameters_dict['hop_size']
    sampling_rate    = parameters_dict['sampling_rate']
    features_annotator = parameters_dict['features_annotator']
    batch_features_annotator = parameters_dict['batch_features_annotator']
    freq_avg_level   = parameters_dict['freq_avg_level']
    hidden_size      = parameters_dict['hidden_size']
    deepness         = parameters_dict['deepness']        
    param_per_env    = parameters_dict['param_per_env']
    N_filter_bank    = parameters_dict['N_filter_bank']
    model_class      = parameters_dict['architecture']
    onset_detection_model_path = parameters_dict['onset_detection_model_path']
    loss_function    = parameters_dict['loss_function']
    regularization   = parameters_dict['regularization']
    batch_size       = parameters_dict['batch_size']       
    num_epochs       = parameters_dict['epochs']      
    models_directory = parameters_dict['directory']     
    
    # Get a name for the model like model_37 or somehing like that
    model_name = get_next_model_folder(models_directory)
    print(f"model id: {model_name}\n")

    # Construct the directory and print it
    directory = os.path.join(models_directory, model_name)
    # print(f"model directory: {os.path.abspath(directory)}\n")

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Save the json file in json_path to the directory
    json_file_new_name = "configurations.json"
    json_file_path = os.path.join(directory, json_file_new_name)
    os.system(f"cp {json_path} {json_file_path}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed creation, saving and sending to device
    seed_path = os.path.join(directory, "seed.pkl")
    seed = seed_maker(frame_size, sampling_rate, N_filter_bank)
    with open(seed_path, 'wb') as file:
        pickle.dump(seed, file)
    seed = seed.to(device)

    # Dataset maker
    dataset        = DDSP_Dataset(sound_path, frame_size, hop_size, sampling_rate, features_annotator, freq_avg_level)
    actual_dataset = dataset.compute_dataset()
    
    # Dataloader
    dataloader = DataLoader(actual_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization (EDIT HERE WHEN DDSP_PVAE IS READY)
    model = model_class(hidden_size, deepness, param_per_env, seed).to(device)
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Filter bank initialization for loss functions
    erb_bank          = fb.EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, 20, sampling_rate // 2)
    new_frame_size, new_sampling_rate   = frame_size // 4, sampling_rate // 4
    log_bank          = fb.Logarithmic(new_frame_size, new_sampling_rate, 6, 10, new_sampling_rate // 4)
    
    # Variables to track the best model and loss history
    best_loss = float('inf')
    loss_dict = {}
    loss_history = []
    main_loss_history = []
    regularizer_history = []

    # Training loop
    print("Training starting!")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_regularizer = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Unpack batch data
            features, segments = batch
            feature_0 = features[0].unsqueeze(1).to(device) # transform vector of size N to matrix of NX1
            feature_1 = features[1].unsqueeze(1).to(device)
            segments  = segments.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed_signal = model(feature_0, feature_1).to(device)

            # Compute main loss
            loss_main = loss_function(segments, reconstructed_signal, N_filter_bank, sampling_rate, erb_bank, log_bank) 
            
            loss_regularizer = 0
            if regularization:
                # Make features from reconstructed signal
                features_reconstructed = batch_features_annotator(reconstructed_signal, sampling_rate).to(device)
                features_og            = torch.cat((feature_0, feature_1), dim=1)
                # print("Features og:\n", features_og)
                # print("Features reconstructed:\n", features_reconstructed)
                loss_regularizer = torch.sqrt(torch.nn.functional.mse_loss(features_og, features_reconstructed)) * 10

            # Final loss
            loss_total = loss_main + loss_regularizer
            
            # Backward pass and optimization
            loss_total.backward()
            optimizer.step()

            # Accumulate the losses
            running_loss        += loss_total.item()
            running_main_loss   += loss_main.item()
            running_regularizer += loss_regularizer.item() if regularization else 0

        epoch_loss = running_loss / len(dataloader)
        epoch_main_loss = running_main_loss / len(dataloader)
        epoch_regularizer = running_regularizer / len(dataloader)
        
        loss_history.append(epoch_loss)
        main_loss_history.append(epoch_main_loss)
        regularizer_history.append(epoch_regularizer)
        
        # put everything inside the loss_dictionary
        loss_dict['loss_total'] = loss_history
        loss_dict['loss_main'] = main_loss_history
        loss_dict['loss_regularizer'] = regularizer_history
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Main Loss: {epoch_main_loss:.4f}, Regularizer: {epoch_regularizer:.4f}")

        # Save the loss dictionary using pickle
        loss_dict_path = os.path.join(directory, "loss_dict.pkl")
        with open(loss_dict_path, 'wb') as file:
            pickle.dump(loss_dict, file)
        
        # Save the model checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, epoch + 1, loss_history, main_loss_history, regularizer_history, directory)

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, loss_history, main_loss_history, regularizer_history, directory, is_best=True)

        # Plot the training losses
        plt.figure(figsize=(15, 5))  # Adjust the size to fit three plots side by side

        # Plot Total Loss
        plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
        plt.plot(loss_history, label='Total Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()

        # Plot Main Loss
        plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
        plt.plot(main_loss_history, label='Main Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Main Loss')
        plt.legend()

        # Plot Regularizer Term
        plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
        plt.plot(regularizer_history, label='Regularizer Term', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Regularizer Term')
        plt.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save and show the plot
        plt.savefig(os.path.join(directory, 'loss_plot.png'))
        plt.show()
    
    print("Training complete.")

    # Rename the model directory to mark training completion
    complete_directory = directory + "_complete"
    os.rename(directory, complete_directory)
    
def trainer_PVAE(json_path):
    return None