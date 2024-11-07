from ddsp_textures.architectures.DDSP import *
from ddsp_textures.auxiliar.seeds import *
from ddsp_textures.auxiliar.filterbanks import *
from ddsp_textures.auxiliar.configuration import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.signal_processors.synthesizers import *

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
    if architecture == 'DDSP_SubEnv':
        print("DDSP_SubEnv trainer")
        trainer_SubEnv(json_path)
    elif architecture == 'DDSP_PVAE':
        print("DDSP_PVAE trainer")
        # trainer_PVAE(json_path)
    else:
        raise ValueError(f"Architecture {architecture} not recognized.")

def trainer_SubEnv(json_path):
    # Get the parameters from the json file
    actual_parameters = model_json_to_parameters(json_path)
    
    # Print parameters in bold
    print("\033[1mModel Parameters:\033[0m\n")
    for key, value in actual_parameters.items():
        print(f"{key}: {value}")
    
    # Unpack parameters
    audio_path                      = actual_parameters['audio_path']
    frame_size                      = actual_parameters['frame_size']
    hop_size                        = actual_parameters['hop_size']
    sampling_rate                   = actual_parameters['sampling_rate']
    features_annotators             = actual_parameters['features']
    input_dimensions                = actual_parameters['input_dimensions']
    hidden_size_enc                 = actual_parameters['hidden_size_enc']
    hidden_size_dec                 = actual_parameters['hidden_size_dec']
    deepness_enc                    = actual_parameters['deepness_enc']
    deepness_dec                    = actual_parameters['deepness_dec']
    param_per_env                   = actual_parameters['param_per_env']
    N_filter_bank                   = actual_parameters['N_filter_bank']
    M_filter_bank                   = actual_parameters['M_filter_bank']
    architecture                    = actual_parameters['architecture']
    stems                           = actual_parameters['stems']
    loss_function                   = actual_parameters['loss_function']
    regularizers                    = actual_parameters['regularizers']
    batch_size                      = actual_parameters['batch_size']
    epochs                          = actual_parameters['epochs']
    models_directory                = actual_parameters['models_directory']  
    
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

    # Dataset maker
    dataset        = DDSP_Dataset(audio_path, frame_size, hop_size, sampling_rate, N_filter_bank, features_annotators)
    actual_dataset = dataset.compute_dataset()
    
    # Dataloader
    dataloader = DataLoader(actual_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    model = architecture(input_dimensions, hidden_size_enc, hidden_size_dec, deepness_enc, deepness_dec, param_per_env, frame_size, N_filter_bank, stems).to(device)
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Frame sizes for downsampling
    new_frame_size, new_sampling_rate   = frame_size // 4, sampling_rate // 4

    # Filter bank initialization for loss functions
    erb_bank    = fb.EqualRectangularBandwidth(frame_size,     sampling_rate, N_filter_bank, 20,     sampling_rate // 2)
    log_bank    = fb.Logarithmic(          new_frame_size, new_sampling_rate, M_filter_bank, 10, new_sampling_rate // 4)

    import torchaudio
    downsampler = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)

    # Variables to track the best model and loss history
    best_loss = float('inf')
    loss_dict = {}
    loss_history = []
    main_loss_history = []
    regularizer_history = []

    # Training loop
    print("Training starting!")

    number_of_features = len(features_annotators)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_regularizer = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Unpack batch data
            segments_batch       = batch[0].to(device)
            segments_stems_batch = batch[1].to(device)
            features_batch = []
            for i in range(2, number_of_features + 2):
                features_batch.append(batch[i]).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed_signal = model(features_batch).to(device)

            # Decide what to use to compare
            if stems==True:
                og_signal = segments_stems_batch
            else:
                og_signal = segments_batch

            # Compute main loss
            loss_main = loss_function(og_signal, reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler) 
            
            loss_regularizer = 0
            for i in range(1,number_of_features+1):
                # Make features from reconstructed signal
                feature_reconstructed = regularizers[i](reconstructed_signal, sampling_rate, erb_bank).to(device)
                feature_og            = features_batch[i].to(device)
                loss_regularizer      += torch.norm(feature_reconstructed - feature_og, p=2) 

            # Final loss
            loss_total = loss_main + loss_regularizer
            
            # Backward pass and optimization
            loss_total.backward()
            optimizer.step()

            # Accumulate the losses
            running_loss        += loss_total.item()
            running_main_loss   += loss_main.item()
            running_regularizer += loss_regularizer.item()

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
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Main Loss: {epoch_main_loss:.4f}, Regularizer: {epoch_regularizer:.4f}")

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