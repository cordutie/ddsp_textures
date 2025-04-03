from ddsp_textures.architectures.DDSP import *
from ddsp_textures.auxiliar.seeds import *
from ddsp_textures.auxiliar.filterbanks import *
from ddsp_textures.auxiliar.configuration import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.evaluation.model_tester import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import signal
import sys

def save_checkpoint(model, optimizer, epoch, loss_history, main_loss_history, regularizer_history, directory, name="checkpoint.pth", last = True):
    if last == True:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'main_loss_history': main_loss_history,
            'regularizer_history': regularizer_history
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict()
            #, 'optimizer_state_dict': optimizer.state_dict()
        }

    checkpoint_path = os.path.join(directory, name)
    torch.save(checkpoint, checkpoint_path)

def train(json_path):
    # Get the parameters from the json file
    with open(json_path, 'r') as file:
        parameters = json.load(file)
    architecture = parameters['architecture']
    if architecture == 'DDSP_TexEnv':
        print("DDSP_TexEnv trainer")
        trainer_TexEnv(json_path)
    elif architecture == 'DDSP_PVAE':
        print("DDSP_PVAE trainer")
        # trainer_PVAE(json_path)
    else:
        raise ValueError(f"Architecture {architecture} not recognized.")

def trainer_TexEnv(json_path):
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
    # stems                           = actual_parameters['stems']
    loss_function                   = actual_parameters['loss_function']
    regularizers                    = actual_parameters['regularizers']
    batch_size                      = actual_parameters['batch_size']
    epochs                          = actual_parameters['epochs']
    models_directory                = actual_parameters['models_directory']  
    alpha                           = actual_parameters['alpha']
    beta                            = actual_parameters['beta']
    data_augmentation               = actual_parameters['data_augmentation'] 
    filter_low_lim                = actual_parameters['filter_low_lim']
    filter_high_lim               = actual_parameters['filter_high_lim']
    
    # Get a name for the model like model_37 or somehing like that
    model_name = get_next_model_folder(models_directory)
    print(f"model id: {model_name}\n")

    # Construct the directory
    directory = os.path.join(models_directory, model_name)

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Save the json file in json_path to the directory
    json_file_new_name = "configurations.json"
    json_file_path = os.path.join(directory, json_file_new_name)
    os.system(f"cp {json_path} {json_file_path}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Dataset maker
    if audio_path[-4:] == ".pkl":
        actual_dataset = pickle.load(open(audio_path, 'rb'))
        # save the dataset
        dataset_path = os.path.join(directory, "dataset.pkl")
        with open(dataset_path, 'wb') as file:
            pickle.dump(actual_dataset, file)
    else:
        precompute_dataset(
        audio_path  = audio_path,
        output_path = os.path.join(directory, "dataset"),
        frame_size  = frame_size,
        hop_size    = hop_size,
        sampling_rate = sampling_rate,
        N_filter_bank = N_filter_bank,
        features_annotators = features_annotators,
        data_augmentation=data_augmentation
        )
        actual_dataset = DDSP_Dataset(os.path.join(directory, "dataset"))        
    
    # Dataloader
    dataloader = DataLoader(actual_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Make seed
    seed = seed_maker(frame_size, sampling_rate, N_filter_bank, filter_low_lim, filter_high_lim).to(device)

    # Pickle save seed
    seed_path = os.path.join(directory, "seed.pkl")
    with open(seed_path, 'wb') as file:
        pickle.dump(seed, file)

    # Model initialization
    model = architecture(input_dimensions, hidden_size_enc, hidden_size_dec, deepness_enc, deepness_dec, param_per_env, frame_size, N_filter_bank, device, seed).to(device)
    
    # Initialize the optimizer
    lr = 0.5 * 1/(10 ** (6 - (deepness_enc + deepness_dec)/2))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Frame sizes for downsampling
    new_frame_size, new_sampling_rate   = frame_size // 4, sampling_rate // 4

    # Filter bank initialization for loss functions
    erb_bank    = fb.EqualRectangularBandwidth(frame_size,     sampling_rate, N_filter_bank, filter_low_lim,     filter_high_lim)
    log_bank    = fb.Logarithmic(          new_frame_size, new_sampling_rate, M_filter_bank, 10, new_sampling_rate // 4)

    import torchaudio
    downsampler = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate).to(device)

    # Variables to track the best model and loss history
    best_loss = float('inf')
    loss_dict = {}
    loss_history = []
    main_loss_history = []
    regularizer_history = []
    
    # Training loop
    print("Training starting!")

    number_of_features  = len(features_annotators)
    print("number_of_features", number_of_features)
    num_of_regularizers = len(regularizers)
    print("num_of_regularizers", num_of_regularizers)

    # Early stopping parameters
    patience = 100  # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_regularizer = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Unpack batch data
            og_signal            = batch[0].to(device, non_blocking=True)
            # segments_stems_batch = batch[1].to(device, non_blocking=True)
            features_batch       = []
            for i in range(1, number_of_features + 1):
                feature = batch[i].to(device, non_blocking=True)
                if feature.ndimension() == 1:  # If feature is 1D (i.e., shape: (batch_size,))
                    feature = feature.unsqueeze(-1)  # Add an extra dimension, making it shape (batch_size, 1)
                features_batch.append(feature)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass (Note that if stems=True, the model will return the stems)
            reconstructed_signal = model(features_batch).to(device)

            # # Decide what to use to compare
            # if stems==True:
            #     og_signal = segments_stems_batch
            # else:
            #     og_signal = segments_batch

            # Compute main loss
            loss_main = loss_function(og_signal, reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha, beta) 
            
            loss_regularizer = torch.tensor(0.0).to(device)

            # # if there are regularizers and the model uses stems lets remake the signal to compute their features
            # if stems==True and num_of_regularizers>0:
            #     reconstructed_signal= TexEnv_stems_to_signals_batches(reconstructed_signal, seed)
            
            for i in range(num_of_regularizers):
                # Make features from reconstructed signal
                feature_reconstructed = regularizers[i](reconstructed_signal, sampling_rate, erb_bank).to(device)
                feature_og            = features_batch[i].to(device)
                loss_regularizer      += 0.2 * 1/(feature_reconstructed.numel()) * torch.norm(feature_reconstructed - feature_og, p=2)

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

        # if epoch % 250 == 0 save it as a checkpoint and call it checkpoint_{epoch}.pth
        if epoch % 250 == 0 and epoch != 0:
            save_checkpoint(model, optimizer, epoch + 1, loss_history, main_loss_history, regularizer_history, directory, name=f"checkpoint_{epoch}.pth", last = False)

        # Save the best model and implement early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, loss_history, main_loss_history, regularizer_history, directory, name="best_model.pth", last = False)
            epochs_no_improve = 0  # Reset counter if loss improves
        else:
            epochs_no_improve += 1  # Increment counter if no improvement

        if len(loss_history)!=1:
            # Plot the training losses
            plt.figure(figsize=(15, 5))  # Adjust the size to fit three plots side by side

            # Plot Total Loss
            plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
            plt.plot(loss_history[1:], label='Total Loss', color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Total Loss')
            plt.legend()

            # Plot Main Loss
            plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
            plt.plot(main_loss_history[1:], label='Main Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Main Loss')
            plt.legend()

            # Plot Regularizer Term
            plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
            plt.plot(regularizer_history[1:], label='Regularizer Term', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Regularizer Term')
            plt.legend()

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Save and show the plot
            plt.savefig(os.path.join(directory, 'loss_plot.png'))
            plt.close()

        print(f"Epochs without improvement: {epochs_no_improve}/{patience}")

        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            sys.exit("Early stopping triggered.")  # Stop training if no improvement after patience epochs
    
    print("Training complete.")

def trainer_from_checkpoint_TexEnv(model_folder):
    # check if inside the folder there is a checkpoint.pth, a configurations.json and a dataset pickle
    checkpoint_path = os.path.join(model_folder, "checkpoint.pth")
    json_path = os.path.join(model_folder, "configurations.json")
    dataset_path = os.path.join(model_folder, "dataset.pkl")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found in {model_folder}.")
    if not os.path.exists(json_path):
        raise ValueError(f"Configuration file not found in {model_folder}.")
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset pickle not found in {model_folder}.")
    
    # parent folder of the model_folder
    models_directory = os.path.dirname(model_folder)
    model_name = os.path.basename(model_folder)
    directory = os.path.join(models_directory, model_name)

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
    # stems                           = actual_parameters['stems']
    loss_function                   = actual_parameters['loss_function']
    regularizers                    = actual_parameters['regularizers']
    batch_size                      = actual_parameters['batch_size']
    epochs                          = actual_parameters['epochs']
    models_directory                = actual_parameters['models_directory']  

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Dataset reader with pickle
    actual_dataset = pickle.load(open(dataset_path, 'rb'))
    
    # Dataloader
    dataloader = DataLoader(actual_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model loading
    model, seed, checkpoint = model_loader(checkpoint_path, json_path, print_parameters=False)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.5*1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Frame sizes for downsampling
    new_frame_size, new_sampling_rate   = frame_size // 4, sampling_rate // 4

    # Filter bank initialization for loss functions
    erb_bank    = fb.EqualRectangularBandwidth(frame_size,     sampling_rate, N_filter_bank, filter_low_limit,     filter_high_limit)
    log_bank    = fb.Logarithmic(          new_frame_size, new_sampling_rate, M_filter_bank, 10, new_sampling_rate // 4)

    import torchaudio
    downsampler = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate).to(device)

    # Variables to track the best model and loss history
    best_loss = float('inf')

    epoch_start = checkpoint['epoch']

    loss_dict = {}
    loss_history        = checkpoint['loss_history']
    main_loss_history   = checkpoint['main_loss_history']
    regularizer_history = checkpoint['regularizer_history']

    best_loss = min(loss_history)

    # Training loop
    print("Training RE-starting!")

    number_of_features  = len(features_annotators)
    print("number_of_features", number_of_features)
    num_of_regularizers = len(regularizers)
    print("num_of_regularizers", num_of_regularizers)

    # Early stopping parameters
    patience = 250  # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0

    for epoch in range(epochs-epoch_start):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_regularizer = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + epoch_start + 1}/{epochs}"):
            # Unpack batch data
            og_signal              = batch[0].to(device, non_blocking=True)
            # segments_stems_batch = batch[1].to(device, non_blocking=True)
            features_batch = []
            for i in range(2, number_of_features + 2):
                feature = batch[i].to(device, non_blocking=True)
                if feature.ndimension() == 1:  # If feature is 1D (i.e., shape: (batch_size,))
                    feature = feature.unsqueeze(-1)  # Add an extra dimension, making it shape (batch_size, 1)
                features_batch.append(feature)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass (Note that if stems=True, the model will return the stems)
            reconstructed_signal = model(features_batch).to(device)

            # # Decide what to use to compare
            # if stems==True:
            #     og_signal = segments_stems_batch
            # else:
            #     og_signal = segments_batch

            # Compute main loss
            loss_main = loss_function(og_signal, reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler) 
            
            loss_regularizer = torch.tensor(0.0).to(device)

            # # if there are regularizers and the model uses stems lets remake the signal to compute their features
            # if stems==True and num_of_regularizers>0:
            #     reconstructed_signal= TexEnv_stems_to_signals_batches(reconstructed_signal, seed)
            
            for i in range(num_of_regularizers):
                # Make features from reconstructed signal
                feature_reconstructed = regularizers[i](reconstructed_signal, sampling_rate, erb_bank).to(device)
                feature_og            = features_batch[i].to(device)
                loss_regularizer      += 0.2 * 1/(feature_reconstructed.numel()) * torch.norm(feature_reconstructed - feature_og, p=2)

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

        # if epoch % 250 == 0 save it as a checkpoint and call it checkpoint_{epoch}.pth
        if epoch % 250 == 0 and epoch != 0:
            save_checkpoint(model, optimizer, epoch + 1, loss_history, main_loss_history, regularizer_history, directory, name=f"checkpoint_{epoch}.pth", last = False)

        # Save the best model and implement early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, loss_history, main_loss_history, regularizer_history, directory, name="best_model.pth", last = False)
            epochs_no_improve = 0  # Reset counter if loss improves
        else:
            epochs_no_improve += 1  # Increment counter if no improvement

        if len(loss_history)!=1:
            # Plot the training losses
            plt.figure(figsize=(15, 5))  # Adjust the size to fit three plots side by side

            # Plot Total Loss
            plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
            plt.plot(loss_history[1:], label='Total Loss', color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Total Loss')
            plt.legend()

            # Plot Main Loss
            plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
            plt.plot(main_loss_history[1:], label='Main Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Main Loss')
            plt.legend()

            # Plot Regularizer Term
            plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
            plt.plot(regularizer_history[1:], label='Regularizer Term', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Regularizer Term')
            plt.legend()

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Save and show the plot
            plt.savefig(os.path.join(directory, 'loss_plot.png'))
            plt.close()

        print(f"Epochs without improvement: {epochs_no_improve}/{patience}")

        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            sys.exit("Early stopping triggered.")  # Stop training if no improvement after patience epochs
    
    print("Training complete.")
