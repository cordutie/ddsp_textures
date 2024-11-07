from ddsp_textures.architectures.DDSP import *
from ddsp_textures.auxiliar.seeds import *
from ddsp_textures.auxiliar.filterbanks import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.training.wrapper import *
from ddsp_textures.training.wrapper import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def model_loader(model_folder_path, print_parameters=True):
    configurations_path = os.path.join(model_folder_path, "configurations.json")
    
    #read configurations
    parameters_dict = model_json_to_parameters(configurations_path)
    
    if print_parameters:
        # Print parameters in bold
        print("\033[1mModel Parameters:\033[0m\n")
        for key, value in parameters_dict.items():
            print(f"{key}: {value}")
    
    # Unpack parameters
    hidden_size      = parameters_dict['hidden_size']
    deepness         = parameters_dict['deepness']        
    param_per_env    = parameters_dict['param_per_env']
    model_class      = parameters_dict['architecture']

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed loading
    seed_path = os.path.join(model_folder_path, "seed.pkl")
    if os.path.exists(seed_path):
        with open(seed_path, 'rb') as file:
            seed = pickle.load(file)
    seed = seed.to(device)

    # Model initialization
    model = model_class(hidden_size, deepness, param_per_env, seed).to(device)
    
    # Loading model
    model_path = os.path.join(model_folder_path, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Loading loss dictionary
    loss_dict_path = os.path.join(model_folder_path, 'loss_dict.pkl')
    with open(loss_dict_path, 'rb') as file:
        loss_dict = pickle.load(file)
    
    return model, parameters_dict, loss_dict

def audio_preprocess(audio_path, frame_size, sampling_rate, features_annotator):
    hop_size = frame_size // 2
    audio_og, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
    
    audio_tensor = torch.tensor(audio_og)
    size = audio_tensor.shape[0]
    N_segments = (size - frame_size) // hop_size
    
    content = []
    for i in range(N_segments):
        segment = audio_tensor[i * hop_size: i * hop_size + frame_size]
        target_loudness = torch.std(segment)
        segment = (segment - torch.mean(segment)) / torch.std(segment)
        features = features_annotator(segment, sampling_rate)
        content.append([features, target_loudness])
    
    return content

def model_synthesizer(content, model, parameters_dict, random_shift=True):
    # Unpack parameters
    frame_size       = parameters_dict['frame_size']
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
    
    hop_size = frame_size // 2
    
    N_segments = len(content)
    
    audio_final = torch.zeros(frame_size + (N_segments-1)*hop_size)
    window = torch.hann_window(frame_size)

    for i in range(N_segments):
        [features, target_loudness] = content[i]
        # transform 0d into 1d features of size 1
        features = [feature.unsqueeze(0) for feature in features]
        synthesized_segment = model.synthesizer(features[0], features[1], target_loudness)
        if random_shift:    
            #shif the synthesized_segment in a random amount of steps
            synthesized_segment = synthesized_segment.roll(shifts=np.random.randint(0, len(synthesized_segment)))
        #Apply window
        synthesized_segment = synthesized_segment * window
        audio_final[i * hop_size: i * hop_size + frame_size] += synthesized_segment

    audio_final = audio_final.detach().cpu().numpy()
    
    return audio_final

# def model_tester(frame_type, model_type, loss_type, audio_path, model_name, best):
#     model = model_loader(frame_type, model_type, loss_type, audio_path, model_name, best)
#     input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size = short_long_decoder(frame_type)
    
#     hop_size = frame_size // 2
    
#     audio_og, _ = librosa.load(audio_path, sr=sampling_rate)
    
#     audio_tensor = torch.tensor(audio_og)
#     size = audio_tensor.shape[0]
#     N_segments = (size - frame_size) // hop_size
    
#     content = []
#     for i in range(N_segments):
#         segment = audio_tensor[i * hop_size: i * hop_size + frame_size]
#         target_loudness = torch.std(segment)
#         segment = (segment - torch.mean(segment)) / torch.std(segment)
#         feature_0 = torchaudio.functional.spectral_centroid(segment, sampling_rate, 0, torch.hamming_window(frame_size), frame_size, frame_size, frame_size)[0].float()
#         feature_1 = torch.tensor([1]).float()
#         features = [feature_0, feature_1]
#         # features = feature_extractor(segment, sampling_rate, N_filter_bank)
#         content.append([features, segment, target_loudness])
    
#     audio_final = torch.zeros(frame_size + (N_segments-1)*hop_size)
#     window = torch.hann_window(frame_size)
    
#     for i in range(N_segments):
#         [features, segment, target_loudness] = content[i]
#         [sp_centroid, rate] = features
#         sp_centroid = sp_centroid.unsqueeze(0)
#         synthesized_segment = model.synthesizer(sp_centroid, rate, target_loudness)
#         #shif the synthesized_segment in a random amount of steps
#         synthesized_segment = synthesized_segment.roll(shifts=np.random.randint(0, len(synthesized_segment)))
#         #Apply window
#         synthesized_segment = synthesized_segment * window
#         audio_final[i * hop_size: i * hop_size + frame_size] += synthesized_segment
    
#     audio_final = audio_final.detach().cpu().numpy()
    
#     return audio_og, audio_final

# def audio_preprocess(frame_type, model_type, loss_type, audio_path, model_name):
#     input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size = short_long_decoder(frame_type)
#     audio_og, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
    
#     hop_size = frame_size // 2

#     audio_tensor = torch.tensor(audio_og)
#     size = audio_tensor.shape[0]
#     N_segments = (size - frame_size) // hop_size
    
#     content = []
#     for i in range(N_segments):
#         segment = audio_tensor[i * hop_size: i * hop_size + frame_size]
#         target_loudness = torch.std(segment)
#         segment = (segment - torch.mean(segment)) / torch.std(segment)
#         feature_0 = torchaudio.functional.spectral_centroid(segment, sampling_rate, 0, torch.hamming_window(frame_size), frame_size, frame_size, frame_size)[0].float()
#         feature_1 = torch.tensor([1]).float()
#         features = [feature_0, feature_1]
#         # features = feature_extractor(segment, sampling_rate, N_filter_bank)
#         content.append([features, segment, target_loudness])
    
#     return content

# def model_synthesizer(model, content, frame_type):
#     input_size, hidden_size, N_filter_bank, frame_size, hop_size, sampling_rate, compression, batch_size = short_long_decoder(frame_type)
#     N_segments = len(content)
    
#     hop_size = frame_size // 2 
    
#     audio_final = torch.zeros(frame_size + (N_segments-1)*hop_size)
#     window = torch.hann_window(frame_size)

#     for i in range(N_segments):
#         [features, segment, target_loudness] = content[i]
#         [sp_centroid, rate] = features
#         sp_centroid = sp_centroid.unsqueeze(0)
#         synthesized_segment = model.synthesizer(sp_centroid, rate, target_loudness)
#         #shif the synthesized_segment in a random amount of steps
#         synthesized_segment = synthesized_segment.roll(shifts=np.random.randint(0, len(synthesized_segment)))
#         #Apply window
#         synthesized_segment = synthesized_segment * window
#         audio_final[i * hop_size: i * hop_size + frame_size] += synthesized_segment

#     audio_final = audio_final.detach().cpu().numpy()
    
#     return audio_final