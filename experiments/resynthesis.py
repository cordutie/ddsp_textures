from ddsp_textures.architectures.DDSP import *
from ddsp_textures.auxiliar.seeds import *
from ddsp_textures.auxiliar.filterbanks import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.training.wrapper import *
from ddsp_textures.training.wrapper import *
from ddsp_textures.auxiliar.configuration import *
from ddsp_textures.tester.model_tester import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
# Function that reads a model from a given path and returns the model and its dictionary of parameters.
def load_model(model_folder_path, print_parameters=False):
    model_path  = os.path.join(model_folder_path, 'best_model.pth')
    config_path = os.path.join(model_folder_path, 'configurations.json')
    # Load the model
    model, seed, checkpoint = model_loader(model_path, config_path, print_parameters)
    parameters_dict         = model_json_to_parameters(config_path)
    return model, seed, parameters_dict

# Function that takes a model with its dictionary of parameters and the path of an audio file and resynthesize the audio through the model
def resynthesis_from_model(audio_path, model_folder_path, input_audio_improver = 1, energy_imposition=1, envelope_follower=1, print_parameters=False):
    model, seed, parameters_dict = load_model(model_folder_path, print_parameters)
    device = seed.device

    # Preprocess the audio file with the corresponding parameters
    frame_size          = parameters_dict["frame_size"]
    features_annotators = parameters_dict["features"]
    sampling_rate       = parameters_dict["sampling_rate"]
    N_filter_bank       = parameters_dict["N_filter_bank"] 
    filter_low_lim      = parameters_dict["filter_low_lim"]
    filter_high_lim     = parameters_dict["filter_high_lim"]
    erb_bank            = EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, filter_low_lim, filter_high_lim)
    # Audio input improving and preprocessing step ---------------------------------------
    # Check that audio_improver is between 0 and 2
    if input_audio_improver < 0 or input_audio_improver > 1:
        raise ValueError(f"Audio improver {input_audio_improver} not recognized. Use a value between 0 and 1.")
    elif input_audio_improver == 0:
        audio_improv = 0
        audio_improv_level = 0
    else:
        audio_improv = 1
        audio_improv_level = input_audio_improver * 4
    content = audio_preprocess(audio_path, frame_size, sampling_rate, features_annotators, erb_bank, audio_improv, audio_improv_level) # content is a list of segments annotated, each segment annotated is a list with the following shape [torch_segment, feature_1, feature_2]
    # Overlap and add starts --------------------------------------------------------------
    N_segments = len(content)
    hop_size = frame_size // 2
    resynthesis = torch.zeros(frame_size + (N_segments-1)*hop_size).to(device)
    window      = torch.hamming_window(frame_size).to(device)
    for i in range(N_segments):
        local_content = content[i]
        segment_loc   = local_content[0]
        features_loc  = local_content[1:]
        # Energy Imposition Step ----------------------------------------------------------
        # Check energy_imposition is either 0 or 1
        if energy_imposition == 0:
            method = "generic"
            target_loudness_loc = torch.norm(segment_loc)
        elif energy_imposition == 1: 
            method = "specific"
            target_loudness_loc = features_energy_bands(segment_loc, 0, erb_bank)
        else:
            raise ValueError(f"Method {method} not recognized. Use 'generic' or 'specific'.")
        synthesized_segment = model.synthesizer(features_loc, method, target_loudness_loc, seed)     
        # Randomization step ---------------------------------------------------------------
        synthesized_segment = synthesized_segment.roll(shifts=np.random.randint(0, frame_size))
        # Apply window and overlap and add -------------------------------------------------
        synthesized_segment = synthesized_segment * window
        resynthesis[i * hop_size: i * hop_size + frame_size] += synthesized_segment
        print(f"Processed segment {i+1}/{N_segments}", end='\r')
    # Envelope follower step --------------------------------------------------------------
    # checkk that envelope_follower is between 0 and 1
    if envelope_follower < 0 or envelope_follower > 1:
        raise ValueError(f"Envelope follower {envelope_follower} not recognized. Use a value between 0 and 1.")
    if envelope_follower != 0:
        kernel_size  = int(sampling_rate * (0.99*(envelope_follower-1)**4+0.01))
        kernel       = torch.ones(1, 1, kernel_size, device=device) / kernel_size  # Moving average kernel
        resynthesis_env = torch.abs(hilbert(resynthesis))
        resynthesis_env = F.conv1d(resynthesis_env[None, None, :], kernel, padding=kernel_size//2)[0, 0, 0:len(resynthesis)]
        input           = librosa.load(audio_path, sr=sampling_rate, mono=True)[0]
        input           = torch.tensor(input, device=device)
        input_env       = torch.abs(hilbert(input))
        input_env       = F.conv1d(input_env[None, None, :], kernel, padding=200//2)[0, 0, 0:len(resynthesis_env)]
        resynthesis     = resynthesis * (input_env / resynthesis_env)
    # Normalize the output
    resynthesis = resynthesis / torch.max(resynthesis)
    return resynthesis.cpu()