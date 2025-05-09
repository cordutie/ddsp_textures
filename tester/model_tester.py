from ddsp_textures.architectures.DDSP import *
from ddsp_textures.auxiliar.seeds import *
from ddsp_textures.auxiliar.filterbanks import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.training.wrapper import *
from ddsp_textures.training.wrapper import *
from ddsp_textures.auxiliar.configuration import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def model_loader(model_path, configurations_path, print_parameters=True):    
    #read configurations
    parameters_dict = model_json_to_parameters(configurations_path)
    
    if print_parameters:
        # Print parameters in bold
        print("\033[1mModel Parameters:\033[0m\n")
        for key, value in parameters_dict.items():
            print(f"{key}: {value}")
    
    # Unpack some parameters
    frame_size = parameters_dict['frame_size']
    sampling_rate = parameters_dict['sampling_rate']  
    hidden_size_enc = parameters_dict['hidden_size_enc']      
    hidden_size_dec = parameters_dict['hidden_size_dec']      
    deepness_enc = parameters_dict['deepness_enc']
    deepness_dec = parameters_dict['deepness_dec']     
    param_per_env = parameters_dict['param_per_env']         
    N_filter_bank = parameters_dict['N_filter_bank']          
    M_filter_bank = parameters_dict['M_filter_bank']    
    architecture  = parameters_dict['architecture']
    input_dimensions = parameters_dict['input_dimensions']
    # print(input_dimensions)
    # stems = parameters_dict['stems']

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading seed
    parent_folder = os.path.dirname(model_path)
    seed_path = os.path.join(parent_folder, "seed.pkl")
    if os.path.exists(seed_path):
        with open(seed_path, 'rb') as file:
            seed = pickle.load(file)

    # Model initialization
    model = architecture(input_dimensions, hidden_size_enc, hidden_size_dec, deepness_enc, deepness_dec, param_per_env, frame_size, N_filter_bank, device, seed).to(device)
    
    # Loading model
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Parent folder of model_path
    parent_folder = os.path.dirname(model_path)

    return model, seed, checkpoint

def plot_loss_history(loss_dict):
    loss_history     = np.array(loss_dict['loss_total'])
    main_loss_histor = np.array(loss_dict['loss_main'])
    regularizer_history = np.array(loss_dict['loss_regularizer'])
    if regularizer_history[0]==0:
        regularizer_history = None
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Total Loss')
    if regularizer_history is not None:
        plt.plot(main_loss_histor, label='Main Loss')
        plt.plot(regularizer_history, label='Regularizer Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Loss History')
    plt.legend()
    plt.show()

def audio_preprocess(audio_path, frame_size, sampling_rate, features_annotators, erb_bank, audio_improv=True, level=2):
    hop_size = frame_size // 2
    audio_og, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
    
    audio_tensor = torch.tensor(audio_og)
    size         = audio_tensor.shape[0]
    N_segments   = (size - frame_size) // hop_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    content = []
    for i in range(N_segments):
        segment_annotated = []
        segment         = audio_tensor[i * hop_size: i * hop_size + frame_size]
        if audio_improv:
            segment = audio_improver(segment, sampling_rate, level)
        # target_loudness = torch.std(segment).to(device)
        segment              = signal_normalizer(segment).to(device)

        # adding the segment to the element
        segment_annotated.append(segment)
        # segment_stems = features_envelopes_stems(segment, 0, erb_bank).to(device) # ESTO TA RAROOOOOOOOOOOO
        # segment_annotated.append(segment_stems)
        # features computation
        for feature_annotator in features_annotators:
            feature_loc = feature_annotator(segment, sampling_rate, erb_bank).to(device)
            # If feature loc is just a tensor number transform into a 1d tensor
            if len(feature_loc.shape) == 0:
                feature_loc = torch.tensor([feature_loc]).to(device)  
            # adding features to the element
            segment_annotated.append(feature_loc)
        # segment_annotated.append(target_loudness)
        content.append(segment_annotated)
    return content

# content[i] = actual torch segment, features, target_loudness

def model_synthesizer(content, model, parameters_dict, type_loudness, seed, random_shift=True):
    # Unpack parameters
    frame_size                      = parameters_dict['frame_size']
    hop_size                        = parameters_dict['hop_size']
    sampling_rate                   = parameters_dict['sampling_rate']
    N_filter_bank                   = parameters_dict['N_filter_bank']
    filter_low_lim      = parameters_dict["filter_low_lim"]
    filter_high_lim     = parameters_dict["filter_high_lim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hop_size = frame_size // 2
    
    N_segments = len(content)
    audio_final = torch.zeros(frame_size + (N_segments-1)*hop_size).to(device)
    window      = torch.hamming_window(frame_size).to(device)

    erb_bank    = fb.EqualRectangularBandwidth(frame_size, sampling_rate, N_filter_bank, filter_low_lim, filter_high_lim)

    for i in range(N_segments):
        local_content = content[i]
        segment_loc   = local_content[0]
        features_loc  = local_content[1:]
        # target_loudness_loc = local_content[-1]
        if type_loudness == "generic_loudness" or type_loudness == "generic":
            target_loudness_loc = torch.norm(segment_loc)
        else:
            target_loudness_loc = features_energy_bands(segment_loc, 0, erb_bank)
        # synthesis time!
        synthesized_segment = model.synthesizer(features_loc, type_loudness, target_loudness_loc, seed)     

        if random_shift:
            # Shift the synthesized_segment in a random amount of steps
            synthesized_segment = synthesized_segment.roll(shifts=np.random.randint(0, frame_size))
        
        # Apply window
        synthesized_segment = synthesized_segment * window
        audio_final[i * hop_size: i * hop_size + frame_size] += synthesized_segment
        # audio_og[i * hop_size: i * hop_size + frame_size]    += segment_loc * window

    # # Normalize the audio_final using window overlap sum
    # window_sum = torch.zeros_like(audio_final).to(device)
    # for i in range(N_segments):
    #     window_sum[i * hop_size: i * hop_size + frame_size] += window
    # audio_final /= torch.clamp(window_sum, min=1e-6)  # Avoid division by zero
    
    audio_final = audio_final.detach().cpu().numpy()
    # audio_og    = audio_og.detach().cpu().numpy()
    
    return audio_final
    # return audio_final, audio_og



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