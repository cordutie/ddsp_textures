import os
import json
from ddsp_textures.auxiliar.features import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.architectures.DDSP import *

# config template ---------------------------------------------------------------------------------------
# {
#     "audio_path": "../sounds/fire_short.wav",
#     "frame_size": 32768,
#     "hop_size": 16384,
#     "sampling_rate": 44100,
#     "features_list": "features_freqavg_rate,features_freqavg,features_freqavg_freqstd,features_rate,features_energy_bands,features_envelopes_stems",
#     "hidden_size_enc": 512,
#     "hidden_size_dec": 512,
#     "deepness_enc": 3,
#     "deepness_dec": 3,
#     "param_per_env": 512,
#     "N_filter_bank": 16,
#     "M_filter_bank": 16,
#     "architecture": "DDSP_TextEnv",
#     "stems": 1,
#     "loss_function": "statistics",
#     "regularization": 1,
#     "batch_size": 16,
#     "epochs": 2,
#     "models_directory": "../trained_models"
# }

# EDIT WHEN NEW STUFF IS ADDED -------------------------------------------------------------------------
def model_json_to_parameters(json_file_path):
    with open(json_file_path, 'r') as file:
        parameters_json = json.load(file)

    # Features options ---------------------------------------------------------------------------------
    features_map = {
        "freqavg_freqstd": features_freqavg_freqstd,
        "freqavg":         features_freqavg,
        "rate":            features_rate,
        "energy_bands":    features_energy_bands,
        "envelopes_stems": features_envelopes_stems 
    }

    regularizers_map = {
        "freqavg_freqstd": batch_features_freqavg_freqstd,
        "freqavg":         batch_features_freqavg,
        "rate":            batch_features_rate,
        "energy_bands":    batch_features_energy_bands,
        "envelopes_stems": batch_features_envelopes_stems 
    }

    N  = int(parameters_json['N_filter_bank'])
    fs = int(parameters_json['frame_size'])

    features_dim_map = {
        "freqavg_freqstd": [2],
        "freqavg":         [1],
        "rate":            [1],
        "energy_bands":    [N],
        "envelopes_stems": [N,fs]
    }
    
    # Architecture options ------------------------------------------------------------------------------
    architecture_map = {
        'DDSP_SubEnv':  DDSP_SubEnv
        # ,'DDSP_PVAE':    DDSP_PVAE
    }
    
    # Loss functions options ---------------------------------------------------------------------------
    loss_function_map = {
        'statistics':       batch_statistics_loss,
        'statistics_stems': batch_statistics_loss_stems,
        'multiscale':       multiscale_spectrogram_loss
    }
    
    def loss_picker(loss_input, stems):
        if stems and loss_input == 'multiscale':
            raise ValueError("Multiscale loss is not compatible with stems.")
        return loss_function_map.get(loss_input)

    actual_parameters = {}
    actual_parameters['audio_path']               = parameters_json['audio_path']
    actual_parameters['frame_size']               = int(parameters_json['frame_size'])
    actual_parameters['hop_size']                 = int(parameters_json['hop_size'])
    actual_parameters['sampling_rate']            = int(parameters_json['sampling_rate'])
    actual_parameters['features']                 = []
    features_strings = parameters_json['features_list'].split(',')
    for features in features_strings:
        actual_parameters['features'].append(features_map[features])
    actual_parameters['input_dimensions']         = []
    for features in features_strings:
        actual_parameters['features']+=features_dim_map[features]
    actual_parameters['hidden_size_enc']          = int(parameters_json['hidden_size_enc'])
    actual_parameters['hidden_size_dec']          = int(parameters_json['hidden_size_dec'])
    actual_parameters['deepness_enc']             = int(parameters_json['deepness_enc'])
    actual_parameters['deepness_dec']             = int(parameters_json['deepness_dec'])
    actual_parameters['param_per_env']            = int(parameters_json['param_per_env'])
    actual_parameters['N_filter_bank']            = int(parameters_json['N_filter_bank'])
    actual_parameters['M_filter_bank']            = int(parameters_json['M_filter_bank'])
    actual_parameters['architecture']             = architecture_map[parameters_json['architecture']]
    stems                                         = bool(int(parameters_json['stems'])) 
    actual_parameters['stems']                    = stems
    actual_parameters['loss_function']            = loss_picker(parameters_json['loss_function'], stems)
    actual_parameters['regularizer']              = []
    regularizers_strings = parameters_json['regularizers_list'].split(',')
    for regularizer in regularizers_strings:
        actual_parameters['regularizer'].append(regularizers_map[regularizer])
    actual_parameters['batch_size']               = int(parameters_json['batch_size'])
    actual_parameters['epochs']                   = int(parameters_json['epochs'])
    actual_parameters['models_directory']         = parameters_json['models_directory']
    
    return actual_parameters

def get_next_model_folder(base_path="trained_models"):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("model_")]
    model_numbers = [int(d.split("_")[1]) for d in model_folders if d.split("_")[1].isdigit()]
    next_model_number = max(model_numbers, default=0) + 1
    next_model_folder = f"model_{next_model_number:02d}"
    return next_model_folder