import os
import json
from ddsp_textures.auxiliar.features import *
from ddsp_textures.dataset.makers import *
from ddsp_textures.loss.functions import *
from ddsp_textures.architectures.DDSP import *

# EDIT WHEN NEW STUFF IS ADDED -------------------------------------------------------------------------
def model_name_to_parameters(json_file_path):
    # Features options ---------------------------------------------------------------------------------
    features_annotator_map = {
        'features_freqavg_freqstd': features_freqavg_freqstd,
        'features_freqavg_rate':    features_freqavg_rate
    }
    
    batch_features_annotator_map = {
        'features_freqavg_freqstd': batch_features_freqavg_freqstd,
        'features_freqavg_rate':    batch_features_freqavg_rate
    }
    
    # Architecture options ------------------------------------------------------------------------------
    architecture_map = {
        'DDSP_TextEnv': DDSP_TextEnv,
        'DDSP_PVAE':    DDSP_PVAE
    }
    
    # Loss functions options ---------------------------------------------------------------------------
    loss_function_map = {
        'statistics_loss':     batch_statistics_loss,
        'sub_statistics_loss': batch_sub_statistics_loss
    }
    
    with open(json_file_path, 'r') as file:
        parameters = json.load(file)
    
    parameters_dict = {}
    parameters_dict['audio_path']               = parameters['audio_path']
    parameters_dict['frame_size']               = int(parameters['frame_size'])
    parameters_dict['hop_size']                 = int(parameters['hop_size'])
    parameters_dict['sampling_rate']            = int(parameters['sampling_rate'])
    parameters_dict['features_annotator']       = features_annotator_map[parameters['features_annotator']]
    parameters_dict['batch_features_annotator'] = batch_features_annotator_map[parameters['features_annotator']]
    parameters_dict['freq_avg_level']           = int(parameters['freq_avg_level'])
    parameters_dict['hidden_size']              = int(parameters['hidden_size'])
    parameters_dict['deepness']                 = int(parameters['deepness'])
    parameters_dict['param_per_env']            = int(parameters['param_per_env'])
    parameters_dict['N_filter_bank']            = int(parameters['N_filter_bank'])
    parameters_dict['architecture']             = architecture_map[parameters['architecture']]
    parameters_dict['onset_detection_model_path'] = parameters['onset_detection_model_path']
    parameters_dict['loss_function']            = loss_function_map[parameters['loss_function']]
    parameters_dict['regularization']           = bool(int(parameters['regularization']))
    parameters_dict['batch_size']               = int(parameters['batch_size'])
    parameters_dict['epochs']                   = int(parameters['epochs'])
    parameters_dict['directory']                = parameters['directory']
    
    return parameters_dict

def get_next_model_folder(base_path="trained_models"):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("model_")]
    model_numbers = [int(d.split("_")[1]) for d in model_folders if d.split("_")[1].isdigit()]
    next_model_number = max(model_numbers, default=0) + 1
    next_model_folder = f"model_{next_model_number:02d}"
    return next_model_folder