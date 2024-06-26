# from initializer import initializer
# from trainer import trainer
import sys
import os

def model_name_to_parameters(model_name):
    parameters = model_name.split("_")
    
    sound_path = os.path.join("sounds", parameters[0]+".wav")
    
    frame_type = parameters[1]
    
    model_type = parameters[2]
    if model_type=='gru':
        model_type = 'DDSP_textenv_gru'
    elif model_type=='mlp':
        model_type = 'DDSP_textenv_mlp'
    elif model_type=='stemsgru':
        model_type = 'DDSP_textenv_stems_gru'
    elif model_type=='stemsmlp':
        model_type = 'DDSP_textenv_stems_mlp'
    else:
        raise NameError("Invalid model type")
    
    loss_type = parameters[3]
    if loss_type=='multispec':
        loss_type = 'multispectrogram_loss'
    elif loss_type=='stats':
        loss_type = 'statistics_loss'
    elif loss_type=='substats':
        loss_type = 'sub_statistics_loss'    
    elif loss_type=='stems':
        loss_type = 'stems_loss'
    else:
        raise NameError("Invalid loss type")
    
    return [sound_path, frame_type, model_type, loss_type, model_name]