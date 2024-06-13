from architectures.DDSP import *
from auxiliar.auxiliar import *
from auxiliar.filterbanks import *
from dataset.dataset_maker import *
from loss.loss_functions import *
from signal_processors.textsynth_env import *
from training.initializer import *
from training.trainer import *
from training.auxiliar import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def wrapper(parameters):
    audio_path, frame_type, model_type, loss_type, model_name = parameters
    
    # Initialize the model
    initializer(frame_type, model_type, loss_type, audio_path, model_name)
    trainer(frame_type, model_type, loss_type, audio_path, model_name)

if __name__ == "__main__":
    # Check if there are two arguments and the second argument is a string
    if len(sys.argv) == 2 and isinstance(sys.argv[1], str):
        model_name = sys.argv[1]
        parameters = model_name_to_parameters(model_name)
    else:
        raise NameError("Invalid arguments")
    audio_path, frame_type, model_type, loss_type, model_name = parameters

    print("Parameters:")
    print(  "audio_path:", audio_path,"\n",
            "frame_type:", frame_type,"\n",
            "model_type:", model_type,"\n",
            "loss_type:", loss_type,  "\n",
            "model_name:", model_name)

    wrapper(parameters)
