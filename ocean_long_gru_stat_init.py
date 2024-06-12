from architectures.DDSP import *
from auxiliar.auxiliar import *
from auxiliar.filterbanks import *
from dataset.dataset_maker import *
from loss.loss_functions import *
from signal_processors.textsynth_env import *
from training.initializer import *
from training.trainer import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Type of frame
frame_type = 'long'

# Type of model
model_type = 'DDSP_textenv_gru'

# type of loss
loss_type = 'statistics_loss'

# Sound path to create dataset
audio_path = 'sounds/ocean.wav'

# model name
model_name = 'ocean_long_gru_stat'

####################### Standard code #######################

initializer(frame_type, model_type, loss_type, audio_path, model_name)