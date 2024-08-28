import torch.nn as nn
import torch
import numpy as np
import librosa
import torchaudio

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + [hidden_size] * n_layers
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

class ReplaceNaN(nn.Module):
    def __init__(self, large_value=1e5, small_value=-1e5):
        super(ReplaceNaN, self).__init__()
        self.large_value = large_value
        self.small_value = small_value

    def forward(self, x):
        # Replace NaN values with the large_value
        x = torch.where(torch.isnan(x), torch.tensor(self.large_value).to(x.device), x)
        # Optionally, clamp the values to prevent extreme values
        x = torch.clamp(x, min=self.small_value, max=self.large_value)
        return x

def mlp_v(in_size, hidden_size, n_layers):
    channels = [in_size] + [hidden_size] * n_layers
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.Sigmoid())
        net.append(ReplaceNaN())  # Add the custom NaN replacement layer
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input, hidden_size, batch_first=True)

class gru_v(nn.Module):
    def __init__(self, n_input, hidden_size, replace_large=1e5, replace_small=-1e5):
        super(gru_v, self).__init__()
        self.gru = nn.GRU(n_input, hidden_size, batch_first=True)
        self.replace_large = replace_large
        self.replace_small = replace_small

    def forward(self, x):
        # Get GRU output and hidden state
        output, hidden = self.gru(x)
        # Handle NaNs in the output
        output = torch.where(torch.isnan(output), torch.tensor(self.replace_large).to(output.device), output)
        output = torch.clamp(output, min=self.replace_small, max=self.replace_large)
        return output, hidden