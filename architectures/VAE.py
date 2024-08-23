import torch
import torch.nn as nn
from ddsp_textures.signal_processors.synthesizers import SubEnv, SubEnv_batches
from ddsp_textures.auxiliar.seeds import seed_maker
from ddsp_textures.auxiliar.nn import mlp

class VAE_SubEnv(nn.Module):
    def __init__(self, hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device):
        super(VAE_SubEnv, self).__init__()
        self.seed = seed.to(device)  # Seed is a tensor, so keep it on the device
        self.device = device
        
        # parameters
        self.N_filter_bank = N_filter_bank
        self.param_per_env = param_per_env
        
        # encoder
        self.encoder_1 = mlp(N_filter_bank, hidden_size, deepness).to(device)
        self.encoder_2 = nn.Linear(hidden_size, latent_dim).to(device)
        
        # decoder    
        self.a_decoder_1 = mlp(latent_dim, hidden_size, deepness).to(device)
        self.a_decoder_2 = nn.Linear(hidden_size, N_filter_bank * param_per_env).to(device)
        self.p_decoder_1 = mlp(latent_dim, hidden_size, deepness).to(device)
        self.p_decoder_2 = nn.Linear(hidden_size, N_filter_bank * param_per_env).to(device)

    def encode(self, x):
        x = x.to(self.device)
        x = self.encoder_1(x)
        l = self.encoder_2(x)
        return torch.sigmoid(l)

    def decode(self, z):
        z = z.to(self.device)
        a = self.a_decoder_1(z)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(z)
        p = self.p_decoder_2(p)
        p = 2 * torch.pi * torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, x):
        x = x.to(self.device)
        l = self.encode(x)
        real_param, imag_param = self.decode(l)
        x_hat = SubEnv_batches(real_param, imag_param, self.seed)
        return x_hat
    
    def generate_batches(self, z):
        z = z.to(self.device)
        real_param, imag_param = self.decode(z)
        x_hat = SubEnv_batches(real_param, imag_param, self.seed)
        return x_hat
    
    def generate(self, z):
        z = z.to(self.device)
        real_param, imag_param = self.decode(z)
        x_hat = SubEnv(real_param, imag_param, self.seed)
        return x_hat

import json

# Function to save the model and settings
def save_model(model, path, settings):
    # Save model state dict
    torch.save(model.state_dict(), path)
    # print(f'Model saved to {path}')
    
    # Save model settings to a JSON file
    settings_path = path.replace('.pth', '_settings.json')
    with open(settings_path, 'w') as f:
        json.dump(settings, f)
    # print(f'Model settings saved to {settings_path}')

# Function to load the model and settings
def load_model(model_path, settings_path, device):
    # Load settings from the JSON file
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Extract settings
    frame_size = settings['frame_size']
    hidden_size = settings['hidden_size']
    deepness = settings['deepness']
    latent_dim = settings['latent_dim']
    N_filter_bank = settings['N_filter_bank']
    param_per_env = settings['param_per_env']
    
    sr = 44100
    seed = seed_maker(frame_size, sr, N_filter_bank)
    seed = seed.to(device)

    # Create model and load state dict
    model = VAE_SubEnv(hidden_size, deepness, latent_dim, N_filter_bank, param_per_env, seed, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Model loaded from {model_path}')
    return model
