from ddsp_textures.signal_processors.synthesizers import *
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

def gru(n_input, hidden_size):
    return nn.GRU(n_input, hidden_size, batch_first=True)

class DDSP_TextEnv(nn.Module):
    def __init__(self, hidden_size, deepness, param_per_env, seed):
        super().__init__()

        frame_size = seed.shape[0]
        N_filter_bank = seed.shape[1]
        
        self.N_filter_bank = N_filter_bank
        self.seed = seed
        self.frame_size = frame_size
        self.param_per_env = param_per_env
        
        self.f0_encoder = mlp(1, hidden_size, deepness)
        self.f1_encoder = mlp(1, hidden_size, deepness)
        self.z_encoder = gru(2 * hidden_size, hidden_size)
    
        self.a_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)
        self.p_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)

    def encoder(self, feature_0, feature_1):
        # print("feature_0:", feature_0.shape)
        # print("feature_1:", feature_1.shape)
        f0 = self.f0_encoder(feature_0)
        f1 = self.f1_encoder(feature_1)
        # print("f0:", f0.shape)
        # print("f1:", f1.shape)
        z, _ = self.z_encoder(torch.cat([f0, f1], dim=-1).unsqueeze(0))
        z = z.squeeze(0)
        # print("z:", z.shape)
        return torch.cat([f0, f1, z], dim=-1)

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2 * torch.pi * torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, feature_0, feature_1):
        latent_vector = self.encoder(feature_0, feature_1)
        real_param, imag_param = self.decoder(latent_vector)

        # Ensure all tensors are on the same device
        device = real_param.device
        latent_vector = latent_vector.to(device)
        feature_0 = feature_0.to(device)
        feature_1 = feature_1.to(device)

        signal = TextEnv_batches(real_param, imag_param, self.seed)
        return signal

    def synthesizer(self, feature_0, feature_1, target_loudness):
        latent_vector = self.encoder(feature_0, feature_1)
        real_param, imag_param = self.decoder(latent_vector)

        # Ensure all tensors are on the same device
        device = real_param.device
        latent_vector = latent_vector.to(device)
        feature_0 = feature_0.to(device)
        feature_1 = feature_1.to(device)

        signal = TextEnv(real_param, imag_param, self.seed, target_loudness)
        return signal
    
# CHANGE WHEN PVAE IS IMPLEMENTED --------------------------------------------------------------
class DDSP_PVAE(nn.Module):
    def __init__(self, hidden_size, deepness, param_per_env, seed):
        super().__init__()

        frame_size = seed.shape[0]
        N_filter_bank = seed.shape[1]
        
        self.N_filter_bank = N_filter_bank
        self.seed = seed
        self.frame_size = frame_size
        self.param_per_env = param_per_env
        
        self.feat0_encoder = mlp(1, hidden_size, deepness)
        self.feat1_encoder = mlp(1, hidden_size, deepness)
        self.z_encoder = gru(2 * hidden_size, hidden_size)
    
        self.a_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)
        self.p_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)

    def encoder(self, feature_0, feature_1):
        f = self.feat0_encoder(feature_0)
        r = self.feat1_encoder(feature_1)
        z, _ = self.z_encoder(torch.cat([f, r], dim=-1).unsqueeze(0))
        z = z.squeeze(0)
        return torch.cat([f, r, z], dim=-1)

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2 * torch.pi * torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, feature_0, feature_1):
        latent_vector = self.encoder(feature_0, feature_1)
        real_param, imag_param = self.decoder(latent_vector)

        # Ensure all tensors are on the same device
        device = real_param.device
        latent_vector = latent_vector.to(device)
        feature_0 = feature_0.to(device)
        feature_1 = feature_1.to(device)

        signal = TextEnv_batches(real_param, imag_param, self.seed)
        return signal

    def synthesizer(self, feature_0, feature_1, target_loudness):
        latent_vector = self.encoder(feature_0, feature_1)
        real_param, imag_param = self.decoder(latent_vector)

        # Ensure all tensors are on the same device
        device = real_param.device
        latent_vector = latent_vector.to(device)
        feature_0 = feature_0.to(device)
        feature_1 = feature_1.to(device)

        signal = TextEnv(real_param, imag_param, self.seed, target_loudness)
        return signal