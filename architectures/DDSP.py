from signal_processors.textsynth_env import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
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

class DDSP_textenv_gru(nn.Module):
    def __init__(self, hidden_size, N_filter_bank, deepness, compression, frame_size, sampling_rate, seed):
        super().__init__()

        self.N_filter_bank = N_filter_bank
        self.seed = seed
        self.frame_size = frame_size
        self.param_per_env = int(frame_size / (2*N_filter_bank*compression))
        
        self.f_encoder = mlp(1, hidden_size, deepness)
        self.l_encoder = mlp(N_filter_bank, hidden_size, deepness)
        self.z_encoder = gru(2 * hidden_size, hidden_size)
    
        self.a_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, 16 * self.param_per_env)
        self.p_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, 16 * self.param_per_env)

    def encoder(self, spectral_centroid, loudness):
        f = self.f_encoder(spectral_centroid)
        # print("f shape: ",f.shape)
        l = self.l_encoder(loudness)
        # print("l shape: ",l.shape)
        z, _ = self.z_encoder(torch.cat([f,l], dim=-1).unsqueeze(0))
        # print("z_1 shape: ",z.shape)
        z = z.squeeze(0)
        # print("z_2 shape: ",z.shape)
        return torch.cat([f,l,z], dim=-1)

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2*torch.pi*torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, spectral_centroid, loudness, _):
        latent_vector = self.encoder(spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        signal = textsynth_env_batches(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size)
        return signal
    
    def synthesizer(self, input_ds, spectral_centroid, loudness, target_loudness):
        latent_vector = self.encoder(input_ds, spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        signal = textsynth_env(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size, target_loudness)
        return signal    
    
class DDSP_textenv_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, N_filter_bank, deepness, compression, frame_size, sampling_rate, seed):
        super().__init__()

        self.N_filter_bank = N_filter_bank
        self.seed = seed
        self.frame_size = frame_size
        self.param_per_env = int(frame_size / (2*N_filter_bank*compression))
        
        self.f_encoder = mlp(1, hidden_size, deepness)
        self.l_encoder = mlp(N_filter_bank, hidden_size, deepness)
        self.z_encoder = mlp(input_size, hidden_size, deepness)
    
        self.a_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)
        self.p_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)

    def encoder(self, spectral_centroid, loudness, input_ds):
        f = self.f_encoder(spectral_centroid)
        # print("f shape: ",f.shape)
        l = self.l_encoder(loudness)
        # print("l shape: ",l.shape)
        z = self.z_encoder(input_ds)
        # print("z_1 shape: ",z.shape)
        return torch.cat([f,l,z], dim=-1)

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2*torch.pi*torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, input_ds, spectral_centroid, loudness):
        latent_vector = self.encoder(input_ds, spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        signal = textsynth_env_batches(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size)
        return signal
    
    def synthesizer(self, input_ds, spectral_centroid, loudness, target_loudness):
        latent_vector = self.encoder(input_ds, spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        signal = textsynth_env(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size, target_loudness)
        return signal
    
    
class DDSP_textenv_stems_gru(nn.Module):
    def __init__(self, hidden_size, N_filter_bank, deepness, compression, frame_size, sampling_rate, seed):
        super().__init__()

        self.N_filter_bank = N_filter_bank
        self.seed = seed
        self.frame_size = frame_size
        self.param_per_env = int(frame_size / (2*N_filter_bank*compression))
        
        self.f_encoder = mlp(1, hidden_size, deepness)
        self.l_encoder = mlp(N_filter_bank, hidden_size, deepness)
        self.z_encoder = gru(2 * hidden_size, hidden_size)
    
        self.a_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, 16 * self.param_per_env)
        self.p_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, 16 * self.param_per_env)

    def encoder(self, spectral_centroid, loudness):
        f = self.f_encoder(spectral_centroid)
        # print("f shape: ",f.shape)
        l = self.l_encoder(loudness)
        # print("l shape: ",l.shape)
        z, _ = self.z_encoder(torch.cat([f,l], dim=-1).unsqueeze(0))
        # print("z_1 shape: ",z.shape)
        z = z.squeeze(0)
        # print("z_2 shape: ",z.shape)
        return torch.cat([f,l,z], dim=-1)

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2*torch.pi*torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, spectral_centroid, loudness, _):
        latent_vector = self.encoder(spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        stems = textsynth_env_stems_batches(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size)
        return stems
    
    def synthesizer(self, input_ds, spectral_centroid, loudness, target_loudness):
        latent_vector = self.encoder(input_ds, spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        stems = textsynth_env(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size, target_loudness)
        return stems    
    
class DDSP_textenv_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, N_filter_bank, deepness, compression, frame_size, sampling_rate, seed):
        super().__init__()

        self.N_filter_bank = N_filter_bank
        self.seed = seed
        self.frame_size = frame_size
        self.param_per_env = int(frame_size / (2*N_filter_bank*compression))
        
        self.f_encoder = mlp(1, hidden_size, deepness)
        self.l_encoder = mlp(N_filter_bank, hidden_size, deepness)
        self.z_encoder = mlp(input_size, hidden_size, deepness)
    
        self.a_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.a_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)
        self.p_decoder_1 = mlp(3 * hidden_size, hidden_size, deepness)
        self.p_decoder_2 = nn.Linear(hidden_size, N_filter_bank * self.param_per_env)

    def encoder(self, spectral_centroid, loudness, input_ds):
        f = self.f_encoder(spectral_centroid)
        # print("f shape: ",f.shape)
        l = self.l_encoder(loudness)
        # print("l shape: ",l.shape)
        z = self.z_encoder(input_ds)
        # print("z_1 shape: ",z.shape)
        return torch.cat([f,l,z], dim=-1)

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2*torch.pi*torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, input_ds, spectral_centroid, loudness):
        latent_vector = self.encoder(input_ds, spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        stems = textsynth_env_stems_batches(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size)
        return stems
    
    def synthesizer(self, input_ds, spectral_centroid, loudness, target_loudness):
        latent_vector = self.encoder(input_ds, spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        stems = textsynth_env(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size, target_loudness)
        return stems