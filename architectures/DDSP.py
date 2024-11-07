from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.auxiliar.nn import mlp, gru, mlp_v, gru_v
import torch.nn as nn
import torch
import numpy as np
import librosa
import torchaudio

# example encoder_sizes=[3,5,1]
class DDSP_SubEnv(nn.Module):
    def __init__(self, input_sizes, enc_hidden_size, dec_hidden_size, enc_deepness, dec_deepness, param_per_env, frame_size, N_filter_bank, stems=True):
        super().__init__()

        self.stems = stems        
        self.N_filter_bank = N_filter_bank
        self.frame_size = frame_size
        self.param_per_env = param_per_env
        
        self.encoders = nn.ModuleList()
        
        # Loop through the input list and create an MLP for each in_size
        for in_size in input_sizes:
            self.encoders.append(mlp(in_size, enc_hidden_size, enc_deepness))
        # Create z encoder
        self.z_encoder = gru(len(input_sizes) * enc_hidden_size, enc_hidden_size)
    
        self.a_decoder_1 = mlp((len(input_sizes)+1) * enc_hidden_size, dec_hidden_size, dec_deepness)
        self.a_decoder_2 = nn.Linear(dec_hidden_size, N_filter_bank * self.param_per_env)
        self.p_decoder_1 = mlp((len(input_sizes)+1) * enc_hidden_size, dec_hidden_size, dec_deepness)
        self.p_decoder_2 = nn.Linear(dec_hidden_size, N_filter_bank * self.param_per_env)

    def encoder(self, features):
        latent_vectors_list = []
        for i in range(len(features)):
            latent_vectors_list.append(self.encoders[i](features[i]))
        
        z, _ = self.z_encoder(torch.cat(latent_vectors_list, dim=-1).unsqueeze(0))
        z = z.squeeze(0)
        
        actual_latent_vector = torch.cat(latent_vectors_list + [z], dim=-1)
        return actual_latent_vector

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

    def forward(self, features):
        latent_vector          = self.encoder(features)
        real_param, imag_param = self.decoder(latent_vector)

        # # Ensure all tensors are on the same device
        # device = real_param.device
        # latent_vector = latent_vector.to(device)

        if self.stems:
            output = SubEnv_stems_batches(real_param, imag_param)
        else:
            output = SubEnv_batches(real_param, imag_param)
        return output

    def synthesizer(self, features, target_loudness, seed):
        latent_vector          = self.encoder(features)
        real_param, imag_param = self.decoder(latent_vector)

        # Ensure all tensors are on the same device
        device = real_param.device
        latent_vector = latent_vector.to(device)
        feature_0 = feature_0.to(device)
        feature_1 = feature_1.to(device)

        signal = SubEnv(real_param, imag_param, seed, target_loudness)
        return signal