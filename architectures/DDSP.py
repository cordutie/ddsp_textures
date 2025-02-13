from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.auxiliar.nn import mlp, gru, mlp_v, gru_v
from ddsp_textures.auxiliar.seeds import seed_maker
import torch.nn as nn
import torch
import numpy as np
import librosa
import torchaudio

# example encoder_sizes=[3,5,1]
class DDSP_SubEnv(nn.Module):
    def __init__(self, input_sizes, enc_hidden_size, dec_hidden_size, enc_deepness, dec_deepness, param_per_env, frame_size, N_filter_bank, device, sampling_rate = 44100, stems=True):
        super().__init__()

        self.stems = stems        
        self.N_filter_bank = N_filter_bank
        self.frame_size = frame_size
        self.param_per_env = param_per_env
        
        self.seed = seed_maker(frame_size, sampling_rate, N_filter_bank).to(device)

        self.encoders = nn.ModuleList()
        
        # Loop through the input list and create an MLP for each in_size in input_sizes
        for in_size in input_sizes:
            print("Creating encoder with input size", in_size)
            self.encoders.append(mlp(in_size, enc_hidden_size, enc_deepness))
        # Create z encoder
        self.z_encoder = gru(len(input_sizes) * enc_hidden_size, enc_hidden_size)
    
        self.a_decoder_1 = mlp((len(input_sizes)+1) * enc_hidden_size, dec_hidden_size, dec_deepness)
        self.a_decoder_2 = nn.Linear(dec_hidden_size, N_filter_bank * self.param_per_env)
        self.p_decoder_1 = mlp((len(input_sizes)+1) * enc_hidden_size, dec_hidden_size, dec_deepness)
        self.p_decoder_2 = nn.Linear(dec_hidden_size, N_filter_bank * self.param_per_env)

    def seed_retrieve(self):
        return self.seed

    def encoder(self, features):
        latent_vectors_list = []
        for i in range(len(features)):
            # print("Encoder ", i, " with input size ", features[i].shape)
            latent_vectors_list.append(self.encoders[i](features[i]))
        
        # print("latent_vector list shape", torch.cat(latent_vectors_list, dim=-1).shape)
        # print("and after unsqueezing", torch.cat(latent_vectors_list, dim=-1).unsqueeze(0).shape)

        z, _ = self.z_encoder(torch.cat(latent_vectors_list, dim=-1).unsqueeze(0))
        z = z.squeeze(0)
        # print("Z shape", z.shape)
        
        actual_latent_vector = torch.cat(latent_vectors_list + [z], dim=-1)
        # print("Actual latent vector shape", actual_latent_vector.shape)
        return actual_latent_vector

    def decoder(self, latent_vector):
        a = self.a_decoder_1(latent_vector)
        a = self.a_decoder_2(a)
        a = self.param_per_env*torch.sigmoid(a) # normalization
        # a = torch.sigmoid(a) # normalization
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2 * torch.pi * torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param

    def forward(self, features):
        latent_vector          = self.encoder(features)
        real_param, imag_param = self.decoder(latent_vector)
        if self.stems:
            output = SubEnv_stems_batches(real_param, imag_param, self.frame_size, self.N_filter_bank)
        else:
            output = SubEnv_batches(real_param, imag_param, self.seed)
        return output

    def synthesizer(self, features, target_loudness, seed):
        latent_vector          = self.encoder(features)
        real_param, imag_param = self.decoder(latent_vector)

        signal = SubEnv(real_param, imag_param, seed, target_loudness)
        return signal