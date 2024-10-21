from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.auxiliar.nn import mlp, gru, mlp_v, gru_v
import torch.nn as nn
import torch
import numpy as np
import librosa
import torchaudio

# example encoder_sizes=[3,5,1]
class DDSP_SubEnv(nn.Module):
    def __init__(self, input_sizes, enc_hidden_size, dec_hidden_size, enc_deepness, dec_deepness, param_per_env, frame_size, N_filter_bank):
        super().__init__()
                
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
        # print("z:", z.shape)
        
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

        signal = SubEnv_stems_batches(real_param, imag_param)
        return signal

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
    
# # CHANGE WHEN PVAE IS IMPLEMENTED --------------------------------------------------------------
# class DDSP_PVAE(nn.Module):
#     def __init__(self, hidden_size, deepness, frame_size, N_filter_bank, sr, model, latent_dim, atoms_size, atoms_number):
#         super().__init__()
        
#         self.sr = sr
#         # Fixing inner VAE
#         self.VAE = model.eval()
#         for param in self.VAE.parameters():
#             param.requires_grad = False
#         self.VAE_latent_dim = latent_dim
#         self.atoms_size = atoms_size
#         self.atoms_number = atoms_number
#         self.N_filter_bank = N_filter_bank
#         self.frame_size = frame_size
#         self.window = torch.hann_window(atoms_size)**0.1
        
#         self.feat0_encoder = mlp_v(1, hidden_size, deepness)
#         self.feat1_encoder = mlp_v(1, hidden_size, deepness)
#         self.z_encoder = gru_v(2 * hidden_size, hidden_size)
        
#         self.a_decoder_1 = mlp_v(3 * hidden_size, hidden_size, deepness)
#         self.a_decoder_2 = nn.Linear(hidden_size, latent_dim * atoms_number)
#         self.t_decoder_1 = mlp_v(3 * hidden_size, hidden_size, deepness)
#         self.t_decoder_2 = nn.Linear(hidden_size, 2)

#     def encoder(self, feature_0, feature_1):
#         # Ensure all tensors are on the same device
#         device = feature_0.device
#         feature_0 = feature_0.to(device)
#         feature_1 = feature_1.to(device)
#         print("feature_0:", feature_0)
#         print("feature_1:", feature_1)
        
#         f = self.feat0_encoder(feature_0)
#         r = self.feat1_encoder(feature_1)
#         z, _ = self.z_encoder(torch.cat([f, r], dim=-1).unsqueeze(0))
#         z = z.squeeze(0)
#         latent_vector = torch.cat([f, r, z], dim=-1)
#         #print if there is a NaN in f
#         if torch.isnan(f).any():
#             print("f has NaN")
#         if torch.isnan(r).any():
#             print("r has NaN")
#         if torch.isnan(z).any():
#             print("z has NaN")
#         return latent_vector

#     def decoder(self, latent_vector):
#         # Ensure all tensors are on the same device
#         device = latent_vector.device
#         latent_vector = latent_vector.to(device)
        
#         a = self.a_decoder_1(latent_vector)
#         a = self.a_decoder_2(a)
#         a = torch.sigmoid(a)*4-2 # atoms are vectors with numbers in (-2,2)
        
#         t = self.t_decoder_1(latent_vector)
#         t = self.t_decoder_2(t)
#         t = torch.sigmoid(t)
                
#         encoded_atoms = a
#         # print("encoded_atoms:", encoded_atoms)
#         # print("encoded_atoms shape:", encoded_atoms.shape)
#         # print("encoded_atoms dim:", encoded_atoms.dim())
        
#         # # rate and alpha are the first two columns of t when in batches
#         # print("t:", t)
#         # print("t shape:", t.shape)
#         # print("t dim:",   t.dim())
#         if t.dim() == 2:
#             rate  = t[:, 0].unsqueeze(1) # Extract and keep the second dimension
#             alpha = t[:, 1].unsqueeze(1)
#         else:
#             rate  = t[0].unsqueeze(0) # Convert to 2D by adding a new dimension
#             alpha = t[1].unsqueeze(0)
            
#         rate  = (rate * 100)+1
#         alpha = (alpha  * 10 + 0.1) ** 2
#         print("rate:", rate)
#         # print("rate shape:", rate.shape)

#         print("alpha:", alpha)
#         # print("alpha shape:", alpha.shape)

#         return rate, alpha, encoded_atoms

#     def forward(self, feature_0, feature_1):
#         latent_vector = self.encoder(feature_0, feature_1)
#         lambda_rate, alpha, encoded_atoms = self.decoder(latent_vector)

#         # Ensure all tensors are on the same device
#         device = latent_vector.device
#         lambda_rate = lambda_rate.to(device)
#         alpha = alpha.to(device)
#         encoded_atoms = encoded_atoms.to(device)
#         signal = P_VAE_batches(self.frame_size, lambda_rate, alpha, self.sr, self.VAE, self.VAE_latent_dim, self.atoms_size, encoded_atoms, window=self.window)
#         return signal

#     def synthesizer(self, feature_0, feature_1, target_loudness):
#         latent_vector = self.encoder(feature_0, feature_1)
#         lambda_rate, alpha, encoded_atoms = self.decoder(latent_vector)

#         # Ensure all tensors are on the same device
#         device = latent_vector.device
#         lambda_rate = lambda_rate.to(device)
#         alpha = alpha.to(device)
#         encoded_atoms = encoded_atoms.to(device)
#         signal = P_VAE(self.frame_size, lambda_rate, alpha, self.sr, self.VAE, self.VAE_latent_dim, self.atoms_size, encoded_atoms, window=self.window)
#         signal = (signal-torch.mean(signal))/ torch.std(signal) * target_loudness
#         return signal
