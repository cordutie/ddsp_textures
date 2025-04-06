from ddsp_textures.signal_processors.synthesizers import *
from ddsp_textures.auxiliar.nn import mlp, gru, mlp_v, gru_v
from ddsp_textures.auxiliar.seeds import seed_maker
import torch.nn as nn
import torch
import numpy as np
import librosa
import torchaudio

# example encoder_sizes=[3,5,1]
class DDSP_TexEnv(nn.Module):
    def __init__(self, input_sizes, enc_hidden_size, dec_hidden_size, enc_deepness, dec_deepness, param_per_env, frame_size, N_filter_bank, device, seed):
        super().__init__()

        self.N_filter_bank = N_filter_bank
        self.frame_size = frame_size
        self.param_per_env = param_per_env
        self.real_normalizing_factor = np.sqrt(frame_size)
        self.imag_normalizing_factor = np.sqrt(np.sqrt(frame_size))

        self.seed = seed.to(device)

        self.encoders = nn.ModuleList()
        
        # Loop through the input list and create an MLP for each in_size in input_sizes
        for in_size in input_sizes:
            # print("Creating encoder with input size", in_size)
            self.encoders.append(mlp(in_size, enc_hidden_size, enc_deepness))
        # Create z encoder
        self.z_encoder = gru(len(input_sizes) * enc_hidden_size, enc_hidden_size)
    
        self.a_decoder_1 = mlp((len(input_sizes)+1) * enc_hidden_size, dec_hidden_size, dec_deepness)
        self.a_decoder_2 = nn.Linear(dec_hidden_size, N_filter_bank * self.param_per_env // 2)
        self.p_decoder_1 = mlp((len(input_sizes)+1) * enc_hidden_size, dec_hidden_size, dec_deepness)
        self.p_decoder_2 = nn.Linear(dec_hidden_size, N_filter_bank * self.param_per_env // 2)

    def seed_retrieve(self):
        return self.seed

    def encoder(self, features):
        latent_vectors_list = []
        for i in range(len(features)):
            # print("Encoder ", i, " with input size ", features[i].shape)
            latent_vectors_list.append(self.encoders[i](features[i]))
        
        latent_vectors_list.reverse()

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
        a = torch.sigmoid(a)
        p = self.p_decoder_1(latent_vector)
        p = self.p_decoder_2(p)
        p = 2*torch.pi*torch.sigmoid(p)
        real_param = a * torch.cos(p)
        imag_param = a * torch.sin(p)
        return real_param, imag_param
    
    # def decoder(self, latent_vector):
    #     real_param = self.a_decoder_1(latent_vector)
    #     real_param = self.a_decoder_2(real_param)
    #     real_param = self.real_normalizing_factor * torch.sigmoid(real_param)
    #     imag_param = self.p_decoder_1(latent_vector)
    #     imag_param = self.p_decoder_2(imag_param)
    #     imag_param = self.imag_normalizing_factor * torch.sigmoid(imag_param)
    #     return real_param, imag_param

    def forward(self, features):
        # Encoding
        latent_vector          = self.encoder(features)

        # Decoding
        # if self.poltocar:
        #     real_param, imag_param = self.decoder_poltocar(latent_vector)
        # else:
        real_param, imag_param = self.decoder(latent_vector)
        
        # Synthesizing
        # if self.stems:
        #     output = TexEnv_stems_batches(real_param, imag_param, self.frame_size, self.N_filter_bank)
        # else:
        output                 = TexEnv_batches(real_param, imag_param, self.seed)

        return output

    def synthesizer(self, features, type_loudness, target_loudness, seed):
        latent_vector          = self.encoder(features)
        real_param, imag_param = self.decoder(latent_vector)
        signal = TexEnv(real_param, imag_param, seed, type_loudness, target_loudness)
        return signal

#     def synthesizer(self, features, target_loudness, seed):
#         latent_vector = self.encoder(features)
#         # print("Latent vector", latent_vector)
#         # print("Latent vector shape", latent_vector.shape)

#         real_param, imag_param = self.decoder_poltocar(latent_vector)
#         # print("Real param", real_param)
#         # print("Real param shape", real_param.shape)
#         # print("Imag param", imag_param)
#         # print("Imag param shape", imag_param.shape)

#         signal = textsynth_env(real_param, imag_param, seed, self.N_filter_bank, self.frame_size, target_loudness)
#         # print("Signal", signal)
#         # print("Signal shape", signal.shape)
#         return signal
    
# def textsynth_env(parameters_real, parameters_imag, seed, N_filter_bank, size, target_loudness=1):
#     N = parameters_real.size(0)
#     parameters_size = N // N_filter_bank
#     signal_final = torch.zeros(size, dtype=torch.float32).to(parameters_real.device)

#     seed.to(parameters_real.device)
    
#     for i in range(N_filter_bank):
#         # Construct the local parameters as a complex array
#         parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
#         # print("parameters_local shape: ", parameters_local.shape)

#         # Initialize FFT coefficients array
#         fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64)
#         fftcoeff_local[:parameters_size] = parameters_local ###########################################3
        
#         # Compute the inverse FFT to get the local envelope
#         env_local = torch.fft.irfft(fftcoeff_local).to(parameters_real.device)
#         # print("env_local shape: ", env_local.shape)

#         # Extract the local noise
#         noise_local = seed[i].to(parameters_real.device)
#         print("noise_local shape: ", noise_local.shape)
        
#         # Generate the texture sound by multiplying the envelope and noise
#         texture_sound_local = env_local * noise_local
        
#         # Accumulate the result
#         signal_final += texture_sound_local
    
#     loudness = torch.sqrt(torch.mean(signal_final ** 2))
#     signal_final = signal_final / loudness

#     signal_final = target_loudness * signal_final

#     return signal_final