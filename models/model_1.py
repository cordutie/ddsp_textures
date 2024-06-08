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

class DDSP_textenv(nn.Module):
    def _init_(self, hidden_size, N_filter_bank, deepness, compression, frame_size, sampling_rate, seed):
        super()._init_()

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

    def forward(self, spectral_centroid, loudness):
        latent_vector = self.encoder(spectral_centroid, loudness)
        real_param, imag_param = self.decoder(latent_vector)

        # Move latent vectors to the same device as real_param and imag_param
        device = real_param.device
        latent_vector = latent_vector.to(device)

        # Ensure all tensors are on the same device
        spectral_centroid = spectral_centroid.to(device)
        loudness = loudness.to(device)

        signal = textsynth_env_batches(real_param, imag_param, self.seed, self.N_filter_bank, self.frame_size)
        return signal, self.seed

# # Initialize model and move it to the appropriate device
# hidden_size = 128  # Example hidden size
# N_filter_bank = 16  # Example filter bank size
# frame_size = 2**15  # Example frame size
# sampling_rate = 44100  # Example sampling rate
# compression = 8  # Placeholder for compression

# model = textsynth_DDSP(hidden_size=128, N_filter_bank=16, deepness=2, compression=8, frame_size=2**15, sampling_rate=44100).to(device)



def feature_extractor(signal, sample_rate, N_filter_bank):
    size = signal.shape[0]
    sp_centroid = torchaudio.functional.spectral_centroid(signal, sample_rate, 0, torch.hamming_window(size), size, size, size) 

    low_lim = 20  # Low limit of filter
    high_lim = sample_rate / 2  # Centre freq. of highest filter

     # Initialize filter bank
    erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
    
    # Generate subbands for noise
    erb_bank.generate_subbands(signal)
    
    # Extract subbands
    erb_subbands_signal = erb_bank.subbands[:, 1:-1]

    loudness = torch.norm(erb_subbands_signal, dim=0)
    return [sp_centroid[0], loudness]

class SoundDataset(Dataset):
    def _init_(self, audio_path, frame_size, hop_size, sampling_rate, N_filter_bank):
        self.audio_path = audio_path
        self.frame_size = frame_size
        self.hop_size   = hop_size
        self.sampling_rate = sampling_rate
        self.N_filter_bank = N_filter_bank
        self.audio, _ = librosa.load(audio_path, sr=sampling_rate)
        self.content = []

    def compute_dataset(self):
        audio_tensor = torch.tensor(self.audio)
        size = audio_tensor.shape[0]
        dataset_size = (size - self.frame_size) // self.hop_size
        for i in range(dataset_size):
            segment = audio_tensor[i * self.hop_size: i * self.hop_size + self.frame_size]
            features = feature_extractor(segment, self.sampling_rate, self.N_filter_bank)
            self.content.append([features, segment])
        # print(dataset_size)

# dataset = SoundDataset(audio_path='noises/fire_long.wav', frame_size=2*15, hop_size=2*10, sampling_rate=44100)
# dataset.compute_dataset()
# actual_dataset = dataset.content
# dataloader = DataLoader(actual_dataset, batch_size=32, shuffle=True)

import torch

def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts

def multispectrogram_loss(original_signal, reconstructed_signal, scales, overlap):
    ori_stft = multiscale_fft(original_signal, scales, overlap)
    rec_stft = multiscale_fft(reconstructed_signal, scales, overlap)

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (torch.log(s_x + 1e-8) - torch.log(s_y + 1e-8)).abs().mean()
        loss += lin_loss + log_loss

    return loss

def correlation_coefficient(tensor1, tensor2):
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_std = torch.std(tensor1)
    tensor2_std = torch.std(tensor2)
    
    standardized_tensor1 = (tensor1 - tensor1_mean) / tensor1_std
    standardized_tensor2 = (tensor2 - tensor2_mean) / tensor2_std
    
    correlation = torch.mean(standardized_tensor1 * standardized_tensor2)
    
    return correlation

import matplotlib.pyplot as plt

def statistics(signal, N_filter_bank, sample_rate):
    size = signal.shape[0]

    low_lim = 20  # Low limit of filter
    high_lim = sample_rate / 2  # Centre freq. of highest filter

    # Initialize filter bank
    erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
    
    # Generate subbands for noise
    erb_bank.generate_subbands(signal)
    
    # Extract subbands
    erb_subbands_signal = erb_bank.subbands[:, 1:-1]

    # Extract envelopes
    env_subbands = torch.abs(hilbert(erb_subbands_signal))
    
    # Plot original signal
    # plt.figure(figsize=(12, 6))
    # plt.plot(signal, label='Original Signal')
    # plt.title('Original Signal')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # # Plot envelopes in env_subbands
    # plt.figure(figsize=(12, 6))
    # for i in range(N_filter_bank):
    #     plt.plot(env_subbands[:, i], label=f'Envelope {i}')
    # plt.title('Envelopes in env_subbands')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    new_sample_rate = 11025
    downsampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate)

    # Downsampling before computing 
    envelopes_downsampled = []
    for i in range(N_filter_bank):
        envelopes_downsampled.append(downsampler(env_subbands[:, i].float()).to(torch.float64))

    # Plot downsampled envelopes
    # plt.figure(figsize=(12, 6))
    # for i in range(N_filter_bank):
    #     plt.plot(envelopes_downsampled[i], label=f'Downsampled Envelope {i}')
    # plt.title('Downsampled Envelopes')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    subenvelopes = []

    new_size = envelopes_downsampled[0].shape[0]

    for i in range(N_filter_bank):
        signal = envelopes_downsampled[i]  
        
        # Initialize filter bank
        log_bank = fb.Logarithmic(new_size, sample_rate, 6, 10, new_sample_rate // 4)
    
        # Generate subbands for noise
        log_bank.generate_subbands(signal)
    
        # Extract subbands
        subenvelopes.append(log_bank.subbands[:, 1:-1])
    
    # Plot envelopes in subenvelopes
    # plt.figure(figsize=(12, 6))
    # for i in range(N_filter_bank):
    #     for j in range(6):
    #         plt.plot(subenvelopes[i][:, j], label=f'Sub Envelope {i}-{j}')
    # plt.title('Envelopes in subenvelopes')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # Extract statistcs up to order 4 and correlations
    statistics_1 = torch.zeros(N_filter_bank, 4)
    for i in range(N_filter_bank):
        mu    = torch.mean(env_subbands[:,i])
        sigma = torch.sqrt(torch.mean((env_subbands[:,i]-mu)**2))
        statistics_1[i,0] = mu * 1000
        statistics_1[i,1] = sigma**2 / mu**2
        statistics_1[i,2] = (torch.mean((env_subbands[:,i]-mu)**3) / sigma**3) / 100
        statistics_1[i,3] = (torch.mean((env_subbands[:,i]-mu)**4) / sigma**4) / 1000

    print("Statistics 1: ", statistics_1)

    # Extract correlations
    statistics_2 = torch.zeros(N_filter_bank*(N_filter_bank-1) // 2)
    index = 0
    for i in range(N_filter_bank):
        for j in range(i+1,N_filter_bank):
            statistics_2[index] = correlation_coefficient(env_subbands[:,i], env_subbands[:,j])
            index += 1

    print("Statistics 2: ", statistics_2)

    # Extract modulation powers
    statistics_3 = torch.zeros(N_filter_bank*6)

    for i in range(N_filter_bank):
        sigma_i = torch.std(envelopes_downsampled[i])
        for j in range(6):
            statistics_3[6*i + j] = torch.std(subenvelopes[i][j]) / sigma_i
    
    print("Statistics 3: ", statistics_3)

    statistics_4 = torch.zeros(15, N_filter_bank)

    for i in range(N_filter_bank):
        counter = 0
        for j in range(6):
            for k in range(j+1,6):
                statistics_4[counter,i] = correlation_coefficient(subenvelopes[i][:,j], subenvelopes[i][:,k])
                counter +=1
    
    print("Statistics 4: ", statistics_4)

    statistics_5 = torch.zeros(6, N_filter_bank*(N_filter_bank-1) // 2)

    for i in range(6):
        counter = 0
        for j in range(N_filter_bank):
            for k in range(j+1,N_filter_bank):
                statistics_5[i,counter] = correlation_coefficient(subenvelopes[j][:,i], subenvelopes[k][:,i])
                counter += 1

    print("Statistics 5: ", statistics_5)

    return [statistics_1, statistics_2, statistics_3, statistics_4, statistics_5] 

def statistics_loss(original_signal, reconstructed_signal):
    original_statistics = statistics(original_signal, 16, 44100)
    reconstructed_statistics = statistics(reconstructed_signal, 16, 44100)
    
    loss = []
    for i in range(5):
        loss_i = torch.sqrt(torch.mean((original_statistics[i] - reconstructed_statistics[i])**2))
        print("Loss ", i, ": ", loss_i)
        loss.append(loss_i)

    final_loss = loss[0] + 20 * loss[1] + 20 * loss[2] + 20 * loss[3] + 20 * loss[4]

    return final_loss

def batch_statistics_loss(original_signals, reconstructed_signals, extra_loss):
    batch_size = original_signals.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        original_signal = original_signals[i]
        reconstructed_signal = reconstructed_signals[i]
        loss = statistics_loss(original_signal, reconstructed_signal, extra_loss)
        total_loss += loss

    average_loss = total_loss / batch_size
    return average_loss