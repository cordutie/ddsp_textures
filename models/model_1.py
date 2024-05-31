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
    def __init__(self, audio_path, frame_size, hop_size, sampling_rate, N_filter_bank):
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

# dataset = SoundDataset(audio_path='noises/fire_long.wav', frame_size=2**15, hop_size=2**10, sampling_rate=44100)
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