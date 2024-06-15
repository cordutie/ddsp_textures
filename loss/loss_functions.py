from signal_processors.textsynth_env import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio
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

def multispectrogram_loss(original_signal, reconstructed_signal, scales=[2048, 1024, 512, 256], overlap=0.5):
    ori_stft = multiscale_fft(original_signal, scales, overlap)
    rec_stft = multiscale_fft(reconstructed_signal, scales, overlap)

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (torch.log(s_x + 1e-8) - torch.log(s_y + 1e-8)).abs().mean()
        loss += lin_loss + log_loss

    return loss

######## statistics loss ########

def correlation_coefficient(tensor1, tensor2):
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_std = torch.std(tensor1)
    tensor2_std = torch.std(tensor2)
    
    standardized_tensor1 = (tensor1 - tensor1_mean) / tensor1_std
    standardized_tensor2 = (tensor2 - tensor2_mean) / tensor2_std
    
    correlation = torch.mean(standardized_tensor1 * standardized_tensor2)
    
    return correlation

def statistics(signal, N_filter_bank, sample_rate):
    device = signal.device  # Get the device of the input signal tensor
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
    
    new_sample_rate = 11025
    downsampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate).to(device)  # Move downsampler to device

    # Downsampling before computing 
    envelopes_downsampled = []
    for i in range(N_filter_bank):
        envelope = env_subbands[:, i].float().to(device)  # Ensure the envelope is on the same device
        envelopes_downsampled.append(downsampler(envelope).to(torch.float64))

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
    
    # Extract statistics up to order 4 and correlations
    statistics_1 = torch.zeros(N_filter_bank, 4, device=device)
    for i in range(N_filter_bank):
        mu = torch.mean(env_subbands[:, i])
        sigma = torch.sqrt(torch.mean((env_subbands[:, i] - mu) ** 2))
        statistics_1[i, 0] = mu * 1000
        statistics_1[i, 1] = sigma ** 2 / mu ** 2
        statistics_1[i, 2] = (torch.mean((env_subbands[:, i] - mu) ** 3) / sigma ** 3) / 100
        statistics_1[i, 3] = (torch.mean((env_subbands[:, i] - mu) ** 4) / sigma ** 4) / 1000

    statistics_2 = torch.zeros(N_filter_bank * (N_filter_bank - 1) // 2, device=device)
    index = 0
    for i in range(N_filter_bank):
        for j in range(i + 1, N_filter_bank):
            statistics_2[index] = correlation_coefficient(env_subbands[:, i], env_subbands[:, j])
            index += 1

    statistics_3 = torch.zeros(N_filter_bank * 6, device=device)
    for i in range(N_filter_bank):
        sigma_i = torch.std(envelopes_downsampled[i])
        for j in range(6):
            statistics_3[6 * i + j] = torch.std(subenvelopes[i][:, j]) / sigma_i

    statistics_4 = torch.zeros(15, N_filter_bank, device=device)
    for i in range(N_filter_bank):
        counter = 0
        for j in range(6):
            for k in range(j + 1, 6):
                statistics_4[counter, i] = correlation_coefficient(subenvelopes[i][:, j], subenvelopes[i][:, k])
                counter += 1

    statistics_5 = torch.zeros(6, N_filter_bank * (N_filter_bank - 1) // 2, device=device)
    for i in range(6):
        counter = 0
        for j in range(N_filter_bank):
            for k in range(j + 1, N_filter_bank):
                statistics_5[i, counter] = correlation_coefficient(subenvelopes[j][:, i], subenvelopes[k][:, i])
                counter += 1

    return [statistics_1, statistics_2, statistics_3, statistics_4, statistics_5]

def statistics_loss(original_signal, reconstructed_signal):
    original_statistics = statistics(original_signal, 16, 44100)
    reconstructed_statistics = statistics(reconstructed_signal, 16, 44100)
    
    loss = []
    for i in range(5):
        loss_i = torch.sqrt(torch.mean((original_statistics[i] - reconstructed_statistics[i])**2))
        # print("Loss ", i, ": ", loss_i)
        loss.append(loss_i)

    final_loss = loss[0] + 20 * loss[1] + 20 * loss[2] + 20 * loss[3] + 20 * loss[4]

    return final_loss

def batch_statistics_loss(original_signals, reconstructed_signals):
    batch_size = original_signals.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        original_signal = original_signals[i]
        reconstructed_signal = reconstructed_signals[i]
        loss = statistics_loss(original_signal, reconstructed_signal)
        total_loss += loss

    average_loss = total_loss / batch_size
    return average_loss

######## stems loss ########

def stems_loss(original_signals, reconstructed_stems, N_filter_bank=16, sample_rate=44100):
    low_lim = 20  # Low limit of filter
    high_lim = sample_rate / 2  # Centre freq. of highest filter

    # print("shape segments: ",original_signals.shape)

    size = original_signals.shape[1]
    n_batch = original_signals.shape[0]

    original_stems = []
    for i in range(N_filter_bank):
        original_stems.append(torch.zeros(0, size))

    for i in range(n_batch):
        signal = original_signals[i,:]

        # Initialize filter bank
        erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
        
        # Generate subbands for noise
        erb_bank.generate_subbands(signal)
        
        # Extract subbands
        erb_subbands_signal = erb_bank.subbands[:, 1: -1]
        
        for j in range(N_filter_bank):
            original_stems[j] = torch.cat((original_stems[j], erb_subbands_signal[:, j].unsqueeze(0)))
    
    loss = 0

    # print("len og stems: ", len(original_stems))
    # print("len rec stems: ", len(reconstructed_stems))

    # print("shape og stems: ", original_stems[0].shape)
    # print("shape rec stems: ", reconstructed_stems[0].shape)

    for i in range(N_filter_bank):
        loss += torch.mean((original_stems[i]-reconstructed_stems[i])**2)
        
    return loss
