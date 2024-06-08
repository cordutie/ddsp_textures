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

def correlation_coefficient(tensor1, tensor2):
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_std = torch.std(tensor1)
    tensor2_std = torch.std(tensor2)
    
    standardized_tensor1 = (tensor1 - tensor1_mean) / tensor1_std
    standardized_tensor2 = (tensor2 - tensor2_mean) / tensor2_std
    
    correlation = torch.mean(standardized_tensor1 * standardized_tensor2)
    
    return correlation

######## statistics loss ########

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