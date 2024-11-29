import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio
import torch
import ddsp_textures.auxiliar.seeds
import ddsp_textures.dataset.makers

# MULTISCALE SPECTOGRAM LOSS

def multiscale_fft(signal, scales=[4096, 2048, 1024, 512, 256, 128], overlap=.75):
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

def safe_log(x):
    return torch.log(x + 1e-7)

def multiscale_spectrogram_loss(x, x_hat):
    ori_stft = multiscale_fft(x)
    rec_stft = multiscale_fft(x_hat)
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss

# STATISTICS LOSS

def correlation_coefficient(tensor1, tensor2):
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_std = torch.std(tensor1)
    tensor2_std = torch.std(tensor2)
    
    standardized_tensor1 = (tensor1 - tensor1_mean) / tensor1_std
    standardized_tensor2 = (tensor2 - tensor2_mean) / tensor2_std
    
    correlation = torch.mean(standardized_tensor1 * standardized_tensor2)
    
    return correlation

#Before using, make both an erb bank and a log bank:
#erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
# new_size = size // 4 and new_sample_rate = sample_rate // 4
#log_bank = fb.Logarithmic(new_size, new_sample_rate, M_filter_bank, 10, new_sample_rate // 4)
# downsampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate).to(device)  # Move downsampler to device
# new_sample_rate = sample_rate // 4

def statistics(signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler):
    device = signal.device  # Get the device of the input signal tensor
    size = signal.shape[0]

    erb_subbands_signal = erb_bank.generate_subbands(signal)[1:-1, :]
    
    # Extract envelopes using erb bank
    env_subbands = torch.abs(ddsp_textures.auxiliar.seeds.hilbert(erb_subbands_signal))
    
    # Downsampling before computing 
    envelopes_downsampled = []
    for i in range(N_filter_bank):
        envelope = env_subbands[i].float().to(device)  # Ensure the envelope is on the same device
        envelopes_downsampled.append(downsampler(envelope))

    subenvelopes = []
    # new_size = envelopes_downsampled[0].shape[0]

    # Extract envelopes using log bank
    for i in range(N_filter_bank):
        signal = envelopes_downsampled[i]
    
        # Extract subbands
        subenvelopes.append(log_bank.generate_subbands(signal)[1:-1, :])
    
    # FROM SUBENVS: extract statistics up to order 4
    statistics_11 = torch.zeros(N_filter_bank, device=device)
    statistics_12 = torch.zeros(N_filter_bank, device=device)
    statistics_13 = torch.zeros(N_filter_bank, device=device)
    statistics_14 = torch.zeros(N_filter_bank, device=device)
    for i in range(N_filter_bank):
        mu = torch.mean(env_subbands[i])
        sigma = torch.sqrt(torch.mean((env_subbands[i] - mu) ** 2))
        statistics_11[i] = mu
        statistics_12[i] = sigma ** 2 / mu ** 2
        statistics_13[i] = (torch.mean((env_subbands[i] - mu) ** 3) / sigma ** 3)
        statistics_14[i] = (torch.mean((env_subbands[i] - mu) ** 4) / sigma ** 4)

    # FROM SUBENVS: extract correlations
    statistics_2 = []
    for i in range(N_filter_bank):
        nice_neighbours = [j for j in range(i+1, N_filter_bank) if j - i < N_filter_bank // 2]
        for j in nice_neighbours:
            statistics_2.append(correlation_coefficient(env_subbands[i], env_subbands[j]))
    statistics_2 = torch.tensor(statistics_2)

    # FROM SUB-SUBENVS: extract weight of each sub-subenv
    statistics_3 = torch.zeros(N_filter_bank * M_filter_bank, device=device)
    for i in range(N_filter_bank):
        sigma_i = torch.std(envelopes_downsampled[i])
        for j in range(M_filter_bank):
            statistics_3[M_filter_bank * i + j] = torch.std(subenvelopes[i][j]) / sigma_i

    # FROM SUB-SUBENVS: extract correlations between sub-subenvs in the same subenv
    statistics_4 = []
    for i in range(N_filter_bank):
        for j in range(i+1, N_filter_bank):
            for n in range(M_filter_bank):
                statistics_4.append(correlation_coefficient(subenvelopes[i][n], subenvelopes[j][n]))
    statistics_4 = torch.tensor(statistics_4)

    # FROM SUB-SUBENVS: extract correlations between sub-subenvs in different subenvs
    statistics_5 = []
    for i in range(N_filter_bank):
        for j in range(M_filter_bank):
            for k in range(j+1, M_filter_bank):
                statistics_5.append(correlation_coefficient(subenvelopes[i][j], subenvelopes[i][k]))
    statistics_5 = torch.tensor(statistics_5)

    return [statistics_11, statistics_12, statistics_13, statistics_14, statistics_2, statistics_3, statistics_4, statistics_5]

def statistics_loss(original_signal, reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha=torch.tensor([0.0070, 0.0035, 0.8993, 0.0049, 0.0431, 0.0265, 0.0067, 0.0089])):
    original_statistics      = statistics(original_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler)
    reconstructed_statistics = statistics(reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler)

    loss = []
    
    # Determine device from the first tensor in original_statistics or reconstructed_statistics
    device = original_statistics[0].device if original_statistics else reconstructed_statistics[0].device

    for i in range(8):
        loss_i = torch.sqrt(torch.sum((original_statistics[i] - reconstructed_statistics[i])**2))
        # Normalize depending on the amount of data (compute data from shape)
        loss_i = loss_i / original_statistics[i].numel()

        # Ensure loss_i is on the same device as the original signal's statistics
        loss.append(loss_i.to(device))

    # Stack all losses on the same device
    loss_tensor = torch.stack(loss).to(device)
    
    # transform alpha to tensor if it is not already
    if not isinstance(alpha, torch.Tensor):
        print("alpha is not a tensor, transforming to tensor")
        alpha = torch.tensor(alpha, dtype=loss_tensor.dtype, device=loss_tensor.device)

    #dot product between lists loss and alpha (ensure equal dtype)
    final_loss = torch.dot(loss_tensor, alpha)
    
    return  final_loss

def batch_statistics_loss(original_signals, reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha=torch.tensor([0.0070, 0.0035, 0.8993, 0.0049, 0.0431, 0.0265, 0.0067, 0.0089])):
    batch_size = original_signals.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        original_signal      = original_signals[i]
        reconstructed_signal = reconstructed_signals[i]
        loss = statistics_loss(original_signal, reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha)
        total_loss += loss

    average_loss = total_loss / batch_size
    
    return average_loss

# Now the Stems version that uses more ram but is faster
# STEMS STATISTICS -------------------------------------------------------------------------------------------------------------------------------------------
# This new functions take the envolopes (already generated with the erb_bank and hilbert transform) as inputs.

def statistics_stems(stems_torch, N_filter_bank, M_filter_bank, log_bank, downsampler):
    device = stems_torch.device  # Get the device of the input signal tensor
    
    # Extract envelopes using erb bank
    env_subbands = stems_torch
    
    # Downsampling before computing 
    envelopes_downsampled = []
    for i in range(N_filter_bank):
        envelope = env_subbands[i].float().to(device)  # Ensure the envelope is on the same device
        envelopes_downsampled.append(downsampler(envelope))

    subenvelopes = []
    # new_size = envelopes_downsampled[0].shape[0]

    # Extract envelopes using log bank
    for i in range(N_filter_bank):
        signal = envelopes_downsampled[i]
    
        # Extract subbands
        subenvelopes.append(log_bank.generate_subbands(signal)[1:-1, :])
    
    # FROM SUBENVS: extract statistics up to order 4
    statistics_11 = torch.zeros(N_filter_bank, device=device)
    statistics_12 = torch.zeros(N_filter_bank, device=device)
    statistics_13 = torch.zeros(N_filter_bank, device=device)
    statistics_14 = torch.zeros(N_filter_bank, device=device)
    for i in range(N_filter_bank):
        mu = torch.mean(env_subbands[i])
        sigma = torch.sqrt(torch.mean((env_subbands[i] - mu) ** 2))
        statistics_11[i] = mu
        statistics_12[i] = sigma ** 2 / mu ** 2
        statistics_13[i] = (torch.mean((env_subbands[i] - mu) ** 3) / sigma ** 3)
        statistics_14[i] = (torch.mean((env_subbands[i] - mu) ** 4) / sigma ** 4)

    # FROM SUBENVS: extract correlations
    statistics_2 = []
    for i in range(N_filter_bank):
        nice_neighbours = [j for j in range(i+1, N_filter_bank) if j - i < N_filter_bank // 2]
        for j in nice_neighbours:
            statistics_2.append(correlation_coefficient(env_subbands[i], env_subbands[j]))
    statistics_2 = torch.tensor(statistics_2)

    # FROM SUB-SUBENVS: extract weight of each sub-subenv
    statistics_3 = torch.zeros(N_filter_bank * M_filter_bank, device=device)
    for i in range(N_filter_bank):
        sigma_i = torch.std(envelopes_downsampled[i])
        for j in range(M_filter_bank):
            statistics_3[M_filter_bank * i + j] = torch.std(subenvelopes[i][j]) / sigma_i

    # FROM SUB-SUBENVS: extract correlations between sub-subenvs in the same subenv
    statistics_4 = []
    for i in range(N_filter_bank):
        for j in range(i+1, N_filter_bank):
            for n in range(M_filter_bank):
                statistics_4.append(correlation_coefficient(subenvelopes[i][n], subenvelopes[j][n]))
    statistics_4 = torch.tensor(statistics_4)

    # FROM SUB-SUBENVS: extract correlations between sub-subenvs in different subenvs
    statistics_5 = []
    for i in range(N_filter_bank):
        for j in range(M_filter_bank):
            for k in range(j+1, M_filter_bank):
                statistics_5.append(correlation_coefficient(subenvelopes[i][j], subenvelopes[i][k]))
    statistics_5 = torch.tensor(statistics_5)

    return [statistics_11, statistics_12, statistics_13, statistics_14, statistics_2, statistics_3, statistics_4, statistics_5]

def statistics_loss_stems(original_stems, reconstructed_stems, N_filter_bank, M_filter_bank, log_bank, downsampler, alpha=torch.tensor([0.0070, 0.0035, 0.8993, 0.0049, 0.0431, 0.0265, 0.0067, 0.0089])):
    original_statistics      = statistics_stems(original_stems,      N_filter_bank, M_filter_bank, log_bank, downsampler)
    reconstructed_statistics = statistics_stems(reconstructed_stems, N_filter_bank, M_filter_bank, log_bank, downsampler)
    
    loss = []
    
    # Determine device from the first tensor in original_statistics or reconstructed_statistics
    device = original_statistics[0].device if original_statistics else reconstructed_statistics[0].device

    for i in range(8):
        loss_i = torch.sqrt(torch.sum((original_statistics[i] - reconstructed_statistics[i])**2))
        # Normalize depending on the amount of data (compute data from shape)
        loss_i = loss_i / original_statistics[i].numel()

        # Ensure loss_i is on the same device as the original signal's statistics
        loss.append(loss_i.to(device))

    # Stack all losses on the same device
    loss_tensor = torch.stack(loss).to(device)

    # transform alpha to tensor if it is not already
    if not isinstance(alpha, torch.Tensor):
        print("alpha is not a tensor, transforming to tensor")
        alpha = torch.tensor(alpha, dtype=loss_tensor.dtype)
    
    alpha = alpha.to(loss_tensor.device)

    #dot product between lists loss and alpha (ensure equal dtype)
    final_loss = torch.dot(loss_tensor, alpha)
    
    return  final_loss

def batch_statistics_loss_stems(original_stems_batch, reconstructed_stems_batch, N_filter_bank, M_filter_bank, _, log_bank, downsampler, alpha=torch.tensor([0.0070, 0.0035, 0.8993, 0.0049, 0.0431, 0.0265, 0.0067, 0.0089])):
    batch_size = original_stems_batch.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        original_stems      = original_stems_batch[i]
        reconstructed_stems = reconstructed_stems_batch[i]
        loss = statistics_loss_stems(original_stems, reconstructed_stems, N_filter_bank, M_filter_bank, log_bank, downsampler, alpha)
        total_loss += loss

    average_loss = total_loss / batch_size
    
    return average_loss