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

def multiscale_spectrogram_loss(x, x_hat, a, b, c, d, e):
    ori_stft = multiscale_fft(x)
    rec_stft = multiscale_fft(x_hat)
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss

# STATISTICS LOSS NEW ---------------------------------------------------------------------

def correlation_coefficient(tensor1, tensor2):
    mean1 = tensor1.mean(dim=-1, keepdim=True)
    mean2 = tensor2.mean(dim=-1, keepdim=True)
    
    tensor1 = tensor1 - mean1
    tensor2 = tensor2 - mean2
    
    std1 = tensor1.norm(dim=-1) / (tensor1.shape[-1] ** 0.5)  # Equivalent to std but avoids computing mean again
    std2 = tensor2.norm(dim=-1) / (tensor2.shape[-1] ** 0.5)
    
    corr = (tensor1 * tensor2).mean(dim=-1) / (std1 * std2)
    return corr

#Before using, make both an erb bank and a log bank:
# erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
# new_size = size // 4 and new_sample_rate = sample_rate // 4
# log_bank = fb.Logarithmic(new_size, new_sample_rate, M_filter_bank, 10, new_sample_rate // 4)
# downsampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate).to(device)  # Move downsampler to device
def statistics_mcds(signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha=torch.tensor([100, 1, 1/10, 1/100])):
    # print("signals.requires_grad:", signals.requires_grad)
    device = signals.device

    # alpha to device
    alpha = alpha.to(device)

    if signals.dim() == 1:  # Single signal case
        signals = signals.unsqueeze(0)  # Add batch dimension (1, Size)
        was_single = True
    else:
        was_single = False
    batch_size = signals.shape[0]

    erb_subbands = erb_bank.generate_subbands(signals)[:, 1:-1, :]
    # print("erb_subbands.requires_grad:", erb_subbands.requires_grad)

    N_filter_bank = erb_subbands.shape[1]

    env_subbands = torch.abs(ddsp_textures.auxiliar.seeds.hilbert(erb_subbands))
    # print("env_subbands.requires_grad:", env_subbands.requires_grad)

    env_subbands_downsampled = downsampler(env_subbands.float())
    # print("env_subbands_downsampled.requires_grad:", env_subbands_downsampled.requires_grad)

    length_downsampled       = env_subbands_downsampled.shape[-1]

    subenvelopes = torch.zeros((batch_size, N_filter_bank, M_filter_bank, length_downsampled), device=device)
    for i in range(N_filter_bank):
        banda     = env_subbands_downsampled[:, i, :]
        subbandas = log_bank.generate_subbands(banda)[:, 1:-1, :]
        subenvelopes[:, i, :, :] = subbandas
    # print("subenvelopes.requires_grad:", subenvelopes.requires_grad)

    mu = env_subbands.mean(dim=-1)
    sigma = env_subbands.std(dim=-1)
    # print("mu.requires_grad:", mu.requires_grad)
    # print("sigma.requires_grad:", sigma.requires_grad)

    # stats_11 = mu
    # stats_12 = (sigma ** 2) / (mu ** 2)
    # normalized_env_subbands = (env_subbands - mu.unsqueeze(-1))
    # stats_13 = (normalized_env_subbands ** 3).mean(dim=-1) / (sigma ** 3)
    # stats_14 = (normalized_env_subbands ** 4).mean(dim=-1) / (sigma ** 4)

    # Comparison version
    stats_1 = torch.zeros(batch_size, N_filter_bank, 4, device=device)
    stats_1[:, :, 0] = mu * alpha[0]
    stats_1[:, :, 1] = ((sigma ** 2) / (mu ** 2) ) * alpha[1]
    normalized_env_subbands = (env_subbands - mu.unsqueeze(-1))
    stats_1[:, :, 2] = ((normalized_env_subbands ** 3).mean(dim=-1) / (sigma ** 3)) * alpha[2]
    stats_1[:, :, 3] = ((normalized_env_subbands ** 4).mean(dim=-1) / (sigma ** 4)) * alpha[3]
 
    # print("stats_11.requires_grad:", stats_11.requires_grad)
    # print("stats_12.requires_grad:", stats_12.requires_grad)
    # print("stats_13.requires_grad:", stats_13.requires_grad)
    # print("stats_14.requires_grad:", stats_14.requires_grad)

    corr_pairs = torch.triu_indices(N_filter_bank, N_filter_bank, 1)
    stats_2 = correlation_coefficient(env_subbands[:, corr_pairs[0]], env_subbands[:, corr_pairs[1]])
    # print("stats_2.requires_grad:", stats_2.requires_grad)

    subenv_sigma = subenvelopes.std(dim=-1)
    # stats_3 = (subenv_sigma / (env_subbands_downsampled.std(dim=-1, keepdim=True))).reshape(-1)
    stats_3 = (subenv_sigma / (env_subbands_downsampled.std(dim=-1, keepdim=True))).view(batch_size, -1)
    # print("stats_3.requires_grad:", stats_3.requires_grad)

    cross_corr_across_subbands = correlation_coefficient(subenvelopes[:, None, :, :, :], subenvelopes[:, :, None, :, :])
    # stats_4 = cross_corr_across_subbands[:, torch.triu_indices(N_filter_bank, N_filter_bank, 1)[0], torch.triu_indices(N_filter_bank, N_filter_bank, 1)[1]].reshape(-1)
    stats_4 = cross_corr_across_subbands[:, torch.triu_indices(N_filter_bank, N_filter_bank, 1)[0], torch.triu_indices(N_filter_bank, N_filter_bank, 1)[1]].view(batch_size, -1)
    # print("stats_4.requires_grad:", stats_4.requires_grad)

    cross_corr_subenvs = correlation_coefficient(subenvelopes[:, :, None, :, :], subenvelopes[:, :, :, None, :])
    # print("corre_shape",cross_corr_subenvs.shape)
    # stats_5 = cross_corr_subenvs[:, :, torch.triu_indices(M_filter_bank, M_filter_bank, 1)[0], torch.triu_indices(M_filter_bank, M_filter_bank, 1)[1]].reshape(-1)
    stats_5 = cross_corr_subenvs[:, :, torch.triu_indices(M_filter_bank, M_filter_bank, 1)[0], torch.triu_indices(M_filter_bank, M_filter_bank, 1)[1]]
    stats_5 = stats_5.permute(0, 2, 1).contiguous().view(batch_size, -1)
    # print("stats_5.requires_grad:", stats_5.requires_grad)

    # return [stats_11, stats_12, stats_13, stats_14, stats_2, stats_3, stats_4, stats_5]
    return [stats_1, stats_2, stats_3, stats_4, stats_5]

# alpha=torch.tensor([0.3, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1])
# alpha = torch.tensor([0.0070, 0.0035, 0.8993, 0.0049, 0.0431, 0.0265, 0.0067, 0.0089])
# alpha_old = torch.tensor([1000, 1, 0.01, 0.0001, 20, 20, 20, 20]) # actually no lol
def statistics_mcds_loss(original_signals, reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha = torch.tensor([100,1,1/10,1/100]), beta=torch.tensor([1, 20, 20, 20, 20])):
    if original_signals.dim() == 1:  # Single signal case
        original_signals      = original_signals.unsqueeze(0)  # Add batch dimension (1, Size)
        reconstructed_signals = reconstructed_signals.unsqueeze(0)

    original_stats      = statistics_mcds(original_signals,      N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha)
    reconstructed_stats = statistics_mcds(reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha)

    # Compute per-statistic loss and take mean over feature dimensions to ensure scalars
    losses = [torch.sqrt(torch.mean((o - r) ** 2, dim=list(range(1, o.dim())))) for o, r in zip(original_stats, reconstructed_stats)]

    # Stack and take batch mean
    losses = torch.stack(losses, dim=-1).mean(dim=0)  

    # Apply weighting
    return (losses * beta.to(losses.device)).sum()

######## moments statistics loss OLD ######## --------------------------------------------------------------------------


# def statistics_mom(signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler):
#     device = signals.device
#     if signals.dim() == 1:  # Single signal case
#         signals = signals.unsqueeze(0)  # Add batch dimension (1, Size)
#         was_single = True
#     else:
#         was_single = False
#     batch_size = signals.shape[0]

#     erb_subbands = erb_bank.generate_subbands(signals)[:, 1:-1, :]

#     N_filter_bank = erb_subbands.shape[1]

#     env_subbands = torch.abs(ddsp_textures.auxiliar.seeds.hilbert(erb_subbands))
#     env_subbands_downsampled = downsampler(env_subbands.float())
#     length_downsampled       = env_subbands_downsampled.shape[-1]

#     subenvelopes = torch.zeros((batch_size, N_filter_bank, M_filter_bank, length_downsampled), device=device)
#     for i in range(N_filter_bank):
#         banda     = env_subbands_downsampled[:, i, :]
#         subbandas = log_bank.generate_subbands(banda)[:, 1:-1, :]
#         subenvelopes[:, i, :, :] = subbandas

#     mu = env_subbands.mean(dim=-1)
#     sigma = env_subbands.std(dim=-1)

#     stats_11 = mu
#     stats_12 = (sigma ** 2) / (mu ** 2)
#     normalized_env_subbands = (env_subbands - mu.unsqueeze(-1))
#     stats_13 = (normalized_env_subbands ** 3).mean(dim=-1) / (sigma ** 3)
#     stats_14 = (normalized_env_subbands ** 4).mean(dim=-1) / (sigma ** 4)
#     stats_15 = (normalized_env_subbands ** 5).mean(dim=-1) / (sigma ** 5)
#     stats_16 = (normalized_env_subbands ** 6).mean(dim=-1) / (sigma ** 6)
#     stats_17 = (normalized_env_subbands ** 7).mean(dim=-1) / (sigma ** 7)
#     stats_18 = (normalized_env_subbands ** 8).mean(dim=-1) / (sigma ** 8)

#     corr_pairs = torch.triu_indices(N_filter_bank, N_filter_bank, 1)
#     stats_2 = correlation_coefficient(env_subbands[:, corr_pairs[0]], env_subbands[:, corr_pairs[1]])

#     subenv_sigma = subenvelopes.std(dim=-1)
#     # stats_3 = (subenv_sigma / (env_subbands_downsampled.std(dim=-1, keepdim=True))).reshape(-1)
#     stats_3 = (subenv_sigma / (env_subbands_downsampled.std(dim=-1, keepdim=True))).view(batch_size, -1)

#     return [stats_11, stats_12, stats_13, stats_14, stats_15, stats_16, stats_17, stats_18, stats_2, stats_3]


# def statistics_mom_loss(original_signals, reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha=torch.tensor([0.3, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1])):
#     if original_signals.dim() == 1:  # Single signal case
#         original_signals      = original_signals.unsqueeze(0)  # Add batch dimension (1, Size)
#         reconstructed_signals = reconstructed_signals.unsqueeze(0)

#     original_stats      = statistics_mom(original_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler)
#     reconstructed_stats = statistics_mom(reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler)

#     # losses = [torch.sqrt((o - r).pow(2).sum(dim=-1)) / o.shape[-1] for o, r in zip(original_stats, reconstructed_stats)]
#     losses = [torch.sqrt(torch.mean((o - r)^2)) for o, r in zip(original_stats, reconstructed_stats)]
#     losses = torch.stack([l.mean() for l in losses])

#     return (losses * alpha.to(losses.device)).sum()

######## statistics loss OLD ######## ----------------------------------------------------------------------------

def correlation_coefficient_old(tensor1, tensor2):
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_std = torch.std(tensor1)
    tensor2_std = torch.std(tensor2)
    
    standardized_tensor1 = (tensor1 - tensor1_mean) / tensor1_std
    standardized_tensor2 = (tensor2 - tensor2_mean) / tensor2_std
    
    correlation = torch.mean(standardized_tensor1 * standardized_tensor2)
    
    return correlation

#Before using, make both and erb bank and a log bank:
#erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
#log_bank = fb.Logarithmic(new_size, sample_rate, 6, 10, new_sample_rate // 4)
def statistics_mcds_old(signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler):
    device = signal.device  # Get the device of the input signal tensor
    size = signal.shape[0]

    #low_lim = 20  # Low limit of filter
    #high_lim = sample_rate / 2  # Centre freq. of highest filter
    #
    ## Initialize filter bank
    #erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
    #
    ## Generate subbands for noise
    #erb_bank.generate_subbands(signal)
    # 
    ## Extract subbands
    #erb_subbands_signal = erb_bank.subbands[:, 1:-1]
    erb_subbands_signal = erb_bank.generate_subbands(signal)[1:-1, :].to(device)
    
    # Extract envelopes
    env_subbands = torch.abs(ddsp_textures.auxiliar.seeds.hilbert(erb_subbands_signal)).to(device)
    
    new_sample_rate = 11025
    
    # Downsampling before computing 
    envelopes_downsampled = []
    for i in range(N_filter_bank):
        envelope = env_subbands[i,:].float().to(device)  # Ensure the envelope is on the same device
        envelopes_downsampled.append(downsampler(envelope).to(torch.float64))

    subenvelopes = []
    new_size = envelopes_downsampled[0].shape[0]

    for i in range(N_filter_bank):
        signal = envelopes_downsampled[i].to(device)
        
        ## Initialize filter bank
        #log_bank = fb.Logarithmic(new_size, sample_rate, 6, 10, new_sample_rate // 4)
        #
        ## Generate subbands for noise
        #log_bank.generate_subbands(signal)
    
        # Extract subbands
        subenvelopes.append(log_bank.generate_subbands(signal)[1:-1, :].to(device))
    
    # Extract statistics up to order 4 and correlations
    statistics_1 = torch.zeros(N_filter_bank, 4, device=device)
    for i in range(N_filter_bank):
        mu = torch.mean(env_subbands[i])
        sigma = torch.sqrt(torch.mean((env_subbands[i] - mu) ** 2))
        statistics_1[i, 0] = mu * 1000
        statistics_1[i, 1] = sigma ** 2 / mu ** 2
        statistics_1[i, 2] = (torch.mean((env_subbands[i] - mu) ** 3) / sigma ** 3) / 100
        statistics_1[i, 3] = (torch.mean((env_subbands[i] - mu) ** 4) / sigma ** 4) / 1000

    statistics_2 = torch.zeros(N_filter_bank * (N_filter_bank - 1) // 2, device=device)
    index = 0
    for i in range(N_filter_bank):
        for j in range(i + 1, N_filter_bank):
            statistics_2[index] = correlation_coefficient_old(env_subbands[i], env_subbands[j])
            index += 1

    statistics_3 = torch.zeros(N_filter_bank * 6, device=device)
    for i in range(N_filter_bank):
        sigma_i = torch.std(envelopes_downsampled[i])
        for j in range(6):
            statistics_3[6 * i + j] = torch.std(subenvelopes[i][j]) / sigma_i

    statistics_4 = torch.zeros(15, N_filter_bank, device=device)
    for i in range(N_filter_bank):
        counter = 0
        for j in range(6):
            for k in range(j + 1, 6):
                statistics_4[counter, i] = correlation_coefficient_old(subenvelopes[i][j], subenvelopes[i][k])
                counter += 1

    statistics_5 = torch.zeros(6, N_filter_bank * (N_filter_bank - 1) // 2, device=device)
    for i in range(6):
        counter = 0
        for j in range(N_filter_bank):
            for k in range(j + 1, N_filter_bank):
                statistics_5[i, counter] = correlation_coefficient_old(subenvelopes[j][i], subenvelopes[k][i])
                counter += 1

    return [statistics_1, statistics_2, statistics_3, statistics_5, statistics_4]

def statistics_mcds_loss_old_single(original_signal, reconstructed_signal, erb_bank, log_bank, downsampler):
    original_statistics      = statistics_mcds_old(original_signal,      16, 44100, erb_bank, log_bank, downsampler)
    reconstructed_statistics = statistics_mcds_old(reconstructed_signal, 16, 44100, erb_bank, log_bank, downsampler)
    
    loss = []
    for i in range(5):
        loss_i = torch.sqrt(torch.mean((original_statistics[i] - reconstructed_statistics[i])**2))
        # print("Loss ", i, ": ", loss_i)
        loss.append(loss_i)

    final_loss = loss[0] + 20 * loss[1] + 20 * loss[2] + 20 * loss[3] + 20 * loss[4]

    return final_loss

# og_signal, reconstructed_signal, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler
def statistics_mcds_loss_old(original_signals, reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler):
    batch_size = original_signals.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        original_signal = original_signals[i]
        reconstructed_signal = reconstructed_signals[i]
        loss = statistics_mcds_loss_old_single(original_signal, reconstructed_signal, erb_bank, log_bank, downsampler)
        total_loss += loss

    average_loss = total_loss / batch_size
    return average_loss