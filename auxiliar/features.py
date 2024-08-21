from signal_processors.synthesizers import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import torchaudio

# All purpose features computation ----------------------------------------------
def computer_freq_avg(signal_filtered, sampling_rate):
    device = signal_filtered.device
    window = torch.hann_window(signal_filtered.shape[-1]).to(device)
    
    # windowing
    signal_filtered = signal_filtered * window

    # Compute the rfft of the signal
    rfft = torch.fft.rfft(signal_filtered)
    
    # Compute the magnitude of the rfft
    magnitude_spectrum = torch.abs(rfft)
    
    # Compute the frequency of each bin of the rfft
    n = signal_filtered.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/sampling_rate)
    
    # Compute the average frequency using the magnitude spectrum as weights
    weighted_sum_freqs = torch.inner(freqs, magnitude_spectrum)
    total_magnitude    = torch.sum(magnitude_spectrum)
    mean_frequency     = weighted_sum_freqs / total_magnitude
    
    return mean_frequency

def computer_freq_avg_and_std(signal_filtered, sampling_rate):# Compute the rfft of the signal
    device = signal_filtered.device
    window = torch.hann_window(signal_filtered.shape[-1]).to(device)
    
    # windowing
    signal_filtered = signal_filtered * window

    # Compute the rfft of the signal
    rfft = torch.fft.rfft(signal_filtered)
    
    # Compute the magnitude of the rfft
    magnitude_spectrum = torch.abs(rfft)
    
    # Compute the frequency of each bin of the rfft
    n = signal_filtered.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/sampling_rate).to(device)
    
    # Compute the average frequency using the magnitude spectrum as weights
    weighted_sum_freqs = torch.inner(freqs, magnitude_spectrum)
    total_magnitude    = torch.sum(magnitude_spectrum)
    mean_frequency     = weighted_sum_freqs / total_magnitude
    
    # Compute the variance of the frequency using the magnitude spectrum as weights
    variance_freqs = torch.sum(magnitude_spectrum * (freqs - mean_frequency) ** 2) / total_magnitude
    std_freqs      = torch.sqrt(variance_freqs)
    
    return mean_frequency, std_freqs

def compute_spectrogram(waveform, n_fft=1024, hop_length=512):
    spectrogram = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=torch.hann_window(n_fft))
    magnitude = torch.abs(spectrogram)
    return magnitude

def calculate_onset_strength(spectrogram):
    # Calculate the difference between consecutive frames (along the time axis)
    diff_spectrogram = spectrogram[:, 1:] - spectrogram[:, :-1]
    
    # Apply ReLU to keep only positive differences
    onset_strength = torch.relu(diff_spectrogram)
    
    # Average across frequency bins (dim=0)
    onset_strength_mean = torch.mean(onset_strength, dim=0)
    
    # Prepend a zero at the start to match the original number of frames
    onset_strength_mean = torch.cat((torch.zeros(1), onset_strength_mean), dim=0)
    
    return onset_strength_mean

def compute_onset_peaks(onset_strength):
    # Normalize onset strength
    onset_strength = (onset_strength - torch.min(onset_strength)) / (torch.max(onset_strength) - torch.min(onset_strength)) - 0.5
    
    # Apply sigmoid function
    onset = torch.sigmoid(onset_strength * 100)
    
    # Compute the derivative and apply ReLU
    onset_shifted = torch.roll(onset, 1)
    onset_derivative = onset - onset_shifted
    onset_peaks = torch.relu(onset_derivative)
    
    return onset_peaks

# So far, this assumes sampling_rate = 44100
def computer_rate(signal_tensor, sampling_rate):
    spectrogram = compute_spectrogram(signal_tensor)
    onset_strength = calculate_onset_strength(spectrogram)
    onset_peaks = compute_onset_peaks(onset_strength)
    rate = torch.sum(onset_peaks)
    return rate

# Features annotators --------------------------------------------------------
def features_freqavg_freqstd(signal_improved, sampling_rate, normalization=True):
    mean_frequency, std_freqs = computer_freq_avg_and_std(signal_improved, sampling_rate)
    if normalization:
        max_freq = torch.tensor(sampling_rate / 2)
        mean_frequency = torch.log2(mean_frequency) / torch.log2(max_freq)
        std_freqs      = torch.log2(std_freqs)      / torch.log2(max_freq)
    return [mean_frequency, std_freqs]

def features_freqavg_rate(signal_improved, sampling_rate, normalization=True):
    freq_avg = computer_freq_avg(signal_improved, sampling_rate)
    rate     = computer_rate(signal_improved, sampling_rate)
    if normalization:
        max_freq = torch.tensor(sampling_rate / 2)
        freq_avg = torch.log2(freq_avg) / torch.log2(max_freq)
        time_length_signal = signal_improved.shape[-1] / sampling_rate # in seconds
        max_rate_per_second = 5 # 5 onsets per second
        max_rate = max_rate_per_second * time_length_signal # max_rate_per_second onsets in the whole signal
        rate = rate / max_rate # normalization
    return [freq_avg, rate]

# Features anotators in batches ----------------------------------------------
def batch_features_freqavg_freqstd(signal_improved_batch, sampling_rate, normalization=True):
    batch_size = signal_improved_batch.shape[0]
    mean_freqs = []
    std_freqs = []
    for i in range(batch_size):
        signal_filtered = signal_improved_batch[i]
        mean_freq, std_freq = features_freqavg_freqstd(signal_filtered, sampling_rate, normalization)
        mean_freqs.append(mean_freq)
        std_freqs.append(std_freq)
    mean_freqs = torch.stack(mean_freqs)
    std_freqs = torch.stack(std_freqs)
    return torch.stack((mean_freqs, std_freqs), dim=1)

def batch_features_freqavg_rate(signal_improved_batch, sampling_rate, normalization=True):
    batch_size = signal_improved_batch.shape[0]
    freq_avgs = []
    rates = []
    for i in range(batch_size):
        signal_filtered = signal_improved_batch[i]
        freq_avg, rate  = features_freqavg_rate(signal_filtered, sampling_rate, normalization)
        freq_avgs.append(freq_avg)
        rates.append(rate)
    freq_avgs = torch.stack(freq_avgs)
    rates = torch.stack(rates)
    return torch.stack((freq_avgs, rates), dim=1)

