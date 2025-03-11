from ddsp_textures.signal_processors.synthesizers import *
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import torchaudio
from   ddsp_textures.auxiliar.seeds import *

# All purpose functions ---------------------------------------------------------
def audio_improver(signal_tensor, sampling_rate, level):
    signal_filtered  = torchaudio.functional.bandpass_biquad(signal_tensor, sample_rate = sampling_rate, central_freq = 10000, Q = 0.1)
    freq_mean        = computer_freq_avg(signal_filtered, sampling_rate)
    # filtered audio is centered around the spectral centroid
    segment_centered = torchaudio.functional.bandpass_biquad(signal_filtered, sample_rate = sampling_rate, central_freq = freq_mean, Q = 1)
    # improved audio is the sum of the centered audio and the original audio
    segment_improved = level*segment_centered + signal_tensor
    return segment_improved

def signal_normalizer(signal):
    signal = (signal - torch.mean(signal))/torch.std(signal)
    return signal
    
# All purpose features computation ----------------------------------------------
def computer_freq_avg(signal_filtered, sampling_rate):
    # device = signal_filtered.device
    # window = torch.hann_window(signal_filtered.shape[-1]).to(device)
    # # windowing
    # signal_filtered = signal_filtered * window
    # # Compute the rfft of the signal
    # rfft = torch.fft.rfft(signal_filtered)
    # # Compute the magnitude of the rfft
    # magnitude_spectrum = torch.abs(rfft)
    # # Compute the frequency of each bin of the rfft
    # n = signal_filtered.shape[-1]
    # freqs = torch.fft.rfftfreq(n, d=1/sampling_rate).to(device)
    # # Compute the average frequency using the magnitude spectrum as weights
    # weighted_sum_freqs = torch.inner(freqs, magnitude_spectrum)
    # total_magnitude    = torch.sum(magnitude_spectrum)
    # mean_frequency     = weighted_sum_freqs / total_magnitude

    size = signal_filtered.shape[0]
    return torchaudio.functional.spectral_centroid(signal_filtered, sampling_rate, 0, torch.hamming_window(size).to(device="cuda"), size, size, size)[0]

def computer_freq_avg_and_std(signal_filtered, sampling_rate):
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

def compute_energy_bands(signal, erb_bank):
    device = signal.device  # Get the device of the input signal tensor
    erb_subbands_signal = erb_bank.generate_subbands(signal)[1:-1, :]
    N_filter_bank = erb_subbands_signal.shape[0]
    energy_bands  = torch.norm(erb_subbands_signal, dim=1)    
    return energy_bands

def compute_spectrogram(waveform, n_fft=1024, hop_length=512):
    device = waveform.device
    spectrogram = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        return_complex=True, 
        window=torch.hann_window(n_fft).to(device)
    )
    magnitude = torch.abs(spectrogram)
    return magnitude

def calculate_onset_strength(spectrogram):
    device = spectrogram.device
    # Calculate the difference between consecutive frames (along the time axis)
    diff_spectrogram = spectrogram[:, 1:] - spectrogram[:, :-1]
    # Apply ReLU to keep only positive differences
    onset_strength = torch.relu(diff_spectrogram)
    # Average across frequency bins (dim=0)
    onset_strength_mean = torch.mean(onset_strength, dim=0)
    # Prepend a zero at the start to match the original number of frames
    onset_strength_mean = torch.cat((torch.zeros(1).to(device), onset_strength_mean), dim=0)
    return onset_strength_mean

def compute_onset_peaks(onset_strength):
    device = onset_strength.device
    # Normalize onset strength
    onset_strength = (onset_strength - torch.min(onset_strength)) / (torch.max(onset_strength) - torch.min(onset_strength)) - 0.5
    # Apply sigmoid function
    onset = torch.sigmoid(onset_strength * 100)
    # Compute the derivative and apply ReLU
    onset_shifted = torch.roll(onset, 1)
    onset_derivative = onset - onset_shifted
    onset_peaks = torch.relu(onset_derivative)
    return onset_peaks

def computer_onset_count(signal_tensor, sampling_rate):
    spectrogram    = compute_spectrogram(signal_tensor)
    onset_strength = calculate_onset_strength(spectrogram)
    onset_peaks    = compute_onset_peaks(onset_strength)
    rate = torch.sum(onset_peaks)
    return rate

def computer_onset_comp(signal_tensor, sampling_rate):
    spectrogram    = compute_spectrogram(signal_tensor)
    onset_strength = calculate_onset_strength(spectrogram)
    onset_peaks    = compute_onset_peaks(onset_strength)
    rate = torch.sum(onset_peaks)
    return rate

def computer_envelopes_stems(signal_tensor, sampling_rate, erb_bank):
    erb_subbands_signal = erb_bank.generate_subbands(signal_tensor)[1:-1, :]
    env_subbands = torch.abs(ddsp_textures.auxiliar.seeds.hilbert(erb_subbands_signal))
    return env_subbands

# Features annotators --------------------------------------------------------
def features_freqavg(signal_improved, sampling_rate, _):
    normalization=False # amazing programming skills
    freq_avg = computer_freq_avg(signal_improved, sampling_rate)
    if normalization:
        max_freq = torch.tensor(sampling_rate / 2)
        mean_frequency = torch.log2(freq_avg) / torch.log2(max_freq)
    return freq_avg # CAREFULLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL

def features_freqavg_freqstd(signal_improved, sampling_rate, _):
    normalization=True # amazing programming skills
    mean_frequency, std_freqs = computer_freq_avg_and_std(signal_improved, sampling_rate)
    if normalization:
        max_freq       = torch.tensor(sampling_rate / 2)
        mean_frequency = torch.log2(mean_frequency) / torch.log2(max_freq)
        std_freqs      = torch.log2(std_freqs)      / torch.log2(max_freq)
    return torch.tensor([mean_frequency, std_freqs])

def features_rate(signal_improved, sampling_rate, _):
    normalization=True # amazing programming skills
    rate     = computer_onset_count(signal_improved, sampling_rate)
    if normalization:
        time_length_signal = signal_improved.shape[-1] / sampling_rate # in seconds
        max_rate_per_second = 5 # 5 onsets per second
        max_rate = max_rate_per_second * time_length_signal # max_rate_per_second onsets in the whole signal
        rate = rate / max_rate # normalization
    return rate

def features_energy_bands(signal, _, erb_bank):
    return compute_energy_bands(signal, erb_bank) # amazing programming skills

def features_envelopes_stems(signal_tensor, _, erb_bank):
    return computer_envelopes_stems(signal_tensor, _, erb_bank) # incredible programming skills

# Features anotators in batches (FOR REGULARIZATION) ----------------------------------------------
def batch_features_freqavg(signals_batch, sampling_rate, _):
    batch_size = signals_batch.shape[0]
    features = []
    for i in range(batch_size):
        signal = signals_batch[i]
        features.append(features_freqavg(signal, sampling_rate, _))
    return torch.stack(features)

def batch_features_freqavg_freqstd(signals_batch, sampling_rate, _):
    batch_size = signals_batch.shape[0]
    features = []
    for i in range(batch_size):
        signal = signals_batch[i]
        features.append(features_freqavg_freqstd(signal, sampling_rate, _))
    return torch.stack(features)

def batch_features_rate(signals_batch, sampling_rate, _):
    batch_size = signals_batch.shape[0]
    features = []
    for i in range(batch_size):
        signal = signals_batch[i]
        features.append(features_rate(signal, sampling_rate, _))
    return torch.stack(features)

def batch_features_energy_bands(signals_batch, _, erb_bank):
    batch_size = signals_batch.shape[0]
    features = []
    for i in range(batch_size):
        signal = signals_batch[i]
        features.append(features_energy_bands(signal, _, erb_bank))
    return torch.stack(features)

def batch_features_envelopes_stems(signals_batch, _, erb_bank):
    batch_size = signals_batch.shape[0]
    features = []
    for i in range(batch_size):
        signal = signals_batch[i]
        features.append(features_envelopes_stems(signal, _, erb_bank))
    return torch.stack(features)