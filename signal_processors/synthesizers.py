import torch
import numpy as np
import ddsp_textures.auxiliar.filterbanks
from   ddsp_textures.auxiliar.seeds import *

# SubEnv ----------------------------------------------------------------------------------------------

def SubEnv_param_extractor(signal, fs, N_filter_bank, param_per_env):
    # send error if param_per_env is not even
    if param_per_env % 2 != 0:
        raise ValueError("param_per_env must be an even number (cause these are complex numbers).")
    low_lim, high_lim = 20, fs / 2  # Low limit of filter
    size = signal.size(0)
    
    # Assuming fb.EqualRectangularBandwidth works with torch tensors
    erb_bank = ddsp_textures.auxiliar.filterbanks.EqualRectangularBandwidth(size, fs, N_filter_bank, low_lim, high_lim)
    subbands = erb_bank.generate_subbands(signal)  # generate subbands for signal y
    
    erb_subbands = subbands[1:-1, :].clone().to(dtype=torch.float32).detach()
    erb_envs = torch.abs(hilbert(erb_subbands))
    
    # print(f"The signal has a size of {size} samples.")
    # print(f"Each envelope of the {N_filter_bank} filters will be approximated by {int(percentage_use * size * 0.5)} complex parameters.")
    # print(f"This gives a total of {int(percentage_use * size * 0.5) * N_filter_bank} parameters in total, and it corresponds to {2 * int(percentage_use * size * 0.5) * N_filter_bank / size} of the entire signal size.")
    
    real_param = []
    imag_param = []
    
    for i in range(N_filter_bank):
        erb_env_local = erb_envs[i]
        fft_coeffs = torch.fft.rfft(erb_env_local)[:param_per_env//2] ###########################
        real_param.append(fft_coeffs.real)
        imag_param.append(fft_coeffs.imag)
    
    real_param = torch.cat(real_param)
    imag_param = torch.cat(imag_param)
    
    return real_param, imag_param

# Actual synth ----------------------------------------------------------------------------------------------

def SubEnv(parameters_real, parameters_imag, seed, target_loudness=1):
    size          = seed.shape[1]
    N_filter_bank = seed.shape[0]
    
    N = parameters_real.size(0)
    parameters_size = N // N_filter_bank
    signal_final = torch.zeros(size, dtype=torch.float32).to(device=parameters_real.device)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array
        parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
        
        # Initialize FFT coefficients array
        fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64).to(parameters_real.device)
        fftcoeff_local[:parameters_size] = parameters_local ###########################################
        
        # Compute the inverse FFT to get the local envelope
        env_local = torch.fft.irfft(fftcoeff_local)
        
        # Extract the local noise
        noise_local = seed[i, :]
        
        # Generate the texture sound by multiplying the envelope and noise
        texture_sound_local = env_local * noise_local
        
        # Accumulate the result
        signal_final += texture_sound_local
    
    loudness = torch.sqrt(torch.mean(signal_final ** 2))
    signal_final = signal_final / loudness

    signal_final = target_loudness * signal_final

    return signal_final

def SubEnv_batches(parameters_real, parameters_imag, seed):
    size          = seed.shape[1]
    N_filter_bank = seed.shape[0]
    
    # Get the batch size
    batch_size = parameters_real.size(0)
    parameters_size = parameters_real.size(1) // N_filter_bank

    # Initialize the final signal tensor for the entire batch
    signal_final = torch.zeros((batch_size, size), dtype=torch.float32, device=parameters_real.device)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array for each filter in the batch
        parameters_local = (parameters_real[:, i * parameters_size : (i + 1) * parameters_size] 
                            + 1j * parameters_imag[:, i * parameters_size : (i + 1) * parameters_size])
        
        # Initialize FFT coefficients array for the entire batch
        fftcoeff_local = torch.zeros((batch_size, int(size / 2) + 1), dtype=torch.complex64, device=parameters_real.device)
        fftcoeff_local[:, :parameters_size] = parameters_local

        # Compute the inverse FFT to get the local envelope for each batch item
        env_local = torch.fft.irfft(fftcoeff_local).real

        # Extract the local noise for each batch item
        noise_local = seed[i, :]

        # Generate the texture sound by multiplying the envelope and noise for each batch item
        texture_sound_local = env_local * noise_local

        # Accumulate the result for each batch item
        signal_final += texture_sound_local
    
    return signal_final

# Actual Synth Stems -------------------------------------------------------------------------------

def SubEnv_stems(parameters_real, parameters_imag, frame_size, N_filter_bank, target_loudness=1):
    size          = frame_size
    N_filter_bank = N_filter_bank
    
    N = parameters_real.size(0)
    parameters_size = N // N_filter_bank
    
    # Initialize a list to store env_locals
    env_locals_list = []
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array
        parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
        
        # Initialize FFT coefficients array
        fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64)
        fftcoeff_local[:parameters_size] = parameters_local
        
        # Compute the inverse FFT to get the local envelope
        env_local = torch.fft.irfft(fftcoeff_local)
        
        # Append the current env_local to the list
        env_locals_list.append(env_local)
    
    # Return the list of env_locals
    return env_locals_list

def SubEnv_stems_batches(parameters_real, parameters_imag, frame_size, N_filter_bank):
    size = frame_size
    batch_size = parameters_real.size(0)
    parameters_size = parameters_real.size(1) // N_filter_bank

    # Initialize a tensor to store the env_locals for each batch item and each filter
    # Shape will be [batch_size, N_filter_bank, size]
    env_locals_tensor = torch.zeros((batch_size, N_filter_bank, size), dtype=torch.float32)

    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array for each filter in the batch
        parameters_local = (parameters_real[:, i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[:, i * parameters_size : (i + 1) * parameters_size])
        
        # Initialize FFT coefficients array for the entire batch
        fftcoeff_local                      = torch.zeros((batch_size, int(size / 2) + 1), dtype=torch.complex64, device=parameters_real.device)
        fftcoeff_local[:, :parameters_size] = parameters_local

        # Compute the inverse FFT to get the local envelope for each batch item
        env_local = torch.fft.irfft(fftcoeff_local).real
        
        # Store the current env_local for each batch item in the tensor
        env_locals_tensor[:, i, :] = env_local

    # Return the tensor of env_locals with shape [batch_size, N_filter_bank, size]
    return env_locals_tensor

# Stems to signal -------------------------------------------------------------------------------

def SubEnv_stems_to_signal(env_locals, seed, target_loudness=1):
    size          = seed.shape[1]
    N_filter_bank = seed.shape[0]

    # Initialize the final signal
    signal_final = torch.zeros(size, dtype=torch.float32)

    # Iterate over the filter bank to reconstruct the signal
    for i in range(N_filter_bank):
        env_local = env_locals[i]  # Get the local envelope
        noise_local = seed[i]   # Get the corresponding noise

        # Reconstruct the signal by multiplying envelope with noise
        texture_sound_local = env_local * noise_local
        
        # Accumulate the result
        signal_final += texture_sound_local

    # Normalize the signal to match the target loudness
    loudness = torch.sqrt(torch.mean(signal_final ** 2))
    signal_final = signal_final / loudness
    signal_final = target_loudness * signal_final

    return signal_final

def SubEnv_stems_to_signals_batches(env_locals_batch, seed, target_loudness=1):
    size          = seed.shape[1]
    N_filter_bank = seed.shape[0]
    batch_size = len(env_locals_batch)  # The number of items in the batch

    device_seed = seed.device

    # Initialize the final signal tensor for the entire batch
    signal_final = torch.zeros((batch_size, size), dtype=torch.float32).to(device_seed)

    # Iterate over the batch items
    for batch_idx in range(batch_size):
        # Iterate over the filter bank to reconstruct each signal
        for i in range(N_filter_bank):
            env_local = env_locals_batch[batch_idx][i]  # Get the local envelope for this batch item
            noise_local = seed[i]                    # Get the corresponding noise

            # Reconstruct the signal by multiplying envelope with noise
            texture_sound_local = env_local * noise_local
            
            # Accumulate the result for this batch item
            signal_final[batch_idx] += texture_sound_local
        
        # Normalize the signal to match the target loudness
        loudness = torch.sqrt(torch.mean(signal_final[batch_idx] ** 2))
        signal_final[batch_idx] = signal_final[batch_idx] / loudness
        signal_final[batch_idx] = target_loudness * signal_final[batch_idx]

    return signal_final