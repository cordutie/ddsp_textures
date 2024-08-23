import torch
import numpy as np
import ddsp_textures.auxiliar.filterbanks
from ddsp_textures.auxiliar.seeds import *

# SubEnv ----------------------------------------------------------------------------------------------

def SubEnv_param_extractor(signal, fs, N_filter_bank, param_per_env):
    # send error if param_per_env is not even
    if param_per_env % 2 != 0:
        raise ValueError("param_per_env must be an even number (cause tese are complex numbers).")
    low_lim, high_lim = 20, fs / 2  # Low limit of filter
    size = signal.size(0)
    
    # Assuming fb.EqualRectangularBandwidth works with torch tensors
    erb_bank = ddsp_textures.auxiliar.filterbanks.EqualRectangularBandwidth(size, fs, N_filter_bank, low_lim, high_lim)
    subbands = erb_bank.generate_subbands(signal)  # generate subbands for signal y
    
    erb_subbands = subbands[:, 1:-1].clone().to(dtype=torch.float32).detach()
    erb_envs = torch.abs(hilbert(erb_subbands.transpose(0, 1)).transpose(0, 1))
    
    # print(f"The signal has a size of {size} samples.")
    # print(f"Each envelope of the {N_filter_bank} filters will be approximated by {int(percentage_use * size * 0.5)} complex parameters.")
    # print(f"This gives a total of {int(percentage_use * size * 0.5) * N_filter_bank} parameters in total, and it corresponds to {2 * int(percentage_use * size * 0.5) * N_filter_bank / size} of the entire signal size.")
    
    real_param = []
    imag_param = []
    
    for i in range(N_filter_bank):
        erb_env_local = erb_envs[:, i]
        fft_coeffs = torch.fft.rfft(erb_env_local)[:param_per_env//2] ###########################
        real_param.append(fft_coeffs.real)
        imag_param.append(fft_coeffs.imag)
    
    real_param = torch.cat(real_param)
    imag_param = torch.cat(imag_param)
    
    return real_param, imag_param


def SubEnv(parameters_real, parameters_imag, seed, target_loudness=1):
    size          = seed.shape[0]
    N_filter_bank = seed.shape[1]
    
    N = parameters_real.size(0)
    parameters_size = N // N_filter_bank
    signal_final = torch.zeros(size, dtype=torch.float32)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array
        parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
        
        # Initialize FFT coefficients array
        fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64)
        fftcoeff_local[:parameters_size] = parameters_local ###########################################3
        
        # Compute the inverse FFT to get the local envelope
        env_local = torch.fft.irfft(fftcoeff_local)
        
        # Extract the local noise
        noise_local = seed[:, i]
        
        # Generate the texture sound by multiplying the envelope and noise
        texture_sound_local = env_local * noise_local
        
        # Accumulate the result
        signal_final += texture_sound_local
    
    loudness = torch.sqrt(torch.mean(signal_final ** 2))
    signal_final = signal_final / loudness

    signal_final = target_loudness * signal_final

    return signal_final

def SubEnv_batches(parameters_real, parameters_imag, seed):
    size          = seed.shape[0]
    N_filter_bank = seed.shape[1]
    
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
        noise_local = seed[:, i]

        # Generate the texture sound by multiplying the envelope and noise for each batch item
        texture_sound_local = env_local * noise_local

        # Accumulate the result for each batch item
        signal_final += texture_sound_local
    
    return signal_final

# P-VAE ----------------------------------------------------------------------------------------------

import torch.nn.functional as F
import ddsp_textures.auxiliar.time_stamps as ts
from ddsp_textures.auxiliar.convolution import *

# REQUIREMENTS
# VAE = object that has two methods
# VAE.generate generates a tensor of size atoms_size from a tensor of size latent_dim
# VAE.generate_batches generates a tensor of size batch_size x atoms_size from a tensor of size batch_size x latent_dim
def P_VAE(time_stamps_size, lambda_rate, alpha, sr, VAE, latent_dim, atoms_size, encoded_atoms, atoms_new_size_factor = 1):
    #time stamps generation
    time_stamps = ts.time_stamps_generator(time_stamps_size, sr, lambda_rate, alpha)
    #number of atoms = size of encoded atoms/latent dim
    K = encoded_atoms.size()[0] // latent_dim
    # print("Number of atoms: ", K)
    #create tensor 1d with all the atom
    atoms_new_size_factor = min(1, atoms_new_size_factor)
    new_atoms_size = int(atoms_size * atoms_new_size_factor)
    atoms = torch.zeros(K*new_atoms_size)
    for i in range(K):
        atom_local = VAE.generate(encoded_atoms[i*latent_dim:(i+1)*latent_dim])
        atom_local = atom_local[:new_atoms_size]
        # print("atom local size: ", atom_local.size())
        atoms[i*new_atoms_size:(i+1)*new_atoms_size] = atom_local
    #convolution step
    # print("convolution_step")
    result = convolution_step(time_stamps, atoms, K)
    return result

def P_VAE_batches(time_stamps_size, lambda_rate, alpha, sr, VAE, latent_dim, atoms_size, encoded_atoms, atoms_new_size_factor = 1):
    # lambda is a number but comes ina btach. Compute the size of the batch
    batch_size = lambda_rate.size()[0]
    # print(batch_size)
    
    # print("let's compute the time stamps")
    #make a batch of time stamps using the batch of lambdas and alphas
    time_stamps_batch = torch.zeros(batch_size, time_stamps_size)
    for i in range(batch_size):
        time_stamps_batch[i] = ts.time_stamps_generator(time_stamps_size, sr, lambda_rate[i], alpha[i])
        # print("time stamps number ", i, " computed")
    
    # number of atoms = size of encoded atoms/latent dim but atoms come in batches
    K = encoded_atoms.size()[1] // latent_dim
    # print("Number of atoms: ", K)
    
    #create tensor 1d with all the atom
    atoms_new_size_factor = min(1, atoms_new_size_factor)
    # print("old atoms size: ", atoms_size)
    new_atoms_size = int(atoms_size * atoms_new_size_factor)
    # print("new atoms size: ", new_atoms_size)
    atoms_batch = torch.zeros(batch_size, K*new_atoms_size)
    
    for i in range(K):
        atom_local = VAE.generate_batches(encoded_atoms[:,i*latent_dim:(i+1)*latent_dim])
        atom_local = atom_local[:, :new_atoms_size]
        # print("atom local size: ", atom_local.size())
        # print("atom local size: ", atom_local.size())
        atoms_batch[:, i*new_atoms_size:(i+1)*new_atoms_size] = atom_local
    #convolution step
    # print("\nCONVOLUTION STEP\n")
    result = convolution_step_batches(time_stamps_batch, atoms_batch, K)
    return result