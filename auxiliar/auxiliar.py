import torch
import torchaudio
import numpy as np
import auxiliar.filterbanks as fb

def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform on PyTorch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Default: `x.shape[axis]`
    axis : int, optional
        Axis along which to do the transformation. Default: -1.

    Returns
    -------
    xa : torch.Tensor
        Analytic signal of `x`, of each 1-D array along `axis`
    """
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = torch.fft.fft(x, n=N, dim=axis)
    h = torch.zeros(N, dtype=Xf.dtype, device=x.device)

    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [None] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]

    x_analytic = torch.fft.ifft(Xf * h, dim=axis)
    return x_analytic

def env_inverter(array):
    """ Takes the envelope of a signal and computes the reciprocal, being careful with 1/0. """
    final = torch.zeros_like(array)
    non_zero_mask = array != 0
    final[non_zero_mask] = 1 / array[non_zero_mask]
    return final

def env_normalizer(signal):
    """ Takes a signal, normalizes its envelope, and applies the reciprocal envelope, normalizing it. """
    envelope = torch.abs(hilbert(signal))
    envelope_inverted = env_inverter(envelope)
    normalization = signal * envelope_inverted
    return normalization

def seed_maker(size, fs, N_filter_bank):
    low_lim = 20  # Low limit of filter
    high_lim = fs / 2  # Centre freq. of highest filter
    
    # Initialize filter bank
    erb_bank = fb.EqualRectangularBandwidth(size, fs, N_filter_bank, low_lim, high_lim)
    
    # Generate noise using PyTorch
    noise = torch.randn(size)
    
    # Generate subbands for noise
    erb_bank.generate_subbands(noise)
    
    # Extract subbands
    erb_subbands_noise = erb_bank.subbands[:, 1:-1]
    
    for i in range(N_filter_bank):
        noise_local = erb_subbands_noise[:, i]
        noise_normalized = env_normalizer(noise_local)
        erb_subbands_noise[:, i] = noise_normalized
    
    return erb_subbands_noise

#def plotter(y, fs):
    ## Detach the tensor from the computation graph and convert to NumPy array
    #y_np = y.detach().cpu().numpy()
    #
    ## Plot the waveform
    #plt.plot(y_np)
    #plt.show()
    #
    ## Display the audio
    #ipd.display(ipd.Audio(data=y_np, rate=fs))
