import torch

# Gamma family of functions
def gamma(x, alpha, p):
    return (torch.relu(x) / p) ** alpha

# Time stamps generator
def time_stamps_generator(size, sr, rate_lambda, alpha):
    p = rate_lambda / sr  # probability of event in one bin
    U = torch.rand(size)  # Tensor of shape `size`
    E = gamma(U - 1 + p, alpha, p)  # Apply gamma function element-wise
    return E