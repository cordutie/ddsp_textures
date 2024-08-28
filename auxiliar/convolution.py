import torch
import torch.nn.functional as F
import ddsp_textures.auxiliar.time_stamps as ts

def convolution_step(time_stamps, atoms, K):
    # Ensure all tensors are on the same device
    device = time_stamps.device
    atoms = atoms.to(device)

    # time stamps size
    N = time_stamps.size(0)
    # time stamps patch size
    n = N // K
    # atom size
    M = atoms.size(0) // K
    # set result
    result = torch.zeros(N + M - 1, device=device)
    
    for i in range(K):
        atom_local = atoms[i * M:(i + 1) * M].view(1, 1, -1)
        time_stamp_local = time_stamps[i * n:(i + 1) * n].view(1, 1, -1)
        # Perform the full convolution
        padding = M - 1
        synthesis_local = F.conv1d(time_stamp_local, atom_local, padding=padding)
        # squeeze
        synthesis_local = synthesis_local.squeeze()
        # overlap and add using hop size = n
        result[i * n:(i + 1) * n + M - 1] += synthesis_local
    
    result_reversed = result.flip(dims=[0])
    final_result = result_reversed[:N]
    
    return final_result


import torch

def convolution_step_batches(time_stamps, atoms, K):
    # Ensure all tensors are on the same device
    device = time_stamps.device
    
    # Move tensors to the correct device
    atoms = atoms.to(device)
    
    # Compute batch size
    batch_size = time_stamps.size(0)
    # Time stamps size
    N = time_stamps.size(1)
    # Time stamps patch size
    n = N // K
    # Atom size
    M = atoms.size(1) // K
    # Set result
    result = torch.zeros(batch_size, N, device=device)
    
    for i in range(batch_size):
        result[i] = convolution_step(time_stamps[i].to(device), atoms[i].to(device), K)
    
    return result
