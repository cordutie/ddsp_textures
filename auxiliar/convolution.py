import torch
import torch.nn.functional as F
import ddsp_textures.auxiliar.time_stamps as ts

def convolution_step(time_stamps, atoms, K):
    #time stamps size
    N = time_stamps.size()[0]
    #time stamps patch size
    n = N // K
    #atom size
    M = atoms.size()[0] // K
    #set result
    result = torch.zeros(N+M-1)
    for i in range(K):
        atom_local       = atoms[i*M:(i+1)*M].view(1, 1, -1)
        # print("atom local size: ", atom_local.size())
        time_stamp_local = time_stamps[i*n:(i+1)*n].view(1, 1, -1)
        # print("time stamp local size: ", time_stamp_local.size())
        # Perform the full convolution
        # padding = min(M, n)-1
        padding = M - 1
        synthesis_local = F.conv1d(time_stamp_local, atom_local, padding=padding)
        # synthesis_local  = torch.convolution(time_stamp_local, atom_local) #size = n+M-1
        # print("synthesis local size: ", synthesis_local.size())
        #squeeze
        synthesis_local = synthesis_local.squeeze()
        #overlap and add using hop size = n
        result[i*n:(i+1)*n+M-1] += synthesis_local
    result_reversed = result.flip(dims=[0])
    return result_reversed

def convolution_step_batches(time_stamps, atoms, K):
    #compute batch size
    batch_size = time_stamps.size()[0]
    #time stamps size
    N = time_stamps.size()[1]
    #time stamps patch size
    n = N // K
    #atom size
    M = atoms.size()[1] // K
    #set result
    result = torch.zeros(batch_size, N+M-1)
    for i in range(batch_size):
        result[i] = convolution_step(time_stamps[i], atoms[i], K)
    return result