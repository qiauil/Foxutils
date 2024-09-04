#usr/bin/python3

#version:0.0.2
#last modified:20231023

import numpy as np
import torch

def conv_outputsize(input_size,kernel,stride,pad):
    print(int((input_size+2*pad-kernel)/stride)+1)

def show_paras(model,print_result=True):
    nn_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in nn_parameters])
    # crucial parameter to keep in view: how many parameters do we have?
    if print_result:
        print("model has {} trainable params".format(params))
    return params

def get_para_vector(network) -> torch.Tensor:
    """
    Returns the parameter vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.

    Returns:
        torch.Tensor: The parameter vector of the network.
    """
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed=par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec
    
def apply_para_vector(network:torch.nn.Module,para_vec:torch.Tensor)->None:
    """
    Applies a parameter vector to the network's parameters.
    
    Args:
        network (torch.nn.Module): The network to apply the parameter vector to.
        para_vec (torch.Tensor): The parameter vector to apply.
    """
    with torch.no_grad():
        start=0
        for par in network.parameters():
            end=start+par.grad.data.view(-1).shape[0]
            par.data=para_vec[start:end].view(par.grad.data.shape)
            start=end
