#usr/bin/python3

#version:0.0.1
#last modified:20230803

import torch
def get_relative_error(a,b):
    with torch.no_grad():
        ori_shape=a.shape
        a=a.reshape(-1)
        b=b.reshape(-1)
        relative_error=torch.zeros(1,device=a.device)
        relative_error_tensor=torch.zeros(b.shape,device=a.device)
        e_0=0
        for i,ai in enumerate(a):
            if ai.item()!=0:
                bi=b[i]
                error=torch.abs((bi-ai)/ai)
                relative_error += error
                relative_error_tensor[i]=error
            else:
                e_0+=1
        mean_relative_error=relative_error/(a.shape[0]-e_0)
        return mean_relative_error.item(),relative_error_tensor.reshape(ori_shape)

def redscale01(ori_tenser):
    return (ori_tenser-torch.min(ori_tenser))/(torch.max(ori_tenser)-torch.min(ori_tenser))
