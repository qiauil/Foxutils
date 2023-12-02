#usr/bin/python3

#version:0.0.1
#last modified:20230803

from . import *
import torch
from matplotlib import animation

def save_animations_from_tensor(tensors,save_path,Transpose=True,inverse_y=True,cmap='viridis',interval=50,timefunc=None):
    vmin=torch.min(tensors)
    vmax=torch.max(tensors)
    if Transpose:
        tensors=tensors.transpose(1,2)
    fig, ax = plt.subplots()
    def animate(i):
        ax.cla()
        ax.set_axis_off()
        frame=ax.imshow(tensors[i],cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
        if timefunc is not None:
            ax.set_title("$t=${}".format(timefunc(i)))
        if inverse_y:
            ax.invert_yaxis()
        return frame,

    animation1 = animation.FuncAnimation(fig=fig, func=animate, frames=tensors.shape[0], interval=interval, blit=False)
    animation1.save(save_path, writer='imagemagick')
