#usr/bin/python3

#version:0.0.11
#last modified:20230804

from . import *
from .style import *
import collections
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from ..helper.coding import *

def sym_colormap(d_min,d_max,d_cen=0,cmap="coolwarm",cmapname="sym_map"):
    if abs(d_max-d_cen)>abs(d_min-d_cen):
        max_v=1
        low_v=0.5-(d_cen-d_min)/(d_max-d_cen)*0.5
    else:
        low_v=0
        max_v=0.5+(d_max-d_cen)/(d_cen-d_min)*0.5
    if isinstance(cmap,str):
        cmap=mlp.cm.get_cmap(cmap)
    return colors.LinearSegmentedColormap.from_list(cmapname,cmap(np.linspace(low_v, max_v, 100)))

def plot3D(z,ztitle="z",xtitle="x",ytitle="y",cmap='viridis',plot2D=False,xlist=None,ylist=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(6, 6))
    delta=1
    if xlist is None:
        xlen=z.shape[0]
        x = np.arange(0, xlen, delta)
    else:
        x=xlist
    if ylist is None:
        ylen=z.shape[1]
        y = np.arange(0, ylen, delta)
    else:
        y=ylist
    Z = z.T
    X, Y = np.meshgrid(x, y)
    surf=ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False,cmap=plt.get_cmap(cmap))  # 设置颜色映射
    plt.xlabel(xtitle,fontsize=12)
    plt.ylabel(ytitle,fontsize=12)
    ax.set_zlabel(ztitle,fontsize=12)
    ax.set_zlim(torch.min(Z).item(), torch.max(Z).item())
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    if plot2D:
        fig, ax = plt.subplots()
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_xticklabels(xlist)
        im=ax.imshow(Z,cmap=plt.get_cmap(cmap))
        plt.colorbar(im,ax=ax)


def plot2D(field,xtitle="j",ytitle="i",cmap='viridis',xlist=None,ylist=None,inverse_y=True):
    delta=1
    if xlist is None:
        xlen=field.shape[0]
        xlabels = np.arange(0, xlen, delta)
    else:
        xlabels=xlist
    if ylist is None:
        ylen=field.shape[1]
        ylabels = np.arange(0, ylen, delta)
    else:
        ylabels=ylist

    fig, ax = plt.subplots()
    im = ax.imshow(field,cmap=plt.get_cmap(cmap))
    fig.colorbar(im,ax=ax)
    ax.set_xticks(ticks=np.arange(xlen),labels=xlabels)
    if inverse_y:
        ax.invert_yaxis()
        ylabels=list(ylabels)
        ylabels.reverse()
    ax.set_yticks(ticks=np.arange(ylen),labels=ylabels)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    
def plot2DX(field,xtitle="j",ytitle="i",cmap='viridis',xlist=None,ylist=None,vmin=None,vmax=None,colorbar=True):
    delta=1
    if xlist is None:
        xlen=field.shape[0]
        x = np.arange(0, xlen, delta)
    else:
        x=xlist
    if ylist is None:
        ylen=field.shape[1]
        y = np.arange(0, ylen, delta)
    else:
        y=ylist
    deltax=(x[1]-x[0])/2
    deltay=(y[1]-y[0])/2
    x=np.array([i+deltax for i in x])
    y=np.array([i+deltay for i in y])
    x=np.insert(x,0,x[0]-deltax*2)
    y=np.insert(y,0,y[0]-deltay*2)

    fig, ax = plt.subplots()
    pcm=ax.pcolormesh(x, y, field,cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    if colorbar:
        fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    fig.set_figheight(len(y)*0.3)
    fig.set_figwidth(len(x)*0.3)

def show_image_from_tensor(image_tensor,title=""):
    if len(image_tensor.shape)>3:
        if image_tensor.shape[0]==1:
            image_tensor=image_tensor.squeeze(dim=0)
        else:
            print("wrong input type")
    n_channels = image_tensor.shape[-3]
    for c in range(n_channels):
        maxv=torch.max(image_tensor[c])
        minv=torch.min(image_tensor[c])
        image_tensor[c] = (image_tensor[c]-minv)/(maxv-minv)
    plt.title(title,y=-0.1)
    plt.imshow(image_tensor.permute(1,2,0))
    plt.axis('off')

class ChannelPloter():
    
    def __init__(self) -> None:
        self.__fig_save_path="./output_figs/"

    def __type_transform(self,fields):
        if isinstance(fields,collections.Sequence):
            if isinstance(fields[0],torch.Tensor):
                fields=[(field.to(torch.device("cpu"))).numpy() for field in fields]
                return fields
            elif isinstance(fields[0],np.ndarray):
                return fields
            else:
                raise Exception("Wrong input type!")
        else:
            if isinstance(fields,torch.Tensor):
                fields=(fields.to(torch.device("cpu"))).numpy()
                return fields
            elif isinstance(fields,np.ndarray):
                return fields
            else:
                raise Exception("Wrong input type!")       
        
    def __cat_fields(self,fields):
        if isinstance(fields,collections.Sequence):
            if len(fields[0].shape)==4:
                return np.concatenate(fields,0)
            elif len(fields[0].shape) == 3:
                return np.concatenate([np.expand_dims(field,0) for field in fields],0)
            elif len(fields[0].shape) == 2:
                return np.concatenate([np.expand_dims(np.expand_dims(field,0),0) for field in fields],0)
            else:
                raise Exception("Wrong input type!")
        else:
            if len(fields.shape) ==2:
                return np.expand_dims(np.expand_dims(fields,0),0)
            if len(fields.shape)==3:
                return np.expand_dims(fields,0)
            elif len(fields.shape)==4:
                return fields
            else:
                raise Exception("Wrong input type!")

    def __find_min_max(self,fields,defaultmin,defaultmax):
        mins=[]
        maxs=[]
        for i in range(fields.shape[1]):
            if defaultmin is not None:
                if defaultmin[i] is not None:
                    mins.append(defaultmin[i])
                else:
                    mins.append(np.min(fields[:,i,:,:]))
            else:
                mins.append(np.min(fields[:,i,:,:]))
            if defaultmax is not None:
                if defaultmax[i] is not None:
                    maxs.append(defaultmax[i])
                else:
                    maxs.append(np.max(fields[:,i,:,:]))
            else:
                maxs.append(np.max(fields[:,i,:,:]))
        return mins,maxs
   
    def __generate_mask(self,mask,transpose,color="white"):
        mask=self.__type_transform(mask)
        if color=="white":
            RGB=np.ones(mask.shape) #zeros=Black, ones=white
        elif color=="black":   
            RGB=np.zeros(mask.shape) 
        if transpose:
            return torch.cat([np.expand_dims(RGB,2),np.expand_dims(RGB,2),np.expand_dims(RGB,2),np.expand_dims(mask.T,2)],-1)
        else:
            return torch.cat([np.expand_dims(RGB,2),np.expand_dims(RGB,2),np.expand_dims(RGB,2),np.expand_dims(mask,2)],-1)

    def fig_save_path (self,path):
        self.__fig_save_path=path  
        
    def plot(self,
            fields,
            channel_names=None,channel_units=None,case_names=None,title="",
            transpose=False,inverse_y=False,
            cmap=CMAP_COOLHOT,
            mask=None,
            size_subfig=3.5,xspace=0.7,yspace=0.1,cbar_pad=0.1,
            title_position=0,
            redraw_ticks=True,num_colorbar_value=4,minvs=None,maxvs=None,tick_format=None,
            data_scale=None,
            rotate_colorbar_with_oneinput=False,
            subfigure_index=None,
            save_name=None,
            use_sym_colormap=True):
        
        fields=self.__cat_fields(self.__type_transform(fields))
        if mask is not None:
            mask=self.__generate_mask(mask,transpose=transpose)
        num_cases=fields.shape[0]
        num_channels=fields.shape[1]
        
        channel_names=default(channel_names,["channel {}".format(i) for i in range(num_channels)])
        channel_units=default(channel_units,["" for i in range(num_channels)])
        case_names=default(case_names,["case {}".format(i) for i in range(num_cases)])
        data_scale=default(data_scale,[1 for i in range(num_channels)])
        fields=np.concatenate([fields[:,i:i+1,:,:]*data_scale[i] for i in range(num_channels)],1)
        mins,maxs=self.__find_min_max(fields,minvs,maxvs)
        
        if num_cases ==1 and rotate_colorbar_with_oneinput:
            cbar_location="right"
            cbar_mode='each'
            ticklocation="right"
        else:
            cbar_location="top"
            cbar_mode='edge'
            ticklocation="top" 
        fig=plt.figure(figsize=(size_subfig*num_channels,size_subfig*num_cases))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(num_cases,num_channels),
                        axes_pad=(xspace,yspace),
                        share_all=True,
                        cbar_location=cbar_location,
                        cbar_mode=cbar_mode,
                        direction = 'row',
                        #cbar_size="10%",
                        cbar_pad=cbar_pad
                        )
        im_cb=[]
        if use_sym_colormap:
            colormaps=[]
            for i in range(num_channels):
                colormaps.append(sym_colormap(mins[i],maxs[i],cmap=cmap))
                
        for i,axis in enumerate(grid):
            i_row=i//num_channels
            i_column=i%num_channels
            datai=fields[i_row,i_column]
            if transpose:
                datai=datai.T
            if use_sym_colormap:
                im=axis.imshow(datai,colormaps[i_column],vmin=mins[i_column],vmax=maxs[i_column])
            else:
                im=axis.imshow(datai,cmap,vmin=mins[i_column],vmax=maxs[i_column])
            if i < num_channels:
                im_cb.append(im)
                    
            if mask is not None:
                axis.imshow(mask)  
            if inverse_y:
                axis.invert_yaxis()      
            axis.set_yticks([])
            axis.set_xticks([])
            if i_column ==0:
                axis.set_ylabel(case_names[i_row])
            if i_row == num_cases-1:
                axis.set_xlabel(channel_names[i_column])   

        for i in range(num_channels):
            cb=grid.cbar_axes[i].colorbar(im_cb[i],label=channel_units[i],ticklocation=ticklocation,format=tick_format)
            cb.ax.minorticks_on()
            if redraw_ticks:
                cb.set_ticks(np.linspace(mins[i],maxs[i],num_colorbar_value,endpoint=True))      
        fig.suptitle(title,y=title_position)
        if subfigure_index is not None:
            plt.suptitle(subfigure_index,x=0.01,y=0.88,fontproperties="Times New Roman")
        if save_name is not None:
            plt.savefig(self.__fig_save_path+save_name+".svg",bbox_inches = 'tight')
        plt.show()

field_plotter=ChannelPloter()

def show_each_channel(
            fields,
            channel_names=None,channel_units=None,case_names=None,title="",
            transpose=False,inverse_y=False,
            cmap=CMAP_COOLHOT,
            mask=None,
            size_subfig=3.5,xspace=0.7,yspace=0.1,cbar_pad=0.1,
            title_position=0,
            redraw_ticks=True,num_colorbar_value=4,minvs=None,maxvs=None,tick_format=None,
            data_scale=None,
            rotate_colorbar_with_oneinput=False,
            save_name=None,
            use_sym_colormap=False
            ):
    field_plotter.plot(
            fields=fields,
            channel_names=channel_names,channel_units=channel_units,case_names=case_names,title=title,
            transpose=transpose,inverse_y=inverse_y,
            cmap=cmap,
            mask=mask,
            size_subfig=size_subfig,xspace=xspace,yspace=yspace,cbar_pad=cbar_pad,
            title_position=title_position,
            redraw_ticks=redraw_ticks,num_colorbar_value=num_colorbar_value,minvs=minvs,maxvs=maxvs,tick_format=tick_format,
            data_scale=data_scale,
            rotate_colorbar_with_oneinput=rotate_colorbar_with_oneinput,
            save_name=save_name,
            use_sym_colormap=use_sym_colormap
    )
