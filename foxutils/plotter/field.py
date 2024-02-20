#usr/bin/python3

#version:0.0.15
#last modified:20240119

from . import *
from .style import *
import collections.abc as collections
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from ..helper.coding import *

def plot3D(z,ztitle="z",xtitle="x",ytitle="y",cmap='viridis',plot2D=False,xlist=None,ylist=None,**kwargs):
    '''
    Plot a 3D surface.

    Args:
        z (torch.Tensor): The input tensor.
        ztitle (str, optional): The title of the z-axis. Defaults to "z".
        xtitle (str, optional): The title of the x-axis. Defaults to "x".
        ytitle (str, optional): The title of the y-axis. Defaults to "y".
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        plot2D (bool, optional): Whether to plot a 2D figure. Defaults to False.
        xlist (list, optional): The list of x-axis values. Defaults to None.
        ylist (list, optional): The list of y-axis values. Defaults to None.
        kwargs: Additional keyword arguments for the plot_surface.
    '''
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
    surf=ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False,cmap=plt.get_cmap(cmap),**kwargs)  # 设置颜色映射
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


def plot_2D_ax(ax,
               data,x_start=None,x_end=None,y_start=None,y_end=None,
               transpose=False,
               x_label=None,y_label=None,title=None,title_loc="center",
               interpolation='none', aspect='auto',
               cmap=CMAP_COOLHOT, use_sym_colormap=True,**kwargs):
    """
    Plot a 2D field on the given axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the field.
    - data (numpy.ndarray or torch.Tensor): The 2D field data to be plotted.
    - x_start, x_end, y_start, y_end (float): The range of x and y values for the field.
    - transpose (bool, optional): Whether to transpose the data before plotting. Default is False.
    - x_label, y_label (str, optional): The labels for the x and y axes. Default is None.
    - title (str, optional): The title of the plot. Default is None.
    - title_loc (str, optional): The location of the title. Default is "center".
    - interpolation (str, optional): The interpolation method for the plot. Default is 'none'.
    - aspect (str, optional): The aspect ratio of the plot. Default is 'auto'.
    - cmap (matplotlib colormap, optional): The colormap for the plot. Default is CMAP_COOLHOT.
    - sym_colormap (bool, optional): Whether to use a symmetric colormap. Default is True.
    - kwargs: Additional keyword arguments for imshow.

    Returns:
    - im (matplotlib.image.AxesImage): The plotted image.
    """
    x_start=default(x_start,0)
    x_end=default(x_end,data.shape[-2])
    y_start=default(y_start,0)
    y_end=default(y_end,data.shape[-1])
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if transpose:
        data = data.T
        _x_start = y_start;_x_end = y_end
        _y_start = x_start;_y_end = x_end
        _x_label = y_label;_y_label = x_label
    else:
        _x_start = x_start;_x_end = x_end
        _y_start = y_start;_y_end = y_end
        _x_label = x_label;_y_label = y_label
    if use_sym_colormap:
        cmap=sym_colormap(np.min(data), np.max(data), cmap=cmap)
    im=ax.imshow(data, interpolation=interpolation, cmap=cmap, extent=[_x_start, _x_end, _y_start, _y_end],
                  origin='lower', aspect=aspect,**kwargs)
    if _x_label is not None:
        ax.set_xlabel(_x_label)
    if _y_label is not None:
        ax.set_ylabel(_y_label)
    if title is not None:
        ax.set_title(title,loc=title_loc)
    return im

def plot_2D(data, x_start=None, x_end=None, y_start=None, y_end=None,
            transpose=False,
            x_label=None, y_label=None, title=None, title_loc="center",
            interpolation='none', aspect='auto',
            cmap=CMAP_COOLHOT, use_sym_colormap=True,
            fig_size=None,
            show_colorbar=True, colorbar_label=None,
            save_path=None,**kwargs):
    """
    Plot a 2D field.

    Parameters:
    - data: 2D array-like object representing the field data.
    - x_start, x_end: Start and end values for the x-axis.
    - y_start, y_end: Start and end values for the y-axis.
    - transpose: Boolean indicating whether to transpose the data.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - title_loc: Location of the title ('center', 'left', or 'right').
    - interpolation: Interpolation method for the plot.
    - aspect: Aspect ratio of the plot.
    - cmap: Colormap for the plot.
    - sym_colormap: Boolean indicating whether to use a symmetric colormap.
    - fig_size: Size of the figure (tuple of width and height).
    - show_colorbar: Boolean indicating whether to show the colorbar.
    - colorbar_label: Label for the colorbar.
    - save_path: File path to save the plot.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    im = plot_2D_ax(ax, data, x_start, x_end, y_start, y_end, transpose=transpose,
                    x_label=x_label, y_label=y_label, title=title, title_loc=title_loc,
                    interpolation=interpolation, cmap=cmap, aspect=aspect,use_sym_colormap=use_sym_colormap,**kwargs)
    if show_colorbar:
        c_bar = fig.colorbar(im)
        if colorbar_label is not None:
            c_bar.set_label(colorbar_label)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
def plot2D_grid_ax(ax,field,xtitle="i",ytitle="j",cmap='viridis',xlist=None,ylist=None,vmin=None,vmax=None,**kwargs):
    '''
    Generate a 2D plot using matplotlib.pcolormesh().
    
    Args:
        field (torch.Tensor): The input tensor.
        xtitle (str, optional): The title of the x-axis. Defaults to "j".
        ytitle (str, optional): The title of the y-axis. Defaults to "i".
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        xlist (list, optional): The list of x-axis values. Defaults to None.
        ylist (list, optional): The list of y-axis values. Defaults to None.
        vmin (float, optional): The minimum value of the colormap. Defaults to None.
        vmax (float, optional): The maximum value of the colormap. Defaults to None.
        colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
    '''
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

    pcm=ax.pcolormesh(x, y, field.T,cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    return pcm
    #fig.set_figheight(len(y)*0.3)
    #fig.set_figwidth(len(x)*0.3)

def plot2D_grid(field,xtitle="i",ytitle="j",cmap='viridis',xlist=None,ylist=None,vmin=None,vmax=None,show_colorbar=True,colorbar_label=None,save_path=None,**kwargs):
    fig, ax = plt.subplots()
    pcm=plot2D_grid_ax(ax,field,xtitle,ytitle,cmap,xlist,ylist,vmin,vmax,**kwargs)
    fig.set_figheight(field.shape[1]*0.3)
    fig.set_figwidth(field.shape[0]*0.3)
    if show_colorbar:
        c_bar=fig.colorbar(pcm)
        if colorbar_label is not None:
            c_bar.set_label(colorbar_label)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def show_image_from_tensor(image_tensor,title=""):
    '''
    Show an image from a tensor.

    Args:
        image_tensor (torch.Tensor): The input tensor. The shape of the tensor should be (3, height, width).
        title (str, optional): The title of the image. Defaults to "".
    '''
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
    """
    A class for plotting channel fields.

    Methods:
    - fig_save_path(self, path): Sets the figure save path.
    - plot(self, fields, channel_names, channel_units, case_names, title, transpose, inverse_y, cmap, mask, size_subfig, xspace, yspace, cbar_pad, title_position, redraw_ticks, num_colorbar_value, minvs, maxvs, tick_format, data_scale, rotate_colorbar_with_oneinput, subfigure_index, save_name, use_sym_colormap): Plots the fields.
    """
    
    def __init__(self) -> None:
        self.__fig_save_path="./output_figs/"

    def __type_transform(self, fields):
        """
        Transforms the input fields to the desired type.

        Args:
        - fields: The input fields.

        Returns:
        - The transformed fields.
        """
        if isinstance(fields, collections.Sequence):
            if isinstance(fields[0], torch.Tensor):
                fields = [(field.to(torch.device("cpu"))).numpy() for field in fields]
                return fields
            elif isinstance(fields[0], np.ndarray):
                return fields
            else:
                raise Exception("Wrong input type!")
        else:
            if isinstance(fields, torch.Tensor):
                fields = (fields.to(torch.device("cpu"))).numpy()
                return fields
            elif isinstance(fields, np.ndarray):
                return fields
            else:
                raise Exception("Wrong input type!")       
        
    def __cat_fields(self, fields):
        """
        Concatenates the fields into a single array.

        Args:
        - fields: The input fields.

        Returns:
        - The concatenated fields.
        """
        if isinstance(fields, collections.Sequence):
            if len(fields[0].shape) == 4:
                return np.concatenate(fields, 0)
            elif len(fields[0].shape) == 3:
                return np.concatenate([np.expand_dims(field, 0) for field in fields], 0)
            elif len(fields[0].shape) == 2:
                return np.concatenate([np.expand_dims(np.expand_dims(field, 0), 0) for field in fields], 0)
            else:
                raise Exception("Wrong input type!")
        else:
            if len(fields.shape) == 2:
                return np.expand_dims(np.expand_dims(fields, 0), 0)
            if len(fields.shape) == 3:
                return np.expand_dims(fields, 0)
            elif len(fields.shape) == 4:
                return fields
            else:
                raise Exception("Wrong input type!")

    def __find_min_max(self, fields, defaultmin, defaultmax):
        """
        Finds the minimum and maximum values for each field.

        Args:
        - fields: The input fields.
        - defaultmin: The default minimum values.
        - defaultmax: The default maximum values.

        Returns:
        - The minimum and maximum values for each field.
        """
        mins = []
        maxs = []
        for i in range(fields.shape[1]):
            if defaultmin is not None:
                if defaultmin[i] is not None:
                    mins.append(defaultmin[i])
                else:
                    mins.append(np.min(fields[:, i, :, :]))
            else:
                mins.append(np.min(fields[:, i, :, :]))
            if defaultmax is not None:
                if defaultmax[i] is not None:
                    maxs.append(defaultmax[i])
                else:
                    maxs.append(np.max(fields[:, i, :, :]))
            else:
                maxs.append(np.max(fields[:, i, :, :]))
        return mins, maxs
   
    def __generate_mask(self, mask, transpose, color="white"):
        """
        Generates a mask for the fields.

        Args:
        - mask: The mask.
        - transpose: Whether to transpose the mask.
        - color: The color of the mask.

        Returns:
        - The generated mask.
        """
        mask = self.__type_transform(mask)
        if color == "white":
            RGB = np.ones(mask.shape)  # zeros=Black, ones=white
        elif color == "black":   
            RGB = np.zeros(mask.shape) 
        if transpose:
            return torch.cat([np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(mask.T, 2)], -1)
        else:
            return torch.cat([np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(mask, 2)], -1)

    def fig_save_path(self, path):
        """
        Sets the figure save path.

        Args:
        - path: The path to save the figures.
        """
        self.__fig_save_path = path  
        
    def plot(self, fields, channel_names=None, channel_units=None, case_names=None, title="", transpose=False, inverse_y=False, cmap=CMAP_COOLHOT, mask=None, size_subfig=3.5, xspace=0.7, yspace=0.1, cbar_pad=0.1, title_position=0, redraw_ticks=True, num_colorbar_value=4, minvs=None, maxvs=None, tick_format=None, data_scale=None, rotate_colorbar_with_oneinput=False, subfigure_index=None, save_name=None, use_sym_colormap=True):
        """
        Plots the fields.

        Args:
        - fields: The input fields.
        - channel_names: The names of the channels.
        - channel_units: The units of the channels.
        - case_names: The names of the cases.
        - title: The title of the plot.
        - transpose: Whether to transpose the fields.
        - inverse_y: Whether to invert the y-axis.
        - cmap: The colormap to use.
        - mask: The mask for the fields.
        - size_subfig: The size of each subfigure.
        - xspace: The spacing between subfigures along the x-axis.
        - yspace: The spacing between subfigures along the y-axis.
        - cbar_pad: The padding of the colorbar.
        - title_position: The position of the title.
        - redraw_ticks: Whether to redraw the colorbar ticks.
        - num_colorbar_value: The number of colorbar values.
        - minvs: The minimum values for each field.
        - maxvs: The maximum values for each field.
        - tick_format: The format of the colorbar ticks.
        - data_scale: The scale of the data for each channel.
        - rotate_colorbar_with_oneinput: Whether to rotate the colorbar when there is only one input.
        - subfigure_index: The index of the subfigure.
        - save_name: The name to save the figure.
        - use_sym_colormap: Whether to use a symmetric colormap.
        """
        fields = self.__cat_fields(self.__type_transform(fields))
        if mask is not None:
            mask = self.__generate_mask(mask, transpose=transpose)
        num_cases = fields.shape[0]
        num_channels = fields.shape[1]
        
        channel_names = default(channel_names, ["channel {}".format(i) for i in range(num_channels)])
        channel_units = default(channel_units, ["" for i in range(num_channels)])
        case_names = default(case_names, ["case {}".format(i) for i in range(num_cases)])
        data_scale = default(data_scale, [1 for i in range(num_channels)])
        fields = np.concatenate([fields[:, i:i+1, :, :] * data_scale[i] for i in range(num_channels)], 1)
        mins, maxs = self.__find_min_max(fields, minvs, maxvs)
        
        if num_cases == 1 and rotate_colorbar_with_oneinput:
            cbar_location = "right"
            cbar_mode = 'each'
            ticklocation = "right"
        else:
            cbar_location = "top"
            cbar_mode = 'edge'
            ticklocation = "top" 
        fig = plt.figure(figsize=(size_subfig * num_channels, size_subfig * num_cases))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(num_cases, num_channels),
                        axes_pad=(xspace, yspace),
                        share_all=True,
                        cbar_location=cbar_location,
                        cbar_mode=cbar_mode,
                        direction='row',
                        cbar_pad=cbar_pad
                        )
        im_cb = []
        if use_sym_colormap:
            colormaps = []
            for i in range(num_channels):
                colormaps.append(sym_colormap(mins[i], maxs[i], cmap=cmap))
                
        for i, axis in enumerate(grid):
            i_row = i // num_channels
            i_column = i % num_channels
            datai = fields[i_row, i_column]
            if transpose:
                datai = datai.T
            if use_sym_colormap:
                im = axis.imshow(datai, colormaps[i_column], vmin=mins[i_column], vmax=maxs[i_column])
            else:
                im = axis.imshow(datai, cmap, vmin=mins[i_column], vmax=maxs[i_column])
            if i < num_channels:
                im_cb.append(im)
                    
            if mask is not None:
                axis.imshow(mask)  
            if inverse_y:
                axis.invert_yaxis()      
            axis.set_yticks([])
            axis.set_xticks([])
            if i_column == 0:
                axis.set_ylabel(case_names[i_row])
            if i_row == num_cases - 1:
                axis.set_xlabel(channel_names[i_column])   

        for i in range(num_channels):
            cb = grid.cbar_axes[i].colorbar(im_cb[i], label=channel_units[i], ticklocation=ticklocation, format=tick_format)
            cb.ax.minorticks_on()
            if redraw_ticks:
                cb.set_ticks(np.linspace(mins[i], maxs[i], num_colorbar_value, endpoint=True))      
        fig.suptitle(title, y=title_position)
        if subfigure_index is not None:
            plt.suptitle(subfigure_index, x=0.01, y=0.88, fontproperties="Times New Roman")
        if save_name is not None:
            plt.savefig(self.__fig_save_path + save_name + ".svg", bbox_inches='tight')
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
    '''
    Show each channel of the fields. Use an instance of ChannelPloter to plot the fields. See ChannelPloter.plot() for more details.

    Args:
        fields: The input fields.
        channel_names: The names of the channels.
        channel_units: The units of the channels.
        case_names: The names of the cases.
        title: The title of the plot.
        transpose: Whether to transpose the fields.
        inverse_y: Whether to invert the y-axis.
        cmap: The colormap to use.
        mask: The mask for the fields.
        size_subfig: The size of each subfigure.
        xspace: The spacing between subfigures along the x-axis.
        yspace: The spacing between subfigures along the y-axis.
        cbar_pad: The padding of the colorbar.
        title_position: The position of the title.
        redraw_ticks: Whether to redraw the colorbar ticks.
        num_colorbar_value: The number of colorbar values.
        minvs: The minimum values for each field.
        maxvs: The maximum values for each field.
        tick_format: The format of the colorbar ticks.
        data_scale: The scale of the data for each channel.
        rotate_colorbar_with_oneinput: Whether to rotate the colorbar when there is only one input.
        save_name: The name to save the figure.
        use_sym_colormap: Whether to use a symmetric colormap.
    '''
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
