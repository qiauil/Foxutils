#usr/bin/python3
# -*- coding: UTF-8 -*-

#version:0.0.1
#last modified:20231221

import numpy as np
import matplotlib.pyplot as plt
from .style import *

def compare_errors(datas, labels, x_items, std=None, title=None, basic_size=1, y_scale='linear', x_label=None, y_label=None, show_values=False, colors=LINE_COLOR):
    """
    Compare errors using a bar plot.

    Parameters:
    - datas (numpy.ndarray or list of lists): The data to be plotted. Should be a 2D array or a list of lists.
    - labels (list): The labels for each set of data.
    - x_items (list): The labels for each x-axis item.
    - std (numpy.ndarray or None): The standard deviation of the data. If provided, error bars will be shown.
    - title (str or None): The title of the plot.
    - basic_size (int): The basic size of the plot. Default is 1.
    - y_scale (str): The scale of the y-axis. Default is 'linear'.
    - x_label (str or None): The label for the x-axis.
    - y_label (str or None): The label for the y-axis.
    - show_values (bool): Whether to show the values on top of the bars. Default is False.
    - colors (list): The colors for each set of data.

    Raises:
    - ValueError: If the input data is not in the correct format.

    Returns:
    - None: The plot is displayed using matplotlib.pyplot.show().
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not isinstance(datas, np.ndarray) or not isinstance(datas[0], np.ndarray):
        datas = np.array(datas)
    if len(datas.shape) != 2:
        raise ValueError("datas should be a 2D array")
    if len(labels) != datas.shape[0]:
        raise ValueError("labels should have the same length as the first dimension of datas")
    if len(labels) > len(colors):
        raise ValueError("number of label datas is larger than the number of colors. Try to use LINE_COLOR_EXTEND or a larger color list.")
    if len(x_items) != datas.shape[1]:
        raise ValueError("x_items should have the same length as the second dimension of datas")

    x = np.arange(len(x_items))  # the label locations
    n_labels = len(labels)
    n_items = len(x_items)
    fig, ax = plt.subplots(layout='constrained', figsize=(n_items * 2 * basic_size, n_items * 2 * basic_size * 0.618))
    width = 1 / (n_labels + 1)  # the width of the bars
    multiplier = 0
    for i, data in enumerate(datas):
        offset = width * multiplier
        if std is not None:
            yerror = std[i]
        else:
            yerror = None
        rects = ax.bar(x + offset, data, width, label=labels[i], color=colors[i], yerr=yerror, capsize=4)
        if show_values:
            ax.bar_label(rects, padding=3)
        multiplier += 1
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x + (n_labels - 1) * width / 2, x_items)
    ax.legend(loc='upper left', ncols=n_labels)
    ax.set_yscale(y_scale)
    plt.show()
