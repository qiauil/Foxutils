#usr/bin/python3

#version:0.0.4
#last modified:20240224

from . import *

def enable_print_style(font_name="Times New Roman", font_size=30):
    #mlp.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    mlp.rcParams['font.sans-serif'] = [font_name, font_name]
    mlp.rcParams['font.size'] = font_size
    #mlp.rcParams['text.usetex'] =True


COOL=mlp.cm.get_cmap("coolwarm")(np.linspace(0, 0.5, 5))
HOT=mlp.cm.get_cmap("coolwarm")(np.linspace(0.5, 1, 5))
WHITE=[[1,1,1,1]]

CMAP_COOL=colors.LinearSegmentedColormap.from_list("COOL",np.vstack((COOL[0:-1],WHITE)))
CMAP_HOT=colors.LinearSegmentedColormap.from_list("HOT",np.vstack((WHITE,HOT[1:])))
CMAP_COOLHOT=colors.LinearSegmentedColormap.from_list("HOT",np.vstack((COOL[0:-1],WHITE,HOT[1:])))

LINE_COLOR=['#FF1F5B', '#009ADE', '#FFC61E', '#AF58BA', '#F28522', '#00CD6C','#A6761D']
LINE_COLOR_EXTEND=LINE_COLOR+["#B2A4FF","#96CEB4","#3C486B"]
#https://github.com/OrdnanceSurvey/GeoDataViz-Toolkit/tree/master/Colours

LINE_STYLE = [
    ('solid', (0, ())), 
     ('dashed', (0, (5, 5))),
     ('dotted', (0, (1, 1))), 
    ('dashdot', (0, (3, 5, 1, 5))), 
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5,1, 5))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
