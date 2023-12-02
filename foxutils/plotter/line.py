#usr/bin/python3

#version:0.0.2
#last modified:20231128


from . import *
from .style import *
from ..helper.coding import *
import os,math

class FormatLinePlotter():
    
    def __init__(self) -> None:
        self.colors=LINE_COLOR
        #https://github.com/OrdnanceSurvey/GeoDataViz-Toolkit/tree/master/Colours
        self.linestyles=[ls[1] for ls in LINE_STYLE]
        self.markers=["o","^","s","X","D","*","p"]
        self.__draws=[]
        #self.__scatters=[]
        #self.__black_lines=[]
        #self.__color_lines=[]
        #self.__scatter_lines=[]
        #self.__color_line_errorbars=[]
        #self.__color_line_errorshadows=[]
        self.__xlabel="$x$"
        self.__ylabel="$y$"
        self.__ncol_legend=None
        self.__fig_save_path="./output_figs/"
        self.__subfigure_index=None
        self.__fig_size=None
        self.__legend_y=None
        self.__xscale="linear"
        self.__yscale="linear"
        self.__grid=False
    
    def scatter(self,x,y,label=None,color_style=None,mark_style=None):
        self.__draws.append(["scatter",(x,y,label,color_style,mark_style)])
    
    def black_line(self,x,y,label=None,line_style=None,lw=2):
        self.__draws.append(["black_line",(x,y,label,line_style,lw)])

    def color_line(self,x,y,label=None,color_style=None,line_style=None,lw=2):
        self.__draws.append(["color_line",(x,y,label,color_style,line_style,lw)])

    def scatter_line(self,x,y,label=None,color_style=None,line_style=None,mark_style=None,lw=2):
        self.__draws.append(["scatter_line",(x,y,label,color_style,line_style,mark_style,lw)])
    
    def color_line_errorbar(self,x,y,y_error=None,x_error=None,label=None,color_style=None,line_style=None,lw=2):
        self.__draws.append(["color_line_errorbar",(x,y,y_error,x_error,label,color_style,line_style,lw)])

    def color_line_errorshadow(self,x,y,y_error=None,x_error=None,label=None,color_style=None,line_style=None,lw=2):
        self.__draws.append(["color_line_errorshadow",(x,y,y_error,x_error,label,color_style,line_style,lw)])

    def get_legend_pos(self,num_legend):
        ncol=self.__ncol_legend
        legend_y=1.02
        if ncol is None:
            if num_legend <=4:
                ncol=num_legend
            ### 如果行数相等
            if math.ceil(num_legend/4)-math.ceil(num_legend/3)==0:
                ### 谁刚好填满选谁
                if num_legend%3 ==0:
                    ncol= 3
                elif num_legend%4 ==0:
                    ncol= 4
                elif 3-(num_legend%3) < 4-(num_legend%4):
                    ncol= 3
                else:
                    ncol= 4
            else:
                ncol= 4  
        n_row=math.ceil(num_legend/ncol)
        if n_row ==2:
            legend_y=1.1
        else:
            legend_y=1.02
        return ncol, default(self.__legend_y,legend_y)

    def plot(self):
        figs=[]
        labels=[]
        plt.figure(figsize=default(self.__fig_size,(8,5)))
        plt.cla()
        num_legend=0
        for i,draws in enumerate(self.__draws):
            name=draws[0]
            if name =="black_line":
                (x,y,label,line_style,lw)=draws[1]      
                fig=plt.plot(x,y,linestyle=self.linestyles[default(line_style,i)],linewidth=lw,color='black')
                if label is not None:
                    figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1
            elif name == "color_line":
                (x,y,label,color_style,line_style,lw)=draws[1] 
                fig=plt.plot(x,y,linewidth=lw,color=self.colors[default(color_style,i)],linestyle=self.linestyles[default(line_style,i)])
                if label is not None:
                    figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1
            elif name == "scatter":
                (x,y,label,color_style,mark_style)=draws[1] 
                fig=plt.scatter(x,y,marker=self.markers[default(mark_style,i)],c=self.colors[default(color_style,i)],s=100)
                if label is not None:
                    figs.append(fig)
                    labels.append(label)
                    num_legend+=1        
            elif name == "scatter_line":
                (x,y,label,color_style,line_style,mark_style,lw)=draws[1]            
                fig=plt.plot(x,y,marker=self.markers[default(mark_style,i)],c=self.colors[default(color_style,i)],linestyle=self.linestyles[default(line_style,i)],markersize=10,linewidth=lw)
                if label is not None:
                    figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1   
            elif name == "color_line_errorbar":
                (x,y,y_error,x_error,label,color_style,line_style,lw)=draws[1] 
                fig=plt.errorbar(x=x,y=y,yerr=y_error,xerr=x_error,c=self.colors[default(color_style,i)],linestyle=self.linestyles[default(line_style,i)],linewidth=lw,elinewidth=2,capsize=4)
                if label is not None:
                    figs.append(fig)
                    labels.append(label)
                    num_legend+=1    
            elif name == "color_line_errorshadow":
                (x,y,y_error,x_error,label,color_style,line_style,lw)=draws[1] 
                handles=[] 
                if y_error is not None:
                    s1=plt.fill_between(x=y,y1=[y[i]-0.5*y_error[i] for i in range(len(y))],y2=[y[i]+0.5*y_error[i] for i in range(len(y))],facecolor=self.colors[default(color_style,i)],alpha=0.2)
                    handles.append(s1)
                if x_error is not None:
                    s2=plt.fill_betweenx(y=y,x1=[x[i]-0.5*x_error[i] for i in range(len(x))],x2=[x[i]+0.5*x_error[i] for i in range(len(x))],facecolor=self.colors[default(color_style,i)],alpha=0.2)
                    handles.append(s2)
                fig=plt.plot(x,y,linewidth=lw,color=self.colors[default(color_style,i)],linestyle=self.linestyles[default(line_style,i)])
                if label is not None:
                    if y_error is not None or x_error is not None:
                        handles.append(fig[0])
                        figs.append(tuple(handles))
                    else:
                        figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1     
            else:
                raise Exception("unknow draw type: {}".format(name))
        if num_legend!=0:
            ncol,legend_y=self.get_legend_pos(num_legend)      
            plt.legend(handles=figs, labels=labels,bbox_to_anchor=(0., legend_y, 1., .102), mode="expand", borderaxespad=0.,ncol=ncol)
        if self.__subfigure_index is not None:
            plt.suptitle(self.__subfigure_index,x=0.01,y=0.88,fontproperties="Times New Roman")
        plt.xscale(self.__xscale)
        plt.yscale(self.__yscale)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.grid(self.__grid)
        plt.minorticks_on()
            
    def xlabel(self,xlabel):
        self.__xlabel=xlabel
    
    def ylabel(self,ylabel):
        self.__ylabel=ylabel     
    
    def ncol_legend(self,ncol):
        self.__ncol_legend=ncol
        
    def clear_all(self):
        self.__init__()    

    def fig_save_path (self,path):
        self.__fig_save_path=path  

    def subfigure_index(self,index):
        self.__subfigure_index=index

    def fig_size(self,size):
        self.__fig_size=size

    def legend_y(self,y):
        self.__legend_y=y
        
    def xscale(self,scale):
        self.__xscale=scale
    
    def yscale(self,scale):
        self.__yscale=scale

    def grid(self,grid):
        self.__grid=grid

    def save(self,filename):
        os.makedirs(self.__fig_save_path,exist_ok=True)
        plt.savefig(self.__fig_save_path+filename+".svg",bbox_inches = 'tight')             

line_plotter=FormatLinePlotter()

class DoubleSidePlotter():
    
    def __init__(self) -> None:
        fig,ax_left=plt.subplots()
        self.left_axis=ax_left
        self.right_axis=ax_left.twinx()
    
    def set_xlabel(self,label):
        self.left_axis.set_xlabel(label)
        
    def fast_plot(self,data_1,data_2,data_x=None,labels=["data_1","data_2"],log=True,xlabel=None,color_data1=None,color_data2=None):
        color_data1 = default(color_data1,LINE_COLOR[0])
        color_data2 = default(color_data2,LINE_COLOR[1])
        if data_x is not None:
            self.left_axis.plot(data_x,data_1, color=color_data1)
            self.right_axis.plot(data_x,data_2, color=color_data2)
        else:
            self.left_axis.plot(data_1, color=color_data1)
            self.right_axis.plot(data_2, color=color_data2)
        self.left_axis.set_ylabel(labels[0], color=color_data1)
        self.right_axis.set_ylabel(labels[1], color=color_data2)
        self.left_axis.tick_params(axis='y', labelcolor=color_data1)
        self.right_axis.tick_params(axis='y', labelcolor=color_data2)
        if log:
            self.left_axis.set_yscale('log')
            self.right_axis.set_yscale('log')
        if xlabel is not None:
            self.left_axis.set_xlabel(xlabel)
        
def plot_double_side(data_1,data_2,data_x=None,labels=["data_1","data_2"],log=True,xlabel=None,color_data1=None,color_data2=None):
    DoubleSidePlotter().fast_plot(data_1,data_2,data_x=data_x,labels=labels,log=log,xlabel=xlabel,color_data1=color_data1,color_data2=color_data2)
