#usr/bin/python3

#version:0.0.2
#last modified:20231020

from inspect import isfunction
import time 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def time_estimation(start_time,index_now,end_index,start_index=0):
    time_now=time.perf_counter()
    def format_time(seconds):
        seconds=int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "{:0>2d}:{:0>2d}:{:0>2d}".format(h, m, s)
    used=time_now-start_time
    if index_now!=end_index:
        left=used*(end_index-index_now)/(index_now-start_index+1)
        return "Time used:{} Time left:{}".format(format_time(used),format_time(left))
    else:
        return "Total time used:{}".format(format_time(used))

class GeneralDataClass():
    
    def __init__(self,generation_dict=None,**kwargs) -> None:
        if generation_dict is not None:
            for key,value in generation_dict.items():
                if isinstance(value,dict):
                    setattr(self,key,GeneralDataClass(value))
                else:
                    setattr(self,key,value)
        for key,value in kwargs.items():
            if isinstance(value,dict):
                setattr(self,key,GeneralDataClass(value))
            else:
                setattr(self,key,value)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self,key):
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__.items())