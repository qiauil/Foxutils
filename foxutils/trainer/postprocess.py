from typing import Optional,Dict,Any
from lightning.fabric import Fabric
from typing import Any,Iterable
from tqdm.auto import tqdm
import torch
import torch.nn as nn

class PostProcessor:
    
    def __init__(self,
                 processor_name:str) -> None:
        self.processor_name=processor_name
    
    def run(self,
            model: Optional[nn.Module]=None,
            config_dict: Optional[dict]=None,
            run_path: Optional[str]=None,
            fabric:Optional[Fabric]=None) -> Any:
        pass
    
    def rank_zero_p_bar(self, 
                        iterable:Iterable, 
                        fabric:Optional[Fabric]=None,
                        desc=None, 
                        leave=True):
        if fabric is None:
            return tqdm(iterable,desc=desc,leave=leave)
        else:
            if fabric.local_rank==0:
                return tqdm(iterable,desc=desc,leave=leave)
            else:
                return iterable
            
    def rank_zero_save(self,
                       data: Any,
                       file_path: str,
                       fabric:Optional[Fabric]=None,):
        if fabric is None:
            torch.save(data,file_path)
        else:
            if fabric.local_rank==0:
                torch.save(data,file_path)