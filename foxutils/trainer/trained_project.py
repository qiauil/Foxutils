import os
import yaml
import torch.nn as nn
import torch
from typing import List,Sequence, Union, Optional, Literal
from warnings import warn
from tensorboard.backend.event_processing import event_accumulator
from tqdm.auto import tqdm
from .postprocess import PostProcessor
from lightning.fabric import Fabric

def read_configs(path_config_file) -> dict:
    '''
    Read the training configurations from a yaml file.
    
    Args:
        path_config_file: str, path to the yaml file of the training configurations.
    
    Returns:
        dict: The training configurations.
    '''
    config_dict={}
    if os.path.isdir(path_config_file):
        paths=[os.path.join(path_config_file,group) for group in os.listdir(path_config_file)]
    else:
        paths=[path_config_file]
    for yaml_path in paths:
        with open(yaml_path,"r") as f:
            yaml_configs=yaml.safe_load(f)
        config_dict.update(yaml_configs)
    return config_dict

def _available_ckpt_ids(ckpt_dir:str) -> List[int]:
    available_ckpts=[]
    for ckpt_name in os.listdir(ckpt_dir):
        if ckpt_name.endswith(".ckpt"):
            try:
                available_ckpts.append(int(ckpt_name.split("_")[1].split(".")[0]))
            except:
                raise ValueError(f"Invalid checkpoint name {ckpt_name}")
    return available_ckpts

class CkptFiles:
    
    def __init__(self,ckpt_dir:str) -> None:
        self.ckpt_dir=ckpt_dir
    
    def __getitem__(self,ckpt_id:int) -> str:
        if ckpt_id not in _available_ckpt_ids(self.ckpt_dir):
            raise ValueError(f"Checkpoint with id {ckpt_id} not found in {self.ckpt_dir}")
        return torch.load(os.path.join(self.ckpt_dir,f"epoch_{ckpt_id}.ckpt"))
        
class TrainedVersion:
    
    def __init__(self,
                 project_path:str,
                 run_name:str,
                 version:str) -> None:
        self.project_path = project_path
        self.run_name = run_name
        self.version = version
        self.run_dir = os.path.join(self.project_path, self.run_name, self.version)
        self.record_path = os.path.join(self.run_dir, "event.log")
        self.logger_dir=os.path.join(self.run_dir, "logs")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoint")
        self.postprocess_dir=os.path.join(self.run_dir,"postprocess")
        self.config_dir = os.path.join(self.run_dir, "configs")
        self.model_structure_path=os.path.join(self.run_dir,"model_structure.pt")
        self.final_weights_path=os.path.join(self.run_dir, "final_weights.ckpt")
        self._config_dict=None
        self._fabric=None
        if not os.path.exists(self.config_dir):
            raise FileNotFoundError(f"Configuration file not found at {self.config_dir}. Not a valid trained version.")
        
    @property 
    def configs(self) -> dict:
        '''
        Read the training configurations from a yaml file.
        
        Returns:
            dict: The training configurations.
        '''
        if self._config_dict is None:
            self._config_dict = read_configs(self.config_dir)
        return self._config_dict
    
    @property
    def network_structure(self) -> nn.Module:
        '''
        Load the network structure.
        
        Returns:
            nn.Module: The network structure.
        '''
        if os.path.exists(self.model_structure_path):
            return torch.load(self.model_structure_path)
        else:
            raise FileNotFoundError(f"Model structure not found at {self.model_structure_path}")
    
    @property
    def final_weights(self) -> dict:
        '''
        Load the final weights of the model.
        
        Returns:
            dict: The final weights of the model.
        '''
        if os.path.exists(self.final_weights_path):
            return torch.load(self.final_weights_path)["model"]
        else:
            raise FileNotFoundError(f"Final weights not found at {self.final_weights_path}")
    
    @property
    def final_network(self) -> nn.Module:
        '''
        Load the final model.
        
        Returns:
            nn.Module: The final model.
        '''
        model=self.network_structure
        model.load_state_dict(self.final_weights)
        return model
    
    @property
    def available_checkpoints_id(self) -> list:
        '''
        Get the list of available checkpoints.
        
        Returns:
            list: The list of ids of available checkpoints.
        '''
        return _available_ckpt_ids(self.ckpt_dir)
    
    @property
    def ckpt(self) -> CkptFiles:
        return CkptFiles(self.ckpt_dir)
    
    @property
    def is_tensorboard_logger(self) -> bool:
        '''
        Check if the logger is tensorboard.
        
        Returns:
            bool: True if the logger is tensorboard, False otherwise.
        '''
        return self.configs["logger"]=="TensorBoard"
    
    @property
    def tb_event_accumulator(self):
        if not self.is_tensorboard_logger:
            raise ValueError("Event accumulator is only available for tensorboard loggers")
        record_path=self.logger_dir
        records=os.listdir(record_path)
        if len(records)==0:
            raise FileNotFoundError("No records found in {}".format(record_path))
        if len(records)>1:
            warn("Multiple records found in {}. Using {}.".format(record_path,records[0]))
        ea= event_accumulator.EventAccumulator(os.path.join(record_path,records[0]))
        ea.Reload()
        return ea

    @property
    def fabric(self):
        if self._fabric is None:        
            self._fabric = Fabric(
                accelerator=self.configs["device_type"],
                strategy=self.configs["multi_devices_strategy"],
                devices=self.configs["num_id_devices"],
                num_nodes=self.configs["num_nodes"],
                precision=self.configs["precision"],
            )
            self._fabric.launch()
        return self._fabric

    def load_postprocessor(self,
                           postprocessor:Union[PostProcessor,Sequence[PostProcessor]],
                           p_bar_leave=True,
                           fabric:Union[Literal["auto"],Fabric,None]="auto"
                           ):
        if not isinstance(postprocessor,Sequence):
            postprocessor=[postprocessor]
        enmu=tqdm(postprocessor,desc="Running postprocessors",leave=p_bar_leave)
        if fabric == "auto":
            fabric=self.fabric
        else:
            fabric=fabric
        for processor in enmu:
            postprocess_dir=os.path.join(self.postprocess_dir,processor.processor_name)
            os.makedirs(postprocess_dir,exist_ok=True)
            enmu.set_description(f"Running postprocessor: {processor.processor_name}")
            model=fabric.setup(self.final_network)
            return_value=processor.run(model,
                                       self.configs,
                                       postprocess_dir,
                                       fabric=fabric)
            if return_value is not None:
                processor.rank_zero_save(return_value,
                                         os.path.join(postprocess_dir,"output.pt"),
                                         fabric)
        
class TrainedRun:
    
    def __init__(self,
                 project_path:str,
                 run_name:str,) -> None:
        self.run_name=run_name
        self.version_names=os.listdir(os.path.join(project_path,run_name))
        self.versions=[TrainedVersion(project_path,run_name,version) for version in self.version_names]
    
    def __getitem__(self,id:int) -> TrainedVersion:
        return self.versions[id]
    
    def __len__(self) -> int:
        return len(self.versions)
    
    def load_postprocessor(self,
                           postprocessor:Union[PostProcessor,Sequence[PostProcessor]],
                           p_bar_leave=True
                           ):
        if len(self) == 1:
            self[0].load_postprocessor(postprocessor,False)
        else:
            enmu=tqdm(self,desc="Versions",leave=p_bar_leave)
            for run in enmu:
                enmu.set_description(f"Version: {run.version}")
                run.load_postprocessor(postprocessor,False)
        
class TrainedProject:
    
    def __init__(self,
                 project_path:str,
                 white_list:Optional[Union[Sequence[str],str]]=None,
                 black_list:Optional[Union[Sequence[str],str]]=None) -> None:
        if black_list is None:
            black_list=[]
        elif isinstance(black_list,str):
                black_list=[black_list]
        if white_list is None:
            white_list=os.listdir(project_path)
        elif isinstance(white_list,str):
            white_list=[white_list]
        self.run_names=[name for name in white_list if name not in black_list]
        self.runs=[TrainedRun(project_path,run_name) for run_name in self.run_names]
        
    def __getitem__(self,id:int) -> TrainedRun:
        return self.runs[id]
    
    def __len__(self) -> int:
        return len(self.runs)
    
    def load_postprocessor(self,
                           postprocessor:Union[PostProcessor,Sequence[PostProcessor]],
                           p_bar_leave=True
                           ):
        if len(self)==1:
            self[0].load_postprocessor(postprocessor,False)
        else:
            enmu=tqdm(self,desc="Runs",leave=p_bar_leave)
            for run in enmu:
                enmu.set_description(f"Run: {run.run_name}")
                run.load_postprocessor(postprocessor,False)