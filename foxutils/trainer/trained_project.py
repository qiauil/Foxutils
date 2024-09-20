import os
import yaml
import torch.nn as nn
import torch
from typing import List
from warnings import warn
from tensorboard.backend.event_processing import event_accumulator

def read_configs(path_config_file) -> dict:
    '''
    Read the training configurations from a yaml file.
    
    Args:
        path_config_file: str, path to the yaml file of the training configurations.
    
    Returns:
        dict: The training configurations.
    '''
    with open(path_config_file,"r") as f:
        yaml_configs=yaml.safe_load(f)
    return yaml_configs

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
        self.logger_dir = os.path.join(self.project_path, "logs",)
        self.logger_file_dir=os.path.join(self.logger_dir, self.run_name, self.version)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoint")
        self.config_path = os.path.join(self.run_dir, "config.yaml")
        self.model_structure_path=os.path.join(self.run_dir,"model_structure.pt")
        self.final_weights_path=os.path.join(self.run_dir, "final_weights.ckpt")
        self._config_dict=None
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}. Not a valid trained version.")
        
    @property 
    def configs(self) -> dict:
        '''
        Read the training configurations from a yaml file.
        
        Returns:
            dict: The training configurations.
        '''
        if self._config_dict is None:
            self._config_dict = read_configs(self.config_path)
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
    def event_accumulator(self):
        if not self.is_tensorboard_logger:
            raise ValueError("Event accumulator is only available for tensorboard loggers")
        record_path=self.logger_file_dir
        records=os.listdir(record_path)
        if len(records)==0:
            raise FileNotFoundError("No records found in {}".format(record_path))
        if len(records)>1:
            warn("Multiple records found in {}. Using {}.".format(record_path,records[0]))
        ea= event_accumulator.EventAccumulator(os.path.join(record_path,records[0]))
        ea.Reload()
        return ea
        