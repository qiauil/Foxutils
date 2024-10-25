from ..helper.coding import *
from typing import List,Any,Union,Dict, Optional
from tqdm.auto import tqdm
from typing import Iterable
from torch.utils.data import DataLoader
from lightning.fabric.strategies import Strategy
from torch.optim import Optimizer

class TrainConfigMixin(GroupedConfigurationsHandler):

    def register_project_info_configs(self):
        self.add_config_item("project_path",
                            group="project_info",
                            value_type=str,
                            mandatory=True,
                            description="Path to save the training results.",
                            )
        self.add_config_item("run_name",
                            group="project_info",
                            value_type=str,
                            mandatory=False,
                            default_value_func=lambda configs:"{}".format(time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))),
                            description="Name of the training run; If not set, the run_name will be set as current time.")
        self.add_config_item("version",
                            group="project_info",
                            value_type=str,
                            mandatory=False,
                            default_value_func=lambda configs:"sd_{}".format(configs["random_seed"]),
                            description='Version of the training run; If not set, the version will be set to "sd_random_seed"')
                
    def register_device_configs(self):
        self.add_config_item("device_type",
                            group="control",
                            default_value="auto",
                            value_type=str,
                            description="The hardware to run on. Will pass to `accelerator` in `Fabric`.",
                            options=["cpu", "cuda", "mps", "gpu", "tpu", "auto"])
        self.add_config_item("num_id_devices",
                            group="distributed_training",
                            default_value="auto",
                            value_type=Union[int,str,List],
                            description='Number of devices to train on (int), which GPUs to train on (list or str), or "auto".The value applies per node. Will pass to `device` in `Fabric`.')
        self.add_config_item("num_nodes",
                            group="distributed_training",
                            default_value=1,
                            value_type=int,
                            description='Number of GPU nodes for distributed training.')
        self.add_config_item("multi_devices_strategy",
                             group="distributed_training",
                             default_value="auto",
                             value_type=str,
                             description='Strategy for how to run across multiple devices. Possible choices include: "dp", "ddp", "ddp_spawn", "deepspeed", "fsdp". Full list of available options can be found at https://lightning.ai/docs/fabric/stable/api/fabric_args.html#strategy. Will pass to `strategy` in `Fabric`.'
                             )  

    def register_control_configs(self):
        self.add_config_item("random_seed",
                             group="control",
                             default_value_func=lambda x:int(time.time()),
                             value_type=int,
                             description="Random seed for training. If not set, the random seed will be the current time.")
        self.add_config_item("num_epochs",
                            group="control",
                            mandatory=True,
                            value_type=int,
                            description="Number of epochs for training.")
        self.add_config_item("validation_frequency",
                             group="control",
                             default_value_func=lambda configs:configs["num_epochs"]//10,
                             value_type=int,
                             description="Frequency of validation. The unit is epoch. If not set, will validate every num_epochs//10 epoch.")
        self.add_config_item("checkpoint_save_frequency",
                             group="control",
                             default_value_func=lambda configs:configs["num_epochs"]//10,
                             value_type=int,
                             description="Frequency of saving checkpoints. The unit is epoch. If not set, the checkpoint will be saved every num_epochs//10 epoch.")    
        self.add_config_item("grad_accum_steps",
                             group="control",
                             default_value=1,
                             value_type=int,
                             description="Number of gradient accumulation steps. The unit is iteration.")
        self.add_config_item("precision",
                            group="control",
                            default_value="32",
                            value_type=str,
                            description='Precision to use. Double precision ("64"), full precision ("32"), half precision AMP ("16-mixed"), or bfloat16 precision AMP ("bf16-mixed"). Full list of available options can be found at https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision. Will pass to `precision` in `Fabric`.')
        self.add_config_item("compile_model",
                             group="control",
                             default_value=False,
                             value_type=bool,
                             description="Whether to compile the model. If you want to compile the model, you need to make sure that the shape of model's input is fixed.")
        

    def register_logger_configs(self):
        self.add_config_item("logger",
                             group="logging",
                             value_type=str,
                             default_value="TensorBoard",
                             options=["TensorBoard","CSV"],
                             description="Logger for training. If you want to use other loggers e.g., wandb, you need to reimplement the `initialize_loggers` method in the trainer class.")
        self.add_config_item("logger_configs",
                             group="logging",
                             default_value={},
                             value_type=dict,
                             description="Additional configuration of loggers",)
        self.add_config_item("run_in_silence",
                            group="logging",
                            value_type=bool,
                            default_value=False,
                            description="Whether to run in silence mode. If set to True, only warning will be printed to the console.")
        self.add_config_item("show_iteration_bar",
                            group="logging",
                            value_type=bool,
                            default_value=True,
                            description="Whether to show training/validation iteration bar.")
        self.add_config_item("save_model_structure",
                            group="logging",
                            value_type=bool,
                            default_value=True,
                            description="Whether to save the model structure.")    
        self.add_config_item("log_iteration_loss",
                            group="logging",
                            value_type=bool,
                            default_value=False,
                            description="Whether to log the loss of each iteration.")

    def register_optimizer_configs(self):
        self.add_config_item("optimizer",
                            group="control",
                             default_value="AdamW",
                             value_type=str,
                             description="Optimizer for training. Can be any optimizer in `torch.optim.`",)
        self.add_config_item("optimizer_configs",
                            group="control",
                             default_value={},
                             value_type=dict,
                             description="Additional configuration of optimizers",)
        self.add_config_item("lr",
                            group="control",
                            mandatory=True,
                            value_type=float,
                            description="Initial learning rate.")

    def register_lr_scheduler_configs(self):
        self.add_config_item("lr_scheduler",
                            group="control",
                             default_value="constant",
                             value_type=str,
                             description="Learning rate scheduler for training. It can be set to 'cosine', 'linear', or 'constant'. If you want to use other learning rate schedulers, you reimplement the `configure_lr_scheduler` method in the trainer class.",
                             options=["cosine","linear","constant"])
        self.add_config_item("final_lr",
                            group="control",
                             default_value_func=lambda configs:configs["lr"],
                             value_type=float,
                             description="Final learning rate for lr_scheduler. If not set, the final learning rate will be the same as the initial learning rate.")
        self.add_config_item("warmup_epoch",
                            group="control",
                             default_value=0,
                             value_type=int,
                             description="Number of epochs for learning rate warm up.")     

    def register_dataloader_configs(self):
        self.add_config_item("batch_size_train",
                            group="control",
                             mandatory=True,
                             value_type=int,
                             description="Batch size for training.")
        self.add_config_item("batch_size_val",
                            group="control",
                             default_value_func=lambda configs:configs["batch_size_train"],
                             value_type=int,
                             description="Batch size for validation. Default is the same as batch_size_train. If not set, the batch size will be the same as the training batch size.")
        self.add_config_item("train_dataloader_configs",
                            group="control",
                             default_value={},
                             value_type=dict,
                             description="Additional configuration of training dataloader",)
        self.add_config_item("validation_dataloader_configs",
                             group="control",
                             default_value={},
                             value_type=dict,
                             description="Additional configuration of validation dataloader",)
              
    def register_configs(self):
        self.register_project_info_configs()
        self.register_device_configs()
        self.register_control_configs()
        self.register_logger_configs()
        self.register_optimizer_configs()
        self.register_lr_scheduler_configs()
        self.register_dataloader_configs()

    def save_configs_to_yaml(self,
                             yaml_path_dir:str,
                             with_description=True,):
        """
        Saves the values of configuration items to a YAML file.

        Args:
            yaml_path_dir (str,): The path to the YAML file. If the path is a directory, the configurations will be saved to multiple files based on the groups. 
            only_optional (bool, optional): Indicates whether to save only the optional configuration items. Defaults to False.
        """
        if os.path.isdir(yaml_path_dir):
            yaml_group=self.to_yaml_group(False,with_description)
            for group_name in yaml_group.keys():
                with open(os.path.join(yaml_path_dir,group_name+".yaml"),"w") as f:
                    f.write("# Foxutils Trainer Config V0.0.1"+os.linesep+yaml_group[group_name])
        else:
            yaml_str=self.to_yaml(False,with_description)
            with open(yaml_path_dir,"w") as f:
                f.write("# Foxutils Trainer Config V0.0.1"+os.linesep+yaml_str)

        
class CallbackMixin:
    
    def on_state_register(self,state_dict:Dict[str,Any]):
        pass
    
    def on_train_start(self):
        pass
    
    def on_train_end(self):
        pass
    
    def on_epoch_end(self,epoch_idx:int):
        pass
    
    def on_train_epoch_start(self,epoch_idx:int):
        pass

    def on_train_epoch_end(self,epoch_idx:int):
        pass

    def on_train_batch_start(self,batch_idx:int,batch:Any):
        pass
    
    def on_train_batch_end(self,batch_idx:int,batch:Any,batch_loss:float):
        pass

    def on_before_optimizer_step(self):
        pass

    def on_after_optimizer_step(self,strategy:Strategy, optimizer:Optimizer):
        pass
    
    def on_before_zero_grad(self):
        pass
    
    def on_validation_epoch_start(self,epoch_idx:int):
        pass
    
    def on_validation_epoch_end(self,epoch_idx:int):
        pass

    def on_validation_batch_start(self,batch_idx:int,batch:Any):
        pass
    
    def on_validation_batch_end(self,batch_idx:int,batch:Any,batch_loss:float):
        pass

class ProgressBarMixin:

    _epoch_bar_msg = {
        "postfix_dict":{},
        "postfix_desc":[],
        "desc":[],
        "default_desc":"Finished Epoch"
    }

    _train_it_bar_msg = {
        "postfix_dict":{},
        "postfix_desc":[],
        "desc":[],
        "default_desc":"Finished Training Iteration"
    }

    _validation_it_bar_msg = {
        "postfix_dict":{},
        "postfix_desc":[],
        "desc":[],
        "default_desc":"Finished Validation Iteration"
    }
    
    _p_bar_epoch: Iterable
    _p_bar_train_it: Iterable
    _p_bar_validation_it: Iterable
    
    def rank_zero_tqdm(self, iterable: Optional[Iterable]=None, **kwargs: Any)->Iterable:
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        
        Returns:
            Iterable: the wrapped iterable

        """
        if self.fabric.is_global_zero and not self.configs.run_in_silence:
            return tqdm(iterable, **kwargs)
        return iterable
    
    def _add_bar_info(self,bar_dict:dict,desc:Optional[str]=None,postfix_desc:Optional[str]=None,**postfix_dict):
        if desc is not None:
            bar_dict["desc"].append(desc)
        if postfix_desc is not None:
            bar_dict["postfix_desc"].append(postfix_desc)
        if postfix_dict is not None:
            bar_dict["postfix_dict"].update(postfix_dict)
        
    def add_epoch_bar_info(self,desc:Optional[str]=None,postfix_desc:Optional[str]=None,**postfix_dict):
        """
        Add information to the epoch
        
        Args:
            desc (Optional[str]): Description of the epoch. Will be cleared after each epoch
            postfix_desc (Optional[str]): Postfix description of the epoch. Will not be cleared after each epoch
            **postfix_dict: Postfix dictionary of the epoch. Will not be cleared after each epoch.
        """
        
        self._add_bar_info(self._epoch_bar_msg,desc,postfix_desc,**postfix_dict)
    
    def add_train_it_bar_info(self,desc:Optional[str]=None,postfix_desc:Optional[str]=None,**postfix_dict):
        """
        Add information to the training iteration

        Args:
            desc (Optional[str]): Description of the training iteration. Will be cleared after each iteration
            postfix_desc (Optional[str]): Postfix description of the training iteration. Will not be cleared after each iteration
            **postfix_dict: Postfix dictionary of the training iteration. Will not be cleared after each iteration.
        """
        self._add_bar_info(self._train_it_bar_msg,desc,postfix_desc,**postfix_dict)
        
    def add_validation_it_bar_info(self,desc:Optional[str]=None,postfix_desc:Optional[str]=None,**postfix_dict):
        """
        Add information to the validation iteration
        
        Args:
            desc (Optional[str]): Description of the validation iteration. Will be cleared after each iteration
            postfix_desc (Optional[str]): Postfix description of the validation iteration. Will not be cleared after each iteration
            **postfix_dict: Postfix dictionary of the validation iteration. Will not be cleared after each iteration
        """
        
        self._add_bar_info(self._validation_it_bar_msg,desc,postfix_desc,**postfix_dict)
        
    def _refresh_bar(self,bar:Iterable,bar_dict:dict):
        if isinstance(bar,tqdm):
            bar_dict["desc"].append(bar_dict["default_desc"])
            bar.set_description(", ".join(bar_dict["desc"]))
            postfix_desc = []
            for key,value in bar_dict["postfix_dict"].items():
                if isinstance(value,float):
                    postfix_desc.append("{}: {:.4e}".format(key,value))
                else:
                    postfix_desc.append("{}: {}".format(key,value))
            postfix_desc = ", ".join(postfix_desc)
            postfix_desc += ", ".join(bar_dict["postfix_desc"])
            bar.set_postfix_str(postfix_desc)
            # only desc will not be reset
            bar_dict.update(desc=[])

    def _refresh_epoch_bar(self):
        self._refresh_bar(self._p_bar_epoch,self._epoch_bar_msg)
    
    def _refresh_train_it_bar(self):
        self._refresh_bar(self._p_bar_train_it,self._train_it_bar_msg)
    
    def _refresh_validation_it_bar(self,):
        self._refresh_bar(self._p_bar_validation_it,self._validation_it_bar_msg)
    
    def update_epoch_bar_position(self,position:int=1):
        if isinstance(self._p_bar_epoch,tqdm):
            self._p_bar_epoch.update(position)
            
    def configure_epoch_bar(self,default_desc:Optional[str]=None):
        if default_desc is not None:
            self._epoch_bar_msg["default_desc"]=default_desc
        self._p_bar_epoch = self.rank_zero_tqdm(None, 
                                                 total=self.configs.num_epochs,
                                                 initial=self.current_epoch,
                                                 unit="epoch",
                                                 desc=self._epoch_bar_msg["default_desc"])
        return self._p_bar_epoch
    
    def configure_train_it_bar(self,train_loader:DataLoader,
                               default_desc:Optional[str]=None):
        if default_desc is not None:
            self._train_it_bar_msg["default_desc"]=default_desc
        if self.configs.show_iteration_bar:
            self._p_bar_train_it=self.rank_zero_tqdm(train_loader,desc=self._train_it_bar_msg["default_desc"],leave=False)
        else:
            self._p_bar_train_it=train_loader
        return self._p_bar_train_it
    
    def configure_validation_it_bar(self,validation_loader:DataLoader,
                                    default_desc:Optional[str]=None):
        if default_desc is not None:
            self._validation_it_bar_msg["default_desc"]=default_desc
        if self.configs.show_iteration_bar:
            self._p_bar_validation_it=self.rank_zero_tqdm(validation_loader,desc=self._validation_it_bar_msg["default_desc"],leave=False)
        else:
            self._p_bar_validation_it=validation_loader
        return self._p_bar_validation_it