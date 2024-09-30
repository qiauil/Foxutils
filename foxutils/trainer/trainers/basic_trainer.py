

# init_module
# print

import logging,os,importlib

import torch.nn as nn

from ..mixin import *
from ..callback import *
from ..lr_lambda import *
from ..callback import TimeSummaryCallback,InfoCallback
from ..postprocess import PostProcessor

from lightning.fabric import Fabric,seed_everything
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from typing import List,Any,Dict,Union,Optional,Tuple
from warnings import warn

class Trainer(TrainConfigMixin,CallbackMixin,ProgressBarMixin):
    callbacks: List[Callback]
    fabric_plugins: List[Any]
    postprocessors: List[PostProcessor]
    state: Dict[str,Any]
    global_step: int=0
    current_epoch: int=0
    should_stop: bool=False
    num_train_loop_called: int=0
    num_validation_loop_called: int=0
    len_train_dataset: int
    len_val_dataset: int
    
    fabric: Fabric
    logger: Union[TensorBoardLogger, CSVLogger, Any]
    _info_recorder: logging.Logger
    train_loader: DataLoader
    validation_loader: DataLoader
    
    run_dir: str
    record_path: str
    config_path: str
    logger_dir: str
    ckpt_dir: str
    ckpt_path: str
    model_structure_path:str
    final_weights_path: str

    def __init__(self) -> None:
        self.register_configs()
        self.postprocessors = []
        self.callbacks = []
        self.fabric_plugins = []

    def info(self, msg):
        if self.fabric.local_rank == 0:
            if not hasattr(self,"_info_recorder"):
                raise RuntimeError("The info recorder is not configured. Please call `configure_info_recorder` before using the `info` method.")
            self._info_recorder.info(msg)
            if not self.configs.run_in_silence:
                print(msg)
    
    def warn(self, msg):
        if self.fabric.local_rank == 0:
            if not hasattr(self,"_info_recorder"):
                raise RuntimeError("The info recorder is not configured. Please call `configure_info_recorder` before using the `warn` method.")
            self._info_recorder.warning(msg)
            warn(msg)

    def configure_info_recorder(self):
        if self.fabric.local_rank == 0:
            self._info_recorder = logging.getLogger(self.configs.run_name)
            disk_handler = logging.FileHandler(filename=self.record_path, mode='a')
            disk_handler.setFormatter(logging.Formatter(fmt="%(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self._info_recorder.addHandler(disk_handler)

    def configure_callbacks(self):
        self.callbacks = [self, InfoCallback(self), TimeSummaryCallback(self)]

    def configure_fabric_plugins(self):
        self.fabric_plugins = []
    
    def configure_postprocessors(self):
        self.postprocessors = []
    
    def configure_loggers(self):
        if self.configs.logger == "TensorBoard":
            self.logger = TensorBoardLogger(root_dir=self.logger_dir,
                                            name=self.configs.run_name,
                                            version=self.configs.version,
                                            **self.configs.logger_configs.to_dict())
        elif self.configs.logger == "CSV":
            self.logger = CSVLogger(root_dir=self.logger_dir,
                                    name=self.configs.run_name,
                                    version=self.configs.version,
                                    **self.configs.logger_configs.to_dict())
        else:
            raise ValueError("Unsupported logger: {}".format(self.configs.logger))
        """
        try:
            os.symlink(
                src=os.path.abspath(os.path.join(self.logger_dir, self.configs.run_name, self.configs.version)),
                dst=os.path.abspath(os.path.join(self.run_dir, "logs")),
                target_is_directory=True
            )
        except Exception as e:
            pass
        """

    def configure_fabric(self):
        self.fabric = Fabric(
            accelerator=self.configs.device_type,
            strategy=self.configs.multi_devices_strategy,
            devices=self.configs.num_id_devices,
            num_nodes=self.configs.num_nodes,
            precision=self.configs.precision,
            plugins=self.fabric_plugins if len(self.fabric_plugins)>0 else None,
            callbacks=self.callbacks,
            loggers=self.logger,
        )
        self.fabric.launch()
        
    def configure_dataloader(self, 
                            train_dataset:Any, 
                            validation_dataset:Optional[Any]=None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Setup dataloaders for training and validation datasets. Recommend to override this method if you want to use custom dataloader.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            validation_dataset (torch.utils.data.Dataset): The validation dataset.

        Returns:
            Tuple[DataLoader, Optional[DataLoader]]: The training dataloader and validation dataloader.
        """

        train_loader = self.fabric.setup_dataloaders(DataLoader(train_dataset, 
                                                    batch_size=self.configs.batch_size_train, 
                                                    **self.configs.train_dataloader_configs.to_dict(),
                                                    drop_last=False
                                                    ))
        self.len_train_dataset = len(train_loader.dataset)
        if validation_dataset is not None:
            validation_loader = self.fabric.setup_dataloaders(DataLoader(validation_dataset, 
                                    batch_size=self.configs.batch_size_val, 
                                    **self.configs.validation_dataloader_configs.to_dict(),
                                    drop_last=False
                                    ))
            self.len_val_dataset = len(validation_loader.dataset)
        else:
            validation_loader = None
        return train_loader, validation_loader
    
    def configure_optimizer(self,model:nn.Module):
        return getattr(importlib.import_module("torch.optim"),
                       self.configs.optimizer)(
                                                params=model.parameters(),
                                                lr=self.configs.lr,
                                                )

    def configure_lr_scheduler(self,optimizer):
        """
        Get the learning rate scheduler based on the configuration.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
        
        Raises:
            ValueError: If the learning rate scheduler is not supported.
        """
        
        if self.configs.lr_scheduler=="cosine":
            lambda_func = get_cosine_lambda
        elif self.configs.lr_scheduler=="linear":
            lambda_func = get_linear_lambda
        elif self.configs.lr_scheduler=="constant":
            lambda_func = get_constant_lambda
        else:
            raise ValueError("Learning rate scheduler '{}' not supported".format(self.configs.lr_scheduler))
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lambda_func(initial_lr=self.configs.lr,
                                                             final_lr=self.configs.final_lr,
                                                             epochs=self.configs.num_epochs,
                                                             warmup_epoch=self.configs.warmup_epoch))
        
    def _set_paths(self,ckpt_path:str):
        self.run_dir = os.path.join(self.configs.project_path, self.configs.run_name, self.configs.version)
        self.record_path = os.path.join(self.run_dir, "event.log")
        self.logger_dir = os.path.join(self.run_dir, "logs",)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoint")
        self.config_dir = os.path.join(self.run_dir, "configs")
        self.model_structure_path=os.path.join(self.run_dir,"model_structure.pt")
        self.final_weights_path=os.path.join(self.run_dir, "final_weights.ckpt")
        latest_ckpt_path=None
        if os.path.exists(self.run_dir):
            if not os.path.exists(self.config_dir):
                raise RuntimeError("The run directory already exists, but the config file does not exist. Can not restart training. Please check the run directory.")
            latest_ckpt_ids = [int(x.split(".")[0].split("_")[-1]) for x in os.listdir(self.ckpt_dir)]
            if len(latest_ckpt_ids)!=0:
                latest_ckpt_ids = max(latest_ckpt_ids)
                if latest_ckpt_ids >= self.configs.num_epochs:
                    raise RuntimeError("The run directory already exists, and the number of checkpoints exceeds the number of epochs. Can not restart training. Please check the run directory.")
                else:
                    latest_ckpt_path = os.path.join(self.ckpt_dir, "epoch_{}.ckpt".format(latest_ckpt_ids))
        for dir_i in [self.run_dir, self.ckpt_dir, self.logger_dir,self.config_dir]:
            os.makedirs(dir_i,exist_ok=True)
        if ckpt_path and latest_ckpt_path:
            raise RuntimeError("Detected unfinished training in the run directory, but also provided a different checkpoint path. Please unset the `ckpt_path` or specify different `project_path`/`run_name`.")
        if latest_ckpt_path and os.path.exists(self.final_weights_path):
            raise RuntimeError("Found final weights. There existed a finished training. Please check the run directory.")
        if not (not ckpt_path and not latest_ckpt_path):
            self.ckpt_path=ckpt_path if ckpt_path else latest_ckpt_path  
        else:
            self.ckpt_path=None

    def _compile_model(self,model:nn.Module):
        if self.configs.compile_model:
            if not self.train_loader.drop_last:
                raise ValueError("Can not compile the model since the training dataloader does not drop the last batch. Please set `drop_last=True` in the training dataloader. Note that compile model need the input shape of the model unchanged during training.")
            if self.len_val_dataset!=0:
                if not self.validation_loader.drop_last:
                    raise ValueError("Can not compile the model since the validation dataloader does not drop the last batch. Please set `drop_last=True` in the validation dataloader. Note that compile model need the input shape of the model unchanged during training.")
                if self.configs.batch_size_val!=self.configs.batch_size_train:
                    raise ValueError("Can not compile the model since the batch size of the training dataloader is different from the validation dataloader. Please set the same batch size for both dataloaders. Note that compile model need the input shape of the model unchanged during training.")
            return torch.compile(model)
        return model    

    def _call_postprocessors(self):
        p_bar=self.rank_zero_tqdm(self.postprocessors,desc="Postprocessing")
        for postprocessor in p_bar:
            if isinstance(p_bar,tqdm):
                p_bar.set_description(f"Postprocessing {postprocessor.processor_name}")
            postprocess_dir=os.path.join(self.run_dir,"postprocess",postprocessor.processor_name)
            os.makedirs(postprocess_dir,exist_ok=True)
            return_value=postprocessor.run(model=self.model,
                              config_dict=self.str_dict(),
                              working_path=self.run_dir,
                              fabric=self.fabric)
            if return_value is not None:
                postprocessor.rank_zero_save(return_value,
                                             os.path.join(postprocess_dir,"output.pt"),
                                             self.fabric)

    def _set_random_seed(self,seed:int):
        seed_everything(self.configs.random_seed,verbose=False,workers=True)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self,
              model:nn.Module,
              train_dataset,
              validation_dataset:Optional[Any]=None,
              ckpt_path:str=None,
              config_path_dir:Optional[str]=None,
              **kwargs): 
        self.global_step=0
        self.current_epoch=0
        # read configs from yaml file and keyword arguments
        if config_path_dir is not None:
            self.read_configs_from_yaml(config_path_dir)
        self.set_config_items(**kwargs)
        self._set_paths(ckpt_path)
        self.save_configs_to_yaml(self.config_dir)
        if self.configs.save_model_structure:
            torch.save(model,self.model_structure_path)
        # setup training environment
        self._set_random_seed(self.configs.random_seed)
        self.configure_callbacks()
        self.configure_fabric_plugins()
        self.configure_postprocessors()
        self.configure_loggers()
        self.configure_fabric()
        self.configure_info_recorder()
        self.train_loader, self.validation_loader = self.configure_dataloader(train_dataset, validation_dataset)
        optimizer=self.configure_optimizer(model)
        self.lr_scheduler=self.configure_lr_scheduler(optimizer)
        self.model, self.optimizer = self.fabric.setup(self._compile_model(model), optimizer)
        self.optimizer.zero_grad()
        # read checkpoint
        self.state=dict(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
        self.fabric.call("on_state_register", self.state)
        if self.ckpt_path:
            self.info(f"Unfinished training detected. Loading checkpoint from {self.ckpt_path}")
            self.load(self.state, self.ckpt_path)
        # initialize training control
        self.fabric.call("on_train_start")
        self.configure_epoch_bar()    
        self.should_stop=False
        # main loop
        while not self.should_stop:
            self.current_epoch += 1
            self.fabric.call("on_train_epoch_start",epoch_idx=self.current_epoch)
            self.train_loop()
            self.fabric.call("on_train_epoch_end",epoch_idx=self.current_epoch)
            if self.current_epoch % self.configs.validation_frequency == 0:
                self.fabric.call("on_validation_epoch_start",epoch_idx=self.current_epoch)
                self.validation_loop()
                self.fabric.call("on_validation_epoch_end",epoch_idx=self.current_epoch)
            if self.validation_loader is not None and self.current_epoch % self.configs.checkpoint_save_frequency == 0 and self.current_epoch != self.configs.num_epochs:
                self.save(self.state)
            self.update_epoch_bar_position()
            self._refresh_epoch_bar()
            self.should_stop = (self.current_epoch >= self.configs.num_epochs)
        self.fabric.save(self.final_weights_path, {"model":self.model})
        self._call_postprocessors()
        self.fabric.call("on_train_end")

    def train_loop(self,
                   epoch_loss_tag:str="losses/train_epoch",
                   iteration_loss_tag:str="losses/train_iteration",
                   epoch_bar_tag:str="train_loss",
                   it_bar_tag:str="loss",
                   default_it_bar_desc:str="Finished Training Iteration"):
        self.model.train()
        losses=[]
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.add_epoch_bar_info(lr=current_lr)
        self.fabric.log("lr",current_lr,step=self.current_epoch)
        for batch_idx,batch in enumerate(self.configure_train_it_bar(self.train_loader,default_desc=default_it_bar_desc)):
            if self.should_stop:
                break
            self.global_step += 1
            self.fabric.call("on_train_batch_start",batch=batch,batch_idx=batch_idx)
            loss=self.train_step(self.model,batch,batch_idx)
            if loss is not None:
                if isinstance(loss,torch.Tensor): #IF you don't want to backpropagate the loss, you can return None or a float value loss
                    self.fabric.backward(loss)
                    losses.append(loss.item())
                else:
                    losses.append(loss)
                self.add_train_it_bar_info(**{it_bar_tag:losses[-1]})
                if self.configs.log_iteration_loss:
                    self.fabric.log(iteration_loss_tag,losses[-1],step=self.global_step)
            self.fabric.call("on_train_batch_end",batch=batch,batch_idx=batch_idx,batch_loss=loss)
            if self.global_step % self.configs.grad_accum_steps  == 0:
                self.fabric.call("on_before_optimizer_step")
                self.optimizer.step()
                #self.fabric.call("on_after_optimizer_step")
                # "on_after_optimizer_step" is automatically called by the fabric
                self.fabric.call("on_before_zero_grad")
                self.optimizer.zero_grad()
            self._refresh_train_it_bar()
        self.lr_scheduler.step()
        if len(losses)!=0:
            losses = sum(losses)/len(losses)
            self.fabric.log(epoch_loss_tag,losses,step=self.current_epoch)
            self.add_epoch_bar_info(**{epoch_bar_tag:losses})
        self.num_train_loop_called += 1

    def validation_loop(self,
                        loss_tag:str="losses/validation_epoch",
                        epoch_bar_tag:str="validation_loss",
                        it_bar_tag:str="loss",
                        default_it_bar_desc:str="Finished Validation Iteration"):
        self.model.eval()
        losses=[]
        with torch.no_grad():
            for batch_idx,batch in enumerate(self.configure_validation_it_bar(self.validation_loader,default_desc=default_it_bar_desc)):
                if self.should_stop:
                    break
                self.fabric.call("on_validation_batch_start",batch=batch,batch_idx=batch_idx)
                loss=self.validation_step(self.model,batch,batch_idx)
                if loss is not None:
                    losses.append(loss.item())
                    self.add_validation_it_bar_info(**{it_bar_tag:losses[-1]})
                self.fabric.call("on_validation_batch_end",batch=batch,batch_idx=batch_idx,batch_loss=loss)
                self._refresh_validation_it_bar()
            if len(losses)!=0:
                losses = sum(losses)/len(losses)
                self.fabric.log(loss_tag,losses,step=self.current_epoch)
                self.add_epoch_bar_info(**{epoch_bar_tag:losses})
        self.num_validation_loop_called += 1

    def train_step(self,model:nn.Module,batch:Any,batch_idx:int):
        inputs,targets=batch
        return torch.nn.functional.mse_loss(model(inputs),targets)
    
    def validation_step(self,model:nn.Module,batch:Any,batch_idx:int):
        inputs,targets=batch
        return torch.nn.functional.mse_loss(model(inputs),targets)

    @property
    def device(self):
        if hasattr(self,"fabric"):
            return self.fabric.device
        else:
            return torch.device(self.configs.device_type)

    def load(self, state: Optional[Dict], path: str) -> None:
        if state is None:
            state = {}
        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")
        if remainder:
            self.warn(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Dict]) -> None:
        if state is None:
            state = {}
        state.update(global_step=self.global_step, current_epoch=self.current_epoch)
        self.fabric.save(os.path.join(self.ckpt_dir, f"epoch_{self.current_epoch}.ckpt"), state)
        
    def add_callbacks(self,callbacks:Union[Callback,Sequence[Callback]]):
        if not isinstance(callbacks,Sequence):
            callbacks = [callbacks]
        self.callbacks.extend(callbacks)
    
    def add_fabric_plugins(self,plugins:Union[Any,Sequence[Any]]):
        if not isinstance(plugins,Sequence):
            plugins = [plugins]
        self.fabric_plugins.extend(plugins)
        
    def add_postprocessors(self,postprocessors:Union[PostProcessor,Sequence[PostProcessor]]):
        if not isinstance(postprocessors,Sequence):
            postprocessors = [postprocessors]
        self.postprocessors.extend(postprocessors)