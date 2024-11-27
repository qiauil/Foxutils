#usr/bin/python3
# -*- coding: UTF-8 -*-
from .basic_trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

class PlateauTrainer(Trainer):
    
    """
    Trainer class with Exponential Moving Average (EMA) weights recording
    """
    
    def __init__(self) -> None:
        super().__init__()

    def register_lr_scheduler_configs(self):
        self.add_config_item("model",
                            group="plateau_lr_scheduler",
                             default_value="min",
                             options=["min", "max"],
                             value_type=str,
                             description="In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing")
        self.add_config_item("factor",
                            group="plateau_lr_scheduler",
                             default_value=0.5,
                             value_type=float,
                             description="Factor by which the learning rate will be reduced. new_lr = lr * factor")
        self.add_config_item("patience",
                            group="plateau_lr_scheduler",
                             default_value=10,
                             value_type=int,
                             description="Number of epochs with no improvement after which learning rate will be reduced")
        self.add_config_item("threshold",
                            group="plateau_lr_scheduler",
                             default_value=0.0001,
                             value_type=float,
                             description="Threshold for measuring the new optimum, to only focus on significant changes")
        self.add_config_item("threshold_mode",
                            group="plateau_lr_scheduler",
                             default_value="rel",
                             options=["rel", "abs"],
                             value_type=str,
                             description="In rel mode, dynamic_threshold = best * (1 + threshold) in 'max' mode or best * (1 - threshold) in 'min' mode. In abs mode, dynamic_threshold = best + threshold in 'max' mode or best - threshold in 'min' mode")
        self.add_config_item("cooldown",
                            group="plateau_lr_scheduler",
                             default_value=0,
                             value_type=int,
                             description="Number of epochs to wait before resuming normal operation after lr has been reduced")
        self.add_config_item("min_lr",
                            group="plateau_lr_scheduler",
                             default_value=0.0,
                             value_type=float,
                             description="A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively")
        self.add_config_item("eps",
                            group="plateau_lr_scheduler",
                             default_value=1e-08,
                             value_type=float,
                             description="A small value to avoid division by zero")
        self.add_config_item("plateau_metric",
                             group="plateau_lr_scheduler",
                             default_value="validation",
                             options=["validation", "train"],
                             value_type=str,
                             description="Metric to monitor for plateau")
        self.add_config_item("stop_lr",
                            group="plateau_lr_scheduler",
                             default_value=1e-08,
                             value_type=float,
                             description="Stop training if learning rate is less than or equal to this value")       

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
        return ReduceLROnPlateau(optimizer,
                                 mode=self.configs.model,
                                 factor=self.configs.factor,
                                 patience=self.configs.patience,
                                 threshold=self.configs.threshold,
                                 threshold_mode=self.configs.threshold_mode,
                                 cooldown=self.configs.cooldown,
                                 min_lr=self.configs.min_lr,
                                 eps=self.configs.eps)
        
    def on_train_start(self):
        super().on_train_start()
        if self.validation_loader is None and self.configs.plateau_metric == "validation":
            raise ValueError("Validation loader is required since plateau metric is set to validation")

    def train_loop(self,
                   epoch_loss_tag:str="losses/train_epoch",
                   iteration_loss_tag:str="losses/train_iteration",
                   epoch_bar_tag:str="train_loss",
                   it_bar_tag:str="loss",
                   default_it_bar_desc:str="Finished Training Iteration"):
        # avoid call lr_scheduler.step() here
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
        #self.lr_scheduler.step()
        if len(losses)!=0:
            losses = sum(losses)/len(losses)
            self.fabric.log(epoch_loss_tag,losses,step=self.current_epoch)
            self.add_epoch_bar_info(**{epoch_bar_tag:losses})
            self.epoch_loss=losses
        elif self.configs.plateau_metric == "train":
            raise RuntimeError("No losses recorded for training epoch but plateau metric is set to train")
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
                self.epoch_validation_loss=losses
            elif self.configs.plateau_metric == "validation":
                raise RuntimeError("No losses recorded for validation epoch but plateau metric is set to validation")
        self.num_validation_loop_called += 1
    
    def on_validation_epoch_end(self, epoch_idx: int):
        super().on_validation_epoch_end(epoch_idx)
        if self.configs.plateau_metric == "validation":
            self.lr_scheduler.step(self.epoch_validation_loss)
        elif self.configs.plateau_metric == "train":
            self.lr_scheduler.step(self.epoch_loss)
        else:
            raise ValueError("Plateau metric not supported")
        if any(abs(param_i["lr"]-self.configs.stop_lr)<1e-10 for param_i in self.optimizer.param_groups):
            self.should_stop = True
            self.info("Stopping training as learning rate is less than or equal to stop_lr")          