
from .callback_abc import Callback
import torch
import copy
import threading
import contextlib
import os

@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, ema_coef, ema_step):
    torch._foreach_mul_(ema_model_tuple, ema_coef)
    torch._foreach_add_(
        ema_model_tuple, current_model_tuple, alpha=(1.0 - ema_coef),
    )
    torch._foreach_mul(ema_model_tuple, 1.0 / (1.0 - ema_coef ** (ema_step)))

def run_ema_update_cpu(ema_model_tuple, current_model_tuple,ema_coef, ema_step, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()
    ema_update(ema_model_tuple, current_model_tuple, ema_coef, ema_step)     

class EMAWeightsCallback(Callback):
    
    def __init__(self, 
                 trainer, 
                 ema_coef:float=0.9,
                 ema_weights_update_freq:int=1,
                 do_ema_validation:bool=True,
                 ) -> None:
        super().__init__(trainer)
        self.ema_coef = ema_coef
        self.ema_params = ()
        self.ema_step = 0
        self.stream = None
        self.thread = None
        self.ema_weights_update_freq = ema_weights_update_freq
        self.do_ema_validation = do_ema_validation
        self.num_called = 0   
        
    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)
        
    def initialized_model_weights(self):
        return tuple(copy.deepcopy(p.data.detach()).to(self.trainer.device) for p in self.trainer.model.parameters())

    @torch.no_grad()
    def update(self):
        self.join()
        if self.ema_step ==0:
            if any(p.is_cuda for p in self.trainer.model.parameters()):
                self.stream = torch.cuda.Stream()
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())
            
        self.ema_step += 1

        with torch.cuda.stream(self.stream):
            current_weight = tuple(p.data.to(self.trainer.device, non_blocking=True) for p in self.trainer.model.parameters())
            if self.trainer.device.type == 'cuda':
                ema_update(self.ema_params, current_weight,self.ema_coef,self.ema_step)

        if self.trainer.device.type == 'cpu':
            self.thread = threading.Thread(
                target=run_ema_update_cpu, args=(self.ema_params, 
                                                 current_weight,
                                                 self.ema_coef,
                                                 self.ema_step,
                                                 self.stream,),)
            self.thread.start()    

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            for p, ema_p in zip(self.trainer.model.parameters(), self.ema_params):
                self.swap_tensors(p.data, ema_p)
        try:
            yield
        finally:
            if enabled:
                for p, ema_p in zip(self.trainer.model.parameters(), self.ema_params):
                    self.swap_tensors(p.data, ema_p)

    def state_dict(self):
        self.join()
        state_dict = {
            'self.ema_params': self.ema_params,
            'ema_step': self.ema_step,
            'ema_coef': self.ema_coef,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()
        self.ema_params = tuple(param.to(self.trainer.device) for param in copy.deepcopy(state_dict['ema']))
        self.ema_step= state_dict['ema_step']
        self.ema_coef = state_dict['ema_coef']
    
    def on_state_register(self, state_dict: torch.Dict[str, torch.Any]):
        state_dict['ema'] = self.state_dict()
        
    def on_train_start(self):
        if len(self.ema_params) == 0:
            self.ema_params = self.initialized_model_weights()
            
    def on_after_optimizer_step(self,strategy,optimizer):
        self.num_called += 1
        if self.num_called % self.ema_weights_update_freq == 0:
            self.update()
    
    def on_validation_epoch_end(self, epoch_idx: int):
        if self.do_ema_validation:
            self.join()
            with self.swap_ema_weights():
                self.trainer.validation_loop(loss_tag="losses/ema_validation_epoch",
                                             epoch_bar_tag="ema_validation_loss",
                                             it_bar_tag="ema_loss",
                                             default_it_bar_desc="Finished EMA Validation Iteration",
                                             )
                
    def on_train_end(self):
        self.join()
        with self.swap_ema_weights():
            self.trainer.fabric.save(os.path.join(self.trainer.run_dir, "ema_final_weights.ckpt"), {"model":self.trainer.model})
        