from ._basis import Callback
import torch
from typing import Optional, List, Tuple, Dict, Union, Iterable
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support, _device_has_foreach_support
import numpy as np

def _clip_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], 
        max_norm: float,
        clip_norm: float, 
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False, 
        foreach: Optional[bool] = None) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed.emagc_grad_coef2 over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            falle total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device) back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector), whether clipping was applied.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    clip_norm = float(clip_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: List[Tensor] = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clipped=total_norm>max_norm
    if clipped:
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_norm / (total_norm + 1e-6), max=1.0)
        for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
            if (
                (foreach is None and _has_foreach_support(device_grads, device))
                or (foreach and _device_has_foreach_support(device))
            ):
                torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in device_grads:
                    g.mul_(clip_coef_clamped_device)

    return total_norm, clipped

def _grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], 
        norm_type: float = 2.0,
        foreach: Optional[bool] = None) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed.emagc_grad_coef2 over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            falle total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: List[Tensor] = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    return torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

class EMAGradClipCallback(Callback):
    """
    Exponential moving average gradient clipping

    Args:
        emagc_grad_coef1: Exponential moving average coefficient for EMA gradient norm recording
        emagc_grad_coef2: Exponential moving average coefficient for EMA gradient norm recording
        emagc_max_norm_ratio: Critical ratio for gradient clipping
        emagc_clip_norm_ratio: Ratio for gradient clipping
        log_clip_info: whether to log the clip info
    
    """

    def __init__(self, 
                 emagc_grad_coef1:float=0.9,
                    emagc_grad_coef2:float=0.99,
                    emagc_max_norm_ratio:float=2.0,
                    emagc_clip_norm_ratio:float=1.1,
                    log_clip_info:bool=True,
                    warmup_epoch=0) -> None:
        """
        Args:
            emagc_grad_coef1: Exponential moving average coefficient for EMA gradient norm recording
            emagc_grad_coef2: Exponential moving average coefficient for EMA gradient norm recording
            emagc_max_norm_ratio: Critical ratio for gradient clipping
            emagc_clip_norm_ratio: Ratio for gradient clipping
            log_clip_info: whether to log the clip info
        """
        super().__init__()
        self.emagc_grad_coef1=emagc_grad_coef1
        self.emagc_grad_coef2=emagc_grad_coef2
        self.emagc_max_norm_ratio=emagc_max_norm_ratio
        self.emagc_clip_norm_ratio=emagc_clip_norm_ratio
        self.log_clip_info=log_clip_info
        self._grad_norm_ema1=0.0
        self._grad_norm_ema2=0.0
        self.ema_index=0
        self.warmup_epoch=warmup_epoch
   
    def _get_norm(self):
        total_norm = 0.0
        for p in self.trainer.model.parameters():
            if p.grad is not None and p.requires_grad:
                total_norm += p.grad.data.item() ** 2
        return total_norm ** 0.5

    def _record_norm(self,new_norm:float):
        self.ema_index+=1
        self._grad_norm_ema1=self.emagc_grad_coef1*self._grad_norm_ema1+(1-self.emagc_grad_coef1)*new_norm
        self._grad_norm_ema2=self.emagc_grad_coef2*self._grad_norm_ema2+(1-self.emagc_grad_coef2)*new_norm

    @property
    def _current_ema1(self):
        return self._grad_norm_ema1/(1-self.emagc_grad_coef1**self.ema_index)

    @property
    def _current_ema2(self):
        return self._grad_norm_ema2/(1-self.emagc_grad_coef2**self.ema_index)

    def on_before_optimizer_step(self):
        if self.trainer.current_epoch>self.warmup_epoch:
            if self._grad_norm_ema2==0.0:
                total_norm, clipped = _clip_grad_norm(self.trainer.model.parameters(), 
                                                            max_norm=10000,
                                                            clip_norm=1,)
            else:
                total_norm, clipped = _clip_grad_norm(
                    self.trainer.model.parameters(),
                    max_norm=self.emagc_max_norm_ratio*self._current_ema2,
                    clip_norm=self.emagc_clip_norm_ratio*self._current_ema1
                )
            norm=self.emagc_clip_norm_ratio*self._current_ema1 if clipped and self.ema_index!=0 else total_norm
            self._record_norm(norm)
            if self.log_clip_info:
                self.trainer.fabric.log("grad_clip/ori_grad_norm",total_norm,step=self.trainer.global_step)
                self.trainer.fabric.log("grad_clip/ema_grad_norm_1",self._current_ema1,step=self.trainer.global_step)
                self.trainer.fabric.log("grad_clip/ema_grad_norm_2",self._current_ema2,step=self.trainer.global_step)
                self.trainer.fabric.log("grad_clip/is_clipped",int(clipped),step=self.trainer.global_step)
                self.trainer.fabric.log("grad_clip/real_grad_norm",norm,step=self.trainer.global_step)

    def load_state_dict(self, state_dict):
        ori_state=self.state_dict()
        ori_state.update(state_dict)
        self._grad_norm_ema1=ori_state["_grad_norm_ema1"]
        self._grad_norm_ema2=ori_state["_grad_norm_ema2"]
        self.ema_index=ori_state["ema_index"]

    def state_dict(self):
        return {
            "_grad_norm_ema1":self._grad_norm_ema1,
            "_grad_norm_ema2":self._grad_norm_ema2,
            "ema_index":self.ema_index
        }

class EpochSTDGradClipCallback(Callback):
    
    def __init__(self, 
                 clip_ratio:float=1.0,
                 log_clip_info:bool=True,) -> None:
        """
        Gradient clipping based on epoch standard deviation
        
        Args:
            clip_ratio: ratio for gradient clipping
            log_clip_info: whether to log the clip info
        """    
        super().__init__()
        self._clip_ratio=clip_ratio
        self._grad_norms=[]
        self._recorded_epoch=1
        self._std=None
        self._mean=None
        self.log_clip_info=log_clip_info
    
    def state_dict(self):
        return {
            "_grad_norms":self._grad_norms,
            "_recorded_epoch":self._recorded_epoch,
            "_std":self._std,
            "_mean":self._mean
        }
        
    def load_state_dict(self, state_dict):
        ori_state=self.state_dict()
        ori_state.update(state_dict)
        self._grad_norms=ori_state["_grad_norms"]
        self._recorded_epoch=ori_state["_recorded_epoch"]
        self._std=ori_state["_std"]
        self._mean=ori_state["_mean"]
        
        
    def on_before_optimizer_step(self):
        if self.trainer.current_epoch!=self._recorded_epoch:
            self._recorded_epoch=self.trainer.current_epoch
            self._mean=np.mean(self._grad_norms)
            self._std=np.std(self._grad_norms)
            self._grad_norms=[]
        if self._mean is None:
            total_norm=_grad_norm(self.trainer.model.parameters())
            clipped=False
        else:
            total_norm, clipped=_clip_grad_norm(self.trainer.model.parameters(), 
                        max_norm=self._mean+self._clip_ratio*self._std,
                        clip_norm=self._mean)
        total_norm=total_norm.item()
        self._grad_norms.append(total_norm)
        if self.log_clip_info:
            self.trainer.fabric.log("grad_clip/ori_grad_norm",total_norm,step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/is_clipped",int(clipped),step=self.trainer.global_step)
            norm=self._mean if clipped else total_norm
            self.trainer.fabric.log("grad_clip/real_grad_norm",norm,step=self.trainer.global_step)
            if self._std is not None:
                self.trainer.fabric.log("grad_clip/mean_grad_norm",self._mean,step=self.trainer.global_step)
                self.trainer.fabric.log("grad_clip/std_grad_norm",self._std,step=self.trainer.global_step)
        
class ConstantGradClipCallback(Callback):
    
    def __init__(self,
                 max_norm:int=1.0,
                 log_clip_info:bool=True) -> None:
        """
        Constant gradient clipping
        
        Args:
            max_norm: max norm of the gradients
            log_clip_info: whether to log the clip info
        """

        super().__init__()
        self.max_norm=max_norm
        self.log_clip_info=log_clip_info
        
    def on_before_optimizer_step(self):
        total_norm, clipped=_clip_grad_norm(self.trainer.model.parameters(), 
                        self.max_norm,
                        self.max_norm)
        if self.log_clip_info:
            self.trainer.fabric.log("grad_clip/ori_grad_norm",total_norm,step=self.trainer.global_step)
            norm = self.max_norm if clipped else total_norm
            self.trainer.fabric.log("grad_clip/real_grad_norm",norm,step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/is_clipped",int(clipped),step=self.trainer.global_step)