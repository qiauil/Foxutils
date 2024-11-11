from .callback_abc import Callback
import torch
from typing import Optional, List, Tuple, Dict, Union, Iterable
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support, _device_has_foreach_support

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
        Total norm of the parameter gradients (viewed as a single vector).
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

class EmaGradClip(Callback):

    def __init__(self, 
                 trainer, 
                 ) -> None:
        super().__init__(trainer)
        trainer.add_config_item(
            name='emagc_grad_coef1',
            group="ema_grad_clip",
            default_value=0.9,
            value_type=float,
            description='Exponential moving average coefficient for EMA gradient norm recording'
        )
        trainer.add_config_item(
            name='emagc_grad_coef2',
            group="ema_grad_clip",
            default_value=0.99,
            value_type=float,
            description='Exponential moving average coefficient for EMA gradient norm recording'
        )
        trainer.add_config_item(
            name='emagc_max_norm_ratio',
            group="ema_grad_clip",
            default_value=2.0,
            value_type=float,
            description='Critical ratio for gradient clipping'
        )
        trainer.add_config_item(
            name='emagc_clip_norm_ratio',
            group="ema_grad_clip",
            default_value=1.1,
            value_type=float,
            description='Ratio for gradient clipping'
        )
        trainer.add_config_item(
            name='log_clip_info',
            group="ema_grad_clip",
            default_value=True,
            value_type=bool,
            description='whether to log the clip info'
        )
        self._grad_norm_ema1=0.0
        self._grad_norm_ema2=0.0
        self.ema_index=0
   
    def _get_norm(self):
        total_norm = 0.0
        for p in self.trainer.model.parameters():
            if p.grad is not None and p.requires_grad:
                total_norm += p.grad.data.item() ** 2
        return total_norm ** 0.5

    def _record_norm(self,new_norm:float):
        self.ema_index+=1
        self._grad_norm_ema1=self.trainer.configs.emagc_grad_coef1*self._grad_norm_ema1+(1-self.trainer.configs.emagc_grad_coef1)*new_norm
        self._grad_norm_ema2=self.trainer.configs.emagc_grad_coef2*self._grad_norm_ema2+(1-self.trainer.configs.emagc_grad_coef2)*new_norm

    @property
    def _current_ema1(self):
        return self._grad_norm_ema1/(1-self.trainer.configs.emagc_grad_coef1**self.ema_index)

    @property
    def _current_ema2(self):
        return self._grad_norm_ema2/(1-self.trainer.configs.emagc_grad_coef2**self.ema_index)

    def on_before_optimizer_step(self):
        if self._grad_norm_ema2==0.0:
            total_norm, clipped = _clip_grad_norm(self.trainer.model.parameters(), 
                                                         max_norm=10000,
                                                         clip_norm=1,)
        else:
            total_norm, clipped = _clip_grad_norm(
                self.trainer.model.parameters(),
                max_norm=self.trainer.configs.emagc_max_norm_ratio*self._current_ema2,
                clip_norm=self.trainer.configs.emagc_clip_norm_ratio*self._current_ema1
            )
        norm=self.trainer.configs.emagc_clip_norm_ratio*self._current_ema1 if clipped else total_norm
        self._record_norm(norm)
        if self.trainer.configs.log_clip_info:
            self.trainer.fabric.log("grad_clip/ori_grad_norm",total_norm,step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/ema_grad_norm_1",self._current_ema1,step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/ema_grad_norm_2",self._current_ema2,step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/is_clipped",int(clipped),step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/real_grad_norm",norm,step=self.trainer.global_step)
        
class ConstantGradClip(Callback):
    
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        trainer.add_config_item(
            name='max_norm',
            group="grad_clip",
            default_value=1.0,
            value_type=float,
            description='Maximum norm for gradient clipping'
        )
        trainer.add_config_item(
            name='log_clip_info',
            group="ema_grad_clip",
            default_value=True,
            value_type=bool,
            description='whether to log the clip info'
        )
        
    def on_before_optimizer_step(self):
        total_norm, clipped=_clip_grad_norm(self.trainer.model.parameters(), 
                        self.trainer.configs.max_norm,
                        self.trainer.configs.max_norm)
        if self.trainer.configs.log_clip_info:
            self.trainer.fabric.log("grad_clip/ori_grad_norm",total_norm,step=self.trainer.global_step)
            norm = self.trainer.configs.max_norm if clipped else total_norm
            self.trainer.fabric.log("grad_clip/real_grad_norm",norm,step=self.trainer.global_step)
            self.trainer.fabric.log("grad_clip/is_clipped",int(clipped),step=self.trainer.global_step)