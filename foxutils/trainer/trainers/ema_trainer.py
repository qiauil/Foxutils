#usr/bin/python3
# -*- coding: UTF-8 -*-
from .basic_trainer import Trainer
from ..callback import EMAWeightsCallback

class EMATrainer(Trainer):
    
    """
    Trainer class with Exponential Moving Average (EMA) weights recording
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    def register_configs(self):
        super().register_configs()
        self.add_config_item(
            name='ema_coef',
            group="ema_weights",
            default_value=0.9,
            value_type=float,
            description='Exponential moving average coefficient for EMA weights recording'
        )
        self.add_config_item(
            name='ema_weights_update_freq',
            group="ema_weights",
            default_value=1,
            value_type=int,
            description='Frequency of EMA weight update. The unit is the number of optimization steps'
        )
        self.add_config_item(
            name='do_ema_validation',
            group="ema_weights",
            default_value=True,
            value_type=bool,
            description='Whether to perform additional validation after EMA weight update'
        )
        
    def configure_callbacks(self):
        super().configure_callbacks()
        self.callbacks.append(
            EMAWeightsCallback(
            trainer=self,
            ema_coef=self.configs.ema_coef,
            ema_weights_update_freq=self.configs.ema_weights_update_freq,
            do_ema_validation=self.configs.do_ema_validation
            )
        )