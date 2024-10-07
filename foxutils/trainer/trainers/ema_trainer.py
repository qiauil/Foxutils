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
        self.add_callbacks(EMAWeightsCallback(trainer=self))
