#usr/bin/python3
# -*- coding: UTF-8 -*-

from ..mixin import CallbackMixin
from ..trainers import Trainer
class Callback(CallbackMixin):
    
    def __init__(self) -> None:
        self._trainer:Trainer = None
    
    def regisiter_trainer(self, trainer:Trainer) -> None:
        self._trainer = trainer

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            raise ValueError("Trainer is not regisitered yet.")
