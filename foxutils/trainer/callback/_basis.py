#usr/bin/python3
# -*- coding: UTF-8 -*-

from ..mixin import CallbackMixin
#from ..trainers import Trainer
class Callback(CallbackMixin):
    
    def __init__(self) -> None:
        self._trainer = None
    
    def register_trainer(self, trainer) -> None:
        self._trainer = trainer

    @property
    def trainer(self):
        if self._trainer is None:
            raise ValueError("Trainer is not registered yet.")
        return self._trainer
