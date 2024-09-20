#usr/bin/python3
# -*- coding: UTF-8 -*-

from ..mixin import CallbackMixin
class Callback(CallbackMixin):
    
    def __init__(self,trainer) -> None:
        self.trainer = trainer