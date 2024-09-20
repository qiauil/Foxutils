#usr/bin/python3
# -*- coding: UTF-8 -*-
from .callback_abc import Callback
import time
import datetime
import yaml
import os

class TimeSummaryCallback(Callback):
    
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.start_time = 0.0
        self.train_time = 0.0
        self.train_time_accumulated = 0.0
        self.validation_time = 0.0
        self.validation_time_accumulated = 0.0
    
    def on_train_start(self):
        self.start_time = time.time()
    
    def on_train_epoch_start(self, epoch_idx: int):
        self.train_time = time.time()
        
    def on_train_epoch_end(self, epoch_idx: int):
        self.train_time_accumulated += time.time() - self.train_time
        
    def on_validation_epoch_start(self, epoch_idx: int):
        self.validation_time = time.time()
    
    def on_validation_epoch_end(self, epoch_idx: int):
        self.validation_time_accumulated += time.time() - self.validation_time
    
    def on_train_end(self):
        summary=dict(
            total_time=time.time() - self.start_time,
            total_training_time=self.train_time_accumulated,
            total_validation_time=self.validation_time_accumulated,
            average_epoch_training_time=self.train_time_accumulated/self.trainer.num_train_loop_called,
            average_epoch_validation_time=self.validation_time_accumulated/self.trainer.num_validation_loop_called
        )
        summary_str={}
        for key, value in summary.items():
            summary_str[key]=str(datetime.timedelta(seconds=value))
        summary_str.update(efficiency=f'{self.train_time_accumulated/summary["total_time"]:.3%}')
        with open(os.path.join(self.trainer.run_dir,"time_summary.yaml"), "w") as f:
            yaml.dump(summary_str, f)
        for key, value in summary_str.items():
            self.trainer.info(f"{key}: {value}")