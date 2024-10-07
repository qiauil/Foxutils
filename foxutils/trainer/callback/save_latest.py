from .callback_abc import Callback
import os

class SaveLatestCallback(Callback):
    
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        trainer.add_config_item("save_latest_checkpoint",
                        group="control",
                        value_type=bool,
                        default_value=True,
                        description="Whether to save the latest checkpoint. It will overwrite the checkpoint in previous epoch. It is helpful for resuming training.")            
        trainer.add_config_item("latest_checkpoint_frequency",
                        group="control",
                        value_type=int,
                        default_value=1,
                        description="The frequency of saving the latest checkpoint. Only effective when save_latest_checkpoint is True.") 


    def on_epoch_end(self,epoch_idx:int):
        if self.trainer.configs.save_latest_checkpoint and not self.trainer.should_save_ckpt:
            self.trainer.save(self.trainer.state,"latest.ckpt")
    
    def on_train_end(self):
        latest_ckpt_path=os.path.join(self.trainer.ckpt_dir,"latest.ckpt")
        if self.trainer.configs.save_latest_checkpoint and os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)