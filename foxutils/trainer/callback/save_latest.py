from ._basis import Callback
import os

class SaveLatestCallback(Callback):
    
    def __init__(self,
                 latest_checkpoint_frequency:int=1) -> None:   
        """
        Save the latest checkpoint every latest_checkpoint_frequency epochs.

        Args:
            latest_checkpoint_frequency (int, optional): The frequency of saving the latest checkpoint. Defaults to 1.
        """        
        super().__init__()          
        self.latest_checkpoint_frequency=latest_checkpoint_frequency

    def on_epoch_end(self,epoch_idx:int):
        if not self.trainer.should_save_ckpt and self.latest_checkpoint_frequency!=0:
            if (epoch_idx+1)%self.latest_checkpoint_frequency==0:
                self.trainer.save(self.trainer.state,"latest.ckpt")
    
    def on_train_end(self):
        latest_ckpt_path=os.path.join(self.trainer.ckpt_dir,"latest.ckpt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)