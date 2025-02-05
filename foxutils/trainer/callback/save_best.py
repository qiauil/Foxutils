from ._basis import Callback
import os
import numpy as np
class SaveBestCallback(Callback):
    
    def __init__(self,
                 num_best_validation=3) -> None:   
        """
        A callback to save the best checkpoints based on the validation loss.
        
        Args:
            num_best_validation (int, optional): The number of best checkpoints to save. Defaults to 3.
        """      
        super().__init__()          
        self.num_best_validation=num_best_validation
        self._best_epochs=[]
        self._loss_at_epochs=[]
    
    def _record_current_epoch(self,epoch_id:int,loss:float):
        self._loss_at_epochs.append(loss)
        self._best_epochs.append(epoch_id)
        indexs=np.argsort(self._loss_at_epochs)
        self._loss_at_epochs=[self._loss_at_epochs[i] for i in indexs]
        self._best_epochs=[self._best_epochs[i] for i in indexs]
        droped_epoch=None
        droped_loss=None
        if len(self._best_epochs)>self.num_best_validation:
            droped_epoch=self._best_epochs.pop()
            droped_loss=self._loss_at_epochs.pop()
        if droped_epoch != epoch_id:
            return epoch_id in self._best_epochs,droped_epoch,droped_loss
        else:
            return epoch_id in self._best_epochs, None, None
    
    def on_validation_epoch_end(self, epoch_idx):
        if hasattr(self.trainer,"validation_epoch_loss"):
            if self.trainer.validation_epoch_loss is not None:
                current_in_best,droped_epoch,droped_loss=self._record_current_epoch(epoch_idx,self.trainer.validation_epoch_loss)
                if droped_epoch is not None:
                    os.remove(os.path.join(self.trainer.ckpt_dir,self._format_ckpt_name(droped_epoch,droped_loss)))
                if current_in_best:
                    self.trainer.save(self.trainer.state,self._format_ckpt_name(epoch_idx,self.trainer.validation_epoch_loss))
            
                
    def _format_ckpt_name(self,epoch_id,loss):
        return f"epoch_{epoch_id}_loss_{loss}.ckpt"