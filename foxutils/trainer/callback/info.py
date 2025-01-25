from ._basis import Callback
import os
import numpy as np
class InfoCallback(Callback):
    
    def __init__(self) -> None:
        super().__init__()
    
    def on_train_start(self):
        """
        self.trainer.info("="*100)
        self.trainer.info("Training configurations:")
        self.trainer.info("="*100)
        output_dict=self.trainer.str_dict(False,sort=True)
        para_group={}
        for key,value in self.trainer._configs_feature.items():
            if value["group"] not in para_group.keys():
                para_group[value["group"]]=""
            para_str=f"    {key}:{output_dict[key]}{os.linesep}"
            para_group[value["group"]]+=para_str
        for key,value in para_group.items():
            self.trainer.info(key+":")
            self.trainer.info(value)
        """
        self.trainer.info("="*100)
        self.trainer.info("Training input:")
        self.trainer.info("="*100)
        nn_parameters = filter(lambda p: p.requires_grad, self.trainer.model.parameters())
        self.trainer.info("number of trainable model parameters: {}".format(sum([np.prod(p.size()) for p in nn_parameters])))
        self.trainer.info(f"training data set has {self.trainer.len_train_dataset} samples")
        self.trainer.info(f"number of training iterations: {len(self.trainer.train_loader)*self.trainer.configs.num_epochs}")
        if self.trainer.len_val_dataset>0:
            self.trainer.info(f"validation data set has {self.trainer.len_val_dataset} samples")
        else:
            self.trainer.info("no validation data set")
        self.trainer.info(f"actual training device: {self.trainer.fabric.device}")
        self.trainer.info("="*100)
        if self.trainer.configs.compile_model:
            self.trainer.info("You requested to compile the model, the model will be compiled at the first forward pass. Thus the first training iteration might be slow.")
        self.trainer.info("Training Start")