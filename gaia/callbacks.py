from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.tensorboard import writer

class WriteGraph(Callback):
    def on_fit_start(self, trainer, pl_module):
        # with torch.no_grad():
        x = torch.ones(1,pl_module.model.input_size).to(pl_module.device)

        sample = x

        if "history" in pl_module.hparams.model_config["model_type"].lower():
            y = torch.ones(1,pl_module.model.memory_size).to(pl_module.device)
            sample = (x,y)


        trainer.logger.experiment.add_graph(pl_module.model,sample,verbose=True)