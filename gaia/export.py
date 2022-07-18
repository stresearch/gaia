from collections import OrderedDict
import torch
from gaia.models import TrainingModel
from gaia.training import get_checkpoint_file
import os

class ModelForExport(torch.nn.Module):
    def __init__(self, training_model, input_order, output_order):
        super().__init__()
        self.input_normalize = training_model.input_normalize
        self.output_normalize = training_model.output_normalize
        self.model = training_model.model

        if input_order is not None:
            input_order_index = OrderedDict()
            i = 0
            
            for k in input_order:
                s,e  = training_model.hparams.input_index[k]
                v_size = e - s
                input_order_index[k] = (i,i + v_size)
                i = i + v_size

            order = torch.cat([torch.arange(*input_order_index[k]) for k in training_model.hparams.input_index.keys()])
        else:
            order = None

        self.register_buffer("input_order",order)


        if output_order is not None:
            order = torch.cat([torch.arange(*training_model.hparams.output_index[k]) for k in output_order])
        else:
            order = None
        
        self.register_buffer("output_order",order)

        
    def forward(self,x):

        if self.input_order is not None:
            x = x[:,self.input_order,...]

        x = self.input_normalize(x)
        
        y = self.model(x)
        y = self.output_normalize(y, normalize=False)

        if self.output_order is not None:
            y = y[:,self.output_order,...]

        return y


def export(model_dir, export_name, inputs=None, outputs=None):

    model = TrainingModel.load_from_checkpoint(
                get_checkpoint_file(model_dir), map_location = "cpu",
            ).eval()

    
            
    model_for_export = ModelForExport(model, inputs, outputs).eval()
    #TODO dont hard code this
    input_dim = list(model.hparams.input_index.values())[-1][-1]
    example = torch.rand(10,input_dim)
    out = model_for_export(example)
    traced_script_module = torch.jit.trace(model_for_export, example)
    traced_script_module.save(os.path.join(model_dir,export_name))

    out_traced = traced_script_module(example)

    with torch.no_grad():
        print((out_traced - out).norm())



