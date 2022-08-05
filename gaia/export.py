from asyncio.log import logger
from collections import OrderedDict
from pathlib import Path
import torch
from gaia.models import TrainingModel
from gaia.training import get_checkpoint_file
import os
from gaia import get_logger
import yaml


logger = get_logger(__name__)


class ModelForExport(torch.nn.Module):
    def __init__(self, training_model, input_order, output_order):
        super().__init__()
        self.input_normalize = training_model.input_normalize
        self.output_normalize = training_model.output_normalize
        self.model = training_model.model

        if input_order is not None:
            if input_order == list(training_model.hparams.input_index.keys()):
                logger.info("inputs align... don't need to reorder")
                self.reorder_input = False
            else:

                input_order_index = OrderedDict()
                i = 0

                for k in input_order:
                    s, e = training_model.hparams.input_index[k]
                    v_size = e - s
                    input_order_index[k] = (i, i + v_size)
                    i = i + v_size

                order = torch.cat(
                    [
                        torch.arange(*input_order_index[k])
                        for k in training_model.hparams.input_index.keys()
                    ]
                )
                self.register_buffer("input_order", order)
                self.reorder_input = True
        else:
            self.reorder_input = False

        # check if we need to pass thru any inputs

        not_in_output = set(output_order) - set(
            training_model.hparams.output_index.keys()
        )

        self.need_pass_thru = False
        if len(not_in_output) > 0:
            if not_in_output.issubset(set(training_model.hparams.input_index.keys())):
                logger.info(
                    f"{not_in_output} not found in output but found in input... assuming pass thru vars"
                )
                self.need_pass_thru = True
            else:
                raise ValueError(f"{not_in_output} not found in input or output")
        

        if self.need_pass_thru:
            # concat (optionally reordered) input and output and then reorder
            # dim of input
            input_size = list(training_model.hparams.input_index.values())[-1][-1]
            self.reorder_output = True
            index_tensor = []
            for k in output_order:
                if k in training_model.hparams.output_index:
                    s, e = training_model.hparams.output_index[k]
                    # shift by size of input
                    s = s + input_size
                    e = e + input_size
                elif k in training_model.hparams.input_index:
                    s, e = training_model.hparams.input_index[k]
                else:
                    raise ValueError(f"{k} is found in neither input nor output")

                index_tensor.append(torch.arange(s, e))

            order = torch.cat(index_tensor)
            self.register_buffer("output_order", order)

        else:
            logger.info("dont need to pass thru anything")


            if output_order is not None:
                if output_order == list(training_model.hparams.output_index.keys()):
                    logger.info("outputs align... dont need to reorder")
                    self.reorder_output = False
                order = torch.cat(
                    [
                        torch.arange(*training_model.hparams.output_index[k])
                        for k in output_order
                    ]
                )
                self.reorder_output = True
                self.register_buffer("output_order", order)
            else:
                self.reorder_output = False

    def forward(self, x):

        if self.reorder_input:
            x = x[:, self.input_order, ...]

        x_norm = self.input_normalize(x)

        y = self.model(x_norm)
        y = self.output_normalize(y, normalize=False)

        if self.reorder_output:
            if self.need_pass_thru:
                y = torch.cat([x, y], dim=1)
            y = y[:, self.output_order, ...]

        return y


def export(model_dir, export_name, inputs=None, outputs=None):

    model = TrainingModel.load_from_checkpoint(
        get_checkpoint_file(model_dir),
        map_location="cpu",
    ).eval()

    model_for_export = ModelForExport(model, inputs, outputs).eval().requires_grad_(False)
    # TODO dont hard code this
    input_dim = list(model.hparams.input_index.values())[-1][-1]
    example = torch.rand(10, input_dim)
    out = model_for_export(example)
    traced_script_module = torch.jit.trace(model_for_export, example)
    traced_script_module.save(os.path.join(model_dir, export_name))

    out_traced = traced_script_module(example)

    with torch.no_grad():

        logger.info((out_traced - out).norm())
        logger.info(out_traced[0].shape)

        logger.info(out_traced[0])

    if inputs is None:
        inputs = list(model.hparams.input_index.keys())
    if outputs is None:
        outputs = list(model.hparams.output_index.keys())

    open(os.path.join(model_dir, export_name.replace(".pt", "_io.yaml")), "w").write(
        yaml.dump(dict(inputs=inputs, outputs=outputs), indent=2)
    )

    test_file = (Path(model_dir) / "test_results.json")
    if test_file.exists():
        import pandas as pd
        temp = pd.read_json(test_file).T
        temp.index = temp.index.str.replace("test_skill_ave_trunc_","")
        temp.columns = ["metric"]
        temp.round(3).to_csv(os.path.join(model_dir, export_name.replace(".pt", "_test_skill.csv")))

