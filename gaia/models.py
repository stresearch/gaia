from collections import OrderedDict
from typing import List
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from gaia.layers import Normalization


class TrainingModel(LightningModule):
    def __init__(
        self,
        lr: float = 0.0002,
        input_index=None,
        output_index=None,
        model_config=OrderedDict(model_type = "fcn"),
        data_stats=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        model_type = model_config.pop("model_type")
        if model_type == "fcn":
            self.model = FcnBaseline(**model_config)
        elif model_type == "conv":
            self.model = ConvNet1x1(**model_config)
        else:
            raise ValueError("unknown model_type")

        self.input_normalize, self.output_normalize = list(self.setup_normalize())

    def setup_normalize(self):
        stats = torch.load(self.hparams.data_stats)
        for k in ["input_stats", "output_stats"]:
            stats[k]["range"] = stats[k]["max"] - stats[k]["min"]
            stats[k]["std_eff"] = torch.where(
                stats[k]["std"] > 1e-9, stats[k]["std"], stats[k]["range"]
            )
            yield Normalization(stats[k]["mean"], stats[k]["std_eff"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return self.model(x)

    def step(self, batch, mode="train"):
        x, y = batch
        x = self.input_normalize(x)
        y = self.output_normalize(y)
        yhat = self(x)
        loss = OrderedDict()
        loss["mse"] = 0.0
        for k, v in self.hparams.output_index.items():
            loss_name = f"mse_{k}"
            loss[loss_name] = F.mse_loss(
                yhat[:, v[0] : v[1], ...], y[:, v[0] : v[1], ...]
            )
            w = 1.0
            loss["mse"] += w * loss[loss_name]
            self.log(
                f"{mode}_{loss_name}", loss[loss_name], on_epoch=True, on_step=False
            )

        self.log(f"{mode}_mse", loss[f"mse"], on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss["mse"]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        self.step(batch, "val")


class FcnBaseline(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 26 * 2,
        num_layers: int = 7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.model = self.make_model()

    def make_model(self):
        def make_layer(ins, outs):
            layer = torch.nn.Sequential(
                torch.nn.Linear(ins, outs),
                torch.nn.BatchNorm1d(outs),
                torch.nn.Dropout(self.dropout),
                torch.nn.LeakyReLU(self.leaky_relu),
            )
            return layer

        input_layer = make_layer(self.input_size, self.hidden_size)
        intermediate_layers = [
            make_layer(self.hidden_size, self.hidden_size)
            for _ in range(self.num_layers - 2)
        ]
        output_layer = torch.nn.Linear(self.hidden_size, self.output_size)
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvNet1x1(torch.nn.Module):
    def __init__(
        self,
        input_size=26 * 2,
        num_layers=7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers

        self.model = self.make_model()

    def make_model(self):
        def make_layer(ins, outs):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ins, outs, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(outs),
                torch.nn.Dropout(self.dropout),
                torch.nn.LeakyReLU(self.leaky_relu),
            )
            return layer

        input_layer = make_layer(self.input_size, self.hidden_size)
        intermediate_layers = [
            make_layer(self.hidden_size, self.hidden_size)
            for _ in range(self.num_layers - 2)
        ]
        output_layer = torch.nn.Conv2d(self.hidden_size, self.output_size, kernel_size=1, bias=True)
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ComputeStats(LightningModule):
    def forward(self, x):
        outs = OrderedDict()
        # TODO dont hard code
        outs["mean"] = x.mean(dim=[0, 2, 3])
        outs["std"] = x.std(dim=[0, 2, 3])
        outs["min"] = x.amin(dim=[0, 2, 3])
        outs["max"] = x.amax(dim=[0, 2, 3])
        return outs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x_stats = self(x)
        y_stats = self(y)
        self.move_to_cpu(x_stats)
        self.move_to_cpu(y_stats)
        return [x_stats, y_stats]

    @staticmethod
    def move_to_cpu(output_dict):
        for k in list(output_dict):
            output_dict[k] = output_dict[k].cpu()

    @staticmethod
    def update_output_dict(output, output_dict):
        for k, v in output.items():
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(v[None, ...])

    @staticmethod
    def average_output_dict(output_dict):
        for k in list(output_dict.keys()):
            output_dict[k] = torch.cat(output_dict[k]).mean(0)

    def process_predictions(self, outputs) -> None:
        x_stats = OrderedDict()
        y_stats = OrderedDict()

        for x, y in outputs:
            self.update_output_dict(x, x_stats)
            self.update_output_dict(y, y_stats)

        self.average_output_dict(x_stats)
        self.average_output_dict(y_stats)

        # outputs[0].clear()
        # outputs[0].append(x_stats)
        # outputs[0].append(y_stats)

        return x_stats, y_stats

        # return
