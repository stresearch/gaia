from asyncio.log import logger
from collections import OrderedDict
from typing import List
from cv2 import normalize
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from gaia.data import SCALING_FACTORS
from gaia.layers import Normalization
import os
import torch_optimizer


class TrainingModel(LightningModule):
    def __init__(
        self,
        lr: float = 0.0002,
        optimizer="adam",
        input_index=None,
        output_index=None,
        model_config=dict(model_type="fcn"),
        data_stats=None,
        use_output_scaling=False,
        replace_std_with_range=False,
        loss_output_weights=None,
        **kwargs,
    ):
        super().__init__()

        if not isinstance(data_stats, str):
            ignore = ["data_stats"]
        else:
            ignore = None

        self.save_hyperparameters(ignore=ignore)
        model_type = model_config["model_type"]
        if model_type == "fcn":
            self.model = FcnBaseline(**model_config)
        elif model_type == "conv":
            self.model = ConvNet1x1(**model_config)
        elif model_type == "fcn_history":
            self.model = FcnHistory(**model_config)
        else:
            raise ValueError("unknown model_type")

        self.input_normalize, self.output_normalize = self.setup_normalize(data_stats)

        if loss_output_weights is not None:
            logger.info(f"using output weights {loss_output_weights}")
            w = torch.tensor(loss_output_weights)
            w *= w.shape[0] / w.sum()
            self.register_buffer("loss_output_weights", w)

        # if min_mean_thres is not None:
        #     if loss_output_weights is not None:
        #         raise ValueError("max_mean_threshold and loss_output_weights cant be both not None")
        #     outputs_to_ignore = self.output_normalize.std 


        if len(kwargs) > 0:
            logger.warning(f"unkown kwargs {list(kwargs.keys())}")

    def setup_normalize(self, data_stats):
        if isinstance(data_stats, str) and os.path.exists(data_stats):
            stats = torch.load(data_stats)
        elif isinstance(data_stats, dict):
            stats = data_stats
        elif data_stats is None:
            logger.warning(
                "no stats provided, assuming will be loaded later, initializing randomly"
            )
            layers = []
            for v in ["input_size", "output_size"]:
                layers.append(
                    Normalization(
                        torch.rand(self.hparams.model_config[v]),
                        torch.rand(self.hparams.model_config[v]),
                    )
                )
            return layers
        else:
            raise ValueError("unsupported stats format")

        layers = []
        input_norm = self.get_normalization(stats["input_stats"])
        # TODO don't hard code but lets use actual training data stats for input normalization

        if self.hparams.use_output_scaling:
            logger.warning("logger using predefined output scaling")
            output_norm = self.get_predefined_output_normalization()
        else:
            output_norm = self.get_normalization(stats["output_stats"], zero_mean=True)

        return input_norm, output_norm

        # for k in vars_to_setup_norm_for:
        #     # stats[k]["range"] = stats[k]["max"] - stats[k]["min"]
        #     stats[k]["range"] = torch.maximum(
        #         stats[k]["max"] - stats[k]["mean"], stats[k]["mean"] - stats[k]["min"]
        #     )
        #     stats[k]["std_eff"] = torch.where(
        #         stats[k]["std"] > 1e-9, stats[k]["std"], stats[k]["range"]
        #     )
        #     layers.append(Normalization(stats[k]["mean"], stats[k]["std_eff"]))

    def get_predefined_output_normalization(self):
        mean = torch.zeros(self.hparams.model_config["output_size"])
        std = torch.ones(self.hparams.model_config["output_size"])
        for k, v in self.hparams.output_index.items():
            s, e = v
            std[s:e] *= 1.0 / SCALING_FACTORS[k]

        return Normalization(mean, std)

    def get_normalization(self, stats, zero_mean=False):
        if self.hparams.replace_std_with_range:
            thr = 1e-9
            stats["range"] = torch.maximum(
                stats["max"] - stats["mean"], stats["mean"] - stats["min"]
            )
            stats["std_eff"] = torch.where(
                stats["std"] > thr, stats["std"], stats["range"]
            )
            mn = stats["mean"]
            std = stats["std_eff"]
        else:
            mn = stats["mean"]
            std = stats["std"]

        if zero_mean:
            mn = torch.zeros_like(std)

        return Normalization(mn, std)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optim = torch.optim.Adam
        elif self.hparams.optimizer == "lamb":
            optim = torch_optimizer.Lamb
        return optim(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return self.model(x)

    def handle_batch(self, batch):
        x,y = batch
        num_dims = len(x.shape)

        if num_dims == 3 or num_dims == 5:

            # have history
            x = x[:,-1,...] #only use last time stemps for state vars
            y1 = y[:,0,...]
            y2 = y[:,1,...]
            x = self.input_normalize(x)
            y2 = self.output_normalize(y2)

            if self.hparams.model_config["model_type"] == "fcn_history":
                y1 = self.output_normalize(y1)
                return [x,y1],y2
            else:
                # dont use history
                return x,y2

        else:
            x = self.input_normalize(x)
            y = self.output_normalize(y)
            return x,y



    def step(self, batch, mode="train"):
        # x, y = batch
        # x = self.input_normalize(x)
        # y = self.output_normalize(y)

        x,y = self.handle_batch(batch)

        if len(y.shape) == 2:
            reduce_dims = [0]
        elif len(y.shape) == 4:
            reduce_dims = [0, 2, 3]
        else:
            raise ValueError("wrong size of x")

        yhat = self(x)
        loss = OrderedDict()
        mse = F.mse_loss(y, yhat, reduction="none")


        with torch.no_grad():
            skill = (
                (1.0 - mse.mean(reduce_dims) / y.var(reduce_dims, unbiased=False))
                .clip(0, 1)
            )

            if self.hparams.loss_output_weights is not None:
                skill = skill * self.loss_output_weights

            loss["skill_ave_clipped"] = skill.mean()

            if mode == "test":
                for k, v in self.hparams.output_index.items():
                    loss_name = f"skill_ave_trunc_{k}"
                    y_v = y[:, v[0] : v[1], ...]
                    mse_v = mse[:, v[0] : v[1], ...]

                    skill = (
                        1.0 - mse_v.mean(reduce_dims) / y_v.var(reduce_dims, unbiased=False)
                    ).clip(0, 1)

                    loss[loss_name] = skill.mean()

                    for i in range(skill.shape[0]):
                        loss_name = f"skill_ave_trunc_{k}_{i:02}"
                        loss[loss_name] = skill[i]

        if self.hparams.loss_output_weights is not None:
            mse = mse * self.loss_output_weights[None, :]

        loss["mse"] = mse.mean()
        for k, v in loss.items():
            self.log(f"{mode}_{k}", v, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss["mse"]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        self.step(batch, "test")

    def _test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y_unnorm = batch
        x = self.input_normalize(x)
        y = self.output_normalize(y_unnorm)
        yhat = self(x)
        loss = OrderedDict()
        loss["mse"] = 0.0

        mode = "test"

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

        ##unnorm

        loss["mse_u"] = 0.0

        mode = "test"

        yhat = self.output_normalize(yhat, normalize=False)
        y = y_unnorm

        for k, v in self.hparams.output_index.items():
            loss_name = f"mse_u_{k}"
            loss[loss_name] = F.mse_loss(
                yhat[:, v[0] : v[1], ...], y[:, v[0] : v[1], ...]
            )
            w = 1.0
            loss["mse_u"] += w * loss[loss_name]
            self.log(
                f"{mode}_{loss_name}", loss[loss_name], on_epoch=True, on_step=False
            )

        self.log(f"{mode}_mse_u", loss[f"mse_u"], on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self.handle_batch(batch)
        yhat = self(x)
        yhat = self.output_normalize(yhat, normalize=False)  # denormalize
        return yhat.cpu()


class FcnBaseline(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 26 * 2,
        num_layers: int = 7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        model_type=None,
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
        if self.num_layers == 1:
            return torch.nn.Linear(self.input_size, self.output_size)

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


class FcnHistory(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 26 * 2,
        num_layers: int = 7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        time_steps: int = 1,
        model_type=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.main_layers = self.make_layers()
        self.output_history_layer = torch.nn.Linear(self.output_size, self.hidden_size)
        self.non_linear_ops = torch.nn.Sequential(
                torch.nn.BatchNorm1d(self.hidden_size),
                torch.nn.Dropout(self.dropout),
                torch.nn.LeakyReLU(self.leaky_relu),
            )
        self.output_layer = torch.nn.Linear(self.hidden_size,self.output_size)



    def make_layers(self):
        if self.num_layers == 1:
            return torch.nn.Linear(self.input_size, self.output_size)

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
            for _ in range(self.num_layers - 3)
        ]
        output_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        x, yh = inputs
        h1 =  self.main_layers(x)
        h2 = self.output_history_layer(yh)
        h = self.non_linear_ops(h1 + h2)
        y = self.output_layer(h)
        return y


class ConvNet1x1(torch.nn.Module):
    def __init__(
        self,
        input_size=26 * 2,
        num_layers=7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        model_type=None,
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
        output_layer = torch.nn.Conv2d(
            self.hidden_size, self.output_size, kernel_size=1, bias=True
        )
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
