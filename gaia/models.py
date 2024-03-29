import os
from collections import OrderedDict
from math import ceil
from typing import Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from gaia import get_logger
from gaia.data import SCALING_FACTORS, flatten_tensor, unflatten_tensor
from gaia.layers import (
    BernoulliGammaOutput,
    Conv2dDS,
    FCLayer,
    InterpolateGrid1D,
    MultiIndexEmbedding,
    Normalization,
    NormalizationBN1D,
    OutputProcesser,
    ResDNNLayer,
    make_interpolation_weights,
)
from gaia.loss import LinearConstraintResidual, RelHumWeight
from gaia.optim import get_cosine_schedule_with_warmup
from gaia.physics import HumidityConversionWithIndex, RelHumConstraint
from gaia.unet.unet import UNet
from torch.nn.utils import clip_grad

logger = get_logger(__name__)


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
        predict_hidden_states=False,
        lr_schedule=None,
        use_batch_norm_for_norm=False,
        fine_tuning=False,
        loss="mse",
        zero_outputs=True,
        noise_sigma=0,
        unit_normalize=False,
        weight_decay=0,
        positive_output_pattern=None,
        positive_func="exp",
        use_rel_hum_constraint=False,
        use_rel_hum_reg=False,
        use_sample_weights=False,
        use_pteq_gradient_reg=0,
        linear_constraints = None,
        **kwargs,
    ):
        super().__init__()

        if not isinstance(data_stats, str):
            ignore = ["data_stats", "fine_tuning"]
        else:
            ignore = None

        self.save_hyperparameters(ignore=ignore)
        model_type = model_config["model_type"]

        self.fine_tuning = fine_tuning

        self.input_normalize, self.output_normalize = self.setup_normalize(data_stats)

        if model_type == "fcn":
            self.model = FcnBaseline(**model_config)
        elif model_type == "conv2d":
            self.model = ConvNet2D(**model_config)
        elif model_type == "fcn_history":
            self.model = FcnHistory(**model_config)
        elif model_type == "conv1d":
            self.model = ConvNet1D(
                input_index=input_index, output_index=output_index, **model_config
            )
        elif model_type == "resdnn":
            self.model = ResDNN(**model_config)
        elif model_type == "encoderdecoder":
            self.model = EncoderDecoder(**model_config)
        elif model_type == "transformer":
            self.model = TransformerModel(
                input_index=input_index, output_index=output_index, **model_config
            )
        elif model_type == "fcn_with_index":
            self.model = FcnWithIndex(**model_config)
        else:
            raise ValueError("unknown model_type")

        if loss_output_weights is not None:
            logger.info(f"using output weights {loss_output_weights}")
            # w = torch.tensor(loss_output_weights)
            # w *= w.shape[0] / w.sum()
            self.make_output_weights(loss_output_weights)
        else:
            self.hparams.zero_outputs = False

        if positive_output_pattern is not None:

            def match(n):
                return any(
                    [n.startswith(p) for p in positive_output_pattern.split(",")]
                )

            # positive_output_mask = torch.cat([torch.ones(e-s).bool() if positive_output_pattern in k else torch.zeros(e-s).bool()  for k,(s,e) in output_index.items()])
            positive_output_mask = torch.cat(
                [
                    torch.ones(e - s).bool() if match(k) else torch.zeros(e - s).bool()
                    for k, (s, e) in output_index.items()
                ]
            )
            logger.info(
                f"adding {positive_output_mask.sum()} positive constraints to outputs"
            )
            if positive_func == "gamma":
                self.output_processor = BernoulliGammaOutput(positive_output_mask)
            else:
                self.output_processor = OutputProcesser(
                    positive_output_mask, positive_func
                )
        else:
            self.output_processor = torch.nn.Identity()

        if use_rel_hum_constraint:
            logger.info("adding rel humidity constraint to moisture tend")
            self.rel_hum_constraint = RelHumConstraint(
                input_index, output_index, self.input_normalize, self.output_normalize
            )

        self.regs = []

        if use_rel_hum_reg:
            logger.info("adding rel humidity reg to moisture tend")
            self.rel_hum_reg = RelHumConstraint(
                input_index,
                output_index,
                self.input_normalize,
                self.output_normalize,
                ub=100,
                lb=0,
            )
            self.regs.append("rel_hum")

        if use_sample_weights:
            logger.info("using rel hum sample weights")
            self.sample_weights_compute = RelHumWeight(input_index=input_index)

        if use_pteq_gradient_reg>0:
            logger.info("using pteq gradient for regularization")
            self.rel_hum_converter = HumidityConversionWithIndex(
                input_index=input_index,
                hum_name="B_Q",
                temp_name="B_T",
                ps_name="B_PS",
                file="/proj/gaia-climate/data/cam4_v3/rF_AMIP_CN_CAM4--torch-test.cam2.h1.1979-01-01-00000.nc",
            )
            self.regs.append("pteq_grad")


        if linear_constraints is not None:
            self.linear_constraints_residuals = torch.nn.ModuleDict() 

            for con in linear_constraints.split(";"): #seperate multiple constaints with ;
                logger.info(f"adding linear constraint {con}")
                ins,outs = con.split(":") # seperate ins and outs with :
                ints = ints.split(",") if len(ins) > 0 else None
                outs = outs.split(",") if len(outs) > 0 else None
                self.linear_constraints_residuals[f"lin_con_{con}"] = LinearConstraintResidual(input_index= input_index, output_index= output_index, input_signs_and_names= ins, output_signs_and_names= outs)
                self.regs.append(f"lin_con_{con}")


        if len(kwargs) > 0:
            logger.warning(f"unkown kwargs {list(kwargs.keys())}")

    def make_output_weights(self, loss_output_weights):
        w = torch.tensor(loss_output_weights)
        w *= w.shape[0] / w.sum()
        self.register_buffer("loss_output_weights", w)
        return w

    def setup_normalize(self, data_stats):
        if self.hparams.use_batch_norm_for_norm:
            logger.info("using batch norm for norm ...")
            input_normalization = NormalizationBN1D(
                self.hparams.model_config["input_size"]
            )
            output_normalization = NormalizationBN1D(
                self.hparams.model_config["output_size"]
            )
            return input_normalization, output_normalization

        if isinstance(data_stats, str) and os.path.exists(data_stats):
            stats = torch.load(data_stats)
        elif isinstance(data_stats, dict):
            stats = data_stats
        elif data_stats is None:
            logger.warning(
                "no stats provided, assuming will be loaded later, initializing randomly"
            )
            layers = []

            # can't rely on model_config since some variables can be dropped,
            # so lets recompute size from index since that never changes
            # last value is the last element
            sizes = [
                list(self.hparams.input_index.values())[-1][-1],
                list(self.hparams.output_index.values())[-1][-1],
            ]

            for v in sizes:
                layers.append(
                    Normalization(
                        torch.rand(v),
                        torch.rand(v),
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

            std = stats["range"]

            # stats["std_eff"] = torch.where(
            #     stats["std"] > thr, stats["std"], stats["range"]
            # )
            mn = stats["mean"]
            # std = stats["std_eff"]
        else:
            mn = stats["mean"]
            std = stats["std"]

        if zero_mean:
            mn = torch.zeros_like(std)

        return Normalization(mn, std)

    def configure_optimizers(self):
        out = {}
        if self.hparams.optimizer == "adam":
            optim = torch.optim.AdamW
        elif self.hparams.optimizer == "lamb":
            import torch_optimizer

            optim = torch_optimizer.Lamb

        if self.hparams.weight_decay > 0:
            logger.info(
                "setting l2 / weight decay regularization 0 for bias terms and batch norm scale"
            )
            params = [
                {
                    "params": p,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay
                    if len(p.shape) > 1
                    else 0.0,
                }
                for p in self.parameters()
            ]
            out["optimizer"] = optim(params)
        else:
            out["optimizer"] = optim(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        if self.hparams.lr_schedule is not None:
            if self.hparams.lr_schedule == "cosine":
                out["lr_scheduler"] = {
                    "scheduler": get_cosine_schedule_with_warmup(
                        out["optimizer"],
                        int(0.1 * self.trainer.estimated_stepping_batches),
                        self.trainer.estimated_stepping_batches,
                    ),
                    "interval": "step",
                }
            else:
                raise ValueError(f"unknown lr scheduler {self.hparams.lr_schedule}")

        return out

    # def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val: Optional[Union[int, float]] = None, gradient_clip_algorithm: Optional[str] = None):
    #     for p in optimizer.param_groups:
    #         clip_grad.clip_grad_norm_(p["params"], gradient_clip_val)

    def forward(self, x, index=None):
        if index is not None:
            y = self.model(x, index=index)
        else:
            y = self.model(x)

        if self.hparams.use_rel_hum_constraint:
            # TODO this needs to be fixed
            y = self.rel_hum_constraint(x, y)

        y = self.output_processor(y)

        if self.hparams.zero_outputs:
            y = y.masked_fill_(self.loss_output_weights[None, :] == 0, 0.0)

        return y

    def handle_batch(self, batch):

        x, y = batch[:2]

        index = None

        if len(batch) > 2:
            index = batch[2]

        num_dims = len(x.shape)

        x = self.input_normalize(x)
        y = self.output_normalize(y)

        # x = self.select_input_variables(x)

        res = x, y

        return res + (index,)

    def step(self, batch, mode="train"):
        # x, y = batch
        # x = self.input_normalize(x)
        # y = self.output_normalize(y)

        if self.fine_tuning:
            self.model.eval()

        if self.hparams.use_sample_weights:
            sample_weights = self.sample_weights_compute(batch[0])

        if mode == "train" and self.hparams.use_pteq_gradient_reg>0:
            rel_hum = self.rel_hum_converter(batch[0])
 
        x, y, index = self.handle_batch(batch)

        if len(y.shape) == 2:
            reduce_dims = [0]
        elif len(y.shape) == 4:
            reduce_dims = [0, 2, 3]
        else:
            raise ValueError("wrong size of x")

        if self.training and (self.hparams.noise_sigma > 0):

            noise = torch.randn_like(x) * self.hparams.noise_sigma

            if len(x.shape) == 4:
                x1 = x[:, :1, :1, :1]
            elif len(x.shape) == 2:
                x1 = x[:, :1]
            else:
                raise ValueError("wrong size of x")

            noise = noise.masked_fill(torch.rand_like(x1) > 0.5, 0.0)
            x = x + noise


        # create leaf nodes
        if mode == "train" and self.hparams.use_pteq_gradient_reg>0:
            #ligthtning disables autograd for val modes
            #TODO dont hard code names of vars
            
            B_Q_si = self.hparams.input_index["B_Q"][0]
            B_Q_ei = self.hparams.input_index["B_Q"][1]

            # there must be a better way
            qs = x[:,B_Q_si:B_Q_ei].clone().requires_grad_(True).split(1,dim = 1)
            # temp = x.clone()
            x[:,B_Q_si:B_Q_ei] = torch.cat(qs, dim = -1)


        yhat = self(x, index=index)

        loss = OrderedDict()

        losses_to_reduce = []

        if self.hparams.use_rel_hum_reg:
            losses_to_reduce.append("rel_hum")
            loss["rel_hum"] = self.rel_hum_reg(x, yhat, mode="as_regularization")


        if mode == "train" and self.hparams.use_pteq_gradient_reg>0:
            #TODO dont hard code names of vars
            PTEQ_si = self.hparams.output_index["A_PTEQ"][0]
            PTEQ_ei = self.hparams.output_index["A_PTEQ"][1]
            temp_out = yhat[:,PTEQ_si:PTEQ_ei].sum(0)
            gs = torch.autograd.grad(outputs= temp_out.split(1), inputs= qs, create_graph=True)
            gs = torch.cat(gs, dim = -1)
            gs = gs.clip(min = .5)
            gs = self.hparams.use_pteq_gradient_reg*gs #batch x num_levels
            loss["pteq_grad"] = torch.zeros_like(yhat)
            loss["pteq_grad"][:,PTEQ_si:PTEQ_ei] = gs.masked_fill(rel_hum<60,0.)
            losses_to_reduce.append("pteq_grad")

        if self.hparams.loss == "mse":
            mse = F.mse_loss(y, yhat, reduction="none")

        elif self.hparams.loss == "smooth_l1":
            loss["smooth_l1"] = F.smooth_l1_loss(yhat, y, reduction="none")
            with torch.no_grad():
                mse = F.mse_loss(y, yhat, reduction="none")

            losses_to_reduce.append("smooth_l1")

        else:
            raise ValueError(f"unknown {self.hparams.loss}")

        loss["mse"] = mse
        losses_to_reduce.append("mse")

        if (
            self.hparams.positive_func == "gamma"
        ):  # compute log likelihood loss for the positive outputs
            loss["gamma"] = self.output_processor.loss()


        if self.hparams.get("linear_constraints",None):

            yhat_unnorm = self.output_normalize(yhat, normalize = False)
            for k, constraint in self.linear_constraints_residuals.items():
                #downweigh them by sum of variances
                weight = 0.
                if constraint.apply_to_input:
                    weight += self.input_normalize.std[:, constraint.input_mask,...].square().sum()

                if constraint.apply_to_output:
                    weight += self.output_normalize.std[:, constraint.output_mask,...].square().sum()
                
                loss[k] = constraint(batch[0], yhat_unnorm).mean() / weight



        with torch.no_grad():
            eps = 1e-18
            y_var = y.var(reduce_dims, unbiased=False).clip(min=eps)
            skill = (1.0 - mse.mean(reduce_dims) / y_var).clip(0, 1)

            for k, v in self.hparams.output_index.items():

                loss_name = f"skill_ave_trunc_{k}"

                skill_v = skill[v[0] : v[1]]
                loss[loss_name] = skill_v.mean()

                if mode == "test":
                    for i, skill_vi in enumerate(skill_v):
                        loss_name = f"skill_ave_trunc_{k}_{i:02}"
                        loss[loss_name] = skill_vi

            if self.hparams.loss_output_weights is not None:
                skill = skill * self.loss_output_weights

            loss["skill_ave_clipped"] = skill.mean()

        if self.hparams.loss_output_weights is not None:

            num_dims = len(mse.shape)

            for n in losses_to_reduce:
                if num_dims == 4:
                    loss[n] = loss[n] * self.loss_output_weights[None, :, None, None]
                elif num_dims == 2:
                    loss[n] = loss[n] * self.loss_output_weights[None, :]
                else:
                    raise ValueError("wrong number of dims in mse")

        if mode == "train" and self.hparams.use_sample_weights:

            num_dims = len(mse.shape)

            for n in losses_to_reduce:
                if num_dims == 4:
                    loss[n] = loss[n] * sample_weights[:, None, None, None]
                elif num_dims == 2:
                    loss[n] = loss[n] * sample_weights[:, None]
                else:
                    raise ValueError("wrong number of dims in mse")

        for n in losses_to_reduce:
            loss[n] = loss[n].mean()

        for k, v in loss.items():
            self.log(f"{mode}_{k}", v, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        out = loss[self.hparams.loss]

        # adding reg terms
        for r in self.regs:
            out += loss[r]

        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        self.step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, index = self.handle_batch(batch)

        if self.hparams.predict_hidden_states:
            y, h = self.model(x, return_hidden_state=True)
            return h.cpu()
        else:
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
        use_index=False,
        per_output_params=1,  # instead of predicting means, we predict prob density parameteters
        model_type=None,
    ):
        super().__init__()

        if use_index:
            # add lon/lat as an additional input
            from gaia.plot import lats, lons

            self.register_buffer("lats", torch.tensor(lats) / 90.0)
            self.register_buffer("lons", torch.tensor(lons) / 180.0)
            input_size = input_size + 2

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.per_output_params = per_output_params
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
        output_layer = torch.nn.Linear(
            self.hidden_size, self.output_size * self.per_output_params
        )
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def reshape_output(self, y):
        if self.per_output_params == 1:
            return y
        else:
            return y.reshape(-1, self.output_size, self.per_output_params)

    def forward(self, x, index: Optional[torch.Tensor] = None):

        if torch.jit.is_scripting():
            y = self.model(x)
            return self.reshape_output(y)
        else:
            if index is None:
                y = self.model(x)
            else:
                lats = self.lats[index[:, 0]]
                lons = self.lons[index[:, 1]]
                x = torch.cat([x, lats[:, None], lons[:, None]], dim=1)
                y = self.model(x)
            return self.reshape_output(y)


class FcnWithIndex(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 26 * 2,
        num_layers: int = 7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        index_shape=None,
        model_type=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.index_shape = index_shape

        self.layers = torch.nn.ModuleList(self.make_model())

    def make_model(self):
        if self.num_layers == 1:
            return [
                FCLayer(self.input_size, self.output_size, index_shape=self.index_shape)
            ]

        input_layer = FCLayer(
            self.input_size,
            self.hidden_size,
            batch_norm=True,
            dropout=self.dropout,
            leaky_relu=self.leaky_relu,
            index_shape=self.index_shape,
        )

        intermediate_layers = [
            FCLayer(
                self.hidden_size,
                self.hidden_size,
                batch_norm=True,
                dropout=self.dropout,
                leaky_relu=self.leaky_relu,
                index_shape=self.index_shape,
            )
            for _ in range(self.num_layers - 2)
        ]
        output_layer = FCLayer(
            self.hidden_size, self.output_size, index_shape=self.index_shape
        )
        layers = [input_layer] + intermediate_layers + [output_layer]
        return layers

    def forward(self, x, index=None):
        for layer in self.layers:
            x = layer(x, index=index)
        return x


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 26 * 2,
        num_layers: int = 7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        bottleneck_dim: int = 32,
        encoder_layers=None,
        index_shape=None,
        model_type=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.scale = 1
        if encoder_layers is None:
            encoder_layers = ceil(self.num_layers / 2)

        decoder_layers = self.num_layers - encoder_layers

        self.encoder = FcnBaseline(
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            input_size=input_size,
            output_size=bottleneck_dim,
            dropout=dropout,
            leaky_relu=leaky_relu,
        )

        self.decoder = FcnBaseline(
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            input_size=bottleneck_dim,
            output_size=output_size,
            dropout=dropout,
            leaky_relu=leaky_relu,
        )

        if index_shape is not None:
            self.index_embedding_scale = MultiIndexEmbedding(
                bottleneck_dim, index_shape, init_value=1.0
            )
            self.index_embedding_bias = MultiIndexEmbedding(
                bottleneck_dim, index_shape, init_value=0.0
            )

    def forward(self, x, return_hidden_state=False, index=None):
        b = self.encoder(x)

        if index is not None:
            scale = self.index_embedding_scale(index)
            bias = self.index_embedding_bias(index)
            b = b * scale + bias

        if self.scale > 1 and not self.training:
            b = unflatten_tensor(b)
            s = self.scale
            b = F.interpolate(b, scale_factor=[1 / s, 1 / s], mode="nearest")
            b = F.interpolate(b, scale_factor=[s, s], mode="bilinear")
            b = flatten_tensor(b)

        y = self.decoder(b)
        if return_hidden_state:
            return y, b
        else:
            return y


class ResDNN(torch.nn.Module):
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
            ResDNNLayer(self.hidden_size, self.leaky_relu, self.dropout)
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
        num_output_layers: int = 1,
        model_type=None,
        memory_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.time_steps = time_steps
        if memory_size is None:
            memory_size = output_size
        self.memory_size = memory_size
        self.num_output_layers = num_output_layers
        self.main_layers = self.make_input_layers()
        self.output_history_layer = torch.nn.Linear(self.memory_size, self.hidden_size)

        self.non_linear_ops = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Dropout(self.dropout),
            torch.nn.LeakyReLU(self.leaky_relu),
        )
        self.output_layer = self.make_output_layers()

    def make_output_layers(self):
        output_layer = torch.nn.Linear(self.hidden_size, self.output_size)
        if self.num_output_layers == 1:
            return output_layer

        def make_layer(ins, outs):
            layer = torch.nn.Sequential(
                torch.nn.Linear(ins, outs),
                torch.nn.BatchNorm1d(outs),
                torch.nn.Dropout(self.dropout),
                torch.nn.LeakyReLU(self.leaky_relu),
            )
            return layer

        intermediate_layers = [
            make_layer(self.hidden_size, self.hidden_size)
            for _ in range(self.num_output_layers - 1)
        ]
        layers = intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def make_input_layers(self):

        num_layers = self.num_layers - self.num_output_layers

        def make_layer(ins, outs):
            layer = torch.nn.Sequential(
                torch.nn.Linear(ins, outs),
                torch.nn.BatchNorm1d(outs),
                torch.nn.Dropout(self.dropout),
                torch.nn.LeakyReLU(self.leaky_relu),
            )
            return layer

        if num_layers == 1:
            return torch.nn.Linear(self.input_size, self.hidden_size)

        input_layer = make_layer(self.input_size, self.hidden_size)
        output_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)

        intermediate_layers = [
            make_layer(self.hidden_size, self.hidden_size)
            for _ in range(num_layers - 2)
        ]
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def forward(self, inputs, memory=None):
        if memory is None:
            x, yh = inputs
        else:
            x = inputs
            yh = memory
        h1 = self.main_layers(x)
        h2 = self.output_history_layer(yh)
        h = self.non_linear_ops(h1 + h2)
        y = self.output_layer(h)
        return y


class ConvNet1D(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 7,
        hidden_size: int = 512,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        kernel_size: int = 3,
        input_index=None,
        output_index=None,
        model_type=None,
        dilation=True,
        **other,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # self.input_size = input_size
        # self.output_size = output_size= ou
        self.input_index = input_index
        self.output_index = output_index
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.set_up_dims()
        self.model = self.make_model()
        # if len(other) > 0:
        #     logger.warn(f"unused kwargs {other}")

    def make_model(self):
        if self.num_layers == 1:
            raise ValueError

        def make_layer(ins, outs, dilation=1):
            layer = torch.nn.Sequential(
                torch.nn.Conv1d(
                    ins,
                    outs,
                    kernel_size=self.kernel_size,
                    bias=False,
                    padding="same",
                    dilation=dilation,
                ),
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
        output_layer = torch.nn.Conv1d(
            self.hidden_size,
            self.output_size,
            kernel_size=self.kernel_size,
            bias=True,
            padding="same",
        )
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.arange_spatially(x, "input")
        y = self.model(x)
        y = self.flatten(y, "output")
        return y

    def set_up_dims(self):
        self.num_levels = max([e - s for s, e in self.input_index.values()])
        self.input_size = len(self.input_index)
        self.output_size = len(self.output_index)

    def flatten(self, x, kind):

        if kind == "input":
            index_dict = self.input_index
        elif kind == "output":
            index_dict = self.output_index

        xout = []
        for i, (k, v) in enumerate(index_dict.items()):
            s, e = v
            if e - s == 1:  # scalar expand
                xout.append(x[:, i, :].mean(-1, keepdim=True))
            else:  # per level
                xout.append(x[:, i, :])

        return torch.cat(xout, dim=1)

    def arange_spatially(self, x, kind):

        if kind == "input":
            index_dict = self.input_index
        elif kind == "output":
            index_dict = self.output_index

        xout = []
        for k, v in index_dict.items():
            s, e = v
            if e - s == 1:  # scalar expand
                xout.append(x[:, None, s:e].expand(-1, -1, self.num_levels))
            else:  # per level
                xout.append(x[:, None, s:e])

        xout = torch.cat(xout, dim=1)

        return xout


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        hidden_size: int = 128,
        nhead: int = 4,
        input_index=None,
        output_index=None,
        model_type=None,
        use_xformer = False,
        **other,
    ):
        super().__init__()

        

        self.hidden_size = hidden_size
        # self.input_size = input_size
        # self.output_size = output_size= ou
        self.input_index = input_index
        self.output_index = output_index
        self.num_layers = num_layers
        self.set_up_dims()

        self.input_encoder = torch.nn.Linear(self.input_size, self.hidden_size)
        self.position_encoder = torch.nn.Embedding(self.num_levels, self.hidden_size)

        if use_xformer:

            from xformers.factory.model_factory import xFormer, xFormerConfig


            xformer_config = [

            {
                "reversible": False,  # Optionally make these layers reversible, to save memory
                "block_type": "encoder",
                "num_layers": num_layers,  # Optional, this means that this config will repeat N times
                "dim_model": hidden_size,
                "residual_norm_style": "pre",  # Optional, pre/post
               
                "multi_head_config": {
                    "num_heads": nhead,
                    "residual_dropout": 0.1,
                    "attention": {
                        "name": "scaled_dot_product",  # whatever attention mechanism
                        "dropout": 0.1,
                        "causal": False,
                        "seq_len": self.num_levels,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": 0.1,
                    "activation": "gelu",
                    "hidden_layer_multiplier": 4,
                },
            },
            ]
            
            self.transformer = xFormer.from_config(xFormerConfig(xformer_config))

        else:

            encoder_layer = torch.nn.TransformerEncoderLayer( 
                self.hidden_size, nhead=nhead, batch_first=True, dim_feedforward=4*hidden_size,
            )
            layer_norm = torch.nn.LayerNorm(self.hidden_size)
            self.transformer = torch.nn.TransformerEncoder(
                encoder_layer, num_layers, layer_norm
            )
        self.output_layer = torch.nn.Linear(hidden_size, self.output_size)

        if other:
            logger.info(f"unknown kwargs {other}")

    def set_up_dims(self):
        self.num_levels = max([e - s for s, e in self.input_index.values()])
        self.input_size = len(self.input_index)
        self.output_size = len(self.output_index)

    def flatten(self, x, kind):

        if kind == "input":
            index_dict = self.input_index
        elif kind == "output":
            index_dict = self.output_index

        xout = []
        # batch x vars
        for i, (k, v) in enumerate(index_dict.items()):
            s, e = v
            if e - s == 1:  # scalar expand
                xout.append(x[:, :, i].mean(1, keepdim=True))
            else:  # per level
                xout.append(x[:, :, i])

        return torch.cat(xout, dim=-1)

    def arange_spatially(self, x, kind):

        if kind == "input":
            index_dict = self.input_index
        elif kind == "output":
            index_dict = self.output_index

        # batch x levels x channels or vars
        xout = []

        for k, v in index_dict.items():
            s, e = v
            if e - s == 1:  # scalar expand
                xout.append(x[:, s:e, None].expand(-1, self.num_levels, -1))
            else:  # per level
                xout.append(x[:, s:e, None])

        xout = torch.cat(xout, dim=-1)

        return xout

    def forward(self, x):
        x = self.arange_spatially(x, "input")
        p = torch.arange(x.shape[1]).to(x.device)
        pos_embedding = self.position_encoder(p)[None, ...]
        input_embedding = self.input_encoder(x)
        x = pos_embedding + input_embedding
        x = self.transformer(x)
        y = self.output_layer(x)
        y = self.flatten(y, "output")
        return y


class ConvNet2D(torch.nn.Module):
    def __init__(
        self,
        input_size=26 * 2,
        num_layers=7,
        hidden_size: int = 512,
        output_size: int = 26 * 2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
        kernel_size: int = 3,
        conv_type: str = "conv2d",
        model_type=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.model = self.make_model()

    def make_model(self):
        if self.conv_type == "conv2d":
            conv = torch.nn.Conv2d
        elif self.conv_type == "conv2d_ds":
            conv = Conv2dDS
        else:
            raise ValueError(f"unknown {self.conv_typ}")

        def make_layer(ins, outs):

            layer = torch.nn.Sequential(
                conv(
                    ins, outs, kernel_size=self.kernel_size, bias=True, padding="same"
                ),
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
        output_layer = conv(
            self.hidden_size,
            self.output_size,
            kernel_size=self.kernel_size,
            padding="same",
            bias=True,
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
