from turtle import forward
from typing import OrderedDict
import torch
from gaia import get_logger

logger = get_logger(__name__)


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        z = std == 0

        if z.any():
            logger.warn(f"found zero {z.sum()} std values, replacing with ones")
            std[z] = 1.0

        self.register_buffer("mean", mean[None, :, None, None])
        self.register_buffer("std", std[None, :, None, None])

    def forward(self, x, normalize=True):
        if normalize:
            if len(x.shape) == 4:
                return (x - self.mean) / self.std
            elif len(x.shape) == 2:
                return (x - self.mean[:, :, 0, 0]) / self.std[:, :, 0, 0]
            else:
                raise ValueError("data must be either 2 or 4 dimensional")
        else:  # demormalize
            if len(x.shape) == 4:
                return x * self.std + self.mean
            elif len(x.shape) == 2:
                return x * self.std[:, :, 0, 0] + self.mean[:, :, 0, 0]
            else:
                raise ValueError("data must be either 2 or 4 dimensional")


class InterpolateGrid1D(torch.nn.Module):
    log_linear_vars = ("Q",)

    def __init__(
        self,
        *,
        input_grid=None,
        output_grid=None,
        input_grid_index=None,
        output_grid_index=None,
        requires_grad = False
    ):
        super().__init__()
        self.linear = torch.nn.Linear(len(input_grid), len(output_grid), bias=False).requires_grad_(requires_grad)
        output_grid, input_grid = torch.tensor(output_grid), torch.tensor(input_grid)
        with torch.no_grad():
            self.linear.weight.data = make_interpolation_weights(
                output_grid, input_grid
            )
        self.input_grid_index = input_grid_index

        if output_grid_index is None:
            n_levels = len(output_grid)
            output_grid_index = OrderedDict()
            s = 0
            for k,v in input_grid_index.items():
                var_size = v[1] - v[0]
                e = s + n_levels if var_size > 1 else s + 1 # change all vector valued 
                output_grid_index[k] =  [s,e]
                s = e
                
        self.output_grid_index = output_grid_index

    def forward(self, x):
        out = []

        squeeze = False

        if x.ndim == 1:
            x = x[None,...]
            squeeze = True

        for k in self.output_grid_index.keys():
            s, e = self.input_grid_index[k]
            if e - s == 1:
                out.append(x[:, s:e])
            else:
                if k in self.log_linear_vars:
                    out.append(self.linear(x[:, s:e].log()).exp())
                else:
                    out.append(self.linear(x[:, s:e]))

        out = torch.cat(out, -1)

        if squeeze:
            out =  out[0]

        return out


def make_interpolation_weights(output_grid, input_grid):
    dist = (input_grid[:, None] - output_grid[None, :]).T.abs()
    w = torch.zeros_like(dist)
    vals, nn_index = dist.topk(2, dim=-1, largest=False)
    for k, ((vi, vj), (i, j)) in enumerate(zip(vals, nn_index)):
        w[k, i] = vj / (vi + vj)
        w[k, j] = vi / (vi + vj)
    return w


class ResDNNLayer(torch.nn.Module):
    def __init__(self, hidden_dim, leaky_relu, dropout):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.LeakyReLU(leaky_relu),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.LeakyReLU(leaky_relu),
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = y + x
        return x


class Conv2dDS(torch.nn.Module):
    def __init__(
        self, nin, nout, kernel_size=3, kernels_per_layer=1, bias=True, padding="same"
    ):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            nin,
            nin * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=nin,
            bias=bias,
        )
        self.pointwise = torch.nn.Conv2d(
            nin * kernels_per_layer, nout, kernel_size=1, bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MultiIndexEmbedding(torch.nn.Module):
    def __init__(self, hidden_dim, index_shape, init_value=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.index_shape = index_shape
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(num_emb, hidden_dim) for num_emb in index_shape]
        )
        if init_value is not None:
            for e in self.embeddings:
                with torch.no_grad():
                    e.weight.data.fill_(init_value)

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for i, emb in enumerate(self.embeddings):
            out += emb(x[:, i])

        return out / len(self.embeddings)


class FCLayer(torch.nn.Module):
    def __init__(
        self,
        ins,
        outs,
        batch_norm=False,
        dropout=0.,
        leaky_relu=-1,
        index_shape=None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(ins, outs)
        self.batch_norm = (
            torch.nn.BatchNorm1d(outs) if batch_norm else torch.nn.Identity()
        )
        self.drop_out = (
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        )
        self.relu = (
            torch.nn.LeakyReLU(leaky_relu) if leaky_relu > -1 else torch.nn.Identity()
        )

        if index_shape is not None:
            self.scale = MultiIndexEmbedding(outs, index_shape, init_value=1.0)
            self.bias = MultiIndexEmbedding(outs, index_shape, init_value=0.0)

    def forward(self, x, index=None):

        x = self.linear(x)
        x = self.batch_norm(x)

        if index is not None:
            x = x * self.scale(index) + self.bias(index)

        x = self.drop_out(x)
        x = self.relu(x)

        return x
