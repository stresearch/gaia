from collections import OrderedDict
import typing
import torch
from gaia import get_logger
from torch.nn import functional as F


logger = get_logger(__name__)


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        logger.warning("hardcoding min std threshold to 1e-10")
        self.min_std = 1e-10
        z = std <= self.min_std

        if z.any():
            logger.warn(f"found {z.sum()} std values smaller than {self.min_std}, replacing with ones")
            std[z] = 1.0

        self.register_buffer("mean", mean[None, :, None, None])
        self.register_buffer("std", std[None, :, None, None])

    def forward(self, x, normalize : bool =True):
        #check for very small std
        # small_stds = self.std <= self.min_std
        # # if small_stds.any():
        # self.std[small_stds] = 1.0

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

class NormalizationBN1D(torch.nn.Module):
    def __init__(self, num_features, eps= 1e-9):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, affine = False, eps= eps)
        
    def forward(self, x, normalize=True):
        if normalize:
            return self.bn(x)
        else:  # demormalize
            return x * self.bn.running_var[None,:].sqrt() + self.bn.running_mean[None,:]




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
    



class OutputProcesser(torch.nn.Module):
    def __init__(self, positive_output_mask, func = "exp", use_stop_grad = False, only_apply_in_test = False):
        super().__init__()
        if not isinstance(positive_output_mask, torch.Tensor):
            positive_output_mask = torch.tensor(positive_output_mask).bool()
        self.register_buffer("positive_output_mask",positive_output_mask.float())
        self.func = func
        self.use_stop_grad = use_stop_grad
        self.only_apply_in_test = only_apply_in_test

        if use_stop_grad:
            logger.info(f"using stop gradient trick for positivity constraints with {func}")

    def forward(self, x):
        # x_exp = x.exp()
        if self.func == "exp":
            x_pos = x.exp()
            x_pos = x_pos.masked_fill_(~self.positive_output_mask[None,:].bool(),0)

        elif self.func == "softplus":
            # x_pos = -F.logsigmoid(-x)
            x_pos = F.softplus(x)
            # x_pos = x_pos.masked_fill_(~self.positive_output_mask[None,:].bool(),0)

        elif self.func == "rectifier":
            if self.only_apply_in_test and self.training:
                return x
            x_pos = x.clip(min = 0)
        else:
            raise ValueError(f"unknown func {x_pos}")
        
        ## doing this way to support auto casting to fp16 and in case there any nans
        x1 = x_pos * self.positive_output_mask + x * (1 - self.positive_output_mask)

        if self.use_stop_grad:
            return x1.detach() - x.detach() + x

        else:
            return x1


       


class BernoulliGammaOutput(torch.nn.Module):
    def __init__(self, positive_output_mask, eps = 1e-9):
        super().__init__()
    
        if not isinstance(positive_output_mask, torch.Tensor):
            positive_output_mask = torch.tensor(positive_output_mask).bool()
        self.register_buffer("output_mask",positive_output_mask.float())
        self.eps = eps


    def mean(self, ins, threshold = True):
        p = torch.sigmoid(ins[:,:,0])
        alpha_div_beta = torch.exp(ins[:,:,1] - ins[:,:,2])
        out = p*alpha_div_beta

        if threshold:
            out = out.masked_fill(p<self.eps, 0)

        return out

    def log_likelihood(self,y,ins):
        p = (y>self.eps).float()
        log_phat = ins[:,:,0]

        # log_y = torch.log(y + self.eps)

        # part of ll corresponding to bernoulli 
        bce_term = F.binary_cross_entropy_with_logits(log_phat, p, reduction="none")

        # part of ll corresponding to gamma (parametrized in terms of shape and rate)

        log_alpha = ins[:,:,1]
        log_beta = ins[:,:,2]
        phat = torch.sigmoid(ins[:,:,0])

        alpha = log_alpha.exp()
        beta = log_beta.exp()

        gamma_term = (alpha-1) * (y + self.eps).log() + alpha * log_beta - torch.lgamma(alpha + self.eps)
        gamma_term = phat*gamma_term - y*beta

        return bce_term - gamma_term # need to maximize log likelihood

    def forward(self, x):
        out = x[:,:,0]
        out[:,self.output_mask] = self.mean(x[:,self.output_mask,:]) # only worry about pos output variables
        return out

    def get_loss(self, ytrue, y):
        pass







    

        

    
