import torch


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
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
