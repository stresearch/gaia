from audioop import bias
from turtle import forward
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F


class Baseline(LightningModule):
    def __init__(
        self,
        input_size: int = 26*2,
        num_layers: int = 7,
        hidden_size: int = 512,
        output_size: int = 26*2,
        lr: float = 0.0002,
        dropout: float = 0.01,
        leaky_relu: float = 0.15,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.layers = self.make_model()

    def make_model(self):
        def make_layer(ins, outs):
            layer = torch.nn.Sequential(
                torch.nn.Linear(ins, outs),
                torch.nn.BatchNorm1d(outs),
                torch.nn.Dropout(self.hparams.dropout),
                torch.nn.LeakyReLU(self.hparams.leaky_relu),
            )
            return layer

        input_layer = make_layer(self.hparams.input_size, self.hparams.hidden_size)
        intermediate_layers = [
            make_layer(self.hparams.hidden_size,self.hparams.hidden_size)
            for _ in range(self.hparams.num_layers - 2)
        ]
        output_layer = torch.nn.Linear(
            self.hparams.hidden_size, self.hparams.output_size
        )
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.hparams.lr)

    def forward(self, x):
        return self.layers(x)

    def step(self, batch, mode="train"):
        x,y = batch
        yhat = self(x)
        mse = F.mse_loss(y,yhat)
        self.log(f"{mode}_mse", mse, on_epoch=True)
        return mse

    def training_step(self,batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self,batch, batch_idx, dataloader_idx=None):
        self.step(batch, "val")

    

class ConvNet1x1(torch.nn.Module):
    def __init__(self, input_size = 26*2, layers = 7, 
        hidden_size: int = 512,
        output_size: int = 26*2,
        dropout: float = 0.01,
        leaky_relu: float = 0.15):

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.layers = layers

        self.model = self.make_model()

    def make_model(self):

        def make_layer(ins, outs):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ins, outs, kernel_size=1, bias=False),
                torch.nn.BatchNorm1d(outs),
                torch.nn.Dropout(self.hparams.dropout),
                torch.nn.LeakyReLU(self.hparams.leaky_relu),
            )
            return layer

        input_layer = make_layer(self.hparams.input_size, self.hparams.hidden_size)
        intermediate_layers = [
            make_layer(self.hparams.hidden_size,self.hparams.hidden_size)
            for _ in range(self.hparams.num_layers - 2)
        ]
        output_layer = torch.nn.Linear(
            self.hparams.hidden_size, self.hparams.output_size
        )
        layers = [input_layer] + intermediate_layers + [output_layer]
        return torch.nn.Sequential(*layers)


    def forward(self,x):
        pass
        