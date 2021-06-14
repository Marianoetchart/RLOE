from typing import Callable
import omegaconf

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


class CNNLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 pre_activation_fn: Callable = identity,
                 activation_fn: Callable = F.relu,
                 post_activation_fn: Callable = identity,
                 ):
        super(CNNLayer, self).__init__()

        self.cnn = nn.Conv2d(input_size,
                             output_size,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)

        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        return x


class CNN(nn.Module):
    """Baseline of Convolution neural network."""

    def __init__(self,
                 configs: omegaconf.dictconfig.DictConfig):
        super(CNN, self).__init__()

        cnn_layers = list(map(CNNLayer, *list(configs.values())))
        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return x
