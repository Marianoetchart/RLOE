from os import NGROUPS_MAX
from typing import Callable
import omegaconf

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Return torch.tanh(x)"""
    return torch.tanh(x)


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer"""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class MLP(nn.Module):
    """Baseline of Multilayer perceptron.

    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)

    """

    def __init__(
            self,
            configs: omegaconf.dictconfig.DictConfig,
            hidden_activation: Callable = F.relu,
            linear_layer: nn.Module = nn.Linear,
            lstm_layer:nn.Module = nn.LSTM,
            use_output_layer: bool = True,
            use_recurrency_layer: bool = False,
            n_category: int = -1,
            init_fn: Callable = init_layer_uniform,
    ):
        """Initialize."""
        super(MLP, self).__init__()

        self.hidden_sizes = configs.hidden_sizes
        self.input_size = configs.input_size
        self.output_size = configs.output_size
        self.hidden_activation = hidden_activation
        self.batch_size = configs.batch_size

        if configs.output_activation == 'identity':
            self.output_activation = identity
        elif configs.output_activation == 'tanh':
            self.output_activation = tanh

        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.use_recurrency_layer = configs.use_recurrency_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(configs.hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)
        
        #Add recurrency with LSTM layer
        if self.use_recurrency_layer:
            self.lstm_hidden = configs.lstm_hidden_size
            lstm=lstm_layer(in_size, hidden_size=self.lstm_hidden, num_layers=1 ) 
            in_size = self.lstm_hidden
            self.hidden_layers.append(lstm)
            self.__setattr__("hidden_lstm{}".format(0), lstm)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, configs.output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def init_lstm_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be determined"

        if training is True:
            return torch.zeros([1, self.batch_size, self.lstm_hidden]), torch.zeros([1, self.batch_size, self.lstm_hidden])
        else:
            return torch.zeros([1, 1, self.lstm_hidden]), torch.zeros([1, 1, self.lstm_hidden])

    def forward(self, x, h=None, c=None, training = True ) -> torch.Tensor:
        """Forward method implementation."""
        for hidden_layer in self.hidden_layers:
    
            if type(hidden_layer) == nn.LSTM:
                x = x.unsqueeze(0)
                h,c = self.init_lstm_hidden_state(self.batch_size, training=training)
                x, (h,c) = hidden_layer(x, (h.cuda(),c.cuda()))
                x = self.hidden_activation(h)
                x = x.view(x.shape[1], -1)
                continue
            x = self.hidden_activation(hidden_layer(x))          
        x = self.output_activation(self.output_layer(x))
        
        return x
