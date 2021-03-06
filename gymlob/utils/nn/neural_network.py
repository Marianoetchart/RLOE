from typing import Union, Tuple
import omegaconf

import torch
import torch.nn as nn

from gymlob.utils.nn.mlp import MLP
from gymlob.utils.nn.cnn import CNN


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def get_nn_architecture(config):
    if config.type == 'MLP':
        return MLP
    elif config.type == 'CNN':
        return CNN
    else:
        raise Exception("Unsupported Neural Network Architecture")


class NeuralNetwork(nn.Module):
    """Class for holding backbone and head networks."""

    def __init__(self,
                 backbone_cfg: omegaconf.dictconfig.DictConfig,
                 head_cfg: omegaconf.dictconfig.DictConfig):

        nn.Module.__init__(self)

        self.backbone_cfg = backbone_cfg
        self.head_cfg = head_cfg

        if not self.backbone_cfg:
            self.backbone = identity
            self.head_cfg.configs.input_size = self.head_cfg.configs.state_size[0]
        else:
            self.backbone = get_nn_architecture(self.backbone_cfg)(configs=self.backbone_cfg.configs)
            self.head_cfg.configs.input_size = self.calculate_fc_input_size(self.head_cfg.configs.state_size)
        self.head = get_nn_architecture(self.head_cfg)(configs=self.head_cfg.configs)

    def forward(self, x, h = None, c = None ) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward method implementation. Use in get_action method in agents."""
        x = self.backbone(x)
        if type(h) == torch.Tensor:
            x = self.head(x.view(-1,self.head_cfg.configs.input_size), h, c)
        else:
            x = self.head(x.view(-1,self.head_cfg.configs.input_size), training = False)
        return x

    def forward_(self, x, h = None, c = None ) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward method implementation. Use in get_action method in agents."""
        x = self.backbone(x)
        if type(h) == torch.Tensor:
            x = self.head.forward_(x.view(-1,self.head_cfg.configs.input_size), h, c)
        else:
            x = self.head.forward_(x)
        return x

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        if type(state_dim) == omegaconf.listconfig.ListConfig:
            state_dim = tuple(state_dim)
        x = torch.zeros(state_dim).unsqueeze(0)
        #x = torch.zeros(1,state_dim[0],1,1) 
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]
