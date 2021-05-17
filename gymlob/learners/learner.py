import os
from abc import ABC, abstractmethod
import typing
from typing import Union
from collections import OrderedDict
import omegaconf

import numpy as np
import torch
import gym

TensorTuple = typing.Tuple[torch.Tensor, ...]


class Learner(ABC):

    def __init__(self,
                 env: Union[gym.Env, str],
                 cfg: omegaconf.dictconfig.DictConfig,
                 observation_space: Union[gym.spaces.Box, np.ndarray],
                 action_space: Union[gym.spaces.Tuple, np.ndarray]
                 ):
        self.env = env
        self.cfg = cfg
        self.observation_space = observation_space
        self.action_space = action_space

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not self.cfg.test:
            self.ckpt_path = f"'../../../../../checkpoint/{self.cfg.project_name}/{self.cfg.experiment_name}"
            os.makedirs(self.ckpt_path, exist_ok=True)

    @abstractmethod
    def _init_network(self):
        """

        :return:
        """
        pass

    def save_params(self,
                    params: dict,
                    n_episode: int
                    ):
        os.makedirs(self.ckpt_path, exist_ok=True)
        path = os.path.join(self.ckpt_path, f"ep_{str(n_episode)}.pt")
        print(path)
        torch.save(params, path)
        print(f"[INFO] Saved the model and optimizer to {path} \n")

    @abstractmethod
    def load_params(self,
                    path: str
                    ):
        if not os.path.exists(path):
            raise Exception(f"[ERROR] the input path does not exist. Wrong path: {path}" )

    @abstractmethod
    def update_model(self,
                     experience: Union[TensorTuple, typing.Tuple[TensorTuple]]
                     ) -> tuple:
        pass

    @abstractmethod
    def get_state_dict(self) -> Union[OrderedDict, typing.Tuple[OrderedDict]]:
        pass

    @abstractmethod
    def get_policy(self) -> torch.nn.Module:
        pass
