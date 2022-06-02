from collections import OrderedDict
from typing import Dict, Tuple, Union

import random
import numpy as np
import gym
import torch


def set_seed(environment: gym.Env,
             seed: int):
    environment.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def state_dict2numpy(state_dict) \
        -> Dict[str, np.ndarray]:
    """Convert Pytorch state dict to list of numpy arrays."""
    np_state_dict = OrderedDict()
    for param in list(state_dict):
        np_state_dict[param] = state_dict[param].numpy()
    return np_state_dict


def numpy2floattensor(
    arrays: Union[np.ndarray, Tuple[np.ndarray]], device_: torch.device) \
        -> Tuple[torch.Tensor]:
    """Convert numpy type to torch FloatTensor.
        - Convert numpy array to torch float tensor.
        - Convert numpy array with Tuple type to torch FloatTensor with Tuple.
    """

    if isinstance(arrays, tuple):  # check Tuple or not
        tensors = []
        for array in arrays:
            tensor = (
                torch.from_numpy(array.copy()).to(device_, non_blocking=True).float()
            )
            tensors.append(tensor)
        return tuple(tensors)
    tensor = torch.from_numpy(arrays.copy()).to(device_, non_blocking=True).float()
    return tensor

def add_widxheight_dim(x):
    """ Adds the width and height dimension when dealing with 1d data and needing to do 2D convs """
    if x.dim() == 1:
        return x.unsqueeze(1).unsqueeze(2)
    return x.unsqueeze(2).unsqueeze(3)


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action