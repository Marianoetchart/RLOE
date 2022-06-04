import numpy as np
from collections import OrderedDict
from gym.spaces import Dict, Box


def get_observation_space(client_order_info):
    return Dict(OrderedDict(
        {
            "time_remaining": Box(low=0,
                                  high=client_order_info['duration'],
                                  shape=(1,),
                                  dtype=np.float32),
            "quantity_remaining": Box(low=0,
                                      high=client_order_info['quantity'],
                                      shape=(1,),
                                      dtype=np.float32),
            "spread": Box(low=0,
                          high=np.iinfo(np.int).max,
                          shape=(1,),
                          dtype=np.float32),
            "order_volume_imbalance": Box(low=-np.iinfo(np.int).min,
                                          high=np.iinfo(np.int).max,
                                          shape=(1,),
                                          dtype=np.float32),
        }
    ))