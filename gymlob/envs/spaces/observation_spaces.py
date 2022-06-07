import numpy as np
from collections import OrderedDict
from gym.spaces import Dict, Box, Discrete, Tuple


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
                "action": Box(low=0,
                              high=client_order_info['quantity'],
                              shape=(1,),
                              dtype=np.float32),
            }
        ))  

"""return Dict(OrderedDict(
        {
            "time_remaining": Discrete(client_order_info['duration']),
            "quantity_remaining": Discrete(client_order_info['quantity']),
            "spread": Discrete(np.iinfo(np.int).max),
            "order_volume_imbalance": Discrete(np.iinfo(np.int).max)
        }
    )) """