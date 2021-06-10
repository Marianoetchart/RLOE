import numpy as np
from collections import OrderedDict
from gym.spaces import Dict, Box, Discrete


def get_discrete_action_space(client_order_info):
    return Discrete(n=client_order_info['quantity'])


def get_continuous_action_space(client_order_info):
    return Dict(OrderedDict(
        {
            #    "order_type": Discrete(n=1),
            #    "price": Box(low=0,
            #                 high=np.inf,
            #                 shape=(1,),
            #                 dtype=np.float32),
            "size": Box(low=0,
                        high=client_order_info['quantity'],
                        shape=(1,),
                        dtype=np.float32),
        }
    ))
