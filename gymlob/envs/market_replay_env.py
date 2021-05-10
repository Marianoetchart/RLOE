from collections import OrderedDict
import numpy as np
import gym
import pandas as pd
from gym.spaces import Dict, Box, Discrete
from gym.spaces.utils import flatten_space, flatten

from gymlob.utils.data.lobster import LOBSTERDataLoader
from gymlob.utils.rewards import get_step_reward


class MarketReplayEnv(gym.Env):

    def __init__(self,
                 instrument: str,
                 date: str,
                 frequency: str,
                 num_levels: int,
                 client_order_info: dict,
                 orders_file_path: str,
                 orderbook_file_path: str,
                 _seed: int):

        self.instrument = instrument
        self.date = date
        self.frequency = frequency
        self.num_levels = num_levels
        self.client_order_info = client_order_info  # direction, quantity, duration, benchmark

        self._seed = _seed
        super().seed(self._seed)
        self.random_state = np.random.RandomState(seed=self._seed)

        self.loader = LOBSTERDataLoader(instrument=instrument,
                                        date=date,
                                        frequency=frequency,
                                        num_levels=num_levels,
                                        orders_file_path=orders_file_path,
                                        orderbook_file_path=orderbook_file_path)
        self.orderbook_df = self.loader.orderbook_df
        self._orderbook_df = self.orderbook_df

        self.timestamps = self.orderbook_df.index
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[1]
        self.current_time = self.start_time

        self.quantity_remaining = np.uint32(self.client_order_info['quantity'])
        self.time_remaining = np.uint32(self.client_order_info['duration'])
        self.executed_orders = []

        self._observation_space = Dict(OrderedDict({
            "time_remaining": Box(low=0,
                                  high=self.client_order_info['duration'],
                                  shape=(1,),
                                  dtype=np.uint32),
            "quantity_remaining": Box(low=0,
                                      high=self.client_order_info['quantity'],
                                      shape=(1,),
                                      dtype=np.uint32),
            "spread": Box(low=0,
                          high=np.inf,
                          shape=(1,),
                          dtype=np.float32),
            "order_volume_imbalance": Box(low=-np.inf,
                                          high=np.inf,
                                          shape=(1,),
                                          dtype=np.float32),
        }))
        self.observation_space = flatten_space(self._observation_space)

        self._action_space = Dict(OrderedDict({
        #    "order_type": Discrete(n=1),
        #    "price": Box(low=0,
        #                 high=np.inf,
        #                 shape=(1,),
        #                 dtype=np.float32),
            "size": Box(low=0,
                        high=self.client_order_info['quantity'],
                        shape=(1,),
                        dtype=np.float32),
        }))
        self.action_space = flatten_space(self._action_space)

    def reset(self):

        allowed_timestamps = pd.date_range(start=self.timestamps[0],
                                           end=self.timestamps[-1] - np.timedelta64(self.client_order_info['duration'],
                                                                                    'm'),
                                           freq=self.frequency)

        self.start_time = self.random_state.choice(allowed_timestamps)
        self.end_time = self.start_time + np.timedelta64(self.client_order_info['duration'], 'm')

        self._orderbook_df = self.orderbook_df.loc[self.start_time:self.end_time]

        self.current_time = self.start_time
        self.time_remaining = self.client_order_info['duration']
        self.quantity_remaining = self.client_order_info['quantity']

        return flatten(space=self._observation_space,
                       x=OrderedDict({"time_remaining": self.time_remaining,
                                      "quantity_remaining": self.quantity_remaining,
                                      "spread": self._orderbook_df.iloc[0]['spread'],
                                      "order_volume_imbalance": self._orderbook_df.iloc[0]['order_volume_imbalance']
                                      })
                       )

    def step(self, action):

        done = False

        current_ob_snapshot = self._orderbook_df.loc[self.current_time]

        step_executed_orders = []
        step_executed_orders = self.handle_market_order(direction=self.client_order_info['direction'],
                                                        #size=int(action[2]),
                                                        size=int(action[0]),
                                                        orderbook_snapshot=current_ob_snapshot)

        # if action['order_type'] == 0: # 'MKT'
        #    step_executed_orders = self.handle_market_order(direction=self.client_order_info['direction'],
        #                                                    size=action['size'][0],
        #                                                    orderbook_snapshot=current_ob_snapshot)
        # elif action['order_type'] == 1: # 'LMT'
        #    step_executed_orders = self.handle_limit_order(direction=self.client_order_info['direction'],
        #                                                   size=action['size'][0],
        #                                                   price=action['price'][0],
        #                                                   orderbook_snapshot=current_ob_snapshot)

        self.executed_orders.append(step_executed_orders)

        self.time_remaining -= 1
        self.quantity_remaining = np.subtract(self.quantity_remaining,
                                              np.sum([order[0] for order in step_executed_orders]))
        self.current_time += np.timedelta64(int(self.frequency[:-1]), self.frequency[-1])

        observation = flatten(
            space=self._observation_space,
            x=OrderedDict({"time_remaining": self.time_remaining,
                           "quantity_remaining": self.quantity_remaining,
                           "spread": self._orderbook_df.loc[self.current_time]['spread'],
                           "order_volume_imbalance": self._orderbook_df.loc[self.current_time]['order_volume_imbalance']
                           })
        )

        reward = get_step_reward(step_executed_orders=step_executed_orders,
                                 order_direction=self.client_order_info['direction'],
                                 arrival_price=self._orderbook_df.iloc[0]['mid_price'])

        if (self.current_time == self.end_time) or (self.quantity_remaining <= 0):
            done = True

        info = {
            "step_executed_orders": step_executed_orders,
            "ob_snapshot": current_ob_snapshot
        }

        return observation, reward, done, info

    def handle_market_order(self, direction, size, orderbook_snapshot):

        remaining_quantity = np.uint32(size)
        book = 'ask' if direction == 'BUY' else 'bid'

        executed_orders = []
        for level in range(1, self.num_levels + 1):
            level_quantity = np.uint32(orderbook_snapshot[f'{book}_size_{level}'])
            level_price = orderbook_snapshot[f'{book}_price_{level}']
            if remaining_quantity <= level_quantity:
                executed_orders.append((remaining_quantity, level_price))
                break
            else:
                executed_orders.append((level_quantity, level_price))
                remaining_quantity = np.subtract(remaining_quantity, level_quantity)
                continue

        executed_orders = [(int(order[0]), order[1]) for order in executed_orders]
        assert sum([order[0] for order in executed_orders]) == size
        return executed_orders

    def handle_limit_order(self, direction, size, price, orderbook_snapshot):
        # TODO: How do we model limit order fill probabilities?!
        return []

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
