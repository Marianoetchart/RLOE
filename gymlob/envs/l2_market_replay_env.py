from collections import OrderedDict
import numpy as np
import gym
from gym.spaces.utils import flatten_space, flatten
import pandas as pd

from gymlob.envs.spaces.observation_spaces import get_observation_space
from gymlob.envs.spaces.action_spaces import get_discrete_action_space, get_continuous_action_space
from gymlob.utils.data.lobster import LOBSTERDataLoader
from gymlob.utils.rewards import get_step_reward


class L2MarketReplayEnv(gym.Env):
    """

    """

    def __init__(self,
                 instrument: str,
                 date: str,
                 frequency: str,
                 num_levels: int,
                 client_order_info: dict,
                 orders_file_path: str,
                 orderbook_file_path: str,
                 discrete_action_space: False,
                 _seed: int):

        self._seed = _seed
        super().seed(self._seed)
        self.random_state = np.random.RandomState(seed=self._seed)

        self.instrument = instrument
        self.date = date
        self.frequency = frequency
        self.num_levels = num_levels
        self.client_order_info = client_order_info  # direction, quantity, duration, benchmark
        self.discrete_action_space = discrete_action_space

        self.loader = LOBSTERDataLoader(instrument=instrument,
                                        date=date,
                                        frequency=frequency,
                                        num_levels=num_levels,
                                        orders_file_path=orders_file_path,
                                        orderbook_file_path=orderbook_file_path)
        self.full_orderbook_df = self.loader.orderbook_df
        self.episode_orderbook_df = self.full_orderbook_df

        self.observation_space = flatten_space(get_observation_space(self.client_order_info))

        if discrete_action_space:
            self.action_space = get_discrete_action_space(self.client_order_info)
        else:
            self.action_space = flatten_space(get_continuous_action_space(self.client_order_info))

        self.timestamps = self.full_orderbook_df.index
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[1]
        self.current_time = self.start_time

        self.quantity_remaining = self.client_order_info['quantity']
        self.time_remaining = self.client_order_info['duration']

        self.executed_orders = []
        self.step_num = 1

    def reset(self):
        """

        """

        allowed_timestamps = pd.date_range(start=self.timestamps[0],
                                           end=self.timestamps[-1] - np.timedelta64(self.client_order_info['duration'], 'm'),
                                           freq=self.frequency)

        self.start_time = self.random_state.choice(allowed_timestamps)
        self.end_time = self.start_time + np.timedelta64(self.client_order_info['duration'], 'm')

        self.episode_orderbook_df = self.full_orderbook_df.loc[self.start_time:self.end_time]

        self.current_time = self.start_time
        self.time_remaining = self.client_order_info['duration']
        self.quantity_remaining = self.client_order_info['quantity']

        self.executed_orders = []
        self.step_num = 1

        return flatten(space=get_observation_space(self.client_order_info),
                       x=OrderedDict(
                           {
                               "time_remaining": self.time_remaining,
                               "quantity_remaining": self.quantity_remaining,
                               "spread": self.episode_orderbook_df.iloc[0]['spread'],
                               "order_volume_imbalance": self.episode_orderbook_df.iloc[0]['order_volume_imbalance']
                           }
                       ))

    def step(self, action):
        """

        """
        self.step_num += 1

        done = False

        current_ob_snapshot = self.episode_orderbook_df.loc[self.current_time]

        size = int(action[0]) if not self.discrete_action_space else action

        step_executed_orders = self.handle_market_order(direction=self.client_order_info['direction'],
                                                        size=size,
                                                        orderbook_snapshot=current_ob_snapshot)

        self.executed_orders.append(step_executed_orders)

        self.time_remaining -= 1
        self.quantity_remaining -= int(sum([order[0] for order in step_executed_orders]))
        self.current_time += np.timedelta64(int(self.frequency[:-1]), self.frequency[-1])

        observation = flatten(
            space=get_observation_space(self.client_order_info),
            x=OrderedDict(
                {
                    "time_remaining": self.time_remaining,
                    "quantity_remaining": self.quantity_remaining,
                    "spread": self.episode_orderbook_df.loc[self.current_time]['spread'],
                    "order_volume_imbalance": self.episode_orderbook_df.loc[self.current_time]['order_volume_imbalance']
                })
        )

        reward = get_step_reward(step_executed_orders=step_executed_orders,
                                 order_direction=self.client_order_info['direction'],
                                 arrival_price=self.episode_orderbook_df.iloc[0]['mid_price'])

        if (self.current_time == self.end_time) or (self.quantity_remaining <= 0) or \
           (self.step_num == self.client_order_info['duration']):
            done = True

        info = {
            "step_executed_orders": step_executed_orders,
            "ob_snapshot": current_ob_snapshot
        }

        return observation, reward, done, info

    def handle_market_order(self, direction, size, orderbook_snapshot):

        remaining_quantity = size
        book = 'ask' if direction == 'BUY' else 'bid'

        executed_orders = []
        for level in range(1, self.num_levels + 1):
            level_quantity = int(orderbook_snapshot[f'{book}_size_{level}'])
            level_price = orderbook_snapshot[f'{book}_price_{level}']
            if remaining_quantity <= level_quantity:
                executed_orders.append((remaining_quantity, level_price))
                break
            else:
                executed_orders.append((level_quantity, level_price))
                remaining_quantity -= level_quantity
                continue

        executed_orders = [(int(order[0]), order[1]) for order in executed_orders]
        return executed_orders

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
