from collections import OrderedDict
import numpy as np
import gym
from gym.spaces.utils import flatten_space, flatten
from gym.spaces import Dict, Box
import pandas as pd

import yfinance as yf


class EquitiesEnv(gym.GoalEnv):
    """
    Class to get equities daily price (and other) data from yahoo finance API
    
    Usage:
    data = EquitiesData(tickers=['AAPL', 'MSFT', 'JPM'],
                        start_date="2019-01-01",
                        end_date="2020-01-01")
    data.prices
    data.returns
    and return means, vols and covariance for percentage and log returns
    """

    def __init__(self,
                 tickers: list = ['AAPL', 'MSFT', 'JPM'],
                 start_date: str = "2000-01-01",
                 end_date: str = "2020-12-31",
                 seed: int = 5422608
                 ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.Tickers(tickers=self.tickers).download(start=self.start_date, end=self.end_date)
        self.data.fillna(method="ffill")

        assert (self.data.isna().sum().sum() == 0), \
            "The given tickers have NaNs in price data, see {}".format(tickers)

        self.prices_open = self.data['Open']
        self.prices_close = self.data['Close']
        self.prices_low = self.data['Low']
        self.prices_high = self.data['High']
        self.volume = self.data['Volume']
        self.dividends = self.data['Dividends']

        self.seed = seed
        self.random_state = np.random.RandomState(seed=self.seed)

    def get_observation_space(self):
        return Dict(OrderedDict(
            {
                "price": Box(low=0, high=self.prices_close, shape=(1,), dtype=np.float32),
                "volume": Box(low=0, high=self.volume, shape=(1,), dtype=np.float32)
            }
        ))

    def get_action_space(self):
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

    def reset(self):
        """

        """
        pass

    def step(self):
        """

        """