import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from typing import Union


class RandomWalk:
    """
    Class for univariate or multivariate random walk data generation
    """

    def __init__(self,
                 N: int = 1,
                 mu: Union[int, np.ndarray] = 1,
                 sigma: Union[int, np.ndarray] = 1,
                 granularity: int = 100,
                 p_start: Union[int, np.ndarray] = 100,
                 walk_type: str = 'univariate'):

        self.N = N # number of (indpendently modeled) assets
        self.mu = mu  # mean (or vector of means) for each asset
        self.p_start = p_start  # starting price (or vector of prices)
        self.sigma = sigma  # standard deviation (or vector of stds) for each asset
        self.granularity = granularity  # number of simulated process steps per real time step
        # (keep the number a few order opf magnitudes >1 for a correct and smoother process)
        self.starting_price = p_start  # starting value/price for the walk
        self.w = np.zeros(shape=(1, self.N))  # initial value for weiner process
        self.type = walk_type  # univariate or multivariate random walk

    def weiner(self, n_step: int) -> np.ndarray:
        """

        :param n_step:
        :return:
        """

        def step():
            
            if self.type == 'multivariate':
                yi = np.dot(sqrtm(self.sigma), np.random.normal(size=self.N))
            elif self.type == 'univariate':
                yi = np.random.normal(size=(1, self.N))
            self.w += (yi * np.sqrt(1 / self.granularity))

        # Generate a Weiner Process
        weiner = np.zeros(shape=(n_step, self.N))
        for i in range(0, n_step):
            step()
            weiner[i, :] = self.w
        return weiner

    def arithmetic(self, steps:int = 1) -> pd.DataFrame:
        """
        Arithmetic correlated random walk with gaussian increments
        :param steps:
        :return:
        """

        n_step = int(steps * self.granularity)
        weiner = self.weiner(n_step=n_step)
        t_vec = np.ones(self.N).reshape((1, -1)) * np.linspace(0, steps, num=n_step).reshape((-1, 1))
        
        if self.type == 'univariate':

            weiner = self.sigma * weiner

        det_var = self.mu * t_vec
        b = self.p_start + det_var + weiner
        self.p_start += det_var[-1, :]
        return self.pandas_wrap(b)

    def geometric(self, steps:int = 1) -> pd.DataFrame:
        """
        Geometric random walk with gaussian increments -
        if step is True then add single step and yield walk
        :param steps:
        :return:
        """

        n_step = int(steps * self.granularity)
        weiner = self.weiner(n_step=n_step)
        t_vec = np.ones(self.N).reshape((1, -1)) * np.linspace(0, steps, num=n_step).reshape((-1, 1))
        
        if self.type == 'univariate':

            mu_star = (self.mu - (self.sigma ** 2 / 2))
            weiner = self.sigma * weiner
            
        elif self.type == 'multivariate':

            mu_star = self.mu
            
        det_var = mu_star * t_vec
        s = self.p_start * np.exp(det_var + weiner)
        self.p_start *= np.exp(det_var[-1, :])
        return self.pandas_wrap(s)

    def pandas_wrap(self, df: np.ndarray) -> pd.DataFrame:
        """

        :param df:
        :return:
        """
        ticker_lst = list(np.arange(self.N))
        df = pd.DataFrame(columns=ticker_lst, data=df)
        return df.iloc[::self.granularity].reset_index(drop=True)
