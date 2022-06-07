import random
import numpy as np
import pandas as pd
import collections
import torch

import gym
from gymlob.envs.spaces.observation_spaces import get_observation_space
from gymlob.envs.spaces.action_spaces import get_discrete_action_space, get_continuous_action_space
from gym.spaces.utils import flatten_space, flatten
from gymlob.utils.data.lobster import LOBSTERDataLoader

# ------------------------------------------------ Financial Parameters --------------------------------------------- #

ANNUAL_VOLAT = 0.12  # Annual volatility in stock price
BID_ASK_SP = 1 / 8  # Bid-ask spread
DAILY_TRADE_VOL = 5e6  # Average Daily trading volume
TRAD_DAYS = 250  # Number of trading days in a year
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)  # Daily volatility in stock price

# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------- #

LLAMBDA = 1e-6  # Trader's risk aversion
LIQUIDATION_TIME = 1  # How many days to sell all the shares.
NUM_N = 60  # Number of trades
EPSILON = BID_ASK_SP / 2  # Fixed Cost of Selling.
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)  # Price Impact for Each 1% of Daily Volume Traded
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)  # Permanent Impact Constant


# ------------------------------------------------------------------------------------------------------------------ #

class AlmgrenChrissEnv(gym.Env):

    def __init__(self,
                 instrument: str,
                 date: str,
                 frequency: str,
                 num_levels: int,
                 client_order_info: dict,
                 orderbook_file_path: str, 
                 orders_file_path: str,
                 discrete_action_space: False,
                 randomSeed=0,
                 lqd_time=LIQUIDATION_TIME,
                 num_tr=NUM_N,
                 lambd=LLAMBDA, 
                ):

        # Set the random seed
        self._seed = randomSeed
        super().seed(self._seed)
        self.random_state = np.random.RandomState(seed=self._seed)

        # Initialize lobster variables
        self.instrument = instrument
        self.date = date
        self.frequency = frequency
        self.num_levels = num_levels

        self.loader = LOBSTERDataLoader(instrument=instrument,
                                        date=date,
                                        frequency=frequency,
                                        num_levels=num_levels,
                                        orders_file_path=orders_file_path,
                                        orderbook_file_path=orderbook_file_path)
        self.full_orderbook_df = self.loader.orderbook_df
        self.episode_orderbook_df = self.full_orderbook_df

        # Initialize the financial parameters so we can access them later
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT

        self.timestamps = self.full_orderbook_df.index
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[1]
        self.current_time = self.start_time

        #initialize state space and action space
        self.discrete_action_space = discrete_action_space
        self.client_order_info = client_order_info
        self.observation_space = flatten_space(get_observation_space(self.client_order_info))
        if discrete_action_space:
            self.action_space = get_discrete_action_space(self.client_order_info)
        else:
            self.action_space = flatten_space(get_continuous_action_space(self.client_order_info))

        # Initialize the Almgren-Chriss parameters so we can access them later
        self.total_shares = self.client_order_info['quantity']
        self.startingPrice = self.episode_orderbook_df.loc[self.current_time].mid_price
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance =  (DAILY_VOLAT * self.startingPrice) ** 2 
        self.eta = ETA
        self.gamma = GAMMA

        # Calculate some Almgren-Chriss parameters
        self.tau = self.liquidation_time / self.num_n
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

        # Set the variables for the initial state
        self.shares_remaining = self.total_shares
        self.time_remaining = self.client_order_info['duration']
        self.logReturns = collections.deque(np.zeros(6))
        self.executed_orders = []

        # store to later do reward normalization
        self.implementation_shortfall_arr = [0]

        # Set the initial impacted price to the starting price
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        self.transacting = False

        # Set a variable to keep trak of the trade number
        self.k = 0

    def reset(self, seed=0 ):

        # # Initialize the environment with the given parameters
        #self.__init__(randomSeed=seed, lqd_time=liquid_time, num_tr=num_trades, lambd=lamb)

        # # Set the initial state to [0,0,0,0,0,0,1,1]
        # self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, \
        #                                                        self.shares_remaining / self.total_shares])
        # return self.initial_state

        # Set the variables for the initial state
        self.shares_remaining = self.total_shares
        self.logReturns = collections.deque(np.zeros(6))


        allowed_timestamps = pd.date_range(start=self.timestamps[0],
                                            end=self.timestamps[-1] - np.timedelta64(self.client_order_info['duration'], 'm'),
                                            freq=self.frequency)

        self.start_time = self.random_state.choice(allowed_timestamps)
        self.end_time = self.start_time + np.timedelta64(self.client_order_info['duration'], 'm')

        self.episode_orderbook_df = self.full_orderbook_df.loc[self.start_time:self.end_time]

        self.current_time = self.start_time
        self.time_remaining = self.client_order_info['duration']
        self.quantity_remaining = self.client_order_info['quantity']
        self.startingPrice = self.episode_orderbook_df.loc[self.current_time].mid_price

        self.executed_orders = []
        self.step_num = 1

        self.start_transactions()

        return flatten(space=get_observation_space(self.client_order_info),
                    x=collections.OrderedDict(
                        {
                            "time_remaining": self.time_remaining,
                            "quantity_remaining": self.quantity_remaining,
                            "spread": self.episode_orderbook_df.iloc[0]['spread'],
                            "order_volume_imbalance": self.episode_orderbook_df.iloc[0]['order_volume_imbalance'],
                            "action": 0
                        }
                    ))



    def start_transactions(self):

        # Set transactions on
        self.transacting = True

        # Set the minimum number of stocks one can sell
        self.tolerance = 1

        # Set the initial capture to zero
        self.totalCapture = 0

        # Set the initial previous price to the starting price
        self.prevPrice = self.startingPrice

        # Set the initial square of the shares to sell to zero
        self.totalSSSQ = 0

        # Set the initial square of the remaing shares to sell to zero
        self.totalSRSQ = 0

        # Set the initial AC utility
        #self.prevUtility = self.compute_AC_utility(self.total_shares)

    def step(self, action):

        # Create a class that will be used to keep track of information about the transaction
        class Info(object):
            pass

        info = Info()

        # Set the done flag to False. This indicates that we haven't sold all the shares yet.
        info.done = False

        current_ob_snapshot = self.episode_orderbook_df.loc[self.current_time]
        info.price = current_ob_snapshot.mid_price

        # During training, if the DDPG fails to sell all the stocks before the given
        # number of trades or if the total number shares remaining is less than 1, then stop transacting,
        # set the done Flag to True, return the current implementation shortfall, and give a negative reward.
        # The negative reward is given in the else statement below.
        if self.transacting and (self.time_remaining == 0 or abs(self.shares_remaining) < self.tolerance):
            self.transacting = False
            info.done = True

            info.share_to_sell_now = self.shares_remaining
            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now, current_ob_snapshot.spread)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)

            # Apply the temporary and permanent impact on the current stock price
            info.exec_price = info.price - info.currentTemporaryImpact - info.currentPermanentImpact

            # Calculate the current total capture
            self.totalCapture += info.share_to_sell_now * info.exec_price

            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
            self.implementation_shortfall_arr.append(info.implementation_shortfall)
            reward = self.getReward(info.implementation_shortfall)

            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)
            info.expected_variance = self.singleStepVariance * self.tau * self.totalSRSQ
            info.utility = info.expected_shortfall + self.llambda * info.expected_variance

        # We don't add noise before the first trade
        #if self.k == 0:
        #    info.price = self.prevImpactedPrice
        #else:
            # Calculate the current stock price using arithmetic brownian motion
            #info.price = self.prevImpactedPrice + np.sqrt(self.singleStepVariance * self.tau) * random.normalvariate(0,
            #                                                                                                         1)
        #    info.price = current_ob_snapshot.mid_price

        # If we are transacting, the stock price is affected by the number of shares we sell. The price evolves
        # according to the Almgren and Chriss price dynamics model.
        if self.transacting:
            
            # If action is an ndarray then extract the number from the array
            if isinstance(action, np.ndarray):
                action = action.item()

            # Convert the action to the number of shares to sell in the current step
            # sharesToSellNow = self.shares_remaining * action
            # sharesToSellNow = min(self.shares_remaining * action, self.shares_remaining)

            size = action  #size = int(action[0]) if not self.discrete_action_space else action

            step_executed_orders = self.handle_market_order(direction=self.client_order_info['direction'],
                                                            size=size,
                                                            orderbook_snapshot=current_ob_snapshot)

            self.executed_orders.append(step_executed_orders)

            sharesToSellNow =  int(sum([order[0] for order in step_executed_orders])) 

            if self.time_remaining < 2:
                sharesToSellNow = self.shares_remaining

            # Since we are not selling fractions of shares, round up the total number of shares to sell to the nearest integer.
            info.share_to_sell_now = np.around(sharesToSellNow)

            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now, current_ob_snapshot.spread)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)

            # Apply the temporary and permanent impact on the current stock price
            info.exec_price = info.price - info.currentTemporaryImpact - info.currentPermanentImpact

            # Calculate the current total capture
            self.totalCapture += info.share_to_sell_now * info.exec_price

            # Calculate the log return for the current step and save it in the logReturn deque
            self.logReturns.append(np.log(info.price / self.prevPrice))
            self.logReturns.popleft()

            # Update the number of shares remaining
            self.shares_remaining -= info.share_to_sell_now

            # Update current time
            self.current_time += np.timedelta64(int(self.frequency[:-1]), self.frequency[-1])

            # Calculate the runnig total of the squares of shares sold and shares remaining
            self.totalSSSQ += info.share_to_sell_now ** 2
            self.totalSRSQ += self.shares_remaining ** 2

            # Update the variables required for the next step
            self.time_remaining -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact

            # Calculate the reward
            #currentUtility = self.compute_AC_utility(self.shares_remaining)
            #reward = (abs(self.prevUtility) - abs(currentUtility)) / abs(self.prevUtility)
            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
            self.implementation_shortfall_arr.append(info.implementation_shortfall)
            reward = self.getReward(info.implementation_shortfall)

            if type(reward) !=  np.float64:
                print('Error')

            #self.prevUtility = currentUtility

            # If all the shares have been sold calculate E, V, and U, and give a positive reward.
            if self.shares_remaining <= 0:
                # Calculate the implementation shortfall
                info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture

                # Set the done flag to True. This indicates that we have sold all the shares
                info.done = True

        self.k += 1
    
        observation = flatten(
            space=get_observation_space(self.client_order_info),
            x=collections.OrderedDict(
                {
                    "time_remaining": self.time_remaining,
                    "quantity_remaining": self.shares_remaining,
                    "spread": self.episode_orderbook_df.loc[self.current_time]['spread'],
                    "order_volume_imbalance": self.episode_orderbook_df.loc[self.current_time]['order_volume_imbalance'], 
                    "action":action
                })
        )

        return (observation , np.array([reward]), info.done, info)


    def getReward(self, impShortFall):

        mean = np.mean(self.implementation_shortfall_arr)
        std = np.std(self.implementation_shortfall_arr)
        impShortFall = impShortFall * -1 
        x_zscored = np.float64((impShortFall - mean) / std)
        x = torch.tanh(torch.tensor([x_zscored], dtype=torch.float64))
        return np.float64(x.item())

        #sigMoid = torch.nn.Sigmoid()
        #x = sigMoid(torch.tensor([impShortFall*-1], dtype=torch.float64))
        #return np.float64(x.item())

    def permanentImpact(self, sharesToSell, spread):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        self.gamma = spread / (0.1 * DAILY_TRADE_VOL)
        pi = self.gamma * sharesToSell
        return pi

    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)
        return ti

    def get_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * self.totalSSSQ
        return ft + st + tt

    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = self.eta_hat * (sharesToSell ** 2)
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)
        fot = nft / dft
        return ft + st + (tt * fot)

    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)
        nst = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(
            self.kappa * (self.liquidation_time - self.tau)) \
              - self.liquidation_time * np.sinh(self.kappa * self.tau)
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)
        st = nst / dst
        return ft * st

    def compute_AC_utility(self, sharesToSell):
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V

    def get_trade_list(self):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        for i in range(1, self.num_n + 1):
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list


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


    def observation_space_dimension(self):
        # Return the dimension of the state
        return 8

    def action_space_dimension(self):
        # Return the dimension of the action
        return 1

    def stop_transactions(self):
        # Stop transacting
        self.transacting = False