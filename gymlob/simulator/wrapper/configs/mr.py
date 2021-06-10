<<<<<<< HEAD
import pandas as pd
import numpy as np
from datetime import datetime

from gymlob.simulator.agent.ExchangeAgent import ExchangeAgent
from gymlob.simulator.agent.marketreplay.MarketReplayAgent import MarketReplayAgent
from gymlob.simulator.agent.marketreplay.SQLMarketReplayAgent import SQLMarketReplayAgent
from gymlob.simulator.agent.marketreplay.LOBSTERMarketReplayAgent import LOBSTERMarketReplayAgent

from gymlob.simulator.util.order import LimitOrder
from gymlob.simulator.util import util

util.silent_mode = True
LimitOrder.silent_mode = True

MKT_OPEN_TIME = '09:30:00'
MKT_CLOSE_TIME = '16:00:00'


def get_marketreplay_config(security, date, start_time=None, end_time=None, book_freq=None, log_orders=False):

    mkt_open = pd.to_datetime('{} {}'.format(date, MKT_OPEN_TIME))
    mkt_close = pd.to_datetime('{} {}'.format(date, MKT_CLOSE_TIME))

    if start_time is None: start_time = mkt_open
    if end_time is None: end_time = mkt_close

    # 1) Exchange Agent
    agents = []
    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 symbols=[security],
                                 mkt_open=start_time,
                                 mkt_close=end_time,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    # 2) Market Replay Agent
    file_name = f'/efs/data/DOW30/{security}/{security}.{date}'
    # file_name = f'/efs/data/R3K/{symbol}/{symbol}/{symbol}_ob.{historical_date}'

    agents.extend([MarketReplayAgent(id=1,
                                     name="MARKET_REPLAY_AGENT",
                                     type='MarketReplayAgent',
                                     symbol=security,
                                     date=pd.to_datetime(date),
                                     start_time=start_time,
                                     end_time=end_time,
                                     log_orders=False,
                                     orders_file_path=file_name,
                                     dict_dump_folder_path='/efs/data/DOW30/abides_marketreplay/',
                                     dict_dump_path=f'/efs/data/DOW30/abides_marketreplay/{security}_{date}.pkl',
                                     starting_cash=0,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    return agents, 2


def get_sql_marketreplay_config(security, date, start_time=None, end_time=None, book_freq=None, log_orders=False):

    mkt_open  = datetime.strptime('{} {}'.format(date, MKT_OPEN_TIME), '%Y%m%d %H:%M:%S')
    mkt_close = datetime.strptime('{} {}'.format(date, MKT_CLOSE_TIME), '%Y%m%d %H:%M:%S')

    if start_time is None: start_time = mkt_open
    if end_time is None: end_time = mkt_close

    # 1) Exchange Agent
    agents = []
    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 symbols=[security],
                                 mkt_open=start_time,
                                 mkt_close=end_time,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    # 2) Market Replay Agent
    db_path = f'/efs/abides-mrrl/data/SQL/{security}/{date}.db'

    agents.extend([SQLMarketReplayAgent(id=1,
                                        name="SQL_MARKET_REPLAY_AGENT",
                                        type='MarketReplayAgent',
                                        symbol=security,
                                        date=date,
                                        start_time=start_time,
                                        end_time=end_time,
                                        db_path=db_path,
                                        log_orders=False,
                                        random_state=np.random.RandomState(
                                            seed=np.random.randint(low=0, high=2 ** 32)))])

    return agents, 2


def get_lobster_marketreplay_config(security, date, start_time=None, end_time=None,
                                    book_freq=None, log_orders=False):

    mkt_open = pd.to_datetime('{} {}'.format(date, MKT_OPEN_TIME))
    mkt_close = pd.to_datetime('{} {}'.format(date, MKT_CLOSE_TIME))

    if start_time is None: start_time = mkt_open
    if end_time is None: end_time = mkt_close

    # 1) Exchange Agent
    agents = []
    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 symbols=[security],
                                 mkt_open=start_time,
                                 mkt_close=end_time,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    # 2) LOBSTER Market Replay Agent
    orderbook_file_path = f'/efs/mm/lobrep/data/{security}_2015-01-01_2015-01-31_10/' \
                          f'{security}_{date}_34200000_57600000_orderbook_10.csv'
    message_file_path = f'/efs/mm/lobrep/data/{security}_2015-01-01_2015-01-31_10/' \
                        f'{security}_{date}_34200000_57600000_message_10.csv'

    agents.extend([LOBSTERMarketReplayAgent(id=1,
                                            name="LOBSTER_MARKET_REPLAY_AGENT",
                                            type='MarketReplayAgent',
                                            symbol=security,
                                            date=pd.to_datetime(date),
                                            starting_cash=0,
                                            orderbook_file_path=orderbook_file_path,
                                            message_file_path=message_file_path,
                                            random_state=np.random.RandomState(
                                                seed=np.random.randint(low=0, high=2 ** 32)))])
    return agents, 2
=======
import pandas as pd
import numpy as np
from datetime import datetime

from gymlob.simulator.agent.ExchangeAgent import ExchangeAgent
from gymlob.simulator.agent.marketreplay.MarketReplayAgent import MarketReplayAgent
from gymlob.simulator.agent.marketreplay.SQLMarketReplayAgent import SQLMarketReplayAgent
from gymlob.simulator.agent.marketreplay.LOBSTERMarketReplayAgent import LOBSTERMarketReplayAgent

from gymlob.simulator.util.order import LimitOrder
from gymlob.simulator.util import util

util.silent_mode = True
LimitOrder.silent_mode = True

MKT_OPEN_TIME = '09:30:00'
MKT_CLOSE_TIME = '16:00:00'


def get_marketreplay_config(security, date, start_time=None, end_time=None, book_freq=None, log_orders=False):

    mkt_open = pd.to_datetime('{} {}'.format(date, MKT_OPEN_TIME))
    mkt_close = pd.to_datetime('{} {}'.format(date, MKT_CLOSE_TIME))

    if start_time is None: start_time = mkt_open
    if end_time is None: end_time = mkt_close

    # 1) Exchange Agent
    agents = []
    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 symbols=[security],
                                 mkt_open=start_time,
                                 mkt_close=end_time,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    # 2) Market Replay Agent
    file_name = f'/efs/data/DOW30/{security}/{security}.{date}'
    # file_name = f'/efs/data/R3K/{symbol}/{symbol}/{symbol}_ob.{historical_date}'

    agents.extend([MarketReplayAgent(id=1,
                                     name="MARKET_REPLAY_AGENT",
                                     type='MarketReplayAgent',
                                     symbol=security,
                                     date=pd.to_datetime(date),
                                     start_time=start_time,
                                     end_time=end_time,
                                     log_orders=False,
                                     orders_file_path=file_name,
                                     dict_dump_folder_path='/efs/data/DOW30/abides_marketreplay/',
                                     dict_dump_path=f'/efs/data/DOW30/abides_marketreplay/{security}_{date}.pkl',
                                     starting_cash=0,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    return agents, 2


def get_sql_marketreplay_config(security, date, start_time=None, end_time=None, book_freq=None, log_orders=False):

    mkt_open  = datetime.strptime('{} {}'.format(date, MKT_OPEN_TIME), '%Y%m%d %H:%M:%S')
    mkt_close = datetime.strptime('{} {}'.format(date, MKT_CLOSE_TIME), '%Y%m%d %H:%M:%S')

    if start_time is None: start_time = mkt_open
    if end_time is None: end_time = mkt_close

    # 1) Exchange Agent
    agents = []
    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 symbols=[security],
                                 mkt_open=start_time,
                                 mkt_close=end_time,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    # 2) Market Replay Agent
    db_path = f'/efs/abides-mrrl/data/SQL/{security}/{date}.db'

    agents.extend([SQLMarketReplayAgent(id=1,
                                        name="SQL_MARKET_REPLAY_AGENT",
                                        type='MarketReplayAgent',
                                        symbol=security,
                                        date=date,
                                        start_time=start_time,
                                        end_time=end_time,
                                        db_path=db_path,
                                        log_orders=False,
                                        random_state=np.random.RandomState(
                                            seed=np.random.randint(low=0, high=2 ** 32)))])

    return agents, 2


def get_lobster_marketreplay_config(security, date, start_time=None, end_time=None,
                                    book_freq=None, log_orders=False):

    mkt_open = pd.to_datetime('{} {}'.format(date, MKT_OPEN_TIME))
    mkt_close = pd.to_datetime('{} {}'.format(date, MKT_CLOSE_TIME))

    if start_time is None: start_time = mkt_open
    if end_time is None: end_time = mkt_close

    # 1) Exchange Agent
    agents = []
    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 symbols=[security],
                                 mkt_open=start_time,
                                 mkt_close=end_time,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])

    # 2) LOBSTER Market Replay Agent
    orderbook_file_path = f'/efs/abides-mrrl/data/LOBSTER/{security}_2015-01-01_2015-01-31_10/' \
                          f'{security}_{date[:4]}-{date[4:6]}-{date[6:]}_34200000_57600000_orderbook_10.csv'
    message_file_path = f'/efs/abides-mrrl/data/LOBSTER/{security}_2015-01-01_2015-01-31_10/' \
                        f'{security}_{date[:4]}-{date[4:6]}-{date[6:]}_34200000_57600000_message_10.csv'

    agents.extend([LOBSTERMarketReplayAgent(id=1,
                                            name="LOBSTER_MARKET_REPLAY_AGENT",
                                            type='MarketReplayAgent',
                                            symbol=security,
                                            date=pd.to_datetime(date),
                                            starting_cash=0,
                                            orderbook_file_path=orderbook_file_path,
                                            message_file_path=message_file_path,
                                            random_state=np.random.RandomState(
                                                seed=np.random.randint(low=0, high=2 ** 32)))])
    return agents, 2
>>>>>>> 2052760fb0c43ebd2c5008144699d9e0e9d2e88d
