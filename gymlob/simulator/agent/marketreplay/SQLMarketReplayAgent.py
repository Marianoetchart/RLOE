import sqlite3
from datetime import datetime
import pandas as pd

from gymlob.simulator.agent.TradingAgent import TradingAgent
from gymlob.simulator.util.order.LimitOrder import LimitOrder
from gymlob.simulator.util.util import log_print


class SQLMarketReplayAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, date, start_time, end_time,
                 db_path, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=0, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.date = date
        self.inmem_db = self.connect_db(db_path)
        self.wakeup_times = self.get_wakeup_times(start_time, end_time)
        self.state = 'AWAITING_WAKEUP'

    @staticmethod
    def connect_db(db_path):
        disk_db = sqlite3.connect('file:{}'.format(db_path), uri=True)
        inmem_db = sqlite3.connect(':memory:')
        inmem_db.executescript("".join(line for line in disk_db.iterdump()))
        return inmem_db

    def get_wakeup_times(self, start_time, end_time):
        db_timestamps = list(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                             for row in self.inmem_db.execute("SELECT timestamp FROM orders"))
        return [t for t in db_timestamps if start_time <= t < end_time]

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if not self.mkt_open or not self.mkt_close:
            return
        try:
            self.setWakeup(self.wakeup_times[0])
            self.wakeup_times.pop(0)

            order = list(self.inmem_db.execute("SELECT order_id, price, size, buy_sell_flag FROM orders "
                                               "WHERE timestamp=='{}'".format(currentTime)))
            self.placeOrder(currentTime, order)
        except IndexError:
            log_print(f"Market Replay Agent submitted all orders - last order @ {currentTime}")

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

    def placeOrder(self, currentTime, order):
        if len(order) == 1:
            order_id = order[0][0]
            price = order[0][1]
            size = order[0][2]
            direction = order[0][3]

            existing_order = self.orders.get(order_id)

            if not existing_order and size > 0:
                self.placeLimitOrder(self.symbol, size, direction == 'BUY', price, order_id=order_id)
            elif existing_order and size == 0:
                self.cancelOrder(existing_order)
            elif existing_order:
                self.modifyOrder(existing_order, LimitOrder(self.id, currentTime, self.symbol, size,
                                                            direction == 'BUY', price,
                                                            order_id=order_id))
        else:
            for ind_order in order:
                self.placeOrder(currentTime, order=[ind_order])

    def getWakeFrequency(self):
        return self.wakeup_times[0] - self.mkt_open


class L3OrdersProcessor:
    COLUMNS = ['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG']
    DIRECTION = {0: 'BUY', 1: 'SELL'}

    def __init__(self, orders_file_path, start_time, end_time):
        self.orders_file_path = orders_file_path
        self.start_time = start_time
        self.end_time = end_time
        self.orders_df = self.processOrders()

    def processOrders(self):
        def convertDate(date_str):
            try:
                return datetime.strptime(date_str, '%Y%m%d%H%M%S.%f')
            except ValueError:
                return convertDate(date_str[:-1])

        orders_df = pd.read_csv(self.orders_file_path).iloc[1:]
        all_columns = orders_df.columns[0].split('|')
        orders_df = orders_df[orders_df.columns[0]].str.split('|', 16, expand=True)
        orders_df.columns = all_columns
        orders_df = orders_df[L3OrdersProcessor.COLUMNS]

        orders_df['TIMESTAMP'] = orders_df['TIMESTAMP'].astype(str).apply(convertDate)
        orders_df = orders_df.loc[(orders_df.TIMESTAMP >= self.start_time) & (orders_df.TIMESTAMP < self.end_time)]

        orders_df['BUY_SELL_FLAG'] = orders_df['BUY_SELL_FLAG'].astype(int).replace(L3OrdersProcessor.DIRECTION)
        orders_df['SIZE'] = orders_df['SIZE'].astype(int)
        orders_df['PRICE'] = (orders_df['PRICE'].astype(float) * 100).astype(int)

        orders_df.set_index('TIMESTAMP', inplace=True)

        return orders_df
