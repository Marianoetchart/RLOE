<<<<<<< HEAD
import pandas as pd

from gymlob.simulator.agent.TradingAgent import TradingAgent
from gymlob.simulator.util.util import log_print, delist
from gymlob.simulator.util.order.LimitOrder import LimitOrder
from gymlob.simulator.util.Message import Message


class LOBSTERMarketReplayAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, date, starting_cash,
                 orderbook_file_path, message_file_path,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.date = date
        self.log_orders = log_orders

        self.data = LOBSTERProcessor(symbol=symbol,
                                     date=date,
                                     orderbook_file_path=orderbook_file_path,
                                     message_file_path=message_file_path)
        self.orderbook_queues_df = pd.DataFrame(columns=['timestamp', 'bids', 'asks'])

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelStopping(self):
        super().kernelStopping()

    def kernelTerminating(self):
        super().kernelTerminating()
        self.writeLog(self.orderbook_queues_df, filename='orderbook_queues_df')

    def wakeup(self, currentTime):
        try:
            super().wakeup(currentTime)
            if not self.mkt_open or not self.mkt_close:
                return
            self.getOrderBook(self.symbol)

            order = self.data.trades_df.loc[self.data.trades_df.timestamp == currentTime]
            wake_up_time = self.data.trades_df.loc[self.data.trades_df.timestamp > currentTime].iloc[0].timestamp
            if currentTime == self.data.orderbook_df.iloc[0].name:
                if len(order) == 1:
                    self.placeMktOpenOrders(order, t=currentTime)
                else:
                    for _, ind_order in order.iterrows():
                        self.placeMktOpenOrders(pd.DataFrame(ind_order).T, t=currentTime)
            elif (currentTime > self.mkt_open) and (currentTime < self.mkt_close):
                try:
                    self.placeOrder(currentTime, order)
                except Exception as e:
                    log_print(e)
            self.setWakeup(wake_up_time)

            self.orderbook_queues_df = self.orderbook_queues_df.append(pd.Series(data={
                "timestamp": currentTime,
                "bids": self.extract_orderbook_queue(self.known_bids[self.symbol]),
                "asks": self.extract_orderbook_queue(self.known_asks[self.symbol])
            }), ignore_index=True)

        except Exception as e:
            print(str(e))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

    def extract_orderbook_queue(self, book):
        orders = {}
        orders_in_price_level = []
        for price_level in range(0, len(book)):
            for order in book[price_level]:
                orders_in_price_level.append(order.obj())
            orders[price_level] = orders_in_price_level
            orders_in_price_level = []
        return orders

    def placeMktOpenOrders(self, snapshot_order, t=0):
        orders_snapshot = self.data.orderbook_df.loc[self.data.orderbook_df.index == t].T
        for i in range(0, len(orders_snapshot) - 1, 4):
            ask_price = orders_snapshot.iloc[i][0]
            ask_vol = orders_snapshot.iloc[i + 1][0]
            bid_price = orders_snapshot.iloc[i + 2][0]
            bid_vol = orders_snapshot.iloc[i + 3][0]

            if snapshot_order.direction.item() == 'BUY' and bid_price == snapshot_order.price.item():
                bid_vol -= snapshot_order.vol.item()
            elif snapshot_order.direction.item() == 'SELL' and ask_price == snapshot_order.price.item():
                ask_vol -= snapshot_order.vol.item()

            self.placeLimitOrder(self.symbol, bid_vol, True, float(bid_price))
            self.placeLimitOrder(self.symbol, ask_vol, False, float(ask_price))
        self.placeOrder(snapshot_order.timestamp.item(), snapshot_order)

    def placeOrder(self, currentTime, order):
        if len(order) == 1:
            type = order.type.item()
            id = order.order_id.item()
            direction = order.direction.item()
            price = order.price.item()
            vol = order.vol.item()
            if type == 'NEW':
                self.placeLimitOrder(self.symbol, vol, direction == 'BUY', float(price), order_id=id)
            elif type in ['CANCELLATION', 'PARTIAL_CANCELLATION']:
                existing_order = self.orders.get(id)
                if existing_order:
                    if type == 'CANCELLATION':
                        self.cancelOrder(existing_order)
                    elif type == 'PARTIAL_CANCELLATION':
                        new_order = LimitOrder(self.id, currentTime, self.symbol, vol, direction == 'BUY', float(price),
                                               order_id=id)
                        self.modifyOrder(existing_order, new_order)
                else:
                    self.replicateOrderbookSnapshot(currentTime)
            elif type in ['EXECUTE_VISIBLE', 'EXECUTE_HIDDEN']:
                existing_order = self.orders.get(id)
                if existing_order:
                    if existing_order.quantity == vol:
                        self.cancelOrder(existing_order)
                    else:
                        new_vol = existing_order.quantity - vol
                        if new_vol == 0:
                            self.cancelOrder(existing_order)
                        else:
                            executed_order = LimitOrder(self.id, currentTime, self.symbol, new_vol, direction == 'BUY',
                                                        float(price), order_id=id)
                            self.modifyOrder(existing_order, executed_order)
                            self.orders.get(id).quantity = new_vol
                else:
                    self.replicateOrderbookSnapshot(currentTime)
        else:
            orders = self.data.trades_df.loc[self.data.trades_df.timestamp == currentTime]
            for index, order in orders.iterrows():
                self.placeOrder(currentTime, order=pd.DataFrame(order).T)

    def replicateOrderbookSnapshot(self, currentTime):
        log_print("Received notification of orderbook snapshot replication at: {}".format(currentTime))
        self.sendMessage(self.exchangeID, Message({"msg": "REPLICATE_ORDERBOOK_SNAPSHOT", "sender": self.id,
                                                   "symbol": self.symbol, "timestamp": str(currentTime)}))
        if self.log_orders: self.logEvent('REPLICATE_ORDERBOOK_SNAPSHOT', currentTime)

    def getWakeFrequency(self):
        return self.data.trades_df.iloc[0].timestamp - self.mkt_open


class LOBSTERProcessor:

    def __init__(self, symbol, date, orderbook_file_path, message_file_path, num_price_levels=10, filter_trades=True):
        self.symbol = symbol
        self.date = date
        self.num_price_levels = num_price_levels
        self.message_df = self.readMessageFile(message_file_path)
        self.orderbook_df = self.readOrderbookFile(orderbook_file_path)
        self.trades_df = self.filter_trades() if filter_trades else self.message_df
        log_print("OrderBookOracle initialized for {} and date: {}".format(self.symbol, self.date))

    def readMessageFile(self, message_file_path):
        """
        :return: a pandas Dataframe of the trade messages file for the given symbol and date
        """
        log_print("OrderBookOracle Message File: {}".format(message_file_path))

        direction = {-1: 'SELL',
                     1: 'BUY'}

        order_type = {
            1: 'NEW',
            2: 'PARTIAL_CANCELLATION',
            3: 'CANCELLATION',
            4: 'EXECUTE_VISIBLE',
            5: 'EXECUTE_HIDDEN',
            7: 'TRADING_HALT'
        }

        message_df = pd.read_csv(message_file_path)
        message_df.columns = ['timestamp', 'type', 'order_id', 'vol', 'price', 'direction']
        message_df['timestamp'] = self.date + pd.to_timedelta(message_df['timestamp'], unit='s')
        message_df['direction'] = message_df['direction'].replace(direction)
        message_df['price'] = message_df['price'] / 10000
        message_df['type'] = message_df['type'].replace(order_type)
        return message_df

    def readOrderbookFile(self, orderbook_file_path):
        """
        :return: a pandas Dataframe of the orderbook file for the given symbol and date
        """
        log_print("OrderBookOracle Orderbook File: {}".format(orderbook_file_path))
        all_cols = delist(
            [[f"ask_price_{level}", f"ask_size_{level}", f"bid_price_{level}", f"bid_size_{level}"] for level in
             range(1, self.num_price_levels + 1)])
        price_cols = delist(
            [[f"ask_price_{level}", f"bid_price_{level}"] for level in range(1, self.num_price_levels + 1)])
        orderbook_df = pd.read_csv(orderbook_file_path)
        orderbook_df.columns = all_cols
        orderbook_df[price_cols] = orderbook_df[price_cols] / 10000
        orderbook_df = orderbook_df.join(self.message_df[['timestamp']])
        orderbook_df = orderbook_df[['timestamp'] + all_cols]
        orderbook_df = orderbook_df.drop_duplicates(subset=['timestamp'], keep='last')
        orderbook_df.set_index('timestamp', inplace=True)
        return orderbook_df

    def filter_trades(self):
        #log_print("Original trades type counts:")
        #log_print(str(self.message_df.type.value_counts()))
        trades_df = self.message_df.loc[
            self.message_df.type.isin(['NEW', 'CANCELLATION', 'PARTIAL_CANCELLATION', 'EXECUTE_VISIBLE'])]
        order_id_types_series = trades_df.groupby('order_id')['type'].apply(list)
        order_id_types_series = order_id_types_series.apply(lambda x: str(x))
        cancel_only_order_ids = list(order_id_types_series[order_id_types_series == "['CANCELLATION']"].index)
        part_cancel_only_order_ids = list(
            order_id_types_series[order_id_types_series == "['PARTIAL_CANCELLATION']"].index)
        trades_df = trades_df.loc[~trades_df.order_id.isin(cancel_only_order_ids + part_cancel_only_order_ids)]
        #log_print("Filtered trades type counts:")
        #log_print(str(trades_df.type.value_counts()))
        return trades_df
=======
import pandas as pd

from gymlob.simulator.agent.TradingAgent import TradingAgent
from gymlob.simulator.util.util import log_print, delist
from gymlob.simulator.util.order.LimitOrder import LimitOrder
from gymlob.simulator.util.Message import Message


class LOBSTERMarketReplayAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, date, starting_cash,
                 orderbook_file_path, message_file_path,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.date = date
        self.log_orders = log_orders

        self.data = LOBSTERProcessor(symbol=symbol,
                                     date=date,
                                     orderbook_file_path=orderbook_file_path,
                                     message_file_path=message_file_path)

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelStopping(self):
        super().kernelStopping()

    def wakeup(self, currentTime):
        try:
            super().wakeup(currentTime)
            if not self.mkt_open or not self.mkt_close:
                return
            order = self.data.trades_df.loc[self.data.trades_df.timestamp == currentTime]
            wake_up_time = self.data.trades_df.loc[self.data.trades_df.timestamp > currentTime].iloc[0].timestamp
            if currentTime == self.data.orderbook_df.iloc[0].name:
                self.placeMktOpenOrders(order, t=currentTime)
            elif (currentTime > self.mkt_open) and (currentTime < self.mkt_close):
                try:
                    self.placeOrder(currentTime, order)
                except Exception as e:
                    log_print(e)
            self.setWakeup(wake_up_time)
        except Exception as e:
            print(str(e))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

    def placeMktOpenOrders(self, snapshot_order, t=0):
        orders_snapshot = self.data.orderbook_df.loc[self.data.orderbook_df.index == t].T
        for i in range(0, len(orders_snapshot) - 1, 4):
            ask_price = orders_snapshot.iloc[i][0]
            ask_vol = orders_snapshot.iloc[i + 1][0]
            bid_price = orders_snapshot.iloc[i + 2][0]
            bid_vol = orders_snapshot.iloc[i + 3][0]

            if snapshot_order.direction.item() == 'BUY' and bid_price == snapshot_order.price.item():
                bid_vol -= snapshot_order.vol.item()
            elif snapshot_order.direction.item() == 'SELL' and ask_price == snapshot_order.price.item():
                ask_vol -= snapshot_order.vol.item()

            self.placeLimitOrder(self.symbol, bid_vol, True, float(bid_price))
            self.placeLimitOrder(self.symbol, ask_vol, False, float(ask_price))
        self.placeOrder(snapshot_order.timestamp.item(), snapshot_order)

    def placeOrder(self, currentTime, order):
        if len(order) == 1:
            type = order.type.item()
            id = order.order_id.item()
            direction = order.direction.item()
            price = order.price.item()
            vol = order.vol.item()
            if type == 'NEW':
                self.placeLimitOrder(self.symbol, vol, direction == 'BUY', float(price), order_id=id)
            elif type in ['CANCELLATION', 'PARTIAL_CANCELLATION']:
                existing_order = self.orders.get(id)
                if existing_order:
                    if type == 'CANCELLATION':
                        self.cancelOrder(existing_order)
                    elif type == 'PARTIAL_CANCELLATION':
                        new_order = LimitOrder(self.id, currentTime, self.symbol, vol, direction == 'BUY', float(price),
                                               order_id=id)
                        self.modifyOrder(existing_order, new_order)
                else:
                    self.replicateOrderbookSnapshot(currentTime)
            elif type in ['EXECUTE_VISIBLE', 'EXECUTE_HIDDEN']:
                existing_order = self.orders.get(id)
                if existing_order:
                    if existing_order.quantity == vol:
                        self.cancelOrder(existing_order)
                    else:
                        new_vol = existing_order.quantity - vol
                        if new_vol == 0:
                            self.cancelOrder(existing_order)
                        else:
                            executed_order = LimitOrder(self.id, currentTime, self.symbol, new_vol, direction == 'BUY',
                                                        float(price), order_id=id)
                            self.modifyOrder(existing_order, executed_order)
                            self.orders.get(id).quantity = new_vol
                else:
                    self.replicateOrderbookSnapshot(currentTime)
        else:
            orders = self.data.trades_df.loc[self.data.trades_df.timestamp == currentTime]
            for index, order in orders.iterrows():
                self.placeOrder(currentTime, order=pd.DataFrame(order).T)

    def replicateOrderbookSnapshot(self, currentTime):
        log_print("Received notification of orderbook snapshot replication at: {}".format(currentTime))
        self.sendMessage(self.exchangeID, Message({"msg": "REPLICATE_ORDERBOOK_SNAPSHOT", "sender": self.id,
                                                   "symbol": self.symbol, "timestamp": str(currentTime)}))
        if self.log_orders: self.logEvent('REPLICATE_ORDERBOOK_SNAPSHOT', currentTime)

    def getWakeFrequency(self):
        return self.data.trades_df.iloc[0].timestamp - self.mkt_open


class LOBSTERProcessor:

    def __init__(self, symbol, date, orderbook_file_path, message_file_path, num_price_levels=10, filter_trades=True):
        self.symbol = symbol
        self.date = date
        self.num_price_levels = num_price_levels
        self.message_df = self.readMessageFile(message_file_path)
        self.orderbook_df = self.readOrderbookFile(orderbook_file_path)
        self.trades_df = self.filter_trades() if filter_trades else self.message_df
        log_print("OrderBookOracle initialized for {} and date: {}".format(self.symbol, self.date))

    def readMessageFile(self, message_file_path):
        """
        :return: a pandas Dataframe of the trade messages file for the given symbol and date
        """
        log_print("OrderBookOracle Message File: {}".format(message_file_path))

        direction = {-1: 'SELL',
                     1: 'BUY'}

        order_type = {
            1: 'NEW',
            2: 'PARTIAL_CANCELLATION',
            3: 'CANCELLATION',
            4: 'EXECUTE_VISIBLE',
            5: 'EXECUTE_HIDDEN',
            7: 'TRADING_HALT'
        }

        message_df = pd.read_csv(message_file_path)
        message_df.columns = ['timestamp', 'type', 'order_id', 'vol', 'price', 'direction']
        message_df['timestamp'] = self.date + pd.to_timedelta(message_df['timestamp'], unit='s')
        message_df['direction'] = message_df['direction'].replace(direction)
        message_df['price'] = message_df['price'] / 10000
        message_df['type'] = message_df['type'].replace(order_type)
        return message_df

    def readOrderbookFile(self, orderbook_file_path):
        """
        :return: a pandas Dataframe of the orderbook file for the given symbol and date
        """
        log_print("OrderBookOracle Orderbook File: {}".format(orderbook_file_path))
        all_cols = delist(
            [[f"ask_price_{level}", f"ask_size_{level}", f"bid_price_{level}", f"bid_size_{level}"] for level in
             range(1, self.num_price_levels + 1)])
        price_cols = delist(
            [[f"ask_price_{level}", f"bid_price_{level}"] for level in range(1, self.num_price_levels + 1)])
        orderbook_df = pd.read_csv(orderbook_file_path)
        orderbook_df.columns = all_cols
        orderbook_df[price_cols] = orderbook_df[price_cols] / 10000
        orderbook_df = orderbook_df.join(self.message_df[['timestamp']])
        orderbook_df = orderbook_df[['timestamp'] + all_cols]
        orderbook_df = orderbook_df.drop_duplicates(subset=['timestamp'], keep='last')
        orderbook_df.set_index('timestamp', inplace=True)
        return orderbook_df

    def filter_trades(self):
        log_print("Original trades type counts:")
        log_print(str(self.message_df.type.value_counts()))
        trades_df = self.message_df.loc[
            self.message_df.type.isin(['NEW', 'CANCELLATION', 'PARTIAL_CANCELLATION', 'EXECUTE_VISIBLE'])]
        order_id_types_series = trades_df.groupby('order_id')['type'].apply(list)
        order_id_types_series = order_id_types_series.apply(lambda x: str(x))
        cancel_only_order_ids = list(order_id_types_series[order_id_types_series == "['CANCELLATION']"].index)
        part_cancel_only_order_ids = list(
            order_id_types_series[order_id_types_series == "['PARTIAL_CANCELLATION']"].index)
        trades_df = trades_df.loc[~trades_df.order_id.isin(cancel_only_order_ids + part_cancel_only_order_ids)]
        log_print("Filtered trades type counts:")
        log_print(str(trades_df.type.value_counts()))
        return trades_df
>>>>>>> 2052760fb0c43ebd2c5008144699d9e0e9d2e88d
