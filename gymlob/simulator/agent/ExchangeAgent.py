import datetime as dt
import warnings

from gymlob.simulator.agent.FinancialAgent import FinancialAgent
from gymlob.simulator.util.Message import Message
from gymlob.simulator.util.OrderBook import OrderBook
from gymlob.simulator.util.util import log_print

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from copy import deepcopy


class ExchangeAgent(FinancialAgent):

    def __init__(self, id, name, type, mkt_open, mkt_close, symbols, book_freq='S', wide_book=False,
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, random_state)

        self.mkt_open = mkt_open
        self.mkt_close = mkt_close

        # Log all order activity?
        self.log_orders = log_orders

        # Create an order book for each symbol.
        self.order_books = {}

        self.symbols = symbols
        for symbol in symbols:
            self.order_books[symbol] = OrderBook(self, symbol)

        # At what frequency will we archive the order books for visualization and analysis?
        self.book_freq = book_freq

        # Store orderbook in wide format? ONLY WORKS with book_freq == 0
        self.wide_book = wide_book

    def kernelInitializing(self, kernel):
        super().kernelInitializing(kernel)

    def kernelTerminating(self):
        super().kernelTerminating()

        if self.book_freq is None:
            return
        else:
            # Iterate over the order books controlled by this exchange.
            for symbol in self.order_books:
                start_time = dt.datetime.now()
                self.logOrderBookSnapshots(symbol)
                end_time = dt.datetime.now()
                print("Time taken to log the order book: {}".format(end_time - start_time))
                print("Order book archival complete.")

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        if currentTime > self.mkt_close:

            if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
                self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))
                return
            elif 'QUERY' in msg.body['msg']:
                pass
            else:
                log_print("{} received {}, discarded: market is closed.", self.name, msg.body['msg'])
                self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))
                return

        if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
            if self.log_orders: self.logEvent(msg.body['msg'], msg.body['order'].to_dict())
        else:
            self.logEvent(msg.body['msg'], msg.body['sender'])

        if msg.body['msg'] == "WHEN_MKT_OPEN":
            self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_OPEN", "data": self.mkt_open}))

        elif msg.body['msg'] == "WHEN_MKT_CLOSE":
            self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_CLOSE", "data": self.mkt_close}))

        elif msg.body['msg'] == "QUERY_LAST_TRADE":
            symbol = msg.body['symbol']
            if symbol not in self.order_books:
                log_print("Last trade request discarded.  Unknown symbol: {}", symbol)
            else:
                log_print("{} received QUERY_LAST_TRADE ({}) request from agents {}", self.name, symbol,
                          msg.body['sender'])
                self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_LAST_TRADE", "symbol": symbol,
                                                              "data": self.order_books[symbol].last_trade,
                                                              "mkt_closed": True if currentTime > self.mkt_close
                                                                                 else False}))
        elif msg.body['msg'] == "QUERY_SPREAD":
            symbol = msg.body['symbol']
            depth = msg.body['depth']
            self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
                                                          "bids": self.order_books[symbol].getInsideBids(depth),
                                                          "asks": self.order_books[symbol].getInsideAsks(depth),
                                                          "data": self.order_books[symbol].last_trade,
                                                          "mkt_closed": True if currentTime > self.mkt_close else False,
                                                          "book": ''}))

        elif msg.body['msg'] == "QUERY_ORDER_STREAM":
            symbol = msg.body['symbol']
            length = msg.body['length']
            self.sendMessage(msg.body['sender'],
                             Message({"msg": "QUERY_ORDER_STREAM", "symbol": symbol, "length": length,
                                      "mkt_closed": True if currentTime > self.mkt_close else False,
                                      "orders": self.order_books[symbol].history[1:length + 1]}))

        elif msg.body['msg'] == "LIMIT_ORDER":
            order = msg.body['order']
            self.order_books[order.symbol].handleLimitOrder(deepcopy(order))

        elif msg.body['msg'] == "MARKET_ORDER":
            order = msg.body['order']
            self.order_books[order.symbol].handleMarketOrder(deepcopy(order))

        elif msg.body['msg'] == "CANCEL_ORDER":
            order = msg.body['order']
            self.order_books[order.symbol].cancelOrder(deepcopy(order))

        elif msg.body['msg'] == 'MODIFY_ORDER':
            order = msg.body['order']
            new_order = msg.body['new_order']
            self.order_books[order.symbol].modifyOrder(deepcopy(order), deepcopy(new_order))

    def logOrderBookSnapshots(self, symbol):
        """
        Log full depth quotes (price, volume) from this order book at some pre-determined frequency. Here we are looking at
        the actual log for this order book (i.e. are there snapshots to export, independent of the requested frequency).
        """

        def get_quote_range_iterator(s):
            """ Helper method for order book logging. Takes pandas Series and returns python range() from first to last
          element.
      """
            forbidden_values = [0, 19999900]  # TODO: Put constant value in more sensible place!
            quotes = sorted(s)
            for val in forbidden_values:
                try:
                    quotes.remove(val)
                except ValueError:
                    pass
            return quotes

        book = self.order_books[symbol]

        if book.book_log:

            import pandas as pd

            print("Logging order book to file...")
            dfLog = book.book_log_to_df()
            dfLog.set_index('QuoteTime', inplace=True)
            dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
            dfLog.sort_index(inplace=True)

            if str(self.book_freq).isdigit() and int(self.book_freq) == 0:  # Save all possible information
                # Get the full range of quotes at the finest possible resolution.
                quotes = get_quote_range_iterator(dfLog.columns.unique())

                # Restructure the log to have multi-level rows of all possible pairs of time and quote
                # with volume as the only column.
                if not self.wide_book:
                    filledIndex = pd.MultiIndex.from_product([dfLog.index, quotes], names=['time', 'quote'])
                    dfLog = dfLog.stack()
                    dfLog = dfLog.reindex(filledIndex)

                filename = f'ORDERBOOK_{symbol}_FULL'

            else:  # Sample at frequency self.book_freq
                # With multiple quotes in a nanosecond, use the last one, then resample to the requested freq.
                dfLog = dfLog.resample(self.book_freq).ffill()
                dfLog.sort_index(inplace=True)

                # Create a fully populated index at the desired frequency from market open to close.
                # Then project the logged data into this complete index.
                time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
                dfLog = dfLog.reindex(time_idx, method='ffill')
                dfLog.sort_index(inplace=True)

                if not self.wide_book:
                    dfLog = dfLog.stack()
                    dfLog.sort_index(inplace=True)

                    # Get the full range of quotes at the finest possible resolution.
                    quotes = get_quote_range_iterator(dfLog.index.get_level_values(1).unique())

                    # Restructure the log to have multi-level rows of all possible pairs of time and quote
                    # with volume as the only column.
                    filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time', 'quote'])
                    dfLog = dfLog.reindex(filledIndex)

                filename = f'ORDERBOOK_{symbol}_FREQ_{self.book_freq}'

            # Final cleanup
            if not self.wide_book:
                dfLog.rename('Volume')
                df = pd.SparseDataFrame(index=dfLog.index)
                df['Volume'] = dfLog
            else:
                df = dfLog
                df = df.reindex(sorted(df.columns), axis=1)

            # Archive the order book snapshots directly to a file named with the symbol, rather than
            # to the exchange agents log.
            self.writeLog(df, filename=filename)
            print("Order book logging complete!")

    def sendMessage(self, recipientID, msg):
        super().sendMessage(recipientID, msg)
        if self.log_orders and msg.body['msg'] in ['ORDER_ACCEPTED', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
            self.logEvent(msg.body['msg'], msg.body['order'].to_dict())