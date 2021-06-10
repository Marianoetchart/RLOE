import pandas as pd


class LOBSTERDataLoader:

    def __init__(self, instrument, date, frequency, num_levels, orders_file_path, orderbook_file_path):
        self.instrument = instrument
        self.date = date
        self.frequency = frequency
        self.num_levels = num_levels

        self.columns = [y for x in [[f"ask_price_{level}", f"ask_size_{level}",
                                     f"bid_price_{level}", f"bid_size_{level}"]
                                    for level in range(1, num_levels + 1)] for y in x]
        self.price_cols = [x for x in self.columns if 'price' in x]
        self.size_cols = [x for x in self.columns if 'size' in x]

        self.orders_df = self.read_orders_file(orders_file_path)
        self.orderbook_df = self.read_orderbook_file(orderbook_file_path)

    def read_orders_file(self, file_path):
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

        orders_df = pd.read_csv(file_path)
        orders_df.columns = ['timestamp', 'type', 'order_id', 'vol', 'price', 'direction']
        orders_df['timestamp'] = pd.to_datetime(self.date) + pd.to_timedelta(orders_df['timestamp'], unit='s')
        orders_df['direction'] = orders_df['direction'].replace(direction)
        orders_df['price'] = orders_df['price'] / 10000
        orders_df['type'] = orders_df['type'].replace(order_type)

        return orders_df

    def read_orderbook_file(self, file_path):
        orderbook_df = pd.read_csv(file_path)
        orderbook_df.columns = self.columns
        orderbook_df[self.price_cols] = orderbook_df[self.price_cols] / 10000
        orderbook_df = orderbook_df.join(self.orders_df[['timestamp']])
        orderbook_df = orderbook_df.drop_duplicates(subset=['timestamp'], keep='last')

        orderbook_df['mid_price'] = (orderbook_df['ask_price_1'] + orderbook_df['bid_price_1']) / 2
        orderbook_df['spread'] = orderbook_df['ask_price_1'] - orderbook_df['bid_price_1']
        orderbook_df['order_volume_imbalance'] = (orderbook_df['ask_size_1'] - orderbook_df['bid_size_1']) / \
                                                 (orderbook_df['ask_size_1'] + orderbook_df['bid_size_1'])

        orderbook_df = orderbook_df[['timestamp', 'mid_price', 'spread', 'order_volume_imbalance'] + self.columns]
        orderbook_df.set_index('timestamp', inplace=True)

        if self.frequency:
            orderbook_df = orderbook_df.resample(self.frequency).bfill()
        return orderbook_df