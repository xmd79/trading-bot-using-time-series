from binance.client import Client
import numpy as np
import talib as ta
import concurrent.futures

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            key = lines[0].strip()
            secret = lines[1].strip()
            self.client = Client(key, secret)

    def get_trading_pairs(self):
        info = self.client.get_exchange_info()
        symbols = info['symbols']
        trading_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDT' 
                         and s['status'] == 'TRADING' and 'SPOT' in s['permissions']]
        return trading_pairs

    def get_volume(self, symbol, interval):
        volume_klines = self.client.get_klines(symbol=symbol, interval=interval)
        volumes = [float(entry[5]) for entry in volume_klines]
        buy_volume = sum([v for v in volumes if v > 0])
        sell_volume = sum([v for v in volumes if v < 0])
        return buy_volume, sell_volume

    def get_close_prices(self, symbol, interval):
        klines = self.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]
        return close

    def get_best_fit_line(self, x):
        y = range(len(x))
        return np.poly1d(np.polyfit(y, x, 1))(y)

    def filter_pairs(self, pairs, interval):
        filtered_pairs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.filter_pair, pair, interval): pair for pair in pairs}
            for future in concurrent.futures.as_completed(futures):
                pair, buy_volume, sell_volume = future.result()
                if pair:
                    filtered_pairs.append((pair, buy_volume, sell_volume))
        return filtered_pairs

    def filter_pair(self, pair, interval):
        close = self.get_close_prices(pair, interval)
        buy_volume, sell_volume = self.get_volume(pair, '1m')
        best_fit_line = self.get_best_fit_line(close)

        if close[-1] < best_fit_line[-1]:
            return pair, buy_volume, sell_volume

        return None, None, None

    def momentum(self, filtered_pairs):
        selected_pair = []
        selected_pairCMO = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.calculate_momentum, pair): pair for pair in filtered_pairs}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    selected_pair.append(result[0])
                    selected_pairCMO.append(result[1])

        return selected_pair, selected_pairCMO

    def calculate_momentum(self, pair):
        symbol, buy_volume, sell_volume = pair
        close = np.asarray(self.get_close_prices(symbol, '1m'))

        # Check if the pair qualifies as a dip on 2h, 15min, and 5min timeframes
        if self.is_dip_on_multiple_timeframes(symbol):
            real = ta.CMO(close, timeperiod=14)
            return symbol, real[-1]

        return None

    def is_dip_on_multiple_timeframes(self, symbol):
        # Check if the pair qualifies as a dip on 2h timeframe
        close_2h = np.asarray(self.get_close_prices(symbol, '2h'))
        best_fit_line_2h = self.get_best_fit_line(close_2h)
        if close_2h[-1] < best_fit_line_2h[-1]:
            # Check if the pair qualifies as a dip on 15min timeframe
            close_15m = np.asarray(self.get_close_prices(symbol, '15m'))
            best_fit_line_15m = self.get_best_fit_line(close_15m)
            if close_15m[-1] < best_fit_line_15m[-1]:
                # Check if the pair qualifies as a dip on 5min timeframe
                close_5m = np.asarray(self.get_close_prices(symbol, '5m'))
                best_fit_line_5m = self.get_best_fit_line(close_5m)
                if close_5m[-1] < best_fit_line_5m[-1]:
                    return True
        return False

if __name__ == "__main__":
    filename = 'credentials.txt'
    trader = Trader(filename)

    trading_pairs = trader.get_trading_pairs()
    filtered_pairs_2h = trader.filter_pairs(trading_pairs, '2h')
    selected_pair_2h, _ = trader.momentum(filtered_pairs_2h)

    if selected_pair_2h:
        print('Dips for selected pairs on 2h timeframe:', selected_pair_2h)

        # Filter pairs for 15min timeframe
        filtered_pairs_15m = trader.filter_pairs(selected_pair_2h, '15m')
        if filtered_pairs_15m:
            selected_pair_15m, _ = trader.momentum(filtered_pairs_15m)
            print('Dips for selected pairs on 15min timeframe:', selected_pair_15m)

            # Filter pairs for 5min timeframe
            filtered_pairs_5m = trader.filter_pairs(selected_pair_15m, '5m')
            if filtered_pairs_5m:
                selected_pair_5m, _ = trader.momentum(filtered_pairs_5m)
                print('Dips for selected pairs on 5min timeframe:', selected_pair_5m)

                # Calculate CMO only for pairs found on multiple timeframes on 1min timeframe
                filtered_pairs_1m = trader.filter_pairs(selected_pair_5m, '1m')
                if filtered_pairs_1m:
                    selected_pair_1m, selected_pairCMO_1m = trader.momentum(filtered_pairs_1m)
                    print('Dips for selected pairs on 1min timeframe:', selected_pair_1m)

                    # If more than one dip found on 1min timeframe, select the dip with lowest CMO value
                    if len(selected_pair_1m) > 1:
                        min_cmo_index = np.argmin(selected_pairCMO_1m)
                        selected_pair_1m = [selected_pair_1m[min_cmo_index]]
                        print('Selected dip with the lowest CMO value on 1min timeframe:', selected_pair_1m)
                    else:
                        print('Dips for selected pairs on 1min timeframe:', selected_pair_1m)

