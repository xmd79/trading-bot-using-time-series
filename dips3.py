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
        real = ta.CMO(np.asarray(self.get_close_prices(symbol, '1m')), timeperiod=14)

        if real[-1] < -50 and buy_volume > sell_volume:
            return symbol, real[-1]

        return None

if __name__ == "__main__":
    filename = 'credentials.txt'
    trader = Trader(filename)

    trading_pairs = trader.get_trading_pairs()
    filtered_pairs = trader.filter_pairs(trading_pairs, '2h')
    selected_pair, selected_pairCMO = trader.momentum(filtered_pairs)

    if len(selected_pair) > 1:
        print('Dips are more than 1 oversold:')
        print(selected_pair)
        print(selected_pairCMO)
        min_index = selected_pairCMO.index(min(selected_pairCMO))
        print(f"Selected pair: {selected_pair[min_index]}")

    elif len(selected_pair) == 1:
        print('1 dip found:')
        print(selected_pair)
        print(selected_pairCMO)

    else:
        print('No oversold dips for the moment.')
