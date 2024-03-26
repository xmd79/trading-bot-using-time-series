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
        selected_pairMomentum = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.calculate_momentum, pair): pair for pair in filtered_pairs}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    selected_pair.append(result[0])
                    selected_pairMomentum.append(result[1])

        return selected_pair, selected_pairMomentum

    def calculate_momentum(self, pair):
        symbol, buy_volume, sell_volume = pair
        close = np.asarray(self.get_close_prices(symbol, '1m'))

        # Calculate the best fit line for close prices
        best_fit_line = self.get_best_fit_line(close)

        # Calculate y-coordinate of the point on the line corresponding to a 45-degree angle
        x = len(close)
        y_at_45_deg = best_fit_line[-1] + (1 / np.sqrt(2)) * x  # Assuming unit vector for the angle

        # Check if the latest close price is below the line corresponding to 45-degree angle
        if close[-1] < y_at_45_deg:
            # Check if TALib momentum is positive for bullish momentum
            momentum = ta.MOM(close, timeperiod=14)[-1]
            if momentum > 0:
                return symbol, 'Bullish'
            else:
                return symbol, 'Bearish'

        return None

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

                # Calculate momentum only for pairs found on multiple timeframes on 1min timeframe
                filtered_pairs_1m = trader.filter_pairs(selected_pair_5m, '1m')
                if filtered_pairs_1m:
                    selected_pair_1m, selected_pairMomentum_1m = trader.momentum(filtered_pairs_1m)
                    print('Momentum for selected pairs on 1min timeframe:')
                    
                    # Filter pairs with positive momentum
                    positive_momentum_pairs = [(pair, momentum) for pair, momentum in zip(selected_pair_1m, selected_pairMomentum_1m) if momentum == 'Bullish']
                    if positive_momentum_pairs:
                        # Sort pairs by momentum if positive momentum pairs exist
                        sorted_pairs = sorted(positive_momentum_pairs, key=lambda x: x[1], reverse=True)
                        most_significant_dip = sorted_pairs[0][0]  # Select the pair with the highest positive momentum
                    else:
                        # If no positive momentum pairs, sort pairs by momentum and select the one with the lowest value
                        sorted_pairs = sorted(zip(selected_pair_1m, selected_pairMomentum_1m), key=lambda x: x[1])
                        most_significant_dip = sorted_pairs[0][0]  # Select the pair with the lowest momentum value
                        
                    print('Most significant dip:', most_significant_dip)
                    
                    for pair, momentum in zip(selected_pair_1m, selected_pairMomentum_1m):
                        print(f'{pair}: {momentum}')
