from binance.client import Client
import numpy as np
import talib as ta
import sys

class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    """ Get all pairs traded against USDC """
    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols'] 
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        return trading_pairs

filename = 'credentials.txt'
trader = Trader(filename)

filtered_pairs_15min = []
filtered_pairs_5min = []
filtered_pairs_3min = []
selected_pair = []
selected_pair_sine = []

# Dynamically fetch trading pairs against USDC
trading_pairs = trader.get_usdc_pairs()

def filter_15min(pair):
    interval = '15m'
    klines = trader.client.get_klines(symbol=pair, interval=interval)
    close = [float(entry[4]) for entry in klines]

    if not close:
        return  # Skip if no close price available

    print(f"on 15m timeframe {pair}")
    
    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line3 = best_fit_line1 * 0.99
    
    if x[-1] < best_fit_line3[-1]:
        filtered_pairs_15min.append(pair)
        print('15m dip found')

def filter_5min(filtered_pairs):
    interval = '5m'
    for symbol in filtered_pairs:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue  # Skip if no close price available

        print(f"on 5m timeframe {symbol}")

        x = close
        y = range(len(x))

        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line3 = best_fit_line1 * 0.99

        if x[-1] < best_fit_line3[-1]:
            filtered_pairs_5min.append(symbol)
            print('5m dip found')

def filter_3min(filtered_pairs):
    interval = '3m'
    for symbol in filtered_pairs:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue  # Skip if no close price available

        print(f"on 3m timeframe {symbol}")

        close_array = np.asarray(close)
        sine_wave, _ = ta.HT_SINE(close_array)

        if np.isfinite(sine_wave[-1]) and sine_wave[-1] < 0:
            filtered_pairs_3min.append(symbol)
            selected_pair_sine.append(sine_wave[-1])
            print('3m dip found')

def momentum(filtered_pairs):
    interval = '1m'
    for symbol in filtered_pairs:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue  # Skip if no close price available

        print(f"on 1m timeframe {symbol}")

        close_array = np.asarray(close)
        sine_wave, _ = ta.HT_SINE(close_array)

        # Check if the latest SINE value indicates a considerable dip
        if np.isfinite(sine_wave[-1]) and sine_wave[-1] < 0:
            selected_pair.append(symbol)
            print('1m dip found')

# Run filters on trading pairs
for pair in trading_pairs:
    filter_15min(pair)

filter_5min(filtered_pairs_15min)
filter_3min(filtered_pairs_5min)
momentum(filtered_pairs_3min)

if len(selected_pair) > 1:
    print('More MTF dips are found:')
    print(selected_pair)

    # Find the pair with the lowest SINE value from the 1m analysis
    if selected_pair_sine:
        min_sine_value = min(selected_pair_sine)
        position = selected_pair_sine.index(min_sine_value)

        print(f'Lowest SINE dip found in: {selected_pair[position]}')

elif len(selected_pair) == 1:
    print('1 MTF dip found:')
    print(selected_pair)

sys.exit(0)