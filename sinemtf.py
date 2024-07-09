import numpy as np
import sys
from datetime import datetime
from binance.client import Client
import pandas as pd
import concurrent.futures
import talib

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        try:
            with open(file, 'r') as f:
                key = f.readline().strip()
                secret = f.readline().strip()
            self.client = Client(key, secret)
            print("Connected to Binance API successfully.")
        except Exception as e:
            print("Error connecting to Binance API:", e)
            sys.exit(1)

def get_active_trading_pairs(client):
    tickers = client.get_all_tickers()
    exchange_info = client.get_exchange_info()
    symbols_info = exchange_info['symbols']
    active_trading_pairs = [symbol['symbol'] for symbol in symbols_info if symbol['status'] == 'TRADING' and symbol['symbol'].endswith("USDT")]
    return active_trading_pairs

def get_klines(client, symbol, interval):
    klines = client.get_klines(symbol=symbol, interval=interval)
    data = {
        'Date': [datetime.fromtimestamp(entry[0] / 1000.0) for entry in klines],
        'Open': [float(entry[1]) for entry in klines],
        'High': [float(entry[2]) for entry in klines],
        'Low': [float(entry[3]) for entry in klines],
        'Close': [float(entry[4]) for entry in klines],
        'Volume': [float(entry[5]) for entry in klines],
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def filter_pair_with_ht_sine(client, pair, interval, filtered_pairs_dips, filtered_pairs_tops, intermediate_pairs):
    df = get_klines(client, pair, interval)
    close_prices = df['Close'].values

    sine, leadsine = talib.HT_SINE(close_prices)
    current_price = close_prices[-1]
    current_sine = sine[-1]
    previous_sine = sine[-2]
    current_leadsine = leadsine[-1]
    previous_leadsine = leadsine[-2]

    if current_sine < current_leadsine and previous_sine > previous_leadsine:
        filtered_pairs_dips.append(pair)
        status = 'Dip'
    elif current_sine > current_leadsine and previous_sine < previous_leadsine:
        filtered_pairs_tops.append(pair)
        status = 'Top'
    else:
        intermediate_pairs.append(pair)
        status = 'In between'

    print(f"{datetime.now()} - Symbol: {pair}, Status: {status}, Current Price: {current_price}, Sine: {current_sine}, Lead Sine: {current_leadsine}")

def rescan_dips_with_ht_sine(client, pair, selected_pair_dips):
    interval = '30m'
    df = get_klines(client, pair, interval)
    close_prices = df['Close'].values

    sine, leadsine = talib.HT_SINE(close_prices)
    current_price = close_prices[-1]
    current_sine = sine[-1]
    current_leadsine = leadsine[-1]

    if current_sine < current_leadsine:
        selected_pair_dips.append(pair)
        status = 'Dip confirmed on second timeframe'
    else:
        status = 'Not a dip on second timeframe'

    print(f"{datetime.now()} - Symbol: {pair}, Status: {status}, Current Price: {current_price}, Sine: {current_sine}, Lead Sine: {current_leadsine}")

def filter_dips_with_ht_sine_momentum(client, pair, selected_pair_dips_momentum):
    interval = '1m'
    df = get_klines(client, pair, interval)
    close_prices = df['Close'].values

    sine, leadsine = talib.HT_SINE(close_prices)
    current_momentum = sine[-1] - leadsine[-1]

    if current_momentum < 0:
        selected_pair_dips_momentum.append((pair, current_momentum))
        status = 'Momentum indicates dip'
    else:
        status = 'Momentum does not indicate dip'

    print(f"{datetime.now()} - Symbol: {pair}, Status: {status}, Current Momentum: {current_momentum}")

# Main execution
filename = 'credentials.txt'
trader = Trader(filename)

filtered_pairs_dips = []
filtered_pairs_tops = []
intermediate_pairs = []
selected_pair_dips = []
selected_pair_dips_momentum = []

try:
    active_trading_pairs = get_active_trading_pairs(trader.client)

    print("All active trading pairs vs USDT (Spot trading only):")
    for pair in active_trading_pairs:
        print(f"Symbol: {pair}")

    print('Scanning all available assets on main timeframe...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda pair: filter_pair_with_ht_sine(trader.client, pair, '2h', filtered_pairs_dips, filtered_pairs_tops, intermediate_pairs), active_trading_pairs)

    print('Filtered dips (2h):', filtered_pairs_dips)
    print('Filtered tops (2h):', filtered_pairs_tops)
    print('Intermediate pairs (2h):', intermediate_pairs)

    if filtered_pairs_dips:
        print('Rescanning dips on lower timeframes...')
        selected_pair_dips = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda pair: rescan_dips_with_ht_sine(trader.client, pair, selected_pair_dips), filtered_pairs_dips)

        print('Dips after second filter:', selected_pair_dips)

        if len(selected_pair_dips) > 1:
            print('Multiple dips found on second timeframe. Analyzing to select the lowest...')
            lowest_dip = None
            lowest_value = float('inf')

            for pair in selected_pair_dips:
                interval = '5m'
                df = get_klines(trader.client, pair, interval)
                close_prices = df['Close'].values

                sine, _ = talib.HT_SINE(close_prices)
                min_sine_value = np.min(sine)

                if min_sine_value < lowest_value:
                    lowest_value = min_sine_value
                    lowest_dip = pair

            print(f'Lowest dip on 5m timeframe is {lowest_dip} with sine value {lowest_value}')
            print(f'Current asset vs USDT: {lowest_dip}')

            print('Applying final momentum filter and sorting...')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(lambda pair: filter_dips_with_ht_sine_momentum(trader.client, pair, selected_pair_dips_momentum), selected_pair_dips)

            selected_pair_dips_momentum.sort(key=lambda x: x[1])
            print('Dips after momentum filter and sorting:')
            for pair, momentum in selected_pair_dips_momentum:
                print(f'Symbol: {pair}, Momentum: {momentum}')
        else:
            print('No dips found on second timeframe.')
    else:
        print('No dips found on main timeframe.')
except Exception as e:
    print("Error during execution:", e)
