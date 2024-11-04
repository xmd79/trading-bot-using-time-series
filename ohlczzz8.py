from binance.client import Client
import numpy as np
import talib as ta
import os
import sys

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def getBalances(self):
        prices = self.client.get_withdraw_history()
        return prices

    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
        return trading_pairs

    def calculate_sma(self, data, period):
        """ Calculate the Simple Moving Average. """
        if len(data) < period:
            return np.nan
        return np.mean(data[-period:])


filename = 'credentials.txt'
trader = Trader(filename)

filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pair_momentum = []

trading_pairs = trader.get_usdc_pairs()

def filter1(pair):
    interval = '1h'
    symbol = pair
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    close = [float(entry[4]) for entry in klines]

    if not close:
        return 

    print("on 1h timeframe " + symbol)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    if x[-1] < best_fit_line3[-1]:
        filtered_pairs1.append(symbol)
        print('found')

def filter2(filtered_pairs1):
    interval = '15m'
    for symbol in filtered_pairs1:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue 

        print("on 15m timeframe " + symbol)

        x = close
        y = range(len(x))

        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line2 = best_fit_line1 * 1.01
        best_fit_line3 = best_fit_line1 * 0.99

        if x[-1] < best_fit_line3[-1]:
            filtered_pairs2.append(symbol)
            print('found')

def filter3(filtered_pairs2):
    interval = '5m'
    for symbol in filtered_pairs2:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue 

        print("on 5m timeframe " + symbol)

        x = close
        y = range(len(x))

        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line2 = best_fit_line1 * 1.01
        best_fit_line3 = best_fit_line1 * 0.99

        if x[-1] < best_fit_line3[-1]:
            filtered_pairs3.append(symbol)
            print('found')

def momentum(filtered_pairs3):
    interval = '1m'
    for symbol in filtered_pairs3:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]
        volume = [float(entry[5]) for entry in klines]

        if len(close) < 14 or len(volume) < 14:
            continue

        print("on 1m timeframe " + symbol)

        close_array = np.asarray(close)
        volume_array = np.asarray(volume)

        # Calculate momentum and check the last two values
        mom = ta.MOM(close_array, timeperiod=14)
        
        # Check if last momentum value is negative
        if mom[-1] < 0:
            # Check if bullish volume is greater than bearish volume
            bullish_volume = sum(v for v in volume_array if v > 0)
            bearish_volume = -sum(v for v in volume_array if v < 0)

            # Condition for selected pairs
            if bullish_volume > bearish_volume:
                print('MTF dip found')
                selected_pair.append(symbol)
                selected_pair_momentum.append(mom[-1])

def check_price_position(symbol):
    """ Check if the latest price is below the 45 degree angle """
    interval = '1m'
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    close = [float(entry[4]) for entry in klines]
    if len(close) < 2:
        return False
    
    y = np.array(range(len(close)))
    best_fit_line = np.poly1d(np.polyfit(y, close, 1))(y)
    
    if close[-1] < best_fit_line[-1] * 0.99:  # Check if price is below a 1% drop from the trendline
        return True
    return False

def analyze_assets():
    for symbol in trading_pairs:
        daily_klines = trader.client.get_klines(symbol=symbol, interval='1d')
        daily_close = [float(entry[4]) for entry in daily_klines]
        sma_50 = trader.calculate_sma(daily_close, 50)
        
        # Check conditions for daily dips
        if daily_close[-1] < sma_50:
            print(f'Daily dip detected for {symbol}')
            filter1(symbol)  # Run further filtering only if the daily criteria are met

# Main execution
analyze_assets()  # Analyze assets based on daily timeframe
filter2(filtered_pairs1)
filter3(filtered_pairs2)
momentum(filtered_pairs3)

if selected_pair:
    print(f'Total selected pairs found: {len(selected_pair)}')
    
    # Sort selected pairs based on momentum
    sorted_selected_pairs = sorted(zip(selected_pair, selected_pair_momentum), key=lambda x: x[1])

    print("\nSorted selected pairs with momentum:")
    for symbol, mom in sorted_selected_pairs:
        print(f'{symbol}: Momentum = {mom}')

        if check_price_position(symbol):
            print(f'Lowest dip with potential to pump: {symbol}, Momentum: {mom}')
else:
    print('No MTF dips found.')

sys.exit(0)
exit()