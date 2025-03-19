from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
import talib as ta
import pandas as pd
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

def find_major_reversals(candles, current_close, min_threshold, max_threshold):
    lows = [float(candle[3]) for candle in candles if float(candle[3]) >= min_threshold]
    highs = [float(candle[2]) for candle in candles if float(candle[2]) <= max_threshold]

    last_bottom = np.nanmin(lows) if lows else None
    last_top = np.nanmax(highs) if highs else None

    closest_reversal = None
    closest_type = None

    if last_bottom is not None and (closest_reversal is None or abs(last_bottom - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_bottom
        closest_type = 'DIP'
    
    if last_top is not None and (closest_reversal is None or abs(last_top - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_top
        closest_type = 'TOP'

    if closest_type == 'TOP' and closest_reversal <= current_close:
        closest_type = None
        closest_reversal = None
    elif closest_type == 'DIP' and closest_reversal >= current_close:
        closest_type = None
        closest_reversal = None

    return last_bottom, last_top, closest_reversal, closest_type

filename = 'credentials.txt'
trader = Trader(filename)

filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pair_info = []

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

        print("on 15min timeframe " + symbol)
        x = close
        y = range(len(x))
        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
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
        best_fit_line3 = best_fit_line1 * 0.99

        if x[-1] < best_fit_line3[-1]:
            filtered_pairs3.append(symbol)
            print('found')

def momentum(filtered_pairs3):
    interval = '1m'
    for symbol in filtered_pairs3:
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue

        print("on 1m timeframe " + symbol)
        
        current_close = close[-1]
        min_threshold = current_close * 0.8
        max_threshold = current_close * 1.2
        
        last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(
            klines, current_close, min_threshold, max_threshold
        )
        
        if closest_type == 'DIP':
            print('mtf dip found')
            selected_pair.append(symbol)
            selected_pair_info.append({
                'last_bottom': last_bottom,
                'last_top': last_top,
                'closest_reversal': closest_reversal,
                'distance_to_reversal': abs(current_close - closest_reversal),
                'closest_type': closest_type
            })

def analyze_market_mood(selected_pairs):
    mood = ""
    if selected_pairs:
        last_info = selected_pair_info[-1]
        current_close = float(trader.client.get_klines(symbol=selected_pairs[-1], interval='1m')[-1][4])
        
        dist_to_min = abs(current_close - last_info['last_bottom']) if last_info['last_bottom'] is not None else float('inf')
        dist_to_max = abs(current_close - last_info['last_top']) if last_info['last_top'] is not None else float('inf')
        
        if dist_to_min < dist_to_max and last_info['closest_type'] == 'DIP':
            mood = "UP CYCLE"
        elif dist_to_max < dist_to_min and last_info['closest_type'] == 'TOP':
            mood = "DOWN CYCLE"
        else:
            mood = "NEUTRAL"
            
        print(f"Market Mood: {mood}")
        print(f"Distance to last bottom: {dist_to_min}")
        print(f"Distance to last top: {dist_to_max}")
        print(f"Last major reversal type: {last_info['closest_type']}")
    else:
        print("Market Mood: NO DATA (no pairs selected)")

for i in trading_pairs:
    filter1(i)

filter2(filtered_pairs1)
filter3(filtered_pairs2)
momentum(filtered_pairs3)

if len(selected_pair) > 1:
    print('more mtf dips are found') 
    print(selected_pair)
    min_distance = min([info['distance_to_reversal'] for info in selected_pair_info])
    position = [info['distance_to_reversal'] for info in selected_pair_info].index(min_distance)
    print(f"Strongest dip: {selected_pair[position]} at distance {min_distance}")

elif len(selected_pair) == 1:
    print('1 mtf dip found')   
    print(f"{selected_pair[0]} at distance {selected_pair_info[0]['distance_to_reversal']}")

analyze_market_mood(selected_pair)

sys.exit(0)