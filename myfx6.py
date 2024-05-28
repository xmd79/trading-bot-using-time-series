import numpy as np
import sys
import talib
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import concurrent.futures

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        try:
            lines = [line.rstrip('\n') for line in open(file)]
            key = lines[0]
            secret = lines[1]
            self.client = Client(key, secret)
            print("Connected to Binance API successfully.")
        except Exception as e:
            print("Error connecting to Binance API:", e)
            sys.exit(1)

filename = 'credentials.txt'
trader = Trader(filename)

filtered_pairs_dips = []
filtered_pairs_tops = []
intermediate_pairs = []

selected_pair_dips = []

trading_pairs = []

# Fetching all trading pairs from Binance spot market trading against USDT
try:
    tickers = trader.client.get_all_tickers()
    exchange_info = trader.client.get_exchange_info()
    symbols_info = exchange_info['symbols']
    active_trading_pairs = [symbol['symbol'] for symbol in symbols_info if symbol['status'] == 'TRADING']

    print("All active trading pairs vs USDT (Spot trading only):")
    for ticker in tickers:
        symbol = ticker['symbol']
        if symbol.endswith("USDT") and symbol in active_trading_pairs:
            trading_pairs.append(symbol)
            print(f"Symbol: {symbol}, Current Price: {ticker['price']}")
except Exception as e:
    print("Error fetching trading pairs:", e)

def filter1(pair):
    interval = '30m'
    symbol = pair
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    current_price = x[-1]
    poly_value = best_fit_line1[-1]

    if current_price < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs_dips.append(symbol)
        status = 'Dip'
    elif current_price > best_fit_line2[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs_tops.append(symbol)
        status = 'Top'
    else:
        intermediate_pairs.append(symbol)
        status = 'In between'

    print(f"Symbol: {symbol}, Status: {status}, Current Price: {current_price}, Poly Value: {poly_value}")

def filter2(pair):
    interval = '5m'
    symbol = pair
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    current_price = x[-1]
    poly_value = best_fit_line1[-1]

    if current_price < best_fit_line1[-1]:
        selected_pair_dips.append(symbol)
        status = 'Dip confirmed on 5m'
    else:
        status = 'Not a dip on 5m'

    print(f"Symbol: {symbol}, Status: {status}, Current Price: {current_price}, Poly Value: {poly_value}")

def generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    harmonics = [1, 2, 3, 5, 7, 9, 12]
    wave = np.zeros_like(t)
    for h in harmonics:
        wave += np.sin(2 * np.pi * frequency * h * t + phase_shift)
        wave += np.sin(2 * np.pi * frequency * h * t - phase_shift)
    ascending_triangle = np.abs((t % (1 / frequency)) * frequency - 0.5) * 2
    descending_triangle = 1 - ascending_triangle
    wave += ascending_triangle
    wave += descending_triangle
    phi = (1 + np.sqrt(5)) / 2
    euler = np.exp(1)
    wave += np.sin(2 * np.pi * frequency * phi * t)
    wave += np.sin(2 * np.pi * frequency * np.pi * t)
    wave += np.sin(2 * np.pi * frequency * euler * t)
    wave += np.sin(2 * np.pi * frequency * phi**2 * t)
    wave += np.sin(2 * np.pi * frequency * np.pi**2 * t)
    wave += np.sin(2 * np.pi * frequency * euler**2 * t)
    primes = [2, 3, 5, 7, 11]
    for p in primes:
        wave += np.sin(2 * np.pi * frequency * p**2 * t)
   
    # Use HT_SINE to generate a stationary wave
    sine_wave, _ = talib.HT_SINE(wave)
    
    # Replace NaN values with 0
    sine_wave = np.nan_to_num(sine_wave)

    return t, sine_wave

def analyze_wave(t, wave, frequency, sampling_rate, close_real_price):
    # Determine the current cycle between reversals
    period = 1 / frequency
    half_period = period / 2
    current_time_index = len(wave) - 1

    # Find the previous reversal point
    prev_reversal_index = current_time_index
    for i in range(current_time_index - 1, -1, -1):
        if wave[i] * wave[i - 1] <= 0:
            prev_reversal_index = i
            break

    # Calculate the cycle start time and end time
    cycle_start_time = t[prev_reversal_index]
    cycle_end_time = t[current_time_index]

    # Calculate the cycle duration
    cycle_duration = cycle_end_time - cycle_start_time

    # Calculate current time and distance to next reversal
    current_time = t[current_time_index]
    next_reversal_time = cycle_start_time + 1800  # 30 minutes
    time_to_next_reversal = next_reversal_time - current_time

    # Determine if the current cycle is up or down
    cycle_direction = "Up" if wave[prev_reversal_index] < wave[current_time_index] else "Down"

    # Determine if it's a dip or top incoming
    incoming = "Dip" if cycle_direction == "Down"  else "Top"

    # Calculate the middle threshold for the stationary sine
    middle_threshold = (np.max(wave) + np.min(wave)) / 2

    # Calculate the current close value on the sine wave
    current_close_sine_value = wave[-1]

    # Calculate the distance from the current close value on the sine wave to the minimum and maximum values
    distance_to_min = np.abs(current_close_sine_value - np.min(wave))
    distance_to_max = np.abs(current_close_sine_value - np.max(wave))

    # Calculate the percentages scaled and normalized symmetrically to each other on a scale from 0 to 100%
    total_distance = distance_to_min + distance_to_max
    percentage_to_min = (distance_to_min / total_distance) * 100
    percentage_to_max = (distance_to_max / total_distance) * 100

    # Calculate the real price values corresponding to the min and max of HT_SINE
    min_real_price = np.min(wave)
    max_real_price = np.max(wave)

    print("Current Cycle between Reversals:")
    print(f"Start Time: {cycle_start_time}, End Time: {cycle_end_time}, Duration: {cycle_duration} seconds")
    print(f"Current Time: {current_time}, Time to Next Reversal: {time_to_next_reversal} seconds")
    print(f"Incoming Reversal: {incoming}")
    print(f"Min Real Price: {min_real_price}, Max Real Price: {max_real_price}")
    print(f"Cycle Direction: {cycle_direction}")

    print("Analysis of Current Cycle:")
    print(f"Current Close Real Price: {close_real_price}")
    print(f"Current Close Value on Sine: {current_close_sine_value}")
    print(f"Distance from Current Close Value to Min: {distance_to_min}")
    print(f"Distance from Current Close Value to Max: {distance_to_max}")
    print(f"Percentage to Min: {percentage_to_min}%")
    print(f"Percentage to Max: {percentage_to_max}%")
    print(f"Middle Threshold for Stationary Sine: {middle_threshold}")

    # Determine and print if the last reversal was confirmed
    last_reversal_value = wave[prev_reversal_index]
    last_reversal_type = "Dip" if last_reversal_value == np.min(wave) else "Top"
    print(f"Last Reversal Type: {last_reversal_type}")

def scan_assets(pair):
    filter1(pair)

# Multithreading for faster asset scanning
print('Scanning all available assets on main timeframe...')
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(scan_assets, trading_pairs)

print('Filtered dips (30m):', filtered_pairs_dips)
print('Filtered tops (30m):', filtered_pairs_tops)
print('Intermediate pairs (30m):', intermediate_pairs)

# Rescan dips on lower timeframes
if len(filtered_pairs_dips) > 0:
    print('Rescanning dips on lower timeframes...')
    selected_pair_dips = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(filter2, filtered_pairs_dips)

    print('Dips after 5m filter:', selected_pair_dips)

    if len(selected_pair_dips) > 1:
        print('Multiple dips found on 5m timeframe. Analyzing to select the lowest...')
        lowest_dip = None
        lowest_value = float('inf')

        for pair in selected_pair_dips:
            interval = '5m'
            klines = trader.client.get_klines(symbol=pair, interval=interval)
            close = [float(entry[4]) for entry in klines]
            close_array = np.asarray(close)

            # Generating a sine wave for comparison
            frequency = 1 / len(close)
            phase_shift = 0
            angle = 0
            duration = len(close) / 12  # Assumed duration
            sampling_rate = len(close)  # Assumed sampling rate

            t, wave = generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate)

            # Compare the lowest value on the sine wave
            min_wave_value = np.min(wave)

            if min_wave_value < lowest_value:
                lowest_value = min_wave_value
                lowest_dip = pair

        print(f'Lowest dip on 5m timeframe is {lowest_dip} with wave value {lowest_value}')
        print(f'Current asset vs USDT: {lowest_dip}')
        interval = '5m'
        klines = trader.client.get_klines(symbol=lowest_dip, interval=interval)
        close = [float(entry[4]) for entry in klines]
        close_array = np.asarray(close)

        frequency = 1  / len(close)
        phase_shift = 0
        angle = 0
        duration = len(close) / 12
        sampling_rate = len(close)

        t, wave = generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate)
        close_real_price = close[-1]
        analyze_wave(t, wave, frequency, sampling_rate, close_real_price)

    else:
        print(f'Selected dip on 5m timeframe: {selected_pair_dips[0]}')
        print(f'Current asset vs USDT: {selected_pair_dips[0]}')
        interval = '5m'
        klines = trader.client.get_klines(symbol=selected_pair_dips[0], interval=interval)
        close = [float(entry[4]) for entry in klines]
        close_array = np.asarray(close)

        frequency = 1 / len(close)
        phase_shift = 0
        angle = 0
        duration = len(close) / 12
        sampling_rate = len(close)

        t, wave = generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate)
        close_real_price = close[-1]
        analyze_wave(t, wave, frequency, sampling_rate, close_real_price)

else:
    print('No dips found in main timeframe.')

