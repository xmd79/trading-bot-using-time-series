import numpy as np
import sys
from datetime import datetime
from binance.client import Client
import pandas as pd
import concurrent.futures
from scipy.signal import hilbert
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

def calculate_best_fit_lines(close_prices):
    y = range(len(close_prices))
    best_fit_line1 = np.poly1d(np.polyfit(y, close_prices, 1))(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99
    return best_fit_line1, best_fit_line2, best_fit_line3

def filter_pair(client, pair, interval, filtered_pairs_dips, filtered_pairs_tops, intermediate_pairs):
    df = get_klines(client, pair, interval)
    close_prices = df['Close'].values

    best_fit_line1, best_fit_line2, best_fit_line3 = calculate_best_fit_lines(close_prices)
    current_price = close_prices[-1]

    if current_price < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs_dips.append(pair)
        status = 'Dip'
    elif current_price > best_fit_line2[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs_tops.append(pair)
        status = 'Top'
    else:
        intermediate_pairs.append(pair)
        status = 'In between'

    print(f"{datetime.now()} - Symbol: {pair}, Status: {status}, Current Price: {current_price}, Poly Value: {best_fit_line1[-1]}")

def rescan_dips(client, pair, selected_pair_dips):
    interval = '30m'
    df = get_klines(client, pair, interval)
    close_prices = df['Close'].values

    best_fit_line1, _, _ = calculate_best_fit_lines(close_prices)
    current_price = close_prices[-1]

    if current_price < best_fit_line1[-1]:
        selected_pair_dips.append(pair)
        status = 'Dip confirmed on second timeframe'
    else:
        status = 'Not a dip on second timeframe'

    print(f"{datetime.now()} - Symbol: {pair}, Status: {status}, Current Price: {current_price}, Poly Value: {best_fit_line1[-1]}")

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

    analytic_signal = hilbert(wave)
    sine_wave = np.sin(np.angle(analytic_signal))
    sine_wave = -sine_wave
    return t, sine_wave

def analyze_wave(t, wave, frequency, sampling_rate, df):
    current_time_index = len(wave) - 1

    prev_reversal_index = current_time_index
    for i in range(current_time_index - 1, -1, -1):
        if wave[i] * wave[i - 1] <= 0:
            prev_reversal_index = i
            break

    current_min = np.min(wave[:current_time_index])
    current_max = np.max(wave[:current_time_index])

    cycle_duration = 30 * 60  # 30 minutes duration

    # Determine cycle start and end times based on the last reversal
    cycle_end_time = t[current_time_index]
    cycle_start_time = t[prev_reversal_index]

    current_time = t[current_time_index]
    time_to_next_reversal = cycle_start_time + (current_max - current_min)  # Dynamic adjustment
    time_to_next_reversal -= current_time  # Remaining time to the end of the cycle

    cycle_direction = "Up" if wave[prev_reversal_index] < wave[current_time_index] else "Down"
    incoming = "Dip" if cycle_direction == "Down" else "Top"

    if incoming == "Top":
        print(f"{datetime.now()} - Symbol: {df.index.name}, Status: {incoming}, Cycle Direction: {cycle_direction}")

def filter_dips_with_momentum(client, pair, selected_pair_dips_momentum):
    interval = '1m'
    df = get_klines(client, pair, interval)
    close_prices = df['Close'].values

    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=10)
    current_momentum = momentum[-1]

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
        executor.map(lambda pair: filter_pair(trader.client, pair, '2h', filtered_pairs_dips, filtered_pairs_tops, intermediate_pairs), active_trading_pairs)

    print('Filtered dips (2h):', filtered_pairs_dips)
    print('Filtered tops (2h):', filtered_pairs_tops)
    print('Intermediate pairs (2h):', intermediate_pairs)

    if filtered_pairs_dips:
        print('Rescanning dips on lower timeframes...')
        selected_pair_dips = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda pair: rescan_dips(trader.client, pair, selected_pair_dips), filtered_pairs_dips)

        print('Dips after second filter:', selected_pair_dips)

        if len(selected_pair_dips) > 1:
            print('Multiple dips found on second timeframe. Analyzing to select the lowest...')
            lowest_dip = None
            lowest_value = float('inf')

            for pair in selected_pair_dips:
                interval = '5m'
                df = get_klines(trader.client, pair, interval)
                close = df['Close'].values

                frequency = 1 / len(close)
                phase_shift = 0
                angle = 0
                duration = len(close) / 12  # Assumed duration
                sampling_rate = len(close)  # Assumed sampling rate

                t, wave = generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate)

                min_wave_value = np.min(wave)

                if min_wave_value < lowest_value:
                    lowest_value = min_wave_value
                    lowest_dip = pair

            print(f'Lowest dip on 5m timeframe is {lowest_dip} with wave value {lowest_value}')
            print(f'Current asset vs USDT: {lowest_dip}')
            interval = '5m'
            df = get_klines(trader.client, lowest_dip, interval)
            frequency = 1 / len(df['Close'].values)
            phase_shift = 0
            angle = 0
            duration = len(df['Close'].values) / 12
            sampling_rate = len(df['Close'].values)
            t, wave = generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate)
            analyze_wave(t, wave, frequency, sampling_rate, df)

            # Final momentum filter and sorting
            print('Applying final momentum filter and sorting...')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(lambda pair: filter_dips_with_momentum(trader.client, pair, selected_pair_dips_momentum), selected_pair_dips)

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
