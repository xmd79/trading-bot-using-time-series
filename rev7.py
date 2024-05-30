import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import talib
from binance.client import Client
import concurrent.futures
from scipy.signal import hilbert

# Define the Fibonacci projection functions
def fibonacci_ratios(price):
    ratios = [0.382, 0.618, 1.0, 1.27, 1.618, 1.73, 2.23]
    return [price * ratio for ratio in ratios]

def project_future_turning_points(df, high_point, low_point, conversion_factor):
    fib_levels_high = fibonacci_ratios(high_point['price'])
    fib_levels_low = fibonacci_ratios(low_point['price'])
    
    projected_dates_high = [high_point['date'] + timedelta(days=level * conversion_factor) for level in fib_levels_high]
    projected_dates_low = [low_point['date'] + timedelta(days=level * conversion_factor) for level in fib_levels_low]
    
    return projected_dates_high, projected_dates_low, fib_levels_high, fib_levels_low

# Define the Trader class and functions
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

def calculate_pivots(df, depth=10):
    def find_pivots(series, window):
        pivots = []
        for i in range(window, len(series) - window):
            if series[i] == max(series[i - window:i + window + 1]):
                pivots.append((i, series[i], 'high'))
            elif series[i] == min(series[i - window:i + window + 1]):
                pivots.append((i, series[i], 'low'))
        return pivots

    highs = find_pivots(df['High'], depth // 2)
    lows = find_pivots(df['Low'], depth // 2)
    return highs + lows

def calculate_deviation(base_price, price):
    return 100 * (price - base_price) / base_price

def price_rotation_diff(p_last, price, mode='Absolute'):
    diff = price - p_last
    if mode == 'Absolute':
        return f"{'+' if diff > 0 else ''}{diff:.2f}"
    else:
        return f"{'+' if diff > 0 else '-'}{(abs(diff) * 100) / p_last:.2f}%"

def cum_volume(df, index1, index2):
    return df['Volume'][index1 + 1:index2].sum()

def hilbert_transform_sine(df):
    analytic_signal = hilbert(df['Close'])
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return np.sin(instantaneous_phase), np.cos(instantaneous_phase)

def fourier_transform_cycle(df, period):
    close_prices = df['Close'].values
    close_prices -= np.mean(close_prices)  # Remove mean
    fourier = np.fft.fft(close_prices)
    frequencies = np.fft.fftfreq(len(close_prices))
    cycle = np.real(fourier * np.exp(1j * 2 * np.pi * frequencies * period))
    return cycle

def calculate_energy_signals(df, period):
    energy = df['Close'].ewm(span=period, adjust=False).mean()
    impulse_momentum = df['Close'].pct_change(periods=period) * 100
    velocity = df['Close'].rolling(window=period).apply(lambda x: np.polyfit(range(period), x, 1)[0], raw=False)
    return energy, impulse_momentum, velocity

def get_klines(symbol, interval):
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
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

def filter1(pair):
    interval = '30m'
    symbol = pair
    df = get_klines(symbol, interval)
    close = df['Close'].values
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

    print(f"{datetime.now()} - Symbol: {symbol}, Status: {status}, Current Price: {current_price}, Poly Value: {poly_value}")

def filter2(pair):
    interval = '5m'
    symbol = pair
    df = get_klines(symbol, interval)
    close = df['Close'].values
    close_array = np.asarray(close)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2  = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    current_price = x[-1]
    poly_value = best_fit_line1[-1]

    if current_price < best_fit_line1[-1]:
        selected_pair_dips.append(symbol)
        status = 'Dip confirmed on 5m'
    else:
        status = 'Not a dip on 5m'

    print(f"{datetime.now()} - Symbol: {symbol}, Status: {status}, Current Price: {current_price}, Poly Value: {poly_value}")

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
    wave *= 0.01
    return t, wave

def analyze_wave(t, wave, frequency, sampling_rate, df):
    period = 1 / frequency
    half_period = period / 2
    current_time_index = len(wave) - 1

    prev_reversal_index = current_time_index
    for i in range(current_time_index - 1, -1, -1):
        if wave[i] * wave[i - 1] <= 0:
            prev_reversal_index = i
            break

    cycle_start_time = t[prev_reversal_index]
    cycle_end_time = t[current_time_index]
    cycle_duration = cycle_end_time - cycle_start_time

    current_time = t[current_time_index]
    next_reversal_time = cycle_start_time + 1800  # 30 minutes
    time_to_next_reversal = next_reversal_time - current_time

    cycle_direction = "Up" if wave[prev_reversal_index] < wave[current_time_index] else "Down"
    incoming = "Dip" if cycle_direction == "Down"  else "Top"

    middle_threshold = (np.max(wave) + np.min(wave)) / 2

    current_close_sine_value = wave[-1]

    distance_to_min = np.abs(current_close_sine_value - np.min(wave))
    distance_to_max = np.abs(current_close_sine_value - np.max(wave))

    total_distance = distance_to_min + distance_to_max
    percentage_to_min = (distance_to_min / total_distance) * 100
    percentage_to_max = (distance_to_max / total_distance) * 100

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

    last_reversal_value = wave[prev_reversal_index]
    last_reversal_type = "Top" if last_reversal_value == np.min(wave) else "Dip"
    print(f"Last Reversal Type: {last_reversal_type}")

def scan_assets(pair):
    filter1(pair)

print('Scanning all available assets on main timeframe...')
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(scan_assets, trading_pairs)

print('Filtered dips (30m):', filtered_pairs_dips)
print('Filtered tops (30m):', filtered_pairs_tops)
print('Intermediate pairs (30m):', intermediate_pairs)

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
            df = get_klines(pair, interval)
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

            # Get projected turning points
            projected_high_dates, projected_low_dates, fib_levels_high, fib_levels_low = project_future_turning_points(df, {'date': pd.Timestamp('2025-01-16 04:53:23.520000'), 'price': 150}, {'date': pd.Timestamp('2025-01-11 18:32:27.840000'), 'price': 120}, 1)

            # Print the projected turning points with real price values
            print(f"Projected high turning points (dates) for {pair}:")
            for date, price in zip(projected_high_dates, fib_levels_high):
                print(f"{date}: {price}")

            print(f"Projected low turning points (dates) for {pair}:")
            for date, price in zip(projected_low_dates, fib_levels_low):
                print(f"{date}: {price}")

        print(f'Lowest dip on 5m timeframe is {lowest_dip} with wave value {lowest_value}')
        print(f'Current asset vs USDT: {lowest_dip}')
        interval = '5m'
        df = get_klines(lowest_dip, interval)
        close_real_price = df['Close'].values[-1]
        analyze_wave(t, wave, frequency, sampling_rate, df)

    else:
        print(f'Selected dip on 5m timeframe: {selected_pair_dips[0]}')
        print(f'Current asset vs USDT: {selected_pair_dips[0]}')
        interval = '5m'
        df = get_klines(selected_pair_dips[0], interval)
        close_real_price = df['Close'].values[-1]
        analyze_wave(t, wave, frequency, sampling_rate, df)

else:
    print('No dips found in main timeframe.')

