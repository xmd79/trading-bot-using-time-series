from binance.client import Client
import numpy as np
import os
import sys
import concurrent.futures
import matplotlib.pyplot as plt
import talib

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        return trading_pairs

def calculate_volume(trader, symbol, interval):
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    volumes = [float(entry[5]) for entry in klines]
    close_prices = [float(entry[4]) for entry in klines]

    if not close_prices or not volumes:
        return {'bullish': 0, 'bearish': 0, 'prices': []}

    bullish_volume = sum(volumes[i] for i in range(len(close_prices) - 1) if close_prices[i] < close_prices[i + 1])
    bearish_volume = sum(volumes[i] for i in range(len(close_prices) - 1) if close_prices[i] > close_prices[i + 1])

    return symbol, {'bullish': bullish_volume, 'bearish': bearish_volume, 'prices': close_prices, 'klines': klines}

def find_reversals(prices):
    maxima = [prices[i] for i in range(1, len(prices) - 1) if prices[i - 1] < prices[i] > prices[i + 1]]
    minima = [prices[i] for i in range(1, len(prices) - 1) if prices[i - 1] > prices[i] < prices[i + 1]]

    major_top = maxima[-1] if maxima else None
    major_bottom = minima[-1] if minima else None

    minor_top = max(minima) if minima else None
    minor_bottom = min(maxima) if maxima else None

    if prices:
        recent_close = prices[-1]
        candidate_min = [m for m in minima if m < recent_close]
        candidate_max = [m for m in maxima if m > recent_close]

        minor_top = max(candidate_max) if candidate_max else minor_top
        minor_bottom = min(candidate_min) if candidate_min else minor_bottom

    return (major_top, major_bottom), (minor_top, minor_bottom)

def calculate_bollinger_bands(prices, window=20, num_std=2):
    if len(prices) < window:
        return None, None
    rolling_mean = np.convolve(prices, np.ones(window) / window, mode='valid')
    rolling_std = np.std(prices[-window:])
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_moving_averages(prices, short_window=10, long_window=50):
    if len(prices) < short_window or len(prices) < long_window:
        return np.array([]), np.array([])
    short_mavg = np.convolve(prices, np.ones(short_window) / short_window, mode='valid')
    long_mavg = np.convolve(prices, np.ones(long_window) / long_window, mode='valid')
    return short_mavg, long_mavg

def envelope_and_forecast(prices, cycles=(5, 10, 20)):
    N = len(prices)
    if N < 10:
        return None, None, None, None

    x = np.arange(N)
    offset = np.mean(prices)

    sine_waves = []
    for cycle in cycles:
        frequency = 2 * np.pi / cycle
        amplitude = (np.max(prices) - np.min(prices)) / 2
        sine_wave = amplitude * np.sin(frequency * x) + offset
        sine_waves.append(sine_wave)

    support = np.min(prices)
    resistance = np.max(prices)
    return sine_waves, support, resistance, offset

def next_reversal_price(amplitude, sym_center, peak=True):
    return sym_center + (amplitude if peak else -amplitude)

def market_mood_metrics(current_price, forecast_price):
    distance = abs(current_price - forecast_price)
    potential_equilibrium = (current_price + forecast_price) / 2
    angle = np.degrees(np.arctan2(forecast_price - current_price, 1))
    intensity = distance / potential_equilibrium if potential_equilibrium else 0
    energy = (current_price + forecast_price) / 2
    angular_momentum = current_price * np.sin(np.radians(angle))

    return intensity, energy, angular_momentum, angle

def sanitize_value(value):
    """Returns 0 if value is NaN, None, or 0, else returns the value."""
    return value if value is not None and (not isinstance(value, float) or not np.isnan(value)) else 0

# Scale current close price to sine wave       
def scale_to_sine(close_prices, historical_min, historical_max):  
    close_prices_np = np.array(close_prices)  # Ensure this is a numpy array
    current_close = close_prices_np[-1]      
        
    # Calculate sine wave        
    sine_wave, leadsine = talib.HT_SINE(close_prices_np)         
    
    # Replace NaN values with 0        
    sine_wave = np.nan_to_num(sine_wave)        
    
    # Filter HT_SINE between historical min and max
    valid_sine_wave = sine_wave[(sine_wave >= historical_min) & (sine_wave <= historical_max)]
    
    current_sine = sine_wave[-1]          
    sine_wave_min = np.min(valid_sine_wave) if valid_sine_wave.size > 0 else historical_min        
    sine_wave_max = np.max(valid_sine_wave) if valid_sine_wave.size > 0 else historical_max

    # Calculate absolute distances
    distance_to_sine_min = abs(current_sine - sine_wave_min)
    distance_to_sine_max = abs(current_sine - sine_wave_max)

    total_distance = distance_to_sine_min + distance_to_sine_max
    
    # Normalize distances to a scale of 0 to 100%
    if total_distance > 0:
        normalized_distance_to_min = (distance_to_sine_min / total_distance) * 100
        normalized_distance_to_max = (distance_to_sine_max / total_distance) * 100
    else:
        normalized_distance_to_min = 0
        normalized_distance_to_max = 0

    return normalized_distance_to_min, normalized_distance_to_max, current_sine
      
filename = 'credentials.txt'
trader = Trader(filename)

trading_pairs = trader.get_usdc_pairs()
volume_data = {}
historical_metrics = {}

# Additional calculations for volume and price using multiple timeframes
timeframes_volume = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
timeframes_sine = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_symbol = {
        executor.submit(calculate_volume, trader, symbol, tf): (symbol, tf)
        for symbol in trading_pairs for tf in timeframes_volume
    }

    for future in concurrent.futures.as_completed(future_to_symbol):
        symbol, tf = future_to_symbol[future]
        try:
            result = future.result()
            if result:
                symbol, vol = result
                if symbol not in volume_data:
                    volume_data[symbol] = {}
                volume_data[symbol][tf] = vol
        except Exception as e:
            print(f'Error fetching data for {symbol} in {tf}: {e}')

highest_ratio = 0
best_symbol = None

for symbol, volumes in volume_data.items():
    total_bullish = sum(volumes[tf]['bullish'] for tf in volumes if 'bullish' in volumes[tf])
    total_bearish = sum(volumes[tf]['bearish'] for tf in volumes if 'bearish' in volumes[tf])
    
    if total_bearish > 0:
        ratio = total_bullish / total_bearish
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_symbol = (symbol, total_bullish, total_bearish)

if best_symbol:
    symbol, total_bullish, total_bearish = best_symbol
    print(f'The asset with the highest bullish/bearish volume ratio is: {symbol}')
    
    total_volume = total_bullish + total_bearish
    bullish_percentage = (total_bullish / total_volume) * 100 if total_volume > 0 else 0
    bearish_percentage = (total_bearish / total_volume) * 100 if total_volume > 0 else 0
    
    print(f'Bullish Volume: {total_bullish:.25f} ({bullish_percentage:.2f}%)')
    print(f'Bearish Volume: {total_bearish:.25f} ({bearish_percentage:.2f}%)')

    for timeframe in timeframes_sine:
        klines = trader.client.get_klines(symbol=symbol, interval=timeframe)
        close_prices = [float(entry[4]) for entry in klines]
        
        if close_prices:
            historical_min = float(np.min(close_prices))
            historical_max = float(np.max(close_prices))
            historical_avg = float(np.mean(close_prices))
            
            historical_metrics[timeframe] = {
                'min': historical_min,
                'max': historical_max,
                'avg': historical_avg
            }
            
            (major_top, major_bottom), (minor_top, minor_bottom) = find_reversals(close_prices)
            current_close = close_prices[-1]
            print(f'\nTimeframe: {timeframe}')
            print(f'Current Close: {current_close:.25f}')
            print(f'Major Top: {sanitize_value(major_top):.25f}' if major_top is not None else 'No Major Top Found')
            print(f'Major Bottom: {sanitize_value(major_bottom):.25f}' if major_bottom is not None else 'No Major Bottom Found')
            print(f'Minor Top: {sanitize_value(minor_top):.25f}' if minor_top is not None else 'No Minor Top Found')
            print(f'Minor Bottom: {sanitize_value(minor_bottom):.25f}' if minor_bottom is not None else 'No Minor Bottom Found')

            # Check distances to major and minor reversals
            distance_to_major_top = sanitize_value(current_close - major_top) if major_top is not None else None
            distance_to_major_bottom = sanitize_value(current_close - major_bottom) if major_bottom is not None else None
            distance_to_minor_top = sanitize_value(current_close - minor_top) if minor_top is not None else None
            distance_to_minor_bottom = sanitize_value(current_close - minor_bottom) if minor_bottom is not None else None

            print(f'Distance to Major Top: {distance_to_major_top:.25f}' if distance_to_major_top is not None else 'Distance to Major Top: 0.0000000000000000')
            print(f'Distance to Major Bottom: {distance_to_major_bottom:.25f}' if distance_to_major_bottom is not None else 'Distance to Major Bottom: 0.0000000000000000')
            print(f'Distance to Minor Top: {distance_to_minor_top:.25f}' if distance_to_minor_top is not None else 'Distance to Minor Top: 0.0000000000000000')
            print(f'Distance to Minor Bottom: {distance_to_minor_bottom:.25f}' if distance_to_minor_bottom is not None else 'Distance to Minor Bottom: 0.0000000000000000')

            # Calculate and print the HT_SINE values and their normalized distances
            dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(close_prices, historical_min, historical_max)

            print(f'Distance to Sine Min: {dist_from_close_to_min:.2f}%')
            print(f'Distance to Sine Max: {dist_from_close_to_max:.2f}%')
            print(f'Current Sine Value: {current_sine:.25f}')

    print('\nHistorical Price Metrics:')
    for timeframe, metrics in historical_metrics.items():
        print(f'Timeframe: {timeframe} - Min: {metrics["min"]:.25f}, Max: {metrics["max"]:.25f}, Avg: {metrics["avg"]:.25f}')

    klines = trader.client.get_klines(symbol=symbol, interval='5m')
    latest_prices = [float(entry[4]) for entry in klines]
    
    sine_waves, support, resistance, offset = envelope_and_forecast(latest_prices)
    
    upper_band, lower_band = calculate_bollinger_bands(latest_prices)
    short_mavg, long_mavg = calculate_moving_averages(latest_prices)

    if sine_waves is not None and len(sine_waves) > 0:
        latest_volume_prices = volume_data[symbol]['1m']['prices']
        current_price = latest_prices[-1]
        amplitude = np.max(latest_prices) - np.min(latest_prices)
        
        forecast_price = next_reversal_price(amplitude, current_price, peak=current_price < (current_price + amplitude))

        # Calculate distance to min and max based on historical metrics
        distance_to_min = abs(current_price - historical_min)
        distance_to_max = abs(historical_max - current_price)

        symmetrical_total = distance_to_min + distance_to_max

        distance_to_min_percentage = (distance_to_min / symmetrical_total * 100) if symmetrical_total > 0 else 0
        distance_to_max_percentage = (distance_to_max / symmetrical_total * 100) if symmetrical_total > 0 else 0

        market_mood = "Up" if current_price < forecast_price else "Down"
        
        print(f'Current Close Price: {current_price:.25f}')
        print(f'Forecasted Price for next reversal: {forecast_price:.25f}')
        print(f'Market Mood: {market_mood}')
        print(f'Distance to Historical Min: {distance_to_min_percentage:.25f}%')
        print(f'Distance to Historical Max: {distance_to_max_percentage:.25f}%')

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(latest_prices, label='Price', color='blue')

        # Plot Bollinger Bands
        if upper_band is not None and lower_band is not None:
            plt.plot(np.arange(len(upper_band)), upper_band, label='Upper Bollinger Band', color='orange', linestyle='--')
            plt.plot(np.arange(len(lower_band)), lower_band, label='Lower Bollinger Band', color='orange', linestyle='--')

        # Plotting Moving Averages
        if len(short_mavg) > 0 and len(long_mavg) > 0:
            plt.plot(np.arange(len(short_mavg)), short_mavg, label='Short Moving Average', color='green')
            plt.plot(np.arange(len(long_mavg)), long_mavg, label='Long Moving Average', color='red')

        for i, sine_wave in enumerate(sine_waves):
            plt.plot(sine_wave, label=f'Sine Wave {i + 1}', linestyle='--')

        plt.axhline(y=support, color='red', label='Support', linestyle='--')
        plt.axhline(y=resistance, color='purple', label='Resistance', linestyle='--')

        plt.axhline(y=current_price, color='orange', label='Current Close', linestyle='-.')
        plt.axhline(y=forecast_price, color='green', label='Forecast Price', linestyle='-.')

        plt.title(f"Price Forecast with Sine Wave Projection for {symbol}")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

else:
    print('No suitable asset found.')

sys.exit(0)
