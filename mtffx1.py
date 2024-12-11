from binance.client import Client
import numpy as np
import os
import sys
import concurrent.futures
import matplotlib.pyplot as plt

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

    return symbol, {'bullish': bullish_volume, 'bearish': bearish_volume, 'prices': close_prices}

def envelope_and_forecast(prices, cycles=(5, 10, 20)):
    N = len(prices)
    if N < 10:  # Ensure there's enough data
        return None, None, None, None

    x = np.arange(N)
    offset = np.mean(prices)

    sine_waves = []
    for cycle in cycles:
        frequency = 2 * np.pi / cycle
        amplitude = (np.max(prices) - np.min(prices)) / 2
        sine_wave = amplitude * np.sin(frequency * x) + offset
        sine_waves.append(sine_wave)

    # Calculate support and resistance levels based on the observed prices
    support = np.min(prices)
    resistance = np.max(prices)
    return sine_waves, support, resistance, offset

def next_reversal_price(amplitude, sym_center, peak=True):
    """Calculate the next expected price for opposite reversal."""
    return sym_center + (amplitude if peak else -amplitude)

def market_mood_metrics(current_price, forecast_price):
    distance = abs(current_price - forecast_price)
    potential_equilibrium = (current_price + forecast_price) / 2
    angle = np.degrees(np.arctan2(forecast_price - current_price, 1))  # Simulating directional momentum
    intensity = distance / potential_equilibrium if potential_equilibrium else 0
    energy = (current_price + forecast_price) / 2
    angular_momentum = current_price * np.sin(np.radians(angle))  # Simplified angular momentum representation

    return intensity, energy, angular_momentum, angle

filename = 'credentials.txt'
trader = Trader(filename)

trading_pairs = trader.get_usdc_pairs()
volume_data = {}

# Include additional timeframes for volume and price calculations
timeframes_volume = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
timeframes_sine = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']  # Sine wave calculations

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

# Calculate bullish-to-bearish ratios and find the asset with the highest ratio
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

# Enhanced feature: Historical min, max, and average calculations for all timeframes
historical_metrics = {}

# Forecast and print current status for the best symbol found
if best_symbol:
    symbol, total_bullish, total_bearish = best_symbol
    print(f'The asset with the highest bullish/bearish volume ratio is: {symbol}')
    
    # Calculate and print dominance ratios
    total_volume = total_bullish + total_bearish
    bullish_percentage = (total_bullish / total_volume) * 100 if total_volume > 0 else 0
    bearish_percentage = (total_bearish / total_volume) * 100 if total_volume > 0 else 0
    
    print(f'Bullish Volume: {total_bullish:.2f} ({bullish_percentage:.2f}%)')
    print(f'Bearish Volume: {total_bearish:.2f} ({bearish_percentage:.2f}%)')

    # Collect historical prices for each timeframe
    for timeframe in timeframes_sine:
        klines = trader.client.get_klines(symbol=symbol, interval=timeframe)
        close_prices = [float(entry[4]) for entry in klines]
        
        # Calculate historical min, max, and avg
        if close_prices:
            historical_min = np.min(close_prices)
            historical_max = np.max(close_prices)
            historical_avg = np.mean(close_prices)
            
            historical_metrics[timeframe] = {
                'min': historical_min,
                'max': historical_max,
                'avg': historical_avg
            }

    # Print historical metrics for all timeframes
    print('\nHistorical Price Metrics:')
    for timeframe, metrics in historical_metrics.items():
        print(f'Timeframe: {timeframe} - Min: {metrics["min"]:.2f}, Max: {metrics["max"]:.2f}, Avg: {metrics["avg"]:.2f}')

    # Get latest price data for the selected symbol using 5-minute timeframe
    klines = trader.client.get_klines(symbol=symbol, interval='5m')
    latest_prices = [float(entry[4]) for entry in klines]  # Only close prices for sine wave analysis
    
    sine_waves, support, resistance, offset = envelope_and_forecast(latest_prices)

    if sine_waves is not None and len(sine_waves) > 0:
        # Get latest price data for volume from 1-minute timeframe
        latest_volume_prices = volume_data[symbol]['1m']['prices']
        current_price = latest_prices[-1]
        amplitude = np.max(latest_prices) - np.min(latest_prices)
        
        # Calculate forecast price
        forecast_price = next_reversal_price(amplitude, current_price, peak=current_price < (current_price + amplitude))

        # Symmetrical percentage calculation
        distance_to_min = max(0, (current_price - historical_min) / (historical_max - historical_min) * 100 if historical_max != historical_min else 0)
        distance_to_max = max(0, (historical_max - current_price) / (historical_max - historical_min) * 100 if historical_max != historical_min else 0)
        symmetrical_total = distance_to_min + distance_to_max

        # Ensure total is 100%
        distance_to_min_percentage = (distance_to_min / symmetrical_total * 100) if symmetrical_total > 0 else 0
        distance_to_max_percentage = (distance_to_max / symmetrical_total * 100) if symmetrical_total > 0 else 0

        # Market Mood
        market_mood = "Up" if current_price > forecast_price else "Down"
        
        # Print metrics
        print(f'Current Close Price: {current_price:.2f}')
        print(f'Forecasted Price for next reversal: {forecast_price:.2f}')
        print(f'Market Mood: {market_mood}')
        print(f'Distance to Historical Min: {distance_to_min_percentage:.2f}%')
        print(f'Distance to Historical Max: {distance_to_max_percentage:.2f}%')

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(latest_prices, label='Price', color='blue')

        # Plot multiple sine waves and determine time from current close
        key_points = {}
        for i, sine_wave in enumerate(sine_waves):
            x = np.arange(len(sine_wave))
            plt.plot(sine_wave, label=f'Sine Wave {i + 1}', linestyle='--')
            
            # Identify key points near current price for each wave
            key_point_index = np.abs(sine_wave - current_price).argmin()
            key_points[f'Wave {i + 1}'] = (x[key_point_index], sine_wave[key_point_index])
            plt.annotate(f'Key Point {i + 1}: {sine_wave[key_point_index]:.2f}', 
                         xy=(x[key_point_index], sine_wave[key_point_index]), 
                         xytext=(x[key_point_index], sine_wave[key_point_index] + 0.5),
                         arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8)

        plt.axhline(y=support, color='red', label='Support', linestyle='--')
        plt.axhline(y=resistance, color='purple', label='Resistance', linestyle='--')

        # Plot current close and forecast price
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