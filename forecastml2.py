import numpy as np
from binance.client import Client as BinanceClient
import datetime
import talib  # Ensure you have TA-Lib installed: pip install TA-Lib
from scipy import fftpack  # Importing fftpack for FFT functionality

# Function to get Binance client credentials
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

# Initialize Binance client
client = get_binance_client()

TRADE_SYMBOL = "ACTUSDT"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h']  # Added new timeframes

# Function to get candles from the Binance API
def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 1000  # Limit the number of candles retrieved
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        for k in klines:
            candle = {
                "time": k[0] / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "timeframe": timeframe
            }
            candles.append(candle)
    return candles

# Clean candle data to remove NaNs and ensure valid prices
def clean_price_data(candles):
    cleaned_candles = []
    for c in candles:
        if all(v is not None and v > 0 for v in [c['open'], c['high'], c['low'], c['close'], c['volume']]):
            cleaned_candles.append(c)
    return cleaned_candles

# Initialize cleaned candle map
cleaned_candle_map = {}
candles = get_candles(TRADE_SYMBOL, timeframes)
for candle in candles:
    timeframe = candle["timeframe"]
    cleaned_candle_map.setdefault(timeframe, []).append(candle)

# Clean each timeframe data
for timeframe in cleaned_candle_map.keys():
    cleaned_candle_map[timeframe] = clean_price_data(cleaned_candle_map[timeframe])

def get_close(timeframe):
    closes = [candle["close"] for candle in cleaned_candle_map[timeframe]]
    return closes

def calculate_thresholds(close_prices):
    """Calculate min, max thresholds and average."""
    if not close_prices:
        return None, None, None
    min_threshold = min(close_prices)
    max_threshold = max(close_prices)
    avg_mtf = np.mean(close_prices)
    return min_threshold, max_threshold, avg_mtf

def calculate_fibonacci_levels(min_threshold, max_threshold):
    """Calculate Fibonacci retracement levels from min to max thresholds."""
    fib_levels = {}
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 1]
    for ratio in fib_ratios:
        fib_levels[ratio] = min_threshold + ratio * (max_threshold - min_threshold)
    return fib_levels

def calculate_gann_medians(candles, gann_lengths):
    """Calculate Gann medians based on the last available candles."""
    gann_medians = {}
    for length in gann_lengths:
        if len(candles) >= length:
            price_data = [candle['close'] for candle in candles[-length:]]
            gann_medians[length] = np.median(price_data)
        else:
            gann_medians[length] = None
    return gann_medians

def analyze_volume(candles):
    """Analyze volume data for market dynamics."""
    bullish_volume = sum(candle['volume'] for candle in candles if candle['close'] > candle['open'])
    bearish_volume = sum(candle['volume'] for candle in candles if candle['close'] < candle['open'])
    total_volume = sum(candle['volume'] for candle in candles)
    # Calculate volume ratios
    bullish_ratio = bullish_volume / total_volume if total_volume > 0 else 0
    bearish_ratio = bearish_volume / total_volume if total_volume > 0 else 0
    return bullish_volume, bearish_volume, total_volume, bullish_ratio, bearish_ratio

def get_most_significant_volume_levels(candles, min_threshold, max_threshold):
    """Get the most significant support and resistance levels based on consistent trading volume."""
    volume_levels = {}
    
    for candle in candles:
        price = candle['close']
        volume = candle['volume']
        
        if min_threshold <= price <= max_threshold:
            if price not in volume_levels:
                volume_levels[price] = []
            volume_levels[price].append(volume)

    support_level = None
    resistance_level = None
    max_avg_volume_below = 0
    max_avg_volume_above = 0
    
    for price, volumes in volume_levels.items():
        avg_volume = np.mean(volumes)
        
        if price < (min_threshold + max_threshold) / 2:  # Below mid-point for support
            if avg_volume > max_avg_volume_below:
                max_avg_volume_below = avg_volume
                support_level = price
        else:  # Above mid-point for resistance
            if avg_volume > max_avg_volume_above:
                max_avg_volume_above = avg_volume
                resistance_level = price

    return support_level, resistance_level

def get_next_minute_targets(closes, n_components):
    """Predict target price levels for the next minute using FFT."""
    fft = fftpack.fft(closes)
    frequencies = fftpack.fftfreq(len(closes))
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = np.real(fftpack.ifft(filtered_fft))
    target_price = filtered_signal[-1]
    entry_price = closes[-1]
    std_dev = np.std(closes)
    targets = [target_price + n * std_dev for n in range(1, 7)]
    return entry_price, target_price, targets

def calculate_reversal_and_forecast(close, min_threshold, max_threshold, avg_threshold, recent_close, current_angle_status):
    """Calculate reversals and forecasts based on thresholds."""
    current_reversal = None
    next_reversal = None
    forecast_dip = None
    forecast_top = None

    last_close = close[-1]
    
    if last_close > min_threshold:
        current_reversal = "DIP"
        next_reversal = "TOP"
    elif last_close < max_threshold:
        current_reversal = "TOP"
        next_reversal = "DIP"
    else:
        if last_close <= (max_threshold + min_threshold) / 2:
            current_reversal = "DIP"
            next_reversal = "TOP"
        else:
            current_reversal = "TOP"
            next_reversal = "DIP"

    if current_reversal == "DIP":
        if (last_close > min_threshold and last_close < (max_threshold + min_threshold) / 2 and last_close < avg_threshold and current_angle_status == "Below 45 Degree Angle"):
            forecast_dip = last_close - (last_close - min_threshold) * 0.2
            forecast_top = last_close + (max_threshold - last_close) * 0.5
    elif current_reversal == "TOP":
        if (last_close < max_threshold and last_close > (max_threshold + min_threshold) / 2 and last_close > avg_threshold and current_angle_status == "Above 45 Degree Angle"):
            forecast_top = last_close + (max_threshold - last_close) * 0.2
            forecast_dip = last_close - (last_close - min_threshold) * 0.5

    forecast_direction = "Up" if forecast_dip is not None else "Down" if forecast_top is not None else "No Clear Direction"

    return current_reversal, next_reversal, forecast_direction, last_close, forecast_dip, forecast_top

def scale_to_sine(timeframe):  
    close_prices = np.array(get_close(timeframe))
    current_close = close_prices[-1]      
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave
    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # Check for the case where max and min sine wave values are equal
    if sine_wave_max == sine_wave_min:
        dist_min = 0  # Handle division by zero
        dist_max = 0  # Handle division by zero
    else:
        dist_min = ((current_sine - sine_wave_min) /  (sine_wave_max - sine_wave_min)) * 100 
        dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    return dist_min, dist_max, current_sine

def calculate_angle_price(min_threshold, max_threshold):
    """Calculate the price corresponding to a 45-degree angle using linear interpolation."""
    price_change = (max_threshold - min_threshold) / 2  # This essentially identifies the midpoint of the two prices
    angle_price = min_threshold + price_change  # The price at a 45-degree angle based on the min and max thresholds
    return angle_price

# New Function to Calculate Pythagorean Price
def calculate_pythagorean_price(min_threshold, max_threshold):
    """Calculate a projected price using the Pythagorean theorem and the current price movement."""
    # Assuming that the price movement can be represented as a right triangle
    price_diff = max_threshold - min_threshold
    pythagorean_price = (min_threshold**2 + price_diff**2) ** 0.5  # Hypotenuse value
    return pythagorean_price

# New Function to Calculate Phi Price Level
def calculate_phi_level(min_threshold, max_threshold):
    """Calculate a price level based on the Golden Ratio (Phi)."""
    phi = (1 + np.sqrt(5)) / 2  # Approximately 1.618
    phi_level = min_threshold + (max_threshold - min_threshold) * (phi - 1)  # Shifted by ratio
    return phi_level

# Main Logic for Timeframes
current_time = datetime.datetime.now()
print("Current local Time is now at:", current_time)
print()

gann_lengths = [5, 7, 9, 12, 14, 17, 21, 27, 45, 56, 100, 126, 147, 200, 258, 369]

for timeframe in timeframes:
    candles = cleaned_candle_map[timeframe]
    close_prices = get_close(timeframe)

    # Calculate thresholds
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close_prices)

    # Determine focus direction based on recent close
    recent_close = close_prices[-1]
    focus_direction = "up" if abs(min_threshold - recent_close) < abs(max_threshold - recent_close) else "down"

    # Calculate Gann medians considering the last available candles
    gann_medians = calculate_gann_medians(candles, gann_lengths)
    sorted_gann_medians = sorted((length, median) for length, median in gann_medians.items() if median is not None)

    # Analyze volume to build support and resistance
    bullish_volume, bearish_volume, total_volume, bullish_ratio, bearish_ratio = analyze_volume(candles)

    # Get the most significant support and resistance levels
    support, resistance = get_most_significant_volume_levels(candles, min_threshold, max_threshold)

    # Get FFT forecasts
    entry_price, target_price, targets = get_next_minute_targets(close_prices, n_components=5)

    # Forecast potential reversals and prices
    avg_threshold = (min_threshold + max_threshold) / 2
    current_angle_price = calculate_angle_price(min_threshold, max_threshold)

    current_angle_status = "Above 45 Degree Angle" if recent_close > current_angle_price else "Below 45 Degree Angle"
    
    (current_reversal, next_reversal, forecast_direction, last_close, forecast_dip, forecast_top) = calculate_reversal_and_forecast(
        close_prices, min_threshold, max_threshold, avg_threshold, recent_close, current_angle_status
    )

    # Scale to sine
    dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)

    # Calculate Fibonacci levels based on min and max thresholds
    fib_levels = calculate_fibonacci_levels(min_threshold, max_threshold)

    # Calculate projected price using Pythagorean Theorem
    pythagorean_price = calculate_pythagorean_price(min_threshold, max_threshold)

    # Calculate potential price based on the Golden Ratio (Phi)
    phi_level = calculate_phi_level(min_threshold, max_threshold)

    # Determine if the current close is below the calculated angle price
    is_below_angle = recent_close < current_angle_price

    # Output details
    print(f"=== Timeframe: {timeframe} ===")
    print(f"Min Threshold: {min_threshold:.25f}, Max Threshold: {max_threshold:.25f}, Avg MTF: {avg_mtf:.25f}")
    print(f"Bullish Volume: {bullish_volume:.25f}, Bearish Volume: {bearish_volume:.25f}, Total Volume: {total_volume:.25f}")
    print(f"Bullish Ratio: {bullish_ratio:.2%}, Bearish Ratio: {bearish_ratio:.2%}")

    if bullish_volume > bearish_volume:
        print("Bullish sentiment prevalent.")
    else:
        print("Bearish sentiment prevalent.")

    if support is None:
        print("Support Level: Not available")
    else:
        print(f"Support Level: {support:.25f}")

    if resistance is None:
        print("Resistance Level: Not available")
    else:
        print(f"Resistance Level: {resistance:.25f}")

    # Print Fibonacci levels
    print("Fibonacci Levels:")
    for level, price in fib_levels.items():
        print(f"Fibonacci Level {level}: {price:.25f}")

    # Determine which threshold is closer to the recent close
    closest_threshold = ""
    if abs(min_threshold - recent_close) < abs(max_threshold - recent_close):
        closest_threshold = f"Min Threshold closer: {min_threshold:.25f} to Current Close: {recent_close:.25f}"
    else:
        closest_threshold = f"Max Threshold closer: {max_threshold:.25f} to Current Close: {recent_close:.25f}"

    # Print sorted Gann medians
    print(closest_threshold)
    print("Sorted Gann Medians:")
    for (length, median) in sorted_gann_medians:
        print(f"Gann Median (Length {length}): {median:.25f}")

    # Print the price corresponding to the 45-degree angle
    print(f"45-Degree Angle Price (Min/Max Thresholds): {current_angle_price:.25f}")

    # Print the Pythagorean price level
    print(f"Pythagorean Price Level: {pythagorean_price:.25f}")

    # Print the Phi level
    print(f"Golden Ratio (Phi) Level: {phi_level:.25f}")

    # Print whether the current close is below the angle price
    if is_below_angle:
        print(f"The current close is below the 45-degree angle price: TRUE")
    else:
        print(f"The current close is above the 45-degree angle price: FALSE")

    # Incoming Reversal Information
    if abs(min_threshold - recent_close) < abs(max_threshold - recent_close):
        print(f"The current close is closer to the min threshold. An upcoming reversal minor might occur around {max_threshold:.25f}.")
    else:
        print(f"The current close is closer to the max threshold. An upcoming reversal minor might occur around {min_threshold:.25f}.")

    # Output projected prices using FFT
    print(f"Entry price: {entry_price:.25f}")
    print(f"Forecast target price: {target_price:.25f}")
    for i, target in enumerate(targets, start=1):
        print(f"Target {i}: {target:.25f}")

    # Print forecast values
    print(f"Current Reversal: {current_reversal}, Next Reversal: {next_reversal}")
    print(f"Forecast Direction: {forecast_direction}, Last Close: {last_close:.25f}")

    # Print forecast dip and top; check for None values before printing
    if forecast_dip is not None:
        print(f"Forecast Dip: {forecast_dip:.25f}")
    else:
        print("Forecast Dip: Not available")

    if forecast_top is not None:
        print(f"Forecast Top: {forecast_top:.25f}")
    else:
        print("Forecast Top: Not available")

    # Print Sine Wave details
    print(f"Distance to Min Sine: {dist_from_close_to_min:.2f}%, Distance to Max Sine: {dist_from_close_to_max:.2f}%")
    print(f"Current Sine value: {current_sine}\n")

    print("\n" + "=" * 30 + "\n")

print("All timeframe calculations completed.")
