import numpy as np
from binance.client import Client as BinanceClient
import datetime
import matplotlib.pyplot as plt
import talib  # Ensure you have TALib installed: pip install TA-Lib
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
timeframes = ['1m', '3m', '5m']

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

# Initialize cleaned_candle_map
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

def identify_significant_highs_lows(candles):
    """Identify significant low and high from the candle data."""
    close_prices = [candle['close'] for candle in candles]
    if close_prices:
        significant_low = min(close_prices)
        significant_high = max(close_prices)
        recent_close = close_prices[-1]
        return significant_low, significant_high, recent_close
    return None, None, None

def calculate_thresholds(close_prices):
    """Calculate min, max thresholds and average."""
    if not close_prices:
        return None, None, None
    min_threshold = min(close_prices)
    max_threshold = max(close_prices)
    avg_mtf = np.mean(close_prices)
    return min_threshold, max_threshold, avg_mtf

def calculate_fibonacci_levels(significant_low, significant_high):
    """Calculate Fibonacci retracement levels from low to high."""
    fib_levels = {}
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 1]
    for ratio in fib_ratios:
        fib_levels[ratio] = significant_low + ratio * (significant_high - significant_low)
    return fib_levels

def calculate_gann_medians(candles, gann_lengths):
    """Calculate Gann medians based on the last available candles."""
    gann_medians = {}
    # Perform Gann calculations based on available data
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

def get_significant_volume_levels(candles, current_close):
    """Get support and resistance levels based on significant volume below and above current close."""
    volume_bullish_map = {}
    volume_bearish_map = {}
    for candle in candles:
        price = candle['close']
        if price < current_close:  # Collect bullish volumes below current close for support
            volume_bullish_map[price] = volume_bullish_map.get(price, 0) + candle['volume']
        elif price > current_close:  # Collect bearish volumes above current close for resistance
            volume_bearish_map[price] = volume_bearish_map.get(price, 0) + candle['volume']

    support_level = max(volume_bullish_map, key=volume_bullish_map.get) if volume_bullish_map else None
    resistance_level = min(volume_bearish_map, key=volume_bearish_map.get) if volume_bearish_map else None

    return support_level, resistance_level

def get_next_minute_targets(closes, n_components):
    """Predict target price levels for the next minute using FFT."""
    # Calculate FFT of closing prices
    fft = fftpack.fft(closes)
    frequencies = fftpack.fftfreq(len(closes))
    # Sort frequencies by magnitude and keep only the top n_components
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]
    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = np.real(fftpack.ifft(filtered_fft))
    # Calculate the next target price as the last value in the filtered signal
    target_price = filtered_signal[-1]
    # Calculate the entry price
    entry_price = closes[-1]
    # Calculate standard deviation for additional target calculations
    std_dev = np.std(closes)
    # Create multiple target levels based on standard deviation
    targets = [target_price + n * std_dev for n in range(1, 7)]
    return entry_price, target_price, targets

def calculate_reversal_and_forecast(close, significant_low, significant_high, avg_threshold, recent_close, current_angle_status):
    """Calculate reversals and forecasts based on significant price levels, avg threshold, and angle status."""
    current_reversal = None
    next_reversal = None
    forecast_dip = None
    forecast_top = None

    last_close = close[-1]
    
    # Determine current reversal based on the rules specified
    if last_close > significant_low:
        current_reversal = "DIP"
        next_reversal = "TOP"
    elif last_close < significant_high:
        current_reversal = "TOP"
        next_reversal = "DIP"
    else:
        if last_close <= (significant_high + significant_low) / 2:
            current_reversal = "DIP"
            next_reversal = "TOP"
        else:
            current_reversal = "TOP"
            next_reversal = "DIP"

    # Execute the logic for upward forecasts
    if current_reversal == "DIP":
        if (last_close > significant_low and
            last_close < (significant_high + significant_low) / 2 and
            last_close < avg_threshold and 
            current_angle_status == "Below 45 Degree Angle"):
            forecast_dip = last_close - (last_close - significant_low) * 0.2
            forecast_top = last_close + (significant_high - last_close) * 0.5

    # Execute the logic for downward forecasts
    elif current_reversal == "TOP":
        if (last_close < significant_high and
            last_close > (significant_high + significant_low) / 2 and 
            last_close > avg_threshold and 
            current_angle_status == "Above 45 Degree Angle"):
            forecast_top = last_close + (significant_high - last_close) * 0.2
            forecast_dip = last_close - (last_close - significant_low) * 0.5

    # Determine final forecast direction
    forecast_direction = "Up" if forecast_dip is not None else "Down" if forecast_top is not None else "No Clear Direction"

    return current_reversal, next_reversal, forecast_direction, last_close, forecast_dip, forecast_top

# Scale current close price to sine wave       
def scale_to_sine(timeframe):  
    close_prices = np.array(get_close(timeframe))
    current_close = close_prices[-1]      
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    
    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    dist_min = ((current_sine - sine_wave_min) /  (sine_wave_max - sine_wave_min)) * 100 
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    return dist_min, dist_max, current_sine

# Main Logic for Timeframes
current_time = datetime.datetime.now()
print("Current local Time is now at:", current_time)
print()

gann_lengths = [5, 7, 9, 12, 14, 17, 21, 27, 45, 56, 100, 126, 147, 200, 258, 369]

for timeframe in timeframes:
    candles = cleaned_candle_map[timeframe]
    close_prices = get_close(timeframe)

    # Identify significant lows and highs
    significant_low, significant_high, recent_close = identify_significant_highs_lows(candles)

    # Calculate thresholds
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close_prices)

    # Determine focus direction based on recent close
    focus_direction = "up" if abs(significant_low - recent_close) < abs(significant_high - recent_close) else "down"

    # Calculate Gann medians considering the last available candles
    gann_medians = calculate_gann_medians(candles, gann_lengths)
    sorted_gann_medians = sorted((length, median) for length, median in gann_medians.items() if median is not None)

    # Analyze volume to build support and resistance
    bullish_volume, bearish_volume, total_volume, bullish_ratio, bearish_ratio = analyze_volume(candles)

    # Get support and resistance that contains strongest volumes
    support, resistance = get_significant_volume_levels(candles, recent_close)

    # Get FFT forecasts
    entry_price, target_price, targets = get_next_minute_targets(close_prices, n_components=5)

    # Forecast potential reversals and prices
    avg_threshold = (min_threshold + max_threshold) / 2
    current_angle_status = "Above 45 Degree Angle" if recent_close > (significant_low + significant_high) / 2 else "Below 45 Degree Angle"
    
    (current_reversal, next_reversal, forecast_direction, last_close, forecast_dip, forecast_top) = calculate_reversal_and_forecast(
        close_prices, significant_low, significant_high, avg_threshold, recent_close, current_angle_status
    )

    # Scale to sine
    dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)

    # Calculate Fibonacci levels
    fib_levels = calculate_fibonacci_levels(significant_low, significant_high)

    # Output details
    print(f"=== Timeframe: {timeframe} ===")
    print(f"Min Threshold: {min_threshold:.25f}, Max Threshold: {max_threshold:.25f}, Avg MTF: {avg_mtf:.25f}")
    print(f"Bullish Volume: {bullish_volume:.25f}, Bearish Volume: {bearish_volume:.25f}, Total Volume: {total_volume:.25f}")
    print(f"Bullish Ratio: {bullish_ratio:.2%}, Bearish Ratio: {bearish_ratio:.2%}")

    if bullish_volume > bearish_volume:
        print("Bullish sentiment prevalent.")
    else:
        print("Bearish sentiment prevalent.")

    print(f"Support Level: {support:.25f}, Resistance Level: {resistance:.25f}")

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

    # Find major highs and lows for current cycle
    print(f"Significant Low: {significant_low:.25f}, Significant High: {significant_high:.25f}")

    # Incoming Reversal Information
    if abs(significant_low - recent_close) < abs(significant_high - recent_close):
        print(f"The current close is closer to the significant low. An upcoming reversal minor might occur around {significant_high:.25f}.")
    else:
        print(f"The current close is closer to the significant high. An upcoming reversal minor might occur around {significant_low:.25f}.")

    # Check if price is above or below the 45-degree angle
    angle_slope = (significant_high - significant_low) / 45  # Assuming the price range represents a 45 degrees change
    angle_intercept = significant_low  # The intersection point with significant low
    current_angle_status = "Above 45 Degree Angle" if recent_close > (angle_slope * len(close_prices) + angle_intercept) else "Below 45 Degree Angle"
    angle_price = angle_slope * len(close_prices) + angle_intercept
    print(current_angle_status + f": {angle_price:.25f}")

    # Output the projected prices for the next minute using FFT
    print(f"Entry price: {entry_price:.25f}")
    print(f"Forecast target price: {target_price:.25f}")
    for i, target in enumerate(targets, start=1):
        print(f"Target {i}: {target:.25f}")

    # Print the forecasted values from reversal calculations
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
