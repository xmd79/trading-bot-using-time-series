#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
import time
from colorama import init, Fore, Style
import ephem
import pytz
from datetime import datetime
from sklearn.linear_model import LinearRegression
import scipy.fftpack as fftpack

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Initialize colorama
init(autoreset=True)

symbol = "BTCUSDC"
timeframes = ["5m", "1m"]
candle_map = {}

# Define a function to get candles
def get_candles(symbol, timeframe, limit=1000):
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        return [{
            "time": k[0] / 1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        } for k in klines]
    except BinanceAPIException as e:
        print(f"Error fetching candles for {symbol} at {timeframe}: {e}")
        return []

def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

def predict_reversals(candles, forecast_minutes=12):
    close_prices = np.array([c["close"] for c in candles])
    if len(close_prices) < 2:
        print("Not enough data for prediction.")
        return None, None

    # Reshape the data for Linear Regression
    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices

    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_x = np.arange(len(close_prices), len(close_prices) + forecast_minutes).reshape(-1, 1)
    predictions = model.predict(future_x)

    # Define minor and major reversal points
    minor_reversal_target = predictions[-1] * 1.02  # 2% above the last predicted price
    major_reversal_target = predictions[-1] * 1.05  # 5% above the last predicted price

    return predictions, minor_reversal_target, major_reversal_target

def calculate_45_degree_price(candles):
    if len(candles) < 50:
        print("Not enough data for 45-degree angle calculation")
        return None

    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    # Remove NaNs and zeros
    close_prices, x = remove_nans_and_zeros(close_prices, x)

    if len(x) < 2:
        print("Not enough valid data points for 45-degree angle calculation")
        return None

    try:
        # Perform linear regression
        slope, intercept, _, _, _ = linregress(x, close_prices)
        # Calculate the price at the end of the dataset
        angle_price = intercept + slope * (len(close_prices) - 1)
        return angle_price
    except Exception as e:
        print(f"Error calculating 45-degree angle price: {e}")
        return None

def enforce_price_area(candles, angle_price, tolerance=0.05):
    close_prices = np.array([c["close"] for c in candles])
    min_price = np.min(close_prices)
    max_price = np.max(close_prices)
    
    lower_bound = angle_price * (1 - tolerance)
    upper_bound = angle_price * (1 + tolerance)
    
    return lower_bound, upper_bound

def calculate_vwap(candles):
    close_prices = np.array([c["close"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    close_prices, volumes = remove_nans_and_zeros(close_prices, volumes)
    return np.sum(close_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else np.nan

def calculate_ema(candles, timeperiod):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    ema = talib.EMA(close_prices, timeperiod=timeperiod)
    return ema[-1] if len(ema) > 0 and not np.isnan(ema[-1]) else np.nan

def calculate_rsi(candles, timeperiod=14):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    rsi = talib.RSI(close_prices, timeperiod=timeperiod)
    return rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else np.nan

def calculate_macd(candles, fastperiod=12, slowperiod=26, signalperiod=9):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    macd, macdsignal, macdhist = remove_nans_and_zeros(macd, macdsignal, macdhist)
    return (
        macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else np.nan,
        macdsignal[-1] if len(macdsignal) > 0 and not np.isnan(macdsignal[-1]) else np.nan,
        macdhist[-1] if len(macdhist) > 0 and not np.isnan(macdhist[-1]) else np.nan
    )

def calculate_momentum(candles, timeperiod=10):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=timeperiod)
    return momentum if len(momentum) > 0 else np.nan

def scale_momentum_to_sine(candles, timeperiod=10):
    momentum_array = calculate_momentum(candles, timeperiod)

    if len(momentum_array) == 0 or np.all(np.isnan(momentum_array)):
        return 0.0, 0.0, 0.0

    sine_wave, leadsine = talib.HT_SINE(momentum_array)

    sine_wave = np.nan_to_num(sine_wave)

    current_sine = sine_wave[-1]

    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0)
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0)

    return dist_min, dist_max, current_sine

def calculate_regression_channels(candles):
    if len(candles) < 50:
        print("Not enough data for regression channel calculation")
        return None, None, None, None

    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))

    close_prices, x = remove_nans_and_zeros(close_prices, x)

    if len(x) < 2:
        print("Not enough valid data points for regression channel calculation")
        return None, None, None, None

    try:
        slope, intercept, _, _, _ = linregress(x, close_prices)
        regression_line = intercept + slope * x
        deviation = close_prices - regression_line
        std_dev = np.std(deviation)

        regression_upper = regression_line + std_dev
        regression_lower = regression_line - std_dev

        regression_upper_value = regression_upper[-1] if not np.isnan(regression_upper[-1]) else None
        regression_lower_value = regression_lower[-1] if not np.isnan(regression_lower[-1]) else None

        current_close_value = close_prices[-1] if len(close_prices) > 0 and not np.isnan(close_prices[-1]) else None

        return regression_lower_value, regression_upper_value, (regression_upper_value + regression_lower_value) / 2, current_close_value

    except Exception as e:
        print(f"Error calculating regression channels: {e}")
        return None, None, None, None

def zigzag_indicator(highs, lows, depth, deviation, backstep):
    zigzag = np.zeros_like(highs)
    last_pivot_low = 0
    last_pivot_high = 0
    current_trend = None

    for i in range(depth, len(highs) - depth):
        if highs[i] == max(highs[i - depth:i + depth]):
            if highs[i] - lows[i] > deviation:
                if last_pivot_high and highs[i] <= highs[last_pivot_high]:
                    continue
                zigzag[i] = highs[i]
                last_pivot_high = i
                current_trend = 'down'

        if lows[i] == min(lows[i - depth:i + depth]):
            if highs[i] - lows[i] > deviation:
                if last_pivot_low and lows[i] >= lows[last_pivot_low]:
                    continue
                zigzag[i] = lows[i]
                last_pivot_low = i
                current_trend = 'up'

    return zigzag, current_trend

def calculate_zigzag_forecast(candles, depth=12, deviation=5, backstep=3):
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])

    highs, lows = remove_nans_and_zeros(highs, lows)

    if len(highs) < depth:
        return None, None, None

    zigzag, current_trend = zigzag_indicator(highs, lows, depth, deviation, backstep)

    pivot_points = zigzag[zigzag != 0]

    if len(pivot_points) < 2:
        return None, None, None

    high = max(pivot_points[-2:])
    low = min(pivot_points[-2:])
    diff = high - low

    if current_trend == 'up':
        fibonacci_levels = {
            "0.0%": low,
            "23.6%": low + 0.236 * diff,
            "38.2%": low + 0.382 * diff,
            "50.0%": low + 0.5 * diff,
            "61.8%": low + 0.618 * diff,
            "76.4%": low + 0.764 * diff,
            "100.0%": high,
        }
        first_incoming_value = fibonacci_levels['76.4%']
    elif current_trend == 'down':
        fibonacci_levels = {
            "0.0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50.0%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "76.4%": high - 0.764 * diff,
            "100.0%": low,
        }
        first_incoming_value = fibonacci_levels['23.6%']
    else:
        return None, None, None

    return first_incoming_value, current_trend, pivot_points

def scale_to_sine(timeframe):
    close_prices = np.array([c["close"] for c in candle_map[timeframe]])
    close_prices, = remove_nans_and_zeros(close_prices)

    current_close = close_prices[-1]

    sine_wave, leadsine = talib.HT_SINE(close_prices)

    sine_wave = np.nan_to_num(sine_wave)

    current_sine = sine_wave[-1]

    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0

    return dist_min, dist_max, current_sine

def calculate_support_resistance(candles):
    closes = np.array([c['close'] for c in candles])
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])

    resistance = np.max(highs[-10:])  # last 10 periods high as resistance
    support = np.min(lows[-10:])      # last 10 periods low as support

    return support, resistance

def forecast_price(candles, n_components, target_distance=0.01):
    closes = np.array([c['close'] for c in candles])

    # Calculate FFT of closing prices
    fft = fftpack.rfft(closes)
    frequencies = fftpack.rfftfreq(len(closes))
    
    # Sort frequencies by magnitude and keep only the top n_components 
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]
    
    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    
    # Calculate the target price as the next value after the last closing price, plus a small constant
    current_close = closes[-1]
    target_price = filtered_signal[-1] + target_distance

    # Calculate the market mood based on the predicted target price and the current close price
    diff = target_price - current_close
    if diff > 0:           
        market_mood = "Bullish"
    else:
        market_mood = "Bearish"
    
    return target_price, market_mood

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    close_prices = np.array(close_prices)

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    momentum = talib.MOM(close_prices, timeperiod=period)

    min_momentum = np.nanmin(momentum)
    max_momentum = np.nanmax(momentum)

    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)

    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)

    avg_mtf = np.nanmean(filtered_close)

    current_momentum = momentum[-1]

    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               
        percent_to_max_momentum = ((current_momentum - min_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan

    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2

    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

def calculate_emas(candles, lengths):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    emas = {}
    for length in lengths:
        ema = talib.EMA(close_prices, timeperiod=length)
        emas[length] = ema[-1] if len(ema) > 0 and not np.isnan(ema[-1]) else np.nan
    return emas

def evaluate_ema_pattern(candles):
    ema_lengths = [9, 12, 21, 27, 45, 56, 100, 150, 200, 369]

    emas = calculate_emas(candles, ema_lengths)

    current_close = candles[-1]["close"]

    below_count = sum(1 for length in ema_lengths if current_close < emas[length])

    max_emas = len(ema_lengths)
    intensity = (below_count / max_emas) * 100

    is_dip = False
    is_top = False

    if current_close < emas[9] and all(emas[ema_lengths[i]] > emas[ema_lengths[i + 1]] for i in range(len(ema_lengths) - 1)):
        is_dip = True
    elif current_close > emas[369] and all(emas[ema_lengths[i]] < emas[ema_lengths[i + 1]] for i in range(len(ema_lengths) - 1)):
        is_top = True

    return current_close, emas, is_dip, is_top, intensity

def analyze_volume_trend(candles, window_size=20):
    if len(candles) < window_size:
        return "Not enough data"

    volumes = np.array([c["volume"] for c in candles[-window_size:]])
    recent_volume = np.array([c["volume"] for c in candles])

    avg_recent_volume = np.mean(volumes)
    avg_previous_volume = np.mean(recent_volume[-2 * window_size:-window_size])

    if avg_recent_volume > avg_previous_volume:
        return "Bullish"
    elif avg_recent_volume < avg_previous_volume:
        return "Bearish"
    else:
        return "Neutral"

def generate_report(timeframe, candles):
    print(f"\nTimeframe: {timeframe}")

    current_close, emas, is_dip, is_top, intensity = evaluate_ema_pattern(candles)

    print(f"Current Close Price: {current_close:.2f}")
    print(f"{'EMA Length':<10} {'EMA Value':<20} {'Close < EMA':<15}")

    for length in sorted(emas.keys()):
        ema_value = emas[length]
        close_below = current_close < ema_value
        print(f"{length:<10} {ema_value:<20.2f} {close_below:<15}")

    # Market mood and reversal predictions
    predictions, minor_reversal_target, major_reversal_target = predict_reversals(candles)
    if predictions is not None:
        print(f"Predicted Close Prices for next {len(predictions)} minutes: {[round(p, 2) for p in predictions]}")
        print(f"Minor Reversal Target: {minor_reversal_target:.2f}")
        print(f"Major Reversal Target: {major_reversal_target:.2f}")

        # Determine market mood
        trend_direction = "Up" if predictions[-1] > current_close else "Down"
        print(f"Market Mood: {trend_direction}")

    # Support and Resistance
    support, resistance = calculate_support_resistance(candles)
    print(f"Support Level: {support:.2f}")
    print(f"Resistance Level: {resistance:.2f}")

    # Price Forecast
    target_price, market_mood = forecast_price(candles, n_components=5, target_distance=0.01)
    print(f"Forecasted Price: {target_price:.2f}")
    print(f"Forecast Market Mood: {market_mood}")

# The main loop for continuous analysis
while True:
    candle_map.clear()

    # Fetch candles for all timeframes
    for timeframe in timeframes:
        candle_map[timeframe] = get_candles(symbol, timeframe)

    # Analyze timeframes
    for timeframe in timeframes:
        candles = candle_map[timeframe]

        if len(candles) > 0:
            generate_report(timeframe, candles)

            vwap = calculate_vwap(candles)
            print(f"VWAP for {timeframe}: {vwap:.2f}")

            ema_50 = calculate_ema(candles, timeperiod=50)
            print(f"EMA 50 for {timeframe}: {ema_50:.2f}")

            rsi = calculate_rsi(candles, timeperiod=14)
            print(f"RSI 14 for {timeframe}: {rsi:.2f}")

            macd, macdsignal, macdhist = calculate_macd(candles)
            print(f"MACD for {timeframe}: {macd:.2f}")
            print(f"MACD Signal for {timeframe}: {macdsignal:.2f}")
            print(f"MACD Histogram for {timeframe}: {macdhist:.2f}")

            momentum = calculate_momentum(candles, timeperiod=10)
            if momentum is not None:
                print(f"Momentum for {timeframe}: {momentum[-1]:.2f}")
            else:
                print(f"Momentum for {timeframe}: not available")

            reg_lower, reg_upper, reg_avg, current_close = calculate_regression_channels(candles)
            print(f"Regression Lower for {timeframe}: {reg_lower:.2f}")
            print(f"Regression Upper for {timeframe}: {reg_upper:.2f}")
            print(f"Regression Avg for {timeframe}: {reg_avg:.2f}")

            first_incoming_value, trend, pivots = calculate_zigzag_forecast(candles)
            print(f"ZigZag Forecast First Incoming Value for {timeframe}: {first_incoming_value:.2f}")
            print(f"ZigZag Forecast Trend for {timeframe}: {trend}")
            print(f"ZigZag Pivot Points for {timeframe}: {pivots}")

            dist_min_close, dist_max_close, current_sine_close = scale_to_sine(timeframe)
            print(f"Sine Scaling Distance to Min (Close Prices) for {timeframe}: {dist_min_close:.2f}%")
            print(f"Sine Scaling Distance to Max (Close Prices) for {timeframe}: {dist_max_close:.2f}%")
            print(f"Sine Scaling Current Sine (Close Prices) for {timeframe}: {current_sine_close:.2f}")

            dist_min_momentum, dist_max_momentum, current_sine_momentum = scale_momentum_to_sine(candles, timeperiod=10)
            print(f"Momentum Scaling Distance to Min for {timeframe}: {dist_min_momentum:.2f}%")
            print(f"Momentum Scaling Distance to Max for {timeframe}: {dist_max_momentum:.2f}%")
            print(f"Current Momentum relative to Sine for {timeframe}: {current_sine_momentum:.2f}")

            min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(
                [c["close"] for c in candles], period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )
            print(f"Thresholds for {timeframe}:")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")

            closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - candles[-1]["close"]))
            if closest_threshold == min_threshold:
                print("The last minimum value is closest to the current close.")
            elif closest_threshold == max_threshold:
                print("The last maximum value is closest to the current close.")
            else:
                print("No threshold value found.")

            angle_price = calculate_45_degree_price(candles)
            if angle_price is not None:
                lower_bound, upper_bound = enforce_price_area(candles, angle_price)
                current_close = candles[-1]["close"]

                print(f"Price Area Around 45-Degree Angle Price: Lower Bound = {lower_bound:.2f}, Upper Bound = {upper_bound:.2f}")

                if current_close > angle_price:
                    print(f"{Fore.RED}Current price ({current_close:.2f}) is ABOVE the 45-degree angle price ({angle_price:.2f}){Style.RESET_ALL}")
                elif current_close < angle_price:
                    print(f"{Fore.GREEN}Current price ({current_close:.2f}) is BELOW the 45-degree angle price ({angle_price:.2f}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Current price ({current_close:.2f}) is EQUAL to the 45-degree angle price ({angle_price:.2f}){Style.RESET_ALL}")

            # Analyze and print the volume trend.
            volume_trend = analyze_volume_trend(candles, window_size=20)
            print(f"Volume Trend for {timeframe}: {volume_trend}")

            print()  # Add a newline for better separation of timeframes.

    time.sleep(5)  # Wait for 5 seconds before the next iteration