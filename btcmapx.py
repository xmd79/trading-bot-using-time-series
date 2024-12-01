#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
import time
import gc

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

symbol = "BTCUSDC"
timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
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

# Helper function to remove NaNs and zeros from arrays
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Function to calculate the Volume Weighted Average Price (VWAP)
def calculate_vwap(candles):
    close_prices = np.array([c["close"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    close_prices, volumes = remove_nans_and_zeros(close_prices, volumes)
    return np.sum(close_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else np.nan

# Define more functions for indicators (all previous functions here)
def calculate_ema(candles, timeperiod):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    ema = talib.EMA(close_prices, timeperiod=timeperiod)
    return ema[-1] if len(ema) > 0 and not np.isnan(ema[-1]) and ema[-1] != 0 else np.nan

def calculate_rsi(candles, timeperiod=14):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    rsi = talib.RSI(close_prices, timeperiod=timeperiod)
    return rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) and rsi[-1] != 0 else np.nan

def calculate_macd(candles, fastperiod=12, slowperiod=26, signalperiod=9):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    macd, macdsignal, macdhist = remove_nans_and_zeros(macd, macdsignal, macdhist)
    return (macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) and macd[-1] != 0 else np.nan,
            macdsignal[-1] if len(macdsignal) > 0 and not np.isnan(macdsignal[-1]) and macdsignal[-1] != 0 else np.nan,
            macdhist[-1] if len(macdhist) > 0 and not np.isnan(macdhist[-1]) and macdhist[-1] != 0 else np.nan)

def calculate_momentum(candles, timeperiod=10):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=timeperiod)
    return momentum[-1] if len(momentum) > 0 and not np.isnan(momentum[-1]) and momentum[-1] != 0 else np.nan

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
        
        regression_upper_value = regression_upper[-1] if not np.isnan(regression_upper[-1]) and regression_upper[-1] != 0 else None
        regression_lower_value = regression_lower[-1] if not np.isnan(regression_lower[-1]) and regression_lower[-1] != 0 else None
        
        current_close_value = close_prices[-1] if len(close_prices) > 0 and not np.isnan(close_prices[-1]) and close_prices[-1] != 0 else None
        
        return regression_lower_value, regression_upper_value, (regression_upper_value + regression_lower_value) / 2, current_close_value

    except Exception as e:
        print(f"Error calculating regression channels: {e}")
        return None, None, None, None

def calculate_polynomial_regression_channels(candles, degree=2):
    if len(candles) < 50:
        print("Not enough data for polynomial regression channel calculation")
        return None, None, None, None
    
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    
    if len(x) < 2:  
        print("Not enough valid data points for polynomial regression channel calculation")
        return None, None, None, None
    
    try:
        coeffs = Polynomial.fit(x, close_prices, degree).convert().coef
        poly = Polynomial(coeffs)
        regression_line = poly(x)
        deviation = close_prices - regression_line
        std_dev = np.std(deviation)
        
        regression_upper = regression_line + std_dev
        regression_lower = regression_line - std_dev
        
        regression_upper_value = regression_upper[-1] if not np.isnan(regression_upper[-1]) and regression_upper[-1] != 0 else None
        regression_lower_value = regression_lower[-1] if not np.isnan(regression_lower[-1]) and regression_lower[-1] != 0 else None
        
        current_close_value = close_prices[-1] if len(close_prices) > 0 and not np.isnan(close_prices[-1]) and close_prices[-1] != 0 else None
        
        return regression_lower_value, regression_upper_value, (regression_upper_value + regression_lower_value) / 2, current_close_value

    except Exception as e:
        print(f"Error calculating polynomial regression channels: {e}")
        return None, None, None, None

def calculate_fibonacci_retracement(high, low):
    diff = high - low
    retracement_levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "76.4%": high - 0.764 * diff,
        "100.0%": low,
    }
    return retracement_levels

def calculate_zigzag_forecast(candles, depth=12, deviation=5, backstep=3):
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    
    highs, lows = remove_nans_and_zeros(highs, lows)

    if len(highs) < depth:
        return None, None, None

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

    zigzag, current_trend = zigzag_indicator(highs, lows, depth, deviation, backstep)
    pivot_points = zigzag[zigzag != 0]
    
    if len(pivot_points) < 2:
        return None, None, None

    high = max(pivot_points[-2:])
    low = min(pivot_points[-2:])
    fibonacci_levels = calculate_fibonacci_retracement(high, low)
    
    first_incoming_value = None
    if current_trend == 'up':
        first_incoming_value = fibonacci_levels['23.6%']
    elif current_trend == 'down':
        first_incoming_value = fibonacci_levels['76.4%']

    return first_incoming_value, current_trend, pivot_points

def scale_to_sine(timeframe):
    close_prices = np.array([c["close"] for c in candle_map[timeframe]])
    close_prices, = remove_nans_and_zeros(close_prices)
    
    current_close = close_prices[-1]

    sine_wave, leadsine = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    return dist_min, dist_max, current_sine
      
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

    return min_threshold, max_threshold, avg_mtf, current_momentum, range_price

def calculate_distances_and_ratios(candles):
    if len(candles) < 2:
        return None, None, None
    
    current_close = candles[-1]['close']
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    
    last_major_high = np.max(highs)
    last_major_low = np.min(lows)

    distance_to_high = last_major_high - current_close
    distance_to_low = current_close - last_major_low

    total_range = last_major_high - last_major_low

    percent_to_high = (distance_to_high / total_range) * 100 if total_range != 0 else np.nan
    percent_to_low = (distance_to_low / total_range) * 100 if total_range != 0 else np.nan

    return distance_to_high, distance_to_low, percent_to_high, percent_to_low, last_major_high, last_major_low

# Function to calculate min/max thresholds based on the last 500 close prices
def calculate_min_max_thresholds(candles):
    if len(candles) < 500:
        close_prices = np.array([c["close"] for c in candles])
    else:
        close_prices = np.array([c["close"] for c in candles[-500:]])  # Limit to last 500
    close_prices, = remove_nans_and_zeros(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    return min_close, max_close

# Main function to analyze timeframes
def analyze_timeframes():
    global candle_map
    
    while True:
        double_bottom_detection_summary = {}
        last_major_reversal_summary = {}

        # Fetch new candles for all timeframes
        for timeframe in timeframes:
            candle_map[timeframe] = get_candles(symbol, timeframe)

        for timeframe in timeframes:
            candles = candle_map[timeframe]
            close_prices = np.array([c["close"] for c in candles])
            
            print(f"\nAnalyzing {timeframe}...")

            vwap = calculate_vwap(candles)
            print(f"VWAP: {vwap:.2f}")
            
            ema_50 = calculate_ema(candles, timeperiod=50)
            print(f"EMA 50: {ema_50:.2f}")
            
            rsi = calculate_rsi(candles, timeperiod=14)
            print(f"RSI 14: {rsi:.2f}")
            
            macd, macdsignal, macdhist = calculate_macd(candles)
            print(f"MACD: {macd:.2f}, Signal: {macdsignal:.2f}, Histogram: {macdhist:.2f}")
            
            momentum = calculate_momentum(candles, timeperiod=10)
            print(f"Momentum: {momentum:.2f}")

            reg_lower, reg_upper, reg_avg, current_close_value = calculate_regression_channels(candles)
            print(f"Regression Lower: {reg_lower:.2f}, Upper: {reg_upper:.2f}, Avg: {reg_avg:.2f}")
            
            poly_reg_lower, poly_reg_upper, poly_reg_avg, current_close_value = calculate_polynomial_regression_channels(candles)
            print(f"Polynomial Regression Lower: {poly_reg_lower:.2f}, Upper: {poly_reg_upper:.2f}, Avg: {poly_reg_avg:.2f}")
            
            fib_high = np.max([c["high"] for c in candles])
            fib_low = np.min([c["low"] for c in candles])
            fibonacci_levels = calculate_fibonacci_retracement(fib_high, fib_low)
            print(f"Fibonacci Retracement Levels: {fibonacci_levels}")

            first_incoming_value, trend, pivots = calculate_zigzag_forecast(candles)
            print(f"ZigZag Forecast First Incoming Value: {first_incoming_value:.2f}, Trend: {trend}, Pivot Points: {pivots}")

            dist_min, dist_max, current_sine = scale_to_sine(timeframe)
            print(f"Sine Scaling Distance to Min: {dist_min:.2f}%, Max: {dist_max:.2f}%, Current Sine: {current_sine:.2f}")

            # Calculate min and max thresholds based on the last 500 close prices
            min_threshold, max_threshold, avg_mtf, current_momentum, range_price = calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)
            print(f"Thresholds: Min {min_threshold:.2f}, Max {max_threshold:.2f}, Avg MTF: {avg_mtf:.2f}, Momentum Signal: {current_momentum:.2f}")

            distance_to_high, distance_to_low, percent_to_high, percent_to_low, last_major_high, last_major_low = calculate_distances_and_ratios(candles)
            print(f"Distance to Last Major High: {distance_to_high:.2f}, Low: {distance_to_low:.2f}, Percent High: {percent_to_high:.2f}%, Low: {percent_to_low:.2f}%")

            distance_between_thresholds = max_threshold - min_threshold
            last_major_reversal = last_major_high if last_major_high > close_prices[-1] else last_major_low
            distance_from_last_reversal = abs(last_major_reversal - close_prices[-1])

            reversal_type = "TOP" if last_major_high == last_major_reversal else "DIP"
            last_major_reversal_summary[timeframe] = reversal_type

            total_distance = distance_between_thresholds + distance_from_last_reversal
            symmetrical_percentage_thresholds = (distance_between_thresholds / total_distance) * 100 if total_distance != 0 else np.nan
            symmetrical_percentage_last_reversal = (distance_from_last_reversal / total_distance) * 100 if total_distance != 0 else np.nan

            print(f"Distance Between Thresholds: {distance_between_thresholds:.2f}")
            print(f"Distance from Last Major Reversal to Current Close: {distance_from_last_reversal:.2f}")
            print(f"Symmetrical Percentage: {symmetrical_percentage_thresholds:.2f}% (Thresholds), {symmetrical_percentage_last_reversal:.2f}% (Last Major Reversal)")
            print(f"Last Major Reversal Found at: {reversal_type}")

            closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close_prices[-1]))
            is_closest_max = (closest_threshold == max_threshold)
            is_closest_min = (closest_threshold == min_threshold)

            print(f"Is the last maximum value closest to current close? {'True' if is_closest_max else 'False'}")
            print(f"Is the last minimum value closest to current close? {'True' if is_closest_min else 'False'}")

            # Check for double bottom detection for the current timeframe
            double_bottom_detected = False
            if symmetrical_percentage_thresholds <= symmetrical_percentage_last_reversal and reversal_type == "DIP":
                print("Potential double bottom detected.")
                double_bottom_detected = True
            elif symmetrical_percentage_thresholds > symmetrical_percentage_last_reversal and reversal_type == "TOP":
                print("Potential double bottom not yet detected.")

            double_bottom_detection_summary[timeframe] = double_bottom_detected

        # Print overall results for each iteration
        print(f"\nOverall Double Bottom Pattern Detected: {'Yes' if any(double_bottom_detection_summary.values()) else 'No'}")
        print(f"Overall Last Major Reversal Found: {last_major_reversal_summary}")

        for timeframe in timeframes:
            print(f"Timeframe: {timeframe} - Double Bottom Detected: {'Yes' if double_bottom_detection_summary[timeframe] else 'No'}, Last Major Reversal: {last_major_reversal_summary[timeframe]}")

        gc.collect()
        time.sleep(5)

if __name__ == "__main__":
    analyze_timeframes()
