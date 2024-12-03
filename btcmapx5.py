#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
import time
import gc
from statsmodels.tsa.arima.model import ARIMA
from numpy.fft import fft, ifft

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

symbol = "BTCUSDC"
timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
candle_map = {}

# Function to get candles
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

    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    return dist_min, dist_max, current_sine

# New Function to Calculate FFT and Forecast Prices
def calculate_fft_forecast(close_prices):
    n = len(close_prices)
    if n < 2:
        return None
    
    # Perform FFT
    yf = fft(close_prices)
    
    # Construct forecast using frequencies
    forecast = np.real(ifft(yf))
    
    return forecast[-1]  # Return last forecasted price

# New Function to calculate ML forecast using ARIMA
def calculate_arima_forecast(close_prices):
    try:
        model = ARIMA(close_prices, order=(5, 1, 0))  # You can adjust the order as needed
        model_fit = model.fit()
        return model_fit.forecast(steps=1)[0]  # Predict next price
    except Exception as e:
        print(f"ARIMA model error: {e}")
        return None

# New Function for Random Walk Forecast
def random_walk_forecast(close_prices):
    if len(close_prices) < 2:
        return None

    # Random walk: Next price could simply be the last price plus a random amount
    last_price = close_prices[-1]
    return last_price + np.random.normal(0, 0.01) * last_price  # Adding randomness based on percentage

# New Function to Calculate Thresholds
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    close_prices = np.array(close_prices)
    
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
        percent_to_min_momentum = ((max_momentum - current_momentum) /   
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               

        percent_to_max_momentum = ((current_momentum - min_momentum) / 
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
 
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

# New function to calculate distances and ratios
def calculate_distances_and_ratios(candles):
    if len(candles) < 2:
        return None, None, None, None, None, None

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

# Function to find major reversals
def find_major_reversals(candles, current_close, min_threshold, max_threshold):
    lows = [candle['low'] for candle in candles if candle['low'] >= min_threshold]
    highs = [candle['high'] for candle in candles if candle['high'] <= max_threshold]

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

# Function to find ABC pattern and X point
def find_abc_x_pattern(lows, highs, last_major_reversal):
    if len(lows) < 3 or len(highs) < 3:
        return None, None  # Not enough points to form a pattern

    # Extract the last three points
    last_dip = lows[-3:]
    last_top = highs[-3:]

    # Assign A, B, C based on last 3 dips and tops
    A = last_dip[0]
    B = last_dip[1]
    C = last_dip[2]
    X = last_major_reversal if last_major_reversal is not None else None

    return (X, (A, B, C), (last_top[0], last_top[1], last_top[2]))

# New Function to determine the closest DIP and TOP
def find_closest_dip_and_top(current_close, dip_values, top_values):
    # Calculate distances to the current close price
    distances_to_dip = {key: abs(current_close - value) for key, value in dip_values.items()}
    distances_to_top = {key: abs(current_close - value) for key, value in top_values.items()}

    # Find the closest DIP
    closest_dip_key = min(distances_to_dip, key=distances_to_dip.get)
    closest_dip_value = dip_values[closest_dip_key]
    
    # Find the closest TOP
    closest_top_key = min(distances_to_top, key=distances_to_top.get)
    closest_top_value = top_values[closest_top_key]

    return (closest_dip_key, closest_dip_value), (closest_top_key, closest_top_value)

# New function to calculate dynamic envelope
def calculate_dynamic_envelope(last_bottom, last_top):
    if last_bottom is None or last_top is None:
        return None, None  # Return None if we don't have valid levels
    
    middle_threshold = (last_bottom + last_top) / 2
    ratio = (last_top - last_bottom) / (last_top + last_bottom) if (last_bottom + last_top) != 0 else 0
    
    return middle_threshold, ratio

# Define the function detect_combined_patterns
def detect_combined_patterns(current_close, closest_reversal, min_threshold, max_threshold, distance_to_high, distance_to_low):
    double_bottom = double_top = False

    # Determine if there's a double bottom
    if (
        current_close < closest_reversal and
        closest_reversal < min_threshold
    ):
        double_bottom = True

    # Determine if there's a double top
    if (
        current_close > closest_reversal and
        closest_reversal > max_threshold
    ):
        double_top = True

    return double_bottom, double_top

# Function to predict triangle formation and next price points
def triangle_prediction(low_points, high_points):
    if len(low_points) < 2 or len(high_points) < 2:
        return None, None  # Need at least 2 points to determine a trend
  
    # Determine the trend for low points
    low_slope = (low_points[-1] - low_points[-2])
    
    # Determine the trend for high points
    high_slope = (high_points[-1] - high_points[-2])
    
    if low_slope > 0 and high_slope > 0:
        triangle_type = "Ascending"
    elif low_slope < 0 and high_slope < 0:
        triangle_type = "Descending"
    else:
        triangle_type = "Symmetrical"

    # Calculate projected next points
    projected_low = low_points[-1] + low_slope
    projected_high = high_points[-1] + high_slope
    
    return triangle_type, (projected_low, projected_high)

# Extended Main function to analyze timeframes
def analyze_timeframes():
    global candle_map, timeframes  # Declare timeframes as global

    while True:
        double_bottom_detection_summary = {}
        double_top_detection_summary = {}
        last_major_reversal_summary = {}
        combined_signals = {}  # Dictionary to store combined signals for each timeframe
        overall_buy_signal = False  # New flag for MTF buy signal

        # Fetch new candles for all timeframes
        for timeframe in timeframes:
            candle_map[timeframe] = get_candles(symbol, timeframe)

        for timeframe in timeframes:
            candles = candle_map[timeframe]
            close_prices = np.array([c["close"] for c in candles])
            lows = np.array([c["low"] for c in candles])
            highs = np.array([c["high"] for c in candles])

            print(f"\nAnalyzing {timeframe}...")

            # Gather indicators
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

            # Calculate thresholds using the provided function
            min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)
            print(f"Momentum signal: {momentum_signal:.2f}")
            print(f"Min Threshold: {min_threshold:.2f}, Max Threshold: {max_threshold:.2f}, Avg MTF: {avg_mtf:.2f}")

            # Calculate distances and ratios
            distance_to_high, distance_to_low, percent_to_high, percent_to_low, last_major_high, last_major_low = calculate_distances_and_ratios(candles)
            print(f"Distance to Last Major High: {distance_to_high:.2f}, Low: {distance_to_low:.2f}, Percent High: {percent_to_high:.2f}%, Low: {percent_to_low:.2f}%")

            # Define current_close
            current_close = close_prices[-1]

            # Find major reversals using the provided function
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candles, current_close, min_threshold, max_threshold)
            last_major_reversal_summary[timeframe] = (closest_type, closest_reversal)
            print(f"Last Major Reversal Found at: {closest_type}, Price: {closest_reversal:.2f}")

            # Display last major found lows and highs for the current timeframe
            last_major_low_message = f"Last Major Low for {timeframe}: {last_bottom:.2f}" if last_bottom is not None else "No Last Major Low Found"
            last_major_high_message = f"Last Major High for {timeframe}: {last_top:.2f}" if last_top is not None else "No Last Major High Found"
            print(last_major_low_message)
            print(last_major_high_message)

            # Print most recent major local reversal
            if last_major_reversal_summary[timeframe][0] == "DIP":
                print(f"Recent Major Local Reversal is at Minima found at Price: {closest_reversal:.2f}")
            elif last_major_reversal_summary[timeframe][0] == "TOP":
                print(f"Recent Major Local Reversal is at Maxima found at Price: {closest_reversal:.2f}")

            # Check for combined pattern detection for the current timeframe
            double_bottom_detected, double_top_detected = detect_combined_patterns(current_close, closest_reversal, min_threshold, max_threshold, distance_to_high, distance_to_low)
            double_bottom_detection_summary[timeframe] = double_bottom_detected
            double_top_detection_summary[timeframe] = double_top_detected

            if double_bottom_detected:
                print("Potential double bottom detected.")
            if double_top_detected:
                print("Potential double top detected.")

            # Find last 3 lows and 3 highs for ABC pattern detection
            last_3_lows = sorted(lows[-3:])  # Sort lows to find A, B, C
            last_3_highs = sorted(highs[-3:])  # Sort highs to find A, B, C

            if len(last_3_lows) >= 3 and len(last_3_highs) >= 3:
                DIP_A, DIP_B, DIP_C = last_3_lows
                print(f"DIP A: {DIP_A}, DIP B: {DIP_B}, DIP C: {DIP_C}")

                TOP_A, TOP_B, TOP_C = last_3_highs
                print(f"TOP A: {TOP_A}, TOP B: {TOP_B}, TOP C: {TOP_C}")

                # Calculate distances from current close price to most recent lows and highs
                distance_to_DIP_A = abs(current_close - DIP_A)
                distance_to_DIP_B = abs(current_close - DIP_B)
                distance_to_DIP_C = abs(current_close - DIP_C)
                
                distance_to_TOP_A = abs(current_close - TOP_A)
                distance_to_TOP_B = abs(current_close - TOP_B)
                distance_to_TOP_C = abs(current_close - TOP_C)

                print(f"Distance to DIP A: {distance_to_DIP_A:.2f}, Distance to DIP B: {distance_to_DIP_B:.2f}, Distance to DIP C: {distance_to_DIP_C:.2f}")
                print(f"Distance to TOP A: {distance_to_TOP_A:.2f}, Distance to TOP B: {distance_to_TOP_B:.2f}, Distance to TOP C: {distance_to_TOP_C:.2f}")

                # Find the closest DIP and TOP
                closest_dip, closest_top = find_closest_dip_and_top(current_close, {'DIP A': DIP_A, 'DIP B': DIP_B, 'DIP C': DIP_C},
                                                                     {'TOP A': TOP_A, 'TOP B': TOP_B, 'TOP C': TOP_C})

                print(f"Most Recent DIP: {closest_dip[0]} with value: {closest_dip[1]:.2f}")
                print(f"Most Recent TOP: {closest_top[0]} with value: {closest_top[1]:.2f}")

                # Find the ABC pattern based on the rules defined
                abc_signal, low_points, high_points = find_abc_x_pattern(last_3_lows, last_3_highs, closest_reversal)
                if abc_signal:
                    print(f"ABC Pattern Signal: {abc_signal}. Low Points: {low_points}, High Points: {high_points}")

                # Predict triangle formation and next price points
                triangle_type, projected_points = triangle_prediction(last_3_lows[-3:], last_3_highs[-3:])
                if projected_points is not None:
                    print(f"Triangle Formation Type: {triangle_type}. Projected Low: {projected_points[0]:.2f}, Projected High: {projected_points[1]:.2f}")

            # Forecasting
            arima_forecast = calculate_arima_forecast(close_prices)
            print(f"ARIMA Forecast for {timeframe}: {arima_forecast:.2f}")

            fft_forecast = calculate_fft_forecast(close_prices)
            print(f"FFT Forecast for {timeframe}: {fft_forecast:.2f}")

            random_walk = random_walk_forecast(close_prices)
            print(f"Random Walk Forecast for {timeframe}: {random_walk:.2f}")

            # Calculate distances and ratios between thresholds and the most recent major reversal
            total_distance = max_threshold - min_threshold + abs(closest_reversal - current_close)

            # Calculate the symmetrical percentages
            symmetrical_percentage_thresholds = (max_threshold - min_threshold) / total_distance * 100 if total_distance != 0 else np.nan
            symmetrical_percentage_last_reversal = (abs(closest_reversal - current_close) / total_distance) * 100 if total_distance != 0 else np.nan

            print(f"Distance Between Thresholds: {max_threshold - min_threshold:.2f}")
            print(f"Distance from Last Major Reversal to Current Close: {abs(closest_reversal - current_close):.2f}")
            print(f"Symmetrical Percentage: {symmetrical_percentage_thresholds:.2f}% (Thresholds), {symmetrical_percentage_last_reversal:.2f}% (Last Major Reversal)")

            # Calculate dynamic envelope
            middle_threshold, ratio = calculate_dynamic_envelope(last_bottom, last_top)
            if middle_threshold is not None:
                print(f"Dynamic Envelope Middle Threshold: {middle_threshold:.2f}, Ratio: {ratio:.2f}")
                predominant_pattern = "Double Bottom" if double_bottom_detected else "Double Top" if double_top_detected else "None"
                print(f"Predominant Pattern: {predominant_pattern}")

        # Determine if an overall buy signal should be issued
        if (
            last_major_reversal_summary.get('1m', (None,))[0] == "DIP" and
            last_major_reversal_summary.get('3m', (None,))[0] == "DIP" and
            last_major_reversal_summary.get('5m', (None,))[0] == "DIP"
        ):
            overall_buy_signal = True
        
        # Print overall results for each iteration
        print(f"\nOverall Double Bottom Pattern Detected: {'Yes' if any(double_bottom_detection_summary.values()) else 'No'}")
        print(f"Overall Double Top Pattern Detected: {'Yes' if any(double_top_detection_summary.values()) else 'No'}")
        print("Overall Last Major Reversal Found:")
        for timeframe, (reversal_type, price) in last_major_reversal_summary.items():
            print(f"{timeframe}: {reversal_type}, Price: {price:.2f}")

        for timeframe in timeframes:
            double_bottom_detected = double_bottom_detection_summary[timeframe]
            double_top_detected = double_top_detection_summary[timeframe]
            reversal_type, reversal_price = last_major_reversal_summary[timeframe]
            print(f"Timeframe: {timeframe} - Double Bottom Detected: {'Yes' if double_bottom_detected else 'No'}, "
                  f"Double Top Detected: {'Yes' if double_top_detected else 'No'}, "
                  f"Last Major Reversal: {reversal_type}, Price: {reversal_price:.2f}")

        # Final output of overall buy signal
        if overall_buy_signal:
            print("Overall MTF Buy Signal: TRUE")
        else:
            print("Overall MTF Buy Signal: FALSE")

        gc.collect()
        time.sleep(5)

if __name__ == "__main__":
    analyze_timeframes()