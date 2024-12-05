#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft, ifft
import time
import gc

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Define symbol and timeframes
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
    return (macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else np.nan,
            macdsignal[-1] if len(macdsignal) > 0 and not np.isnan(macdsignal[-1]) else np.nan,
            macdhist[-1] if len(macdhist) > 0 and not np.isnan(macdhist[-1]) else np.nan)

def calculate_momentum(candles, timeperiod=10):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=timeperiod)
    return momentum[-1] if len(momentum) > 0 and not np.isnan(momentum[-1]) else np.nan

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

        regression_upper_value = regression_upper[-1] if not np.isnan(regression_upper[-1]) else None
        regression_lower_value = regression_lower[-1] if not np.isnan(regression_lower[-1]) else None

        current_close_value = close_prices[-1] if len(close_prices) > 0 and not np.isnan(close_prices[-1]) else None

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
    if len(close_prices) < period:  # Ensure there's enough data
        print("Not enough data to calculate thresholds.")
        return None, None, None, None, None  # Return None for all thresholds

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    close_prices = np.array(close_prices)

    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=period)

    if len(momentum) == 0:  # Ensure there is momentum data
        print("Not enough data for momentum calculation.")
        return None, None, None, None, None

    min_momentum = np.nanmin(momentum)
    max_momentum = np.nanmax(momentum)

    # Calculate thresholds
    min_percentage_custom = minimum_percentage / 100
    max_percentage_custom = maximum_percentage / 100

    min_threshold = min_close - (max_close - min_close) * min_percentage_custom
    max_threshold = max_close + (max_close - min_close) * max_percentage_custom

    # Check for NaNs in calculated thresholds
    if np.isnan(min_threshold) or np.isnan(max_threshold):
        print("Calculated thresholds are NaN.")
        return None, None, None, None, None

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

    last_bottom = np.nanmin(lows) if len(lows) > 0 else None
    last_top = np.nanmax(highs) if len(highs) > 0 else None

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

# Function to assess if double tops or bottoms were detected based on current close
def assess_double_pattern(current_close, last_major_reversal, last_highs, last_lows):
    if last_highs is not None:
        print(f"Checking last highs: {last_highs}")  # Print for debugging

    last_bottom = np.nanmin(last_lows) if len(last_lows) > 0 else None
    last_top = np.nanmax(last_highs) if len(last_highs) > 0 else None

    # Make comparison logic for double top: all highs should be less than or equal to last_major_reversal
    if last_major_reversal == "DIP" and last_bottom is not None:
        # Check if all lows after the dip are above the last major reversal found
        all_above_dip = all(l > last_bottom for l in last_lows)
        close_above_last_low = current_close > last_bottom

        # Print checks
        print(f"Double Bottom Check - Last Bottom: {last_bottom:.2f}")
        print(f"All Lows: {[f'{l:.2f}' for l in last_lows]}, All Lows > Last Major Reversal Found: {all_above_dip}, Current Close > Last Low: {close_above_last_low}")

        if all_above_dip and close_above_last_low:
            return "DIP"  # Double Bottom confirmed

    elif last_major_reversal == "TOP" and last_top is not None:
        # Check if all highs after the top are below or equal to the last major reversal found
        all_below_or_equal_top = all(h <= last_top for h in last_highs)
        close_below_last_top = current_close < last_top

        # Print checks
        print(f"Double Top Check - Last Top: {last_top:.2f}")
        print(f"All Highs: {[f'{h:.2f}' for h in last_highs]}, All Highs <= Last Major Reversal Found: {all_below_or_equal_top}, Current Close < Last Top: {close_below_last_top}")

        if all_below_or_equal_top and close_below_last_top:
            return "TOP"  # Double Top confirmed

    return None

# Function for triangle prediction
def triangle_prediction(lows, highs, closest_reversal):
    # Consider the last three lows (A, B, C)
    A, B, C = lows
    # Consider the last three highs (NP1, NP2, NP3)
    NP1, NP2, NP3 = highs

    # Calculate projected lows and highs based on the range between them
    projected_low = B + (C - A)
    projected_high = B + (NP2 - NP1)

    # Determine triangle type based on prices
    triangle_type = "ASCENDING" if projected_low < closest_reversal else "DESCENDING"

    return triangle_type, (projected_low, projected_high)

# New function to calculate dynamic envelope
def calculate_dynamic_envelope(last_bottom, last_top):
    if last_bottom is None or last_top is None:
        return None, None  # Return None if inputs are not valid

    middle_threshold = (last_bottom + last_top) / 2
    ratio = (last_top - last_bottom) / last_bottom if last_bottom != 0 else np.nan  # Avoid division by zero

    return middle_threshold, ratio

# Extended Main function to analyze timeframes
def analyze_timeframes():
    global candle_map  # Use global declaration for the candle_map

    while True:
        double_bottom_detection_summary = {tf: False for tf in timeframes}
        double_top_detection_summary = {tf: False for tf in timeframes}
        last_major_reversal_summary = {}
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

            # Adding check for None values here
            if min_threshold is None or max_threshold is None:
                print(f"Skipping distance calculations for {timeframe} due to invalid thresholds.")
                continue  # Skip this timeframe if thresholds are invalid

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

            # Adjust the print statement to handle None values gracefully
            if closest_reversal is not None:
                print(f"Last Major Reversal Found at: {closest_type}, Price: {closest_reversal:.2f}")
            else:
                print(f"Last Major Reversal Found at: {closest_type}, Price: Not Available")

            # Display last major found lows and highs for the current timeframe
            last_major_low_message = f"Last Major Low for {timeframe}: {last_bottom:.2f}" if last_bottom is not None else "No Last Major Low Found"
            last_major_high_message = f"Last Major High for {timeframe}: {last_top:.2f}" if last_top is not None else "No Last Major High Found"
            print(last_major_low_message)
            print(last_major_high_message)

            # Most recent minor reversal determination based on current close and last three reversals only
            recent_minima = lows[-3:]  # Last 3 lows
            recent_maxima = highs[-3:]  # Last 3 highs

            most_recent_minor_reversal = None
            if current_close < np.min(recent_minima):
                most_recent_minor_reversal = np.min(recent_minima)
                print(f"Most Recent Minor Reversal found at Price: {most_recent_minor_reversal:.2f} (Minima)")
            elif current_close > np.max(recent_maxima):
                most_recent_minor_reversal = np.max(recent_maxima)
                print(f"Most Recent Minor Reversal found at Price: {most_recent_minor_reversal:.2f} (Maxima)")

            # Ensure we select the most recent between the last minor reversals and the most recent low/high reversal
            if closest_reversal is not None:
                if most_recent_minor_reversal is None or closest_reversal > most_recent_minor_reversal:
                    most_recent_minor_reversal = closest_reversal

            # Check for double bottom or double top based on current close, min, max thresholds
            last_highs = highs[-3:]  # Only consider last 3 highs
            last_lows = lows[-3:]    # Only consider last 3 lows
            double_pattern = assess_double_pattern(current_close, closest_type, last_highs, last_lows)

            if double_pattern == "DIP":
                double_bottom_detection_summary[timeframe] = True
                print("Double Bottom confirmed.")
            elif double_pattern == "TOP":
                double_top_detection_summary[timeframe] = True
                print("Double Top confirmed.")

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

                # Predict triangle formation and next price points with the last major reversal
                triangle_type, projected_points = triangle_prediction(last_3_lows[-3:], last_3_highs[-3:], closest_reversal)
                if projected_points is not None:
                    middle_threshold = (projected_points[0] + projected_points[1]) / 2
                    print(f"Triangle Formation Type: {triangle_type}. Projected Low: {projected_points[0]:.2f}, Projected High: {projected_points[1]:.2f}")
                    print(f"Forecasted Triangle Price (Threshold): {middle_threshold:.2f}")

            # Forecasting
            arima_forecast = calculate_arima_forecast(close_prices)
            print(f"ARIMA Forecast for {timeframe}: {arima_forecast:.2f}")

            fft_forecast = calculate_fft_forecast(close_prices)
            print(f"FFT Forecast for {timeframe}: {fft_forecast:.2f}")

            random_walk = random_walk_forecast(close_prices)
            print(f"Random Walk Forecast for {timeframe}: {random_walk:.2f}")

            # Calculate distances and ratios between thresholds and the most recent major reversal
            if min_threshold is not None and max_threshold is not None:
                if closest_reversal is not None:  # Ensure closest_reversal is valid for calculations
                    total_distance = max_threshold - min_threshold + abs(closest_reversal - current_close)

                    # Calculate the symmetrical percentages
                    symmetrical_percentage_thresholds = (max_threshold - min_threshold) / total_distance * 100 if total_distance != 0 else np.nan
                    symmetrical_percentage_last_reversal = (abs(closest_reversal - current_close) / total_distance) * 100 if total_distance != 0 else np.nan

                    print(f"Distance Between Thresholds: {max_threshold - min_threshold:.2f}")
                    print(f"Distance from Last Major Reversal to Current Close: {abs(closest_reversal - current_close):.2f}")
                    print(f"Symmetrical Percentage: {symmetrical_percentage_thresholds:.2f}% (Thresholds), {symmetrical_percentage_last_reversal:.2f}% (Last Major Reversal)")
                else:
                    # Handle case where closest_reversal is None
                    print("Cannot calculate distance from Last Major Reversal to Current Close: closest_reversal is not available.")
            else:
                print(f"Skipping distance calculations due to invalid thresholds for {timeframe}.")

            # Calculate dynamic envelope
            middle_threshold, ratio = calculate_dynamic_envelope(last_bottom, last_top)
            if middle_threshold is not None:
                print(f"Dynamic Envelope Middle Threshold: {middle_threshold:.2f}, Ratio: {ratio:.2f}")
                predominant_pattern = "Double Bottom" if double_bottom_detection_summary.get(timeframe) else "Double Top" if double_top_detection_summary.get(timeframe) else "None"
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
            if price is not None:
                print(f"{timeframe}: {reversal_type}, Price: {price:.2f}")
            else:
                print(f"{timeframe}: {reversal_type}, Price: Not Available")

        for timeframe in timeframes:
            double_bottom_detected = double_bottom_detection_summary[timeframe]
            double_top_detected = double_top_detection_summary[timeframe]
            reversal_type, reversal_price = last_major_reversal_summary[timeframe]
            print(f"Timeframe: {timeframe} - Double Bottom Detected: {'Yes' if double_bottom_detected else 'No'}, "
                  f"Double Top Detected: {'Yes' if double_top_detected else 'No'}, "
                  f"Last Major Reversal: {reversal_type}, Price: {reversal_price:.2f}" if reversal_price is not None else f"Last Major Reversal: {reversal_type}, Price: Not Available")

        # Final output of overall buy signal
        if overall_buy_signal:
            print("Overall MTF Buy Signal: TRUE")
        else:
            print("Overall MTF Buy Signal: FALSE")

        gc.collect()
        time.sleep(5)

if __name__ == "__main__":
    analyze_timeframes()