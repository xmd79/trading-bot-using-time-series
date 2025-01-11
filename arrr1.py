#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial

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

# Fetch candles for all timeframes
for timeframe in timeframes:
    candle_map[timeframe] = get_candles(symbol, timeframe)

# Helper function to remove NaNs and zeros from arrays
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Define functions for indicators
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

def calculate_regression_channels(candles, min_threshold, max_threshold):
    # Filter out close prices that are within the thresholds
    close_prices = np.array([c["close"] for c in candles])
    valid_mask = (close_prices >= min_threshold) & (close_prices <= max_threshold)
    close_prices = close_prices[valid_mask]
    
    if len(close_prices) < 50:
        print("Not enough valid data for regression channel calculation")
        return None, None, None, None  # Not enough data to calculate regression channels
    
    x = np.arange(len(valid_mask))  # Create x as a series

    try:
        # Perform linear regression using scipy.stats.linregress
        slope, intercept, _, _, _ = linregress(x[valid_mask], close_prices)
        regression_line = intercept + slope * x
        deviation = close_prices - regression_line[valid_mask]
        std_dev = np.std(deviation)
        
        regression_upper = regression_line + std_dev
        regression_lower = regression_line - std_dev
        
        # Return the last values of regression channels
        regression_upper_value = regression_upper[-1] if not np.isnan(regression_upper[-1]) and regression_upper[-1] != 0 else None
        regression_lower_value = regression_lower[-1] if not np.isnan(regression_lower[-1]) and regression_lower[-1] != 0 else None
        
        # Get the current close value
        current_close_value = close_prices[-1] if len(close_prices) > 0 and not np.isnan(close_prices[-1]) and close_prices[-1] != 0 else None
        
        return regression_lower_value, regression_upper_value, (regression_upper_value + regression_lower_value) / 2, current_close_value

    except Exception as e:
        print(f"Error calculating regression channels: {e}")
        return None, None, None, None

# New function for polynomial regression channels
def calculate_polynomial_regression_channels(candles, min_threshold, max_threshold, degree=2):
    # Filter out close prices that are within the thresholds
    close_prices = np.array([c["close"] for c in candles])
    valid_mask = (close_prices >= min_threshold) & (close_prices <= max_threshold)
    close_prices = close_prices[valid_mask]
    
    if len(close_prices) < 50:
        print("Not enough valid data for polynomial regression channel calculation")
        return None, None, None, None  # Not enough data to calculate polynomial regression channels
    
    x = np.arange(len(valid_mask))  # Create x based on valid mask

    try:
        # Perform polynomial regression using numpy's Polynomial
        coeffs = Polynomial.fit(x[valid_mask], close_prices, degree).convert().coef
        poly = Polynomial(coeffs)
        regression_line = poly(x)
        deviation = close_prices - regression_line[valid_mask]
        std_dev = np.std(deviation)
        
        regression_upper = regression_line + std_dev
        regression_lower = regression_line - std_dev
        
        # Return the last values of regression channels
        regression_upper_value = regression_upper[-1] if not np.isnan(regression_upper[-1]) and regression_upper[-1] != 0 else None
        regression_lower_value = regression_lower[-1] if not np.isnan(regression_lower[-1]) and regression_lower[-1] != 0 else None
        
        # Get the current close value
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
    # Extracting the high and low prices
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    
    # Remove NaNs and zeros
    highs, lows = remove_nans_and_zeros(highs, lows)

    if len(highs) < depth:
        return None, None, None  # Not enough data to calculate ZigZag forecast

    def zigzag_indicator(highs, lows, depth, deviation, backstep):
        """ Simple ZigZag indicator """
        zigzag = np.zeros_like(highs)
        last_pivot_low = 0
        last_pivot_high = 0
        current_trend = None  # None, 'up', 'down'

        for i in range(depth, len(highs) - depth):
            # Find local high
            if highs[i] == max(highs[i - depth:i + depth]):
                if highs[i] - lows[i] > deviation:
                    if last_pivot_high and highs[i] <= highs[last_pivot_high]:
                        continue
                    zigzag[i] = highs[i]
                    last_pivot_high = i
                    current_trend = 'down'  # Trend is going down after a high

            # Find local low
            if lows[i] == min(lows[i - depth:i + depth]):
                if highs[i] - lows[i] > deviation:
                    if last_pivot_low and lows[i] >= lows[last_pivot_low]:
                        continue
                    zigzag[i] = lows[i]
                    last_pivot_low = i
                    current_trend = 'up'  # Trend is going up after a low

        return zigzag, current_trend

    zigzag, current_trend = zigzag_indicator(highs, lows, depth, deviation, backstep)
    # Extracting the pivot points
    pivot_points = zigzag[zigzag != 0]
    
    if len(pivot_points) < 2:
        return None, None, None  # Not enough pivot points to calculate forecast

    # Calculate Fibonacci levels based on last two pivot points
    high = max(pivot_points[-2:])
    low = min(pivot_points[-2:])
    fibonacci_levels = calculate_fibonacci_retracement(high, low)
    
    # Determine the first incoming Fibonacci level based on current trend
    first_incoming_value = None
    if current_trend == 'up':
        first_incoming_value = fibonacci_levels['23.6%']
    elif current_trend == 'down':
        first_incoming_value = fibonacci_levels['76.4%']

    return first_incoming_value, current_trend, pivot_points

def scale_to_sine(timeframe):
    close_prices = np.array([c["close"] for c in candle_map[timeframe]])
    close_prices, = remove_nans_and_zeros(close_prices)
    
    # Get last close price 
    current_close = close_prices[-1]
        
    # Calculate sine wave        
    sine_wave, leadsine = talib.HT_SINE(close_prices)
        
    # Replace NaN values with 0        
    sine_wave = np.nan_to_num(sine_wave)
    #sine_wave = -sine_wave
        
    # Get the sine value for last close      
    current_sine = sine_wave[-1]
        
    # Calculate the min and max sine           
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # Calculate % distances            
    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    return dist_min, dist_max, current_sine
      
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate thresholds and averages based on min and max percentages. 
    """
    close_prices = np.array(close_prices)
    
    # Get min/max close    
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=period)
    
    # Get min/max momentum    
    min_momentum = np.nanmin(momentum)   
    max_momentum = np.nanmax(momentum)
    
    # Calculate custom percentages 
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    # Calculate thresholds       
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    # Calculate range of prices within a certain distance from the current close price
    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)

    # Filter close prices
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
        
    # Calculate avg    
    avg_mtf = np.nanmean(filtered_close)

    # Get current momentum       
    current_momentum = momentum[-1]

    # Calculate % to min/max momentum    
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               
        percent_to_max_momentum = ((current_momentum - min_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
 
    # Calculate combined percentages              
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    # Combined momentum signal     
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

# Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
def analyze_timeframes():
    for timeframe in timeframes:
        candles = candle_map[timeframe]
        close_prices = np.array([c["close"] for c in candles])
        
        # Calculate minimum and maximum thresholds
        min_threshold, max_threshold, _, _, _ = calculate_thresholds(close_prices)
        
        # Print VWAP
        vwap = calculate_vwap(candles)
        print(f"VWAP for {timeframe}: {vwap:.2f}")
        
        # Print EMA
        ema_50 = calculate_ema(candles, timeperiod=50)
        print(f"EMA 50 for {timeframe}: {ema_50:.2f}")
        
        # Print RSI
        rsi = calculate_rsi(candles, timeperiod=14)
        print(f"RSI 14 for {timeframe}: {rsi:.2f}")
        
        # Print MACD
        macd, macdsignal, macdhist = calculate_macd(candles)
        print(f"MACD for {timeframe}: {macd:.2f}")
        print(f"MACD Signal for {timeframe}: {macdsignal:.2f}")
        print(f"MACD Histogram for {timeframe}: {macdhist:.2f}")
        
        # Print Momentum
        momentum = calculate_momentum(candles, timeperiod=10)
        print(f"Momentum for {timeframe}: {momentum:.2f}")
        
        # Print Regression Channels
        reg_lower, reg_upper, reg_avg, current_close = calculate_regression_channels(candles, min_threshold, max_threshold)
        print(f"Regression Lower for {timeframe}: {reg_lower:.2f}")
        print(f"Regression Upper for {timeframe}: {reg_upper:.2f}")
        print(f"Regression Avg for {timeframe}: {reg_avg:.2f}")
        
        # Print Polynomial Regression Channels
        poly_reg_lower, poly_reg_upper, poly_reg_avg, current_close = calculate_polynomial_regression_channels(candles, min_threshold, max_threshold)
        print(f"Polynomial Regression Lower for {timeframe}: {poly_reg_lower:.2f}")
        print(f"Polynomial Regression Upper for {timeframe}: {poly_reg_upper:.2f}")
        print(f"Polynomial Regression Avg for {timeframe}: {poly_reg_avg:.2f}")
        
        # Print Fibonacci Retracement
        fib_high = np.max([c["high"] for c in candles])
        fib_low = np.min([c["low"] for c in candles])
        fibonacci_levels = calculate_fibonacci_retracement(fib_high, fib_low)
        print(f"Fibonacci Retracement Levels for {timeframe}: {fibonacci_levels}")
        
        # Print ZigZag Forecast
        first_incoming_value, trend, pivots = calculate_zigzag_forecast(candles)
        print(f"ZigZag Forecast First Incoming Value for {timeframe}: {first_incoming_value:.2f}")
        print(f"ZigZag Forecast Trend for {timeframe}: {trend}")
        print(f"ZigZag Pivot Points for {timeframe}: {pivots}")
        
        # Print Sine Scaling
        dist_min, dist_max, current_sine = scale_to_sine(timeframe)
        print(f"Sine Scaling Distance to Min for {timeframe}: {dist_min:.2f}%")
        print(f"Sine Scaling Distance to Max for {timeframe}: {dist_max:.2f}%")
        print(f"Sine Scaling Current Sine for {timeframe}: {current_sine:.2f}")
        
        # Calculate thresholds
        min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)
        print(f"Thresholds for {timeframe}:")
        print(f"Minimum Threshold: {min_threshold:.2f}")
        print(f"Maximum Threshold: {max_threshold:.2f}")
        print(f"Average MTF: {avg_mtf:.2f}")
        print(f"Momentum Signal: {momentum_signal:.2f}")

        # Determine which threshold is closest to the current close
        closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close_prices[-1]))
        if closest_threshold == min_threshold:
            print("The last minimum value is closest to the current close.")
        elif closest_threshold == max_threshold:
            print("The last maximum value is closest to the current close.")
        else:
            print("No threshold value found.")

        print()

analyze_timeframes()