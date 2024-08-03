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

def calculate_regression_channels(candles):
    if len(candles) < 50:
        print("Not enough data for regression channel calculation")
        return None, None, None, None  # Not enough data to calculate regression channels
    
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    # Remove NaNs and zeros
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    
    if len(x) < 2:  # Ensure there are at least 2 valid points for regression
        print("Not enough valid data points for regression channel calculation")
        return None, None, None, None
    
    try:
        # Perform linear regression using scipy.stats.linregress
        slope, intercept, _, _, _ = linregress(x, close_prices)
        regression_line = intercept + slope * x
        deviation = close_prices - regression_line
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
def calculate_polynomial_regression_channels(candles, degree=2):
    if len(candles) < 50:
        print("Not enough data for polynomial regression channel calculation")
        return None, None, None, None  # Not enough data to calculate polynomial regression channels
    
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    # Remove NaNs and zeros
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    
    if len(x) < 2:  # Ensure there are at least 2 valid points for regression
        print("Not enough valid data points for polynomial regression channel calculation")
        return None, None, None, None
    
    try:
        # Perform polynomial regression using numpy's Polynomial
        coeffs = Polynomial.fit(x, close_prices, degree).convert().coef
        poly = Polynomial(coeffs)
        regression_line = poly(x)
        deviation = close_prices - regression_line
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

# Output results for all timeframes
for timeframe, candles in candle_map.items():
    print(f"Timeframe: {timeframe}")
    print("------------------")
    print(f"VWAP: {calculate_vwap(candles)}")
    print(f"EMA (50): {calculate_ema(candles, timeperiod=50)}")
    print(f"EMA (100): {calculate_ema(candles, timeperiod=100)}")
    print(f"EMA (200): {calculate_ema(candles, timeperiod=200)}")
    print(f"RSI: {calculate_rsi(candles)}")
    
    macd, macdsignal, macdhist = calculate_macd(candles)
    print(f"MACD: {macd}")
    print(f"MACD Signal: {macdsignal}")
    print(f"MACD Histogram: {macdhist}")

    print(f"Momentum: {calculate_momentum(candles)}")
    
    lower, upper, mean, close = calculate_regression_channels(candles)
    print(f"Linear Regression Channel Lower: {lower}")
    print(f"Linear Regression Channel Upper: {upper}")
    print(f"Linear Regression Channel Mean: {mean}")
    print(f"Current Close: {close}")

    poly_lower, poly_upper, poly_mean, poly_close = calculate_polynomial_regression_channels(candles, degree=2)
    print(f"Polynomial Regression Channel Lower: {poly_lower}")
    print(f"Polynomial Regression Channel Upper: {poly_upper}")
    print(f"Polynomial Regression Channel Mean: {poly_mean}")
    print(f"Current Close: {poly_close}")
    
    high_price = max([c["high"] for c in candles])
    low_price = min([c["low"] for c in candles])
    fib_levels = calculate_fibonacci_retracement(high_price, low_price)
    print("Fibonacci Retracement Levels:")
    for level, price in fib_levels.items():
        print(f"  {level}: {price}")
    
    first_incoming_value, current_trend, pivot_points = calculate_zigzag_forecast(candles)
    if first_incoming_value:
        print(f"First Incoming ZigZag Fibonacci Level: {first_incoming_value}")
        print(f"Current ZigZag Cycle: {current_trend}")

    print("\n")
