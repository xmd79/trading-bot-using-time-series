#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import datetime
import matplotlib.pyplot as plt

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
    return {
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "76.4%": high - 0.764 * diff
    }

# Determine the trend for each indicator
def determine_trend(indicator_value, previous_value):
    if previous_value is None:
        return "Neutral"
    return "Up" if indicator_value > previous_value else "Down"

def print_indicator_values(timeframe, candles):
    if len(candles) == 0:
        print(f"No candles data for {timeframe}")
        return
    
    vwap = calculate_vwap(candles)
    ema_10 = calculate_ema(candles, timeperiod=10)
    rsi_14 = calculate_rsi(candles, timeperiod=14)
    macd, macdsignal, macdhist = calculate_macd(candles)
    momentum = calculate_momentum(candles)
    regression_channels = calculate_regression_channels(candles)
    poly_regression_channels = calculate_polynomial_regression_channels(candles)
    
    # Previous values for trend determination (use last values as proxies)
    previous_candles = candles[:-1] if len(candles) > 1 else []
    previous_vwap = calculate_vwap(previous_candles) if previous_candles else None
    previous_ema_10 = calculate_ema(previous_candles, timeperiod=10) if previous_candles else None
    previous_rsi_14 = calculate_rsi(previous_candles, timeperiod=14) if previous_candles else None
    previous_macd, previous_macdsignal, previous_macdhist = calculate_macd(previous_candles) if previous_candles else (None, None, None)
    previous_momentum = calculate_momentum(previous_candles) if previous_candles else None

    # Determine trends
    vwap_trend = determine_trend(vwap, previous_vwap)
    ema_10_trend = determine_trend(ema_10, previous_ema_10)
    rsi_14_trend = determine_trend(rsi_14, previous_rsi_14)
    macd_trend = determine_trend(macd, previous_macd)
    macdsignal_trend = determine_trend(macdsignal, previous_macdsignal)
    macdhist_trend = determine_trend(macdhist, previous_macdhist)
    momentum_trend = determine_trend(momentum, previous_momentum)
    
    print(f"VWAP for {timeframe}: {vwap:.2f} ({vwap_trend})")
    print(f"EMA 10 for {timeframe}: {ema_10:.2f} ({ema_10_trend})")
    print(f"RSI 14 for {timeframe}: {rsi_14:.2f} ({rsi_14_trend})")
    print(f"MACD for {timeframe}: {macd:.2f} ({macd_trend})")
    print(f"MACD Signal for {timeframe}: {macdsignal:.2f} ({macdsignal_trend})")
    print(f"MACD Histogram for {timeframe}: {macdhist:.2f} ({macdhist_trend})")
    print(f"Momentum for {timeframe}: {momentum:.2f} ({momentum_trend})")
    
    if regression_channels:
        regression_lower, regression_upper, regression_middle, current_close = regression_channels
        print(f"Regression Channels for {timeframe}:")
        print(f"Lower: {regression_lower:.2f}")
        print(f"Middle: {regression_middle:.2f}")
        print(f"Upper: {regression_upper:.2f}")
        print(f"Current Close: {current_close:.2f}")
    
    if poly_regression_channels:
        poly_regression_lower, poly_regression_upper, poly_regression_middle, current_close = poly_regression_channels
        print(f"Polynomial Regression Channels for {timeframe}:")
        print(f"Lower: {poly_regression_lower:.2f}")
        print(f"Middle: {poly_regression_middle:.2f}")
        print(f"Upper: {poly_regression_upper:.2f}")
        print(f"Current Close: {current_close:.2f}")

    high = max([c["high"] for c in candles])
    low = min([c["low"] for c in candles])
    fib_retracements = calculate_fibonacci_retracement(high, low)
    print(f"Fibonacci Retracements for {timeframe}:")
    for level, value in fib_retracements.items():
        print(f"{level}: {value:.2f}")

    # Print overall cycle status
    trends = {
        "VWAP": vwap_trend,
        "EMA 10": ema_10_trend,
        "RSI 14": rsi_14_trend,
        "MACD": macd_trend,
        "MACD Signal": macdsignal_trend,
        "MACD Histogram": macdhist_trend,
        "Momentum": momentum_trend
    }
    
    # Determine the overall cycle status
    trend_count = sum(1 for trend in trends.values() if trend == "Up")
    trend_count = min(trend_count, len(trends) - trend_count)  # Ensure balance for neutral cases
    if trend_count > len(trends) / 2:
        overall_cycle = "Up"
    else:
        overall_cycle = "Down"
    
    print(f"Overall Cycle Status for {timeframe}: {overall_cycle}")

def forecast(candles, forecast_period=24):
    if len(candles) < 50:
        print("Not enough data for forecasting")
        return
    
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    # Remove NaNs and zeros
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    
    if len(x) < 2:  # Ensure there are at least 2 valid points for regression
        print("Not enough valid data points for forecasting")
        return
    
    try:
        # Perform polynomial regression using numpy's Polynomial for forecasting
        degree = 2
        coeffs = Polynomial.fit(x, close_prices, degree).convert().coef
        poly = Polynomial(coeffs)
        future_x = np.arange(len(close_prices), len(close_prices) + forecast_period)
        forecasted_values = poly(future_x)
        
        print(f"Forecast for the next {forecast_period} hours:")
        for i, value in enumerate(forecasted_values, 1):
            print(f"Hour {i}: {value:.2f}")
        
    except Exception as e:
        print(f"Error forecasting future values: {e}")

# Print indicator values and forecast for each timeframe
for timeframe, candles in candle_map.items():
    print(f"\nTimeframe: {timeframe}")
    print_indicator_values(timeframe, candles)
    forecast(candles, forecast_period=24)
