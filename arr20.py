#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.fft import fft, ifft
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
        return None, None, None  # Not enough data to calculate regression channels
    
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    # Remove NaNs and zeros
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    
    if len(x) < 2:  # Ensure there are at least 2 valid points for regression
        print("Not enough valid data points for regression channel calculation")
        return None, None, None
    
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
        
        return regression_lower_value, regression_upper_value, current_close_value

    except Exception as e:
        print(f"Error calculating regression channels: {e}")
        return None, None, None

# New function for polynomial regression channels
def calculate_polynomial_regression_channels(candles, degree=2):
    if len(candles) < 50:
        print("Not enough data for polynomial regression channel calculation")
        return None, None, None  # Not enough data to calculate polynomial regression channels
    
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    
    # Remove NaNs and zeros
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    
    if len(x) < 2:  # Ensure there are at least 2 valid points for regression
        print("Not enough valid data points for polynomial regression channel calculation")
        return None, None, None
    
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
        
        return regression_lower_value, regression_upper_value, current_close_value

    except Exception as e:
        print(f"Error calculating polynomial regression channels: {e}")
        return None, None, None

def calculate_fibonacci_retracement(high, low):
    diff = high - low
    return {
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "61.8%": high - 0.618 * diff
    }

def calculate_fft(candles):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    return fft(close_prices)

def calculate_ifft(fft_result):
    return ifft(fft_result).real

def calculate_all_talib_indicators(candles):
    results = {}
    close_prices = np.array([c["close"] for c in candles])
    high_prices = np.array([c["high"] for c in candles])
    low_prices = np.array([c["low"] for c in candles])
    open_prices = np.array([c["open"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    close_prices, high_prices, low_prices, open_prices, volumes = remove_nans_and_zeros(
        close_prices, high_prices, low_prices, open_prices, volumes
    )

    # Overlap Studies
    results['HT_TRENDLINE'] = talib.HT_TRENDLINE(close_prices)
    results['SAR'] = talib.SAR(high_prices, low_prices)
    results['SMA'] = talib.SMA(close_prices)
    results['T3'] = talib.T3(close_prices)
    results['TRIMA'] = talib.TRIMA(close_prices)
    results['WMA'] = talib.WMA(close_prices)

    # Momentum Indicators
    results['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
    results['ADXR'] = talib.ADXR(high_prices, low_prices, close_prices)
    results['APO'] = talib.APO(close_prices)
    results['CCI'] = talib.CCI(high_prices, low_prices, close_prices)
    results['CMO'] = talib.CMO(close_prices)
    results['DX'] = talib.DX(high_prices, low_prices, close_prices)
    macd, macdsignal, macdhist = talib.MACD(close_prices)
    results['MACD'] = macd
    results['MACD_SIGNAL'] = macdsignal
    results['MACD_HIST'] = macdhist
    results['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volumes)
    results['MINUS_DI'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
    results['MINUS_DM'] = talib.MINUS_DM(high_prices, low_prices)
    results['MOM'] = talib.MOM(close_prices)
    results['PLUS_DI'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
    results['PLUS_DM'] = talib.PLUS_DM(high_prices, low_prices)
    results['PPO'] = talib.PPO(close_prices)
    results['ROC'] = talib.ROC(close_prices)

    # Convert results to NaN and zero-free
    for key in results:
        if isinstance(results[key], (np.ndarray, list)):
            results[key] = np.nan_to_num(results[key])
            results[key] = results[key][results[key] != 0]
        else:
            results[key] = np.nan

    return results

# Calculate indicators for all timeframes
results = {}
for timeframe in timeframes:
    candles = candle_map[timeframe]
    vwap = calculate_vwap(candles)
    ema = calculate_ema(candles, 21)
    rsi = calculate_rsi(candles)
    macd, macdsignal, macdhist = calculate_macd(candles)
    momentum = calculate_momentum(candles)
    regression_lower, regression_upper, current_close = calculate_regression_channels(candles)
    poly_regression_lower, poly_regression_upper, poly_current_close = calculate_polynomial_regression_channels(candles)
    fft_result = calculate_fft(candles)
    ifft_result = calculate_ifft(fft_result)
    fib_retracement = calculate_fibonacci_retracement(max(c["high"] for c in candles), min(c["low"] for c in candles))
    talib_indicators = calculate_all_talib_indicators(candles)
    
    results[timeframe] = {
        "VWAP": vwap,
        "EMA": ema,
        "RSI": rsi,
        "MACD": macd,
        "MACD_SIGNAL": macdsignal,
        "MACD_HIST": macdhist,
        "Momentum": momentum,
        "Regression Channels": {
            "Lower": regression_lower,
            "Upper": regression_upper,
            "Current Close": current_close
        },
        "Polynomial Regression Channels": {
            "Lower": poly_regression_lower,
            "Upper": poly_regression_upper,
            "Current Close": poly_current_close
        },
        "FFT": fft_result,
        "IFFT": ifft_result,
        "Fibonacci Retracement": fib_retracement,
        "TA-Lib Indicators": {k: v[-1] if isinstance(v, (np.ndarray, list)) and len(v) > 0 and not np.isnan(v[-1]) and v[-1] != 0 else None for k, v in talib_indicators.items()}
    }

# Print results with separation between timeframes
for timeframe, indicators in results.items():
    print(f"Timeframe: {timeframe}")
    for indicator_name, value in indicators.items():
        if isinstance(value, (np.ndarray, list, tuple, dict)):
            if isinstance(value, dict):
                for sub_indicator, sub_value in value.items():
                    print(f"  {indicator_name} - {sub_indicator}: {sub_value}")
            else:
                if isinstance(value, tuple):
                    value = [v[-1] if isinstance(v, (np.ndarray, list)) and len(v) > 0 and not np.isnan(v[-1]) and v[-1] != 0 else None for v in value]
                else:
                    value = value[-1] if len(value) > 0 and not np.isnan(value[-1]) and value[-1] != 0 else None
                print(f"  {indicator_name}: {value}")
        else:
            print(f"  {indicator_name}: {value}")
    print()  # Print an empty line for separation between timeframes
