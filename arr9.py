import numpy as np
import talib
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.fft import fft, ifft

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Define a function to get candles
def get_candles(symbol, timeframe, limit=1000):
    try:
        klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        candles = [{
            "time": k[0] / 1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        } for k in klines]
        return candles
    except BinanceAPIException as e:
        print(f"Error fetching candles for {symbol} at {timeframe}: {e}")
        return []

# Define functions for indicators
def calculate_vwap(candles):
    close_prices = np.array([c["close"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    return np.sum(close_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else np.nan

def calculate_ema(candles, timeperiod):
    close_prices = np.array([c["close"] for c in candles])
    return talib.EMA(close_prices, timeperiod=timeperiod)

def calculate_regression_channels(candles):
    if len(candles) < 50:
        return np.nan, np.nan
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    coeffs = np.polyfit(x, close_prices, 1)
    regression_line = np.polyval(coeffs, x)
    deviation = close_prices - regression_line
    regression_upper = regression_line + np.std(deviation)
    regression_lower = regression_line - np.std(deviation)
    return regression_lower[-1], regression_upper[-1]

def calculate_fibonacci_retracement(high, low):
    diff = high - low
    return {
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "61.8%": high - 0.618 * diff
    }

def calculate_fft(candles):
    close_prices = np.array([c["close"] for c in candles])
    fft_result = fft(close_prices)
    return fft_result

def calculate_ifft(fft_result):
    return ifft(fft_result).real

# Main logic
symbol = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

for timeframe in timeframes:
    candles = get_candles(symbol, timeframe)
    if not candles:
        continue

    current_price = candles[-1]["close"]
    poc = np.max([c["high"] for c in candles])  # Simplified POC approximation
    vwap = calculate_vwap(candles)
    ema = calculate_ema(candles, timeperiod=14)[-1] if len(candles) >= 14 else np.nan
    regression_lower, regression_upper = calculate_regression_channels(candles)
    fib_levels = calculate_fibonacci_retracement(np.max([c["high"] for c in candles]), np.min([c["low"] for c in candles]))
    fft_result = calculate_fft(candles)
    ifft_result = calculate_ifft(fft_result)

    # Print details
    print(f"Timeframe: {timeframe}")
    print(f"Current Price: {current_price}")
    print(f"POC: {poc}")
    print(f"VWAP: {vwap}")
    print(f"EMA: {ema}")
    print(f"Regression Lower: {regression_lower}")
    print(f"Regression Upper: {regression_upper}")
    print(f"Fibonacci Retracement Levels: {fib_levels}")
    print(f"FFT Result (first 5 components): {fft_result[:5]}")
    print(f"IFFT Result (first 5 components): {ifft_result[:5]}")
    
    # Plot indicators
    plt.figure(figsize=(14, 7))
    plt.plot([c["time"] for c in candles], [c["close"] for c in candles], label='Price')
    plt.axhline(y=vwap, color='r', linestyle='--', label='VWAP')
    plt.fill_between([c["time"] for c in candles], regression_lower, regression_upper, color='gray', alpha=0.2, label='Regression Channel')
    plt.legend()
    plt.title(f"Price and Indicators for {timeframe}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

    print("\n" + "="*30 + "\n")
