import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from decimal import Decimal, getcontext
import requests
import pandas as pd

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Load credentials from file (ensure credentials.txt exists)
try:
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
except FileNotFoundError:
    raise Exception("credentials.txt not found. Please create it with API key and secret.")

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})

# Utility Functions
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=100, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = []
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
        except BinanceAPIException as e:
            print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts. Skipping timeframe.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.get_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                return price
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            print(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (1 + attempt))
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    return Decimal('0.0')

def analyze_market_data_wave():
    """
    Fetches BTC/USDC market data, computes velocity of non-stationary price changes,
    transforms data into a stationary wave, and displays plots.
    """
    # Define timeframes to fetch
    timeframes = [client.KLINE_INTERVAL_1HOUR]  # Use 1-hour candles
    limit = 100  # Number of candles

    # Fetch candlestick data
    print("Fetching candlestick data...")
    candles_dict = fetch_candles_in_parallel(timeframes, TRADE_SYMBOL, limit)
    
    # Process 1-hour candles
    if not candles_dict[client.KLINE_INTERVAL_1HOUR]:
        raise Exception("No data fetched for 1-hour timeframe.")
    
    candles = candles_dict[client.KLINE_INTERVAL_1HOUR]
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Extract close prices and compute log returns
    prices = df['close'].values
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)  # Log returns: ln(P_t / P_{t-1})
    
    # Compute velocity (rate of change of log returns)
    velocity = np.diff(log_returns)  # Second difference of log prices
    time_points = np.arange(len(prices))
    velocity_time = time_points[2:]  # Adjust for differencing
    
    # Transform to stationary wave using FFT
    # Detrend the log prices to make them more stationary
    log_prices_detrended = log_prices - np.polyval(np.polyfit(time_points, log_prices, 1), time_points)
    
    # Apply FFT
    N = len(log_prices_detrended)
    yf = fftpack.fft(log_prices_detrended)
    freq = fftpack.fftfreq(N, d=3600)  # Assume 1-hour spacing (3600 seconds)
    
    # Keep top 5 frequencies for reconstruction
    power = np.abs(yf) ** 2
    top_indices = power.argsort()[-5:][::-1]  # Top 5 frequencies
    yf_filtered = np.zeros_like(yf)
    yf_filtered[top_indices] = yf[top_indices]
    
    # Inverse FFT to get stationary wave
    stationary_wave = fftpack.ifft(yf_filtered).real
    
    # Compute velocity of stationary wave (numerical derivative)
    wave_velocity = np.diff(stationary_wave)
    wave_velocity_time = time_points[1:]
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Raw Prices
    plt.subplot(2, 2, 1)
    plt.plot(df['time'], prices, label='BTC/USDC Close Price', color='blue')
    plt.title('Non-Stationary Market Data (BTC/USDC)')
    plt.xlabel('Time')
    plt.ylabel('Price (USDC)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Velocity of Market Data
    plt.subplot(2, 2, 2)
    plt.plot(df['time'][2:], velocity, label='Velocity (Δ Log Returns)', color='red')
    plt.title('Velocity of Non-Stationary Market Data')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Stationary Wave
    plt.subplot(2, 2, 3)
    plt.plot(df['time'], stationary_wave, label='Stationary Wave (FFT Reconstruction)', color='green')
    plt.title('Transformed Stationary Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: FFT Spectrum
    plt.subplot(2, 2, 4)
    plt.plot(freq[:N//2], power[:N//2], label='Power Spectrum', color='purple')
    plt.scatter(freq[top_indices], power[top_indices], color='red', label='Top Frequencies')
    plt.title('FFT Power Spectrum')
    plt.xlabel('Frequency (cycles/hour)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("Plots displayed.")

if __name__ == "__main__":
    try:
        analyze_market_data_wave()
    except Exception as e:
        print(f"Error in analysis: {e}")