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
from scipy import signal
import talib

# Set Decimal precision
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Load credentials
try:
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
except FileNotFoundError:
    raise Exception("credentials.txt not found.")

# Initialize Binance client
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
            print(f"Error fetching {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Timeout {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch {timeframe}.")
    return []

def find_reversals(highs, lows, times, window=5):
    high_peaks = signal.argrelextrema(highs, np.greater, order=window)[0]
    low_peaks = signal.argrelextrema(lows, np.less, order=window)[0]
    return high_peaks, low_peaks

def fit_sinusoidal_wave(times, prices, period):
    A = (np.max(prices) - np.min(prices)) / 2
    omega = 2 * np.pi / period
    offset = np.mean(prices)
    return A * np.sin(omega * (times - times[0])) + offset

def compute_technicals(df):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    technicals = {
        'RSI': talib.RSI(closes, timeperiod=14),
        'MACD': talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)[0],
        'BB_upper': talib.BBANDS(closes, timeperiod=20)[0],
        'BB_lower': talib.BBANDS(closes, timeperiod=20)[2],
        'ATR': talib.ATR(highs, lows, closes, timeperiod=14),
        'OBV': talib.OBV(closes, volumes),
        'ADX': talib.ADX(highs, lows, closes, timeperiod=14)
    }
    return technicals

def analyze_timeframe(df, timeframe, times):
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    if len(prices) < 50:
        print(f"{timeframe}: Insufficient data (<50 candles).")
        return None

    # Find reversals and major dip/top
    high_peaks, low_peaks = find_reversals(highs, lows, times)
    reversal_times = np.sort(np.concatenate([times[high_peaks], times[low_peaks]]))
    reversal_prices = np.concatenate([highs[high_peaks], lows[low_peaks]])
    reversal_types = ['High'] * len(high_peaks) + ['Low'] * len(low_peaks)
    major_top_idx = np.argmax(highs)
    major_dip_idx = np.argmin(lows)
    major_top = (times[major_top_idx], highs[major_top_idx])
    major_dip = (times[major_dip_idx], lows[major_dip_idx])

    # Sinusoidal wave over recent times
    window = min(50, len(prices))
    recent_times = times[-window:]
    recent_prices = prices[-window:]
    period = window
    sin_wave = fit_sinusoidal_wave(recent_times, recent_prices, period)
    forecast_times = np.arange(times[-1], times[-1] + 10)
    sin_forecast = fit_sinusoidal_wave(forecast_times, recent_prices, period)

    # FFT analysis
    yf = fftpack.fft(sin_wave)
    freq = fftpack.fftfreq(len(sin_wave), d=3600)
    power = np.abs(yf) ** 2
    top_indices = power.argsort()[-5:][::-1]
    yf_filtered = np.zeros_like(yf)
    yf_filtered[top_indices] = yf[top_indices]
    fft_wave = fftpack.ifft(yf_filtered).real
    fft_forecast = np.zeros_like(forecast_times, dtype=float)
    for idx in top_indices:
        A = np.abs(yf[idx]) / len(sin_wave)
        omega = 2 * np.pi * freq[idx]
        phi = np.angle(yf[idx])
        fft_forecast += 2 * A * np.cos(omega * (forecast_times - times[0]) + phi)

    # Volume confirmation
    obv = talib.OBV(prices, volumes)
    volume_trend = np.diff(obv)
    volume_stages = np.where(volume_trend > 0, 'Accumulation', 'Distribution')

    # Energy motion flow
    atr = talib.ATR(highs, lows, prices, timeperiod=14)
    macd = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    energy_flow = atr * macd

    # Technical indicators
    technicals = compute_technicals(df)

    # Print detailed data
    print(f"\n=== {timeframe} Analysis ===")
    print("Reversals:")
    for t, p, r_type in zip(reversal_times, reversal_prices, reversal_types):
        print(f"  Time: {pd.to_datetime(t, unit='s')}, Price: {p:.2f}, Type: {r_type}")
    print(f"Major Top: Time: {pd.to_datetime(major_top[0], unit='s')}, Price: {major_top[1]:.2f}")
    print(f"Major Dip: Time: {pd.to_datetime(major_dip[0], unit='s')}, Price: {major_dip[1]:.2f}")
    print("Sinusoidal Forecast (10 hours):")
    for t, p in zip(forecast_times, sin_forecast):
        print(f"  Time: {pd.to_datetime(t, unit='s')}, Price: {p:.2f}")
    print("FFT Forecast (10 hours):")
    for t, p in zip(forecast_times, fft_forecast):
        print(f"  Time: {pd.to_datetime(t, unit='s')}, Price: {p:.2f}")
    print("Volume Stages (Last 10):")
    for i in range(-10, 0):
        print(f"  Time: {df['time'].iloc[i]}, Stage: {volume_stages[i]}")
    print("Energy Flow (Last 5):")
    for i in range(-5, 0):
        print(f"  Time: {df['time'].iloc[i]}, Energy: {energy_flow[i]:.2f}")
    print("Technicals (Last 5):")
    for key in technicals:
        print(f"  {key}: {technicals[key][-5:].tolist()}")

    return {
        'df': df,
        'times': times,
        'reversal_times': reversal_times,
        'reversal_prices': reversal_prices,
        'reversal_types': reversal_types,
        'major_top': major_top,
        'major_dip': major_dip,
        'sin_wave': sin_wave,
        'sin_forecast': sin_forecast,
        'forecast_times': forecast_times,
        'fft_wave': fft_wave,
        'fft_forecast': fft_forecast,
        'volume_stages': volume_stages,
        'energy_flow': energy_flow,
        'technicals': technicals
    }

def analyze_market_data_wave():
    timeframes = [
        client.KLINE_INTERVAL_1MINUTE,
        client.KLINE_INTERVAL_5MINUTE,
        client.KLINE_INTERVAL_15MINUTE,
        client.KLINE_INTERVAL_1HOUR,
        client.KLINE_INTERVAL_4HOUR,
        client.KLINE_INTERVAL_1DAY,
        client.KLINE_INTERVAL_1WEEK
    ]
    limit = 100

    print("Fetching candlestick data...")
    candles_dict = fetch_candles_in_parallel(timeframes, TRADE_SYMBOL, limit)

    plt.figure(figsize=(14, 10 * len(timeframes)))
    results = {}

    for idx, timeframe in enumerate(timeframes):
        if not candles_dict[timeframe]:
            print(f"No data for {timeframe}.")
            continue

        candles = candles_dict[timeframe]
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        times = np.arange(len(df))

        results[timeframe] = analyze_timeframe(df, timeframe, times)
        if not results[timeframe]:
            continue

        r = results[timeframe]
        plt.subplot(len(timeframes), 2, 2 * idx + 1)
        plt.plot(df['time'], df['close'], label='Close Price', color='blue')
        for t, p, r_type in zip(r['reversal_times'], r['reversal_prices'], r['reversal_types']):
            plt.scatter(df['time'].iloc[int(t)], p, color='red' if r_type == 'High' else 'green', label='Reversals' if t == r['reversal_times'][0] else "")
        plt.scatter(df['time'].iloc[int(r['major_top'][0])], r['major_top'][1], color='purple', marker='^', s=100, label='Major Top')
        plt.scatter(df['time'].iloc[int(r['major_dip'][0])], r['major_dip'][1], color='purple', marker='v', s=100, label='Major Dip')
        window = min(50, len(df))
        plt.plot(df['time'].iloc[-window:], r['sin_wave'], 'g--', label='Sinusoidal Wave')
        forecast_timestamps = df['time'].iloc[-1] + pd.to_timedelta(r['forecast_times'] - times[-1], unit='h')
        plt.plot(forecast_timestamps, r['sin_forecast'], 'g:', label='Sin Forecast')
        plt.title(f'{timeframe} - Price and Sinusoidal Analysis')
        plt.xlabel('Time')
        plt.ylabel('Price (USDC)')
        plt.legend()
        plt.grid(True)

        plt.subplot(len(timeframes), 2, 2 * idx + 2)
        plt.plot(df['time'].iloc[-window:], r['fft_wave'], label='FFT Wave', color='purple')
        plt.plot(forecast_timestamps, r['fft_forecast'], 'purple', linestyle=':', label='FFT Forecast')
        plt.plot(df['time'], r['energy_flow'], 'orange', label='Energy Flow')
        plt.title(f'{timeframe} - FFT and Energy Flow')
        plt.xlabel('Time')
        plt.ylabel('Amplitude/Energy')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    print("Plots displayed.")

if __name__ == "__main__":
    try:
        analyze_market_data_wave()
    except Exception as e:
        print(f"Error: {e}")