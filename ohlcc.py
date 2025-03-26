#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
from datetime import datetime, timedelta
import pytz
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
volume_amplitude = 100000  # Global variable for volume scaling

# Get local time zone
local_tz = datetime.now().astimezone().tzinfo

# Fetch candles
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

# Populate candle_map
for timeframe in timeframes:
    candle_map[timeframe] = get_candles(symbol, timeframe)

# Helper function to remove NaNs and zeros
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Indicator calculation functions
def calculate_vwap(candles):
    close_prices, volumes = np.array([c["close"] for c in candles]), np.array([c["volume"] for c in candles])
    close_prices, volumes = remove_nans_and_zeros(close_prices, volumes)
    return np.sum(close_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else np.nan

def calculate_ema(candles, timeperiod):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    ema = talib.EMA(close_prices, timeperiod=timeperiod)
    return ema[-1] if len(ema) > 0 else np.nan

def calculate_rsi(candles, timeperiod=14):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    rsi = talib.RSI(close_prices, timeperiod=timeperiod)
    return rsi[-1] if len(rsi) > 0 else np.nan

def calculate_macd(candles, fastperiod=12, slowperiod=26, signalperiod=9):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return (macd[-1] if len(macd) > 0 else np.nan,
            macdsignal[-1] if len(macdsignal) > 0 else np.nan,
            macdhist[-1] if len(macdhist) > 0 else np.nan)

def calculate_momentum(candles, timeperiod=10):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=timeperiod)
    return momentum[-1] if len(momentum) > 0 else np.nan

def calculate_regression_channels(candles):
    if len(candles) < 50:
        return None, None, None
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    if len(x) < 2:
        return None, None, None
    slope, intercept, _, _, _ = linregress(x, close_prices)
    regression_line = intercept + slope * x
    std_dev = np.std(close_prices - regression_line)
    return (regression_line[-1] - std_dev, regression_line[-1] + std_dev, regression_line[-1])

def calculate_polynomial_regression_channels(candles, degree=2):
    if len(candles) < 50:
        return None, None, None
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    close_prices, x = remove_nans_and_zeros(close_prices, x)
    if len(x) < 2:
        return None, None, None
    coeffs = Polynomial.fit(x, close_prices, degree).convert().coef
    poly = Polynomial(coeffs)
    regression_line = poly(x)
    std_dev = np.std(close_prices - regression_line)
    return (regression_line[-1] - std_dev, regression_line[-1] + std_dev, regression_line[-1])

def calculate_fibonacci_retracement(high, low):
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "76.4%": high - 0.764 * diff,
        "100.0%": low
    }

def calculate_zigzag_forecast(candles, depth=12, deviation=5, backstep=3):
    highs, lows = np.array([c['high'] for c in candles]), np.array([c['low'] for c in candles])
    times = np.array([c['time'] for c in candles])
    highs, lows, times = remove_nans_and_zeros(highs, lows, times)
    if len(highs) < depth:
        return None, None, None, None, None
    zigzag = np.zeros_like(highs)
    last_high, last_low = 0, 0
    last_high_time, last_low_time = None, None
    trend = None
    for i in range(depth, len(highs)):
        if highs[i] == max(highs[i-depth:i+depth]) and i - last_high > backstep:
            zigzag[i] = highs[i]
            last_high = i
            last_high_time = times[i]
            trend = 'down'
        elif lows[i] == min(lows[i-depth:i+depth]) and i - last_low > backstep:
            zigzag[i] = lows[i]
            last_low = i
            last_low_time = times[i]
            trend = 'up'
    pivots = zigzag[zigzag != 0]
    pivot_times = times[zigzag != 0]
    if len(pivots) < 2:
        return None, None, None, None, None
    high, low = max(pivots[-2:]), min(pivots[-2:])
    fib = calculate_fibonacci_retracement(high, low)
    last_pivot_time = last_high_time if trend == 'down' else last_low_time
    last_pivot_price = highs[last_high] if trend == 'down' else lows[last_low]
    return fib['23.6%'] if trend == 'up' else fib['76.4%'], trend, pivots, last_pivot_time, last_pivot_price

def scale_to_sine(candles):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave, 0)
    current_sine = sine_wave[-1] if len(sine_wave) > 0 else 0
    min_sine, max_sine = np.min(sine_wave), np.max(sine_wave)
    return ((current_sine - min_sine) / (max_sine - min_sine) * 100 if max_sine != min_sine else 50,
            (max_sine - current_sine) / (max_sine - min_sine) * 100 if max_sine != min_sine else 50,
            current_sine)

def calculate_bollinger_bands(candles, timeperiod=20, nbdevup=2, nbdevdn=2):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    upper, middle, lower = talib.BBANDS(close_prices, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    return (upper[-1] if len(upper) > 0 else np.nan,
            middle[-1] if len(middle) > 0 else np.nan,
            lower[-1] if len(lower) > 0 else np.nan)

def calculate_atr(candles, timeperiod=14):
    high_prices, low_prices, close_prices = (np.array([c["high"] for c in candles]),
                                             np.array([c["low"] for c in candles]),
                                             np.array([c["close"] for c in candles]))
    high_prices, low_prices, close_prices = remove_nans_and_zeros(high_prices, low_prices, close_prices)
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=timeperiod)
    return atr[-1] if len(atr) > 0 else np.nan

def calculate_stochrsi(candles, timeperiod=14, fastk_period=5, fastd_period=3):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    stochrsi, stochrsi_d = talib.STOCHRSI(close_prices, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period)
    return (stochrsi[-1] if len(stochrsi) > 0 else np.nan,
            stochrsi_d[-1] if len(stochrsi_d) > 0 else np.nan)

def find_major_reversals(candles):
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    times = np.array([c["time"] for c in candles])
    top_idx = np.argmax(highs)
    dip_idx = np.argmin(lows)
    return highs[top_idx], times[top_idx], lows[dip_idx], times[dip_idx]

# Updated market_cycle_sinewave with FFT and volume confirmation
def market_cycle_sinewave(candles, timeframe, plot=True):
    if not candles:
        return {}
    
    top_price, top_time, dip_price, dip_time = find_major_reversals(candles)
    
    if top_time > dip_time:
        start_date_local = datetime.fromtimestamp(top_time, tz=pytz.utc).astimezone(local_tz)
        start_price = top_price
        trend = 'down'
        last_reversal = 'top'
    else:
        start_date_local = datetime.fromtimestamp(dip_time, tz=pytz.utc).astimezone(local_tz)
        start_price = dip_price
        trend = 'up'
        last_reversal = 'dip'

    post_reversal_candles = [c for c in candles if c["time"] >= (top_time if top_time > dip_time else dip_time)]
    if not post_reversal_candles:
        post_reversal_candles = candles
    times = [datetime.fromtimestamp(c["time"], tz=pytz.utc).astimezone(local_tz) for c in post_reversal_candles]
    time_diffs = [(t - start_date_local).total_seconds() / 3600 for t in times]
    period = time_diffs[-1] if time_diffs else 24
    time_labels = [t.strftime('%Y-%m-%d %H:%M:%S') for t in times]

    fib_high, fib_low = max([c["high"] for c in post_reversal_candles]), min([c["low"] for c in post_reversal_candles])
    price_amplitude = (fib_high - fib_low) / 2
    base_price = fib_low + price_amplitude

    time = np.array(time_diffs)
    degrees = (360 / period) * time if period > 0 else np.zeros_like(time)
    price = base_price + price_amplitude * np.sin(np.radians(degrees))
    volume = np.gradient(price, time) if len(time) > 1 else np.zeros_like(price)
    volume = volume_amplitude * np.abs(volume) / (max(np.abs(volume)) + 1e-10) + volume_amplitude * 0.5

    # Volume confirmation
    actual_volumes = np.array([c["volume"] for c in post_reversal_candles])
    top_idx = np.argmax(price)
    dip_idx = np.argmin(price)
    vol_at_top = actual_volumes[top_idx] if top_idx < len(actual_volumes) else 0
    vol_at_dip = actual_volumes[dip_idx] if dip_idx < len(actual_volumes) else 0
    avg_volume = np.mean(actual_volumes) if len(actual_volumes) > 0 else 0
    vol_confirmed_top = vol_at_top > avg_volume * 1.5
    vol_confirmed_dip = vol_at_dip > avg_volume * 1.5

    # FFT forecast
    close_prices = np.array([c["close"] for c in post_reversal_candles])
    if len(close_prices) > 1:
        fft_result = np.fft.fft(close_prices)
        fft_freq = np.fft.fftfreq(len(close_prices))
        positive_freqs = fft_freq > 0
        dominant_freq = fft_freq[positive_freqs][np.argmax(np.abs(fft_result)[positive_freqs])] if any(positive_freqs) else 0
        fft_result_filtered = fft_result.copy()
        fft_result_filtered[np.abs(fft_freq) > dominant_freq * 2] = 0
        forecast_price = np.real(np.fft.ifft(fft_result_filtered))
        fft_forecast = forecast_price[-1] if len(forecast_price) > 0 else close_prices[-1]
    else:
        fft_forecast = close_prices[0] if len(close_prices) > 0 else np.nan

    # Cycle and reversal
    current_cycle = "UP" if last_reversal == 'dip' else "DOWN"
    incoming_reversal = "TOP" if current_cycle == "UP" else "DIP"
    incoming_price = fib_high if incoming_reversal == "TOP" else fib_low

    # Output
    print(f"\n=== {timeframe} Market Cycle ===")
    print(f"Last Major Reversal: {last_reversal.capitalize()} at {start_price:.2f} ({start_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
    print(f"Current Cycle: {current_cycle}")
    print(f"Incoming Reversal: {incoming_reversal} at approx. {incoming_price:.2f}")
    print(f"Volume Confirmation at Last Top: {'Yes' if vol_confirmed_top else 'No'} (Vol: {vol_at_top:.2f}, Avg: {avg_volume:.2f})")
    print(f"Volume Confirmation at Last Dip: {'Yes' if vol_confirmed_dip else 'No'} (Vol: {vol_at_dip:.2f}, Avg: {avg_volume:.2f})")
    print(f"FFT Forecast Price for Incoming Reversal: {fft_forecast:.2f}")

    # Plotting
    if plot and len(time) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(time, price, label='Sine Price ($)', color='blue')
        ax1.axhline(calculate_vwap(candles), color='purple', linestyle='--', label='VWAP')
        ax1.plot(time[top_idx] if top_idx < len(time) else time[-1], price[top_idx], 'ro', label=f"Top")
        ax1.plot(time[dip_idx] if dip_idx < len(time) else time[-1], price[dip_idx], 'bo', label=f"Dip")
        ax1.plot(0, start_price, 'ko', label=f"Last Reversal")
        ax1.set_title(f"{timeframe} Price Cycle")
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time, volume, label='Sine Volume', color='orange')
        ax2.plot(time, actual_volumes, label='Actual Volume', color='green', alpha=0.5)
        ax2.set_title('Volume Intensity')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return {
        'current_cycle': current_cycle,
        'incoming_reversal': incoming_reversal,
        'incoming_price': incoming_price,
        'fft_forecast': fft_forecast,
        'vol_confirmed_top': vol_confirmed_top,
        'vol_confirmed_dip': vol_confirmed_dip
    }

def analyze_timeframes():
    print(f"Current Local Datetime: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    for timeframe in timeframes:
        candles = candle_map[timeframe]
        if not candles:
            print(f"No data for {timeframe}")
            continue
        print(f"\n=== {timeframe} Analysis ===")
        top_price, top_time, dip_price, dip_time = find_major_reversals(candles)
        top_date_local = datetime.fromtimestamp(top_time, tz=pytz.utc).astimezone(local_tz)
        dip_date_local = datetime.fromtimestamp(dip_time, tz=pytz.utc).astimezone(local_tz)
        print(f"Major Top: Price = {top_price:.2f}, Time = {top_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Major Dip: Price = {dip_price:.2f}, Time = {dip_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"VWAP: {calculate_vwap(candles):.2f}")
        print(f"EMA 50: {calculate_ema(candles, 50):.2f}")
        print(f"RSI 14: {calculate_rsi(candles):.2f}")
        macd, macdsignal, macdhist = calculate_macd(candles)
        print(f"MACD: {macd:.2f}, Signal: {macdsignal:.2f}, Histogram: {macdhist:.2f}")
        print(f"Momentum: {calculate_momentum(candles):.2f}")
        reg_lower, reg_upper, reg_avg = calculate_regression_channels(candles)
        print(f"Regression Channels: Lower = {reg_lower:.2f}, Upper = {reg_upper:.2f}, Avg = {reg_avg:.2f}")
        poly_lower, poly_upper, poly_avg = calculate_polynomial_regression_channels(candles)
        print(f"Polynomial Regression: Lower = {poly_lower:.2f}, Upper = {poly_upper:.2f}, Avg = {poly_avg:.2f}")
        fib_high, fib_low = max([c["high"] for c in candles]), min([c["low"] for c in candles])
        fib_levels = calculate_fibonacci_retracement(fib_high, fib_low)
        print(f"Fibonacci Levels: {', '.join([f'{k}: {v:.2f}' for k, v in fib_levels.items()])}")
        zig_first, zig_trend, zig_pivots, _, _ = calculate_zigzag_forecast(candles)
        print(f"ZigZag Forecast: First = {zig_first:.2f}, Trend = {zig_trend}")
        dist_min, dist_max, current_sine = scale_to_sine(candles)
        print(f"Sine Scaling: Dist to Min = {dist_min:.2f}%, Dist to Max = {dist_max:.2f}%, Current = {current_sine:.2f}")
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(candles)
        print(f"Bollinger Bands: Upper = {upper_bb:.2f}, Middle = {middle_bb:.2f}, Lower = {lower_bb:.2f}")
        print(f"ATR 14: {calculate_atr(candles):.2f}")
        stochrsi_k, stochrsi_d = calculate_stochrsi(candles)
        print(f"Stochastic RSI: K = {stochrsi_k:.2f}, D = {stochrsi_d:.2f}")

        # Sinewave analysis
        market_cycle_sinewave(candles, timeframe)

if __name__ == "__main__":
    analyze_timeframes()