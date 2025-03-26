#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

symbol = "BTCUSDC"
timeframes = ["1m", "3m", "5m"]  # Updated to use only 1m, 3m, 5m
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
    volumes = np.array([c["volume"] for c in candles])
    top_idx = np.argmax(highs)
    dip_idx = np.argmin(lows)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    vol_confirmed_top = volumes[top_idx] > avg_volume * 1.5 if top_idx < len(volumes) else False
    vol_confirmed_dip = volumes[dip_idx] > avg_volume * 1.5 if dip_idx < len(volumes) else False
    return {
        'top': {'price': highs[top_idx], 'time': times[top_idx], 'volume_confirmed': vol_confirmed_top},
        'dip': {'price': lows[dip_idx], 'time': times[dip_idx], 'volume_confirmed': vol_confirmed_dip}
    }

def calculate_sma(candles, period=50):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    sma = talib.SMA(close_prices, timeperiod=period)
    return sma[-1] if len(sma) > 0 else np.nan

def market_cycle_sinewave(candles, timeframe):
    if not candles:
        return {}
    reversals = find_major_reversals(candles)
    return {'reversals': reversals}

def analyze_timeframes():
    print(f"Current Local Datetime: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    mtf_reversals = []

    for timeframe in timeframes:
        candles = candle_map[timeframe]
        if not candles:
            print(f"No data for {timeframe}")
            continue
        print(f"\n=== {timeframe} Analysis ===")
        reversals = find_major_reversals(candles)
        top_price, top_time = reversals['top']['price'], reversals['top']['time']
        dip_price, dip_time = reversals['dip']['price'], reversals['dip']['time']
        top_date_local = datetime.fromtimestamp(top_time, tz=pytz.utc).astimezone(local_tz)
        dip_date_local = datetime.fromtimestamp(dip_time, tz=pytz.utc).astimezone(local_tz)
        print(f"Major Top: Price = {top_price:.2f}, Time = {top_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')}, Volume Confirmed = {reversals['top']['volume_confirmed']}")
        print(f"Major Dip: Price = {dip_price:.2f}, Time = {dip_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')}, Volume Confirmed = {reversals['dip']['volume_confirmed']}")
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

        cycle_data = market_cycle_sinewave(candles, timeframe)
        if cycle_data:
            mtf_reversals.append((timeframe, cycle_data['reversals']))

    # Multi-timeframe harmonic oscillator analysis
    if mtf_reversals:
        print("\n=== Multi-Timeframe Harmonic Oscillator Analysis ===")
        confirmed_tops = [(tf, r['top']['price'], r['top']['time']) for tf, r in mtf_reversals if r['top']['volume_confirmed']]
        confirmed_dips = [(tf, r['dip']['price'], r['dip']['time']) for tf, r in mtf_reversals if r['dip']['volume_confirmed']]

        if not confirmed_tops or not confirmed_dips:
            print("Insufficient volume-confirmed reversals for MTF analysis.")
            return

        # ATH and ATL across all TFs
        ath_price = max([p for _, p, _ in confirmed_tops])
        ath_time = max([(t, p) for _, p, t in confirmed_tops if p == ath_price])[0]
        atl_price = min([p for _, p, _ in confirmed_dips])
        atl_time = min([(t, p) for _, p, t in confirmed_dips if p == atl_price])[0]

        # Current close from 1m
        current_close = candle_map["1m"][-1]["close"] if "1m" in candle_map and candle_map["1m"] else 0

        # Find the most recent major reversal (top or dip)
        all_reversals = [(tf, p, t, 'top') for tf, p, t in confirmed_tops] + [(tf, p, t, 'dip') for tf, p, t in confirmed_dips]
        latest_reversal = max(all_reversals, key=lambda x: x[2])  # Most recent by time
        latest_tf, latest_price, latest_time, latest_type = latest_reversal
        latest_date_local = datetime.fromtimestamp(latest_time, tz=pytz.utc).astimezone(local_tz)

        # Determine current cycle and incoming target based on specific TF
        current_cycle = "UP" if latest_type == 'dip' else "DOWN"
        incoming_reversal = "TOP" if current_cycle == "UP" else "DIP"
        tf_reversals = next(r for tf, r in mtf_reversals if tf == latest_tf)
        incoming_target = tf_reversals['top']['price'] if incoming_reversal == "TOP" else tf_reversals['dip']['price']

        # MTF range and symmetry
        mtf_high = ath_price
        mtf_low = atl_price
        mtf_amplitude = (mtf_high - mtf_low) / 4
        mtf_middle = (mtf_high + mtf_low) / 2

        # FFT to determine dominant period
        reversal_times = sorted([t for _, _, t in confirmed_tops + confirmed_dips])
        if len(reversal_times) > 2:
            time_diffs = np.diff(reversal_times) / 3600
            fft_result = np.fft.fft(time_diffs)
            fft_freq = np.fft.fftfreq(len(time_diffs))
            positive_freqs = fft_freq > 0
            dominant_freq = fft_freq[positive_freqs][np.argmax(np.abs(fft_result)[positive_freqs])] if any(positive_freqs) else 0
            period_hours = 1 / dominant_freq if dominant_freq != 0 else np.mean(time_diffs)
        else:
            period_hours = 24

        # Stationary harmonic wave
        current_time = datetime.now(pytz.utc).astimezone(local_tz)
        time_since_start = (current_time - latest_date_local).total_seconds() / 3600
        forecast_hours = 24
        time_future = np.linspace(0, forecast_hours, 100)
        degrees = (360 / period_hours) * time_future
        
        # Symmetrical harmonic oscillator centered at middle
        mtf_osc = mtf_middle + mtf_amplitude * np.sin(np.radians(degrees))
        mtf_middle_wave = np.full_like(time_future, mtf_middle)
        osc_min = mtf_middle - mtf_amplitude
        osc_max = mtf_middle + mtf_amplitude
        support_offset = (mtf_high - mtf_low) / 4
        mtf_support = np.full_like(time_future, osc_min - support_offset)
        mtf_top = np.full_like(time_future, osc_max + support_offset)

        # Thresholds
        middle_threshold = mtf_middle
        min_threshold = osc_min - support_offset
        max_threshold = osc_max + support_offset

        # Print statements
        print(f"Most Recent Major Reversal: {latest_type.capitalize()} at {latest_price:.2f} on {latest_tf} ({latest_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
        print(f"Current Cycle: {current_cycle}")
        print(f"Incoming Target: {incoming_reversal} at {incoming_target:.2f} (from {latest_tf})")
        print(f"MTF Price Range: Min = {min_threshold:.2f}, Middle = {middle_threshold:.2f}, Max = {max_threshold:.2f}")
        print(f"Average Cycle Period (FFT): {period_hours:.2f} hours")
        print(f"Confirmed Tops: {len(confirmed_tops)} across TFs")
        print(f"Confirmed Dips: {len(confirmed_dips)} across TFs")
        print(f"Current Close (1m): {current_close:.2f}")

        # Plot MTF harmonic oscillator
        plt.figure(figsize=(12, 6))
        plt.plot(time_future, mtf_osc, label='MTF Harmonic Oscillator', color='blue', linewidth=2)
        plt.plot(time_future, mtf_middle_wave, label=f'Middle Wave ({middle_threshold:.2f})', color='gray', linestyle='--')
        plt.plot(time_future, mtf_support, label=f'Support (Min: {min_threshold:.2f})', color='green', linestyle='--')
        plt.plot(time_future, mtf_top, label=f'Top (Max: {max_threshold:.2f})', color='red', linestyle='--')
        plt.axvline(time_since_start, color='black', linestyle='--', label='Current Time')
        plt.scatter(time_since_start, current_close, color='purple', label=f'Current Close: {current_close:.2f}', zorder=5)
        plt.title("Multi-Timeframe Symmetrical Harmonic Oscillator")
        plt.xlabel("Time (hours into forecast)")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    analyze_timeframes()