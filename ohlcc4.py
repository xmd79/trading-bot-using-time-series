#!/usr/bin/env python3

import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
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
timeframes = ["1m", "3m", "5m"]
candle_map = {}

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
    candle_map[timeframe] = get_candles(symbol, timeframe, limit=2000)  # Increased limit for more data

# Helper function to remove NaNs and zeros
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Find major reversals with volume confirmation and bullish/bearish volume
def find_major_reversals(candles):
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    times = np.array([c["time"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    opens = np.array([c["open"] for c in candles])
    closes = np.array([c["close"] for c in candles])
    
    top_idx = np.argmax(highs)
    dip_idx = np.argmin(lows)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    vol_confirmed_top = volumes[top_idx] > avg_volume * 1.2  # Lowered threshold to 1.2x
    vol_confirmed_dip = volumes[dip_idx] > avg_volume * 1.2
    
    # Bullish vs Bearish volume
    bullish_vol = sum(v for o, c, v in zip(opens, closes, volumes) if c > o)
    bearish_vol = sum(v for o, c, v in zip(opens, closes, volumes) if c < o)
    total_vol = bullish_vol + bearish_vol
    bull_ratio = (bullish_vol / total_vol * 100) if total_vol > 0 else 50
    bear_ratio = (bearish_vol / total_vol * 100) if total_vol > 0 else 50
    
    return {
        'top': {'price': highs[top_idx], 'time': times[top_idx], 'volume_confirmed': vol_confirmed_top},
        'dip': {'price': lows[dip_idx], 'time': times[dip_idx], 'volume_confirmed': vol_confirmed_dip},
        'bull_ratio': bull_ratio,
        'bear_ratio': bear_ratio
    }

def analyze_timeframes():
    print(f"Current Local Datetime: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    mtf_reversals = []

    for timeframe in timeframes:
        candles = candle_map[timeframe]
        if not candles:
            print(f"No data for {timeframe}")
            continue
        reversals = find_major_reversals(candles)
        mtf_reversals.append((timeframe, reversals))
        print(f"\n=== {timeframe} Analysis ===")
        print(f"Bullish Volume Ratio: {reversals['bull_ratio']:.2f}%")
        print(f"Bearish Volume Ratio: {reversals['bear_ratio']:.2f}%")

    # Multi-timeframe harmonic oscillator analysis
    if mtf_reversals:
        print("\n=== Multi-Timeframe Harmonic Oscillator Analysis ===")
        confirmed_tops = [(tf, r['top']['price'], r['top']['time']) for tf, r in mtf_reversals if r['top']['volume_confirmed']]
        confirmed_dips = [(tf, r['dip']['price'], r['dip']['time']) for tf, r in mtf_reversals if r['dip']['volume_confirmed']]

        if not confirmed_tops or not confirmed_dips:
            print("Insufficient volume-confirmed reversals. Check API data or lower volume threshold.")
            return

        # ATH and ATL across all TFs
        ath_price = max([p for _, p, _ in confirmed_tops])
        ath_time = max([(t, p) for _, p, t in confirmed_tops if p == ath_price])[0]
        atl_price = min([p for _, p, _ in confirmed_dips])
        atl_time = min([(t, p) for _, p, t in confirmed_dips if p == atl_price])[0]

        # Current close from 1m
        current_close = candle_map["1m"][-1]["close"] if "1m" in candle_map and candle_map["1m"] else 0

        # Most recent reversal
        all_reversals = [(tf, p, t, 'top') for tf, p, t in confirmed_tops] + [(tf, p, t, 'dip') for tf, p, t in confirmed_dips]
        latest_reversal = max(all_reversals, key=lambda x: x[2])
        latest_tf, latest_price, latest_time, latest_type = latest_reversal
        latest_date_local = datetime.fromtimestamp(latest_time, tz=pytz.utc).astimezone(local_tz)

        # Current cycle and target
        current_cycle = "UP" if latest_type == 'dip' else "DOWN"
        tf_reversals = next(r for tf, r in mtf_reversals if tf == latest_tf)
        incoming_target = tf_reversals['top']['price'] if current_cycle == "UP" else tf_reversals['dip']['price']

        # MTF range and symmetry
        mtf_high = ath_price
        mtf_low = atl_price
        mtf_amplitude = (mtf_high - mtf_low) / 4
        mtf_middle = (mtf_high + mtf_low) / 2

        # FFT trend analysis
        reversal_times = sorted([t for _, _, t in confirmed_tops + confirmed_dips])
        if len(reversal_times) > 1:
            time_diffs = np.diff(reversal_times) / 3600
            fft_result = np.fft.fft(time_diffs)
            fft_freq = np.fft.fftfreq(len(time_diffs))
            spectral_power = np.abs(fft_result)
            positive_freqs = fft_freq > 0
            dominant_freq = fft_freq[positive_freqs][np.argmax(spectral_power[positive_freqs])] if any(positive_freqs) else 0
            fft_trend = "UP" if dominant_freq > 0 else "DOWN"
            avg_period_hours = 1 / dominant_freq if dominant_freq != 0 else np.mean(time_diffs)
        else:
            fft_trend = "NEUTRAL"
            avg_period_hours = 24

        # Cycle strength and MTF dip confirmation
        tf_weights = {"5m": 0.5, "3m": 0.3, "1m": 0.2}
        dip_confirmed_tfs = [tf for tf, r in mtf_reversals if r['dip']['time'] == latest_time and r['dip']['volume_confirmed']]
        cycle_strength = sum(tf_weights[tf] for tf in dip_confirmed_tfs if latest_type == 'dip') if dip_confirmed_tfs else 0
        mtf_dip_confirmed = len(dip_confirmed_tfs) == 3 and latest_type == 'dip'

        # 1m volume confirmation
        one_min_reversals = next(r for tf, r in mtf_reversals if tf == "1m")
        bull_ratio_1m = one_min_reversals['bull_ratio']
        bear_ratio_1m = one_min_reversals['bear_ratio']
        dip_volume_confirmed = bull_ratio_1m > bear_ratio_1m and latest_type == 'dip'

        # MTF bullish/bearish volume ratio
        mtf_bull_vol = sum(r['bull_ratio'] * tf_weights[tf] for tf, r in mtf_reversals)
        mtf_bear_vol = sum(r['bear_ratio'] * tf_weights[tf] for tf, r in mtf_reversals)
        mtf_total_vol = mtf_bull_vol + mtf_bear_vol
        mtf_bull_ratio = (mtf_bull_vol / mtf_total_vol * 100) if mtf_total_vol > 0 else 50
        mtf_bear_ratio = (mtf_bear_vol / mtf_total_vol * 100) if mtf_total_vol > 0 else 50

        # Signal pattern trigger
        signal = "BUY" if (mtf_dip_confirmed and dip_volume_confirmed and fft_trend == "UP" and cycle_strength > 0.5) else \
                "SELL" if (latest_type == 'top' and fft_trend == "DOWN" and mtf_bear_ratio > mtf_bull_ratio) else "HOLD"

        # Harmonic wave
        time_since_start = (datetime.now(pytz.utc).astimezone(local_tz) - latest_date_local).total_seconds() / 3600
        forecast_hours = 24
        time_future = np.linspace(0, forecast_hours, 100)
        degrees = (360 / avg_period_hours) * time_future
        mtf_osc = mtf_middle + mtf_amplitude * np.sin(np.radians(degrees))
        
        # Support and resistance
        support_offset = (mtf_high - mtf_low) / 4
        mtf_support = np.full_like(time_future, mtf_low - support_offset)
        mtf_top = np.full_like(time_future, mtf_high + support_offset)
        mtf_middle_wave = np.full_like(time_future, mtf_middle)

        # Print statements
        print(f"Most Recent Major Reversal: {latest_type.capitalize()} at {latest_price:.2f} on {latest_tf} ({latest_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
        print(f"Current Cycle: {current_cycle}")
        print(f"Incoming Target: {('Top' if current_cycle == 'UP' else 'Dip')} at {incoming_target:.2f} (from {latest_tf})")
        print(f"Cycle Strength (5m/3m/1m): {cycle_strength:.2f}")
        print(f"MTF Dip Confirmed: {mtf_dip_confirmed}")
        print(f"1m Volume - Bullish: {bull_ratio_1m:.2f}%, Bearish: {bear_ratio_1m:.2f}%")
        print(f"MTF Volume - Bullish: {mtf_bull_ratio:.2f}%, Bearish: {mtf_bear_ratio:.2f}%")
        print(f"FFT Trend: {fft_trend}")
        print(f"Average Cycle Period: {avg_period_hours:.2f} hours")
        print(f"MTF Price Range: Min = {mtf_low - support_offset:.2f}, Middle = {mtf_middle:.2f}, Max = {mtf_high + support_offset:.2f}")
        print(f"Signal Pattern Trigger: {signal}")
        print(f"Confirmed Tops: {len(confirmed_tops)} across TFs")
        print(f"Confirmed Dips: {len(confirmed_dips)} across TFs")
        print(f"Current Close (1m): {current_close:.2f}")

        # Plot MTF harmonic oscillator
        plt.figure(figsize=(12, 6))
        plt.plot(time_future, mtf_osc, label='MTF Harmonic Oscillator', color='blue', linewidth=2)
        plt.plot(time_future, mtf_middle_wave, label=f'Middle Wave ({mtf_middle:.2f})', color='gray', linestyle='--')
        plt.plot(time_future, mtf_support, label=f'Support ({mtf_low - support_offset:.2f})', color='green', linestyle='--')
        plt.plot(time_future, mtf_top, label=f'Top ({mtf_high + support_offset:.2f})', color='red', linestyle='--')
        plt.axvline(time_since_start, color='black', linestyle='--', label='Current Time')
        plt.scatter(time_since_start, current_close, color='purple', label=f'Current Close: {current_close:.2f}', zorder=5)
        plt.title("Multi-Timeframe Harmonic Oscillator")
        plt.xlabel("Time (hours into forecast)")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    analyze_timeframes()
