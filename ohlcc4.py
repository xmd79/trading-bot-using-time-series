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
    candle_map[timeframe] = get_candles(symbol, timeframe)

# Helper function to remove NaNs and zeros
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Find major reversals with volume confirmation
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

        # Most recent reversal
        all_reversals = [(tf, p, t, 'top') for tf, p, t in confirmed_tops] + [(tf, p, t, 'dip') for tf, p, t in confirmed_dips]
        latest_reversal = max(all_reversals, key=lambda x: x[2])
        latest_tf, latest_price, latest_time, latest_type = latest_reversal
        latest_date_local = datetime.fromtimestamp(latest_time, tz=pytz.utc).astimezone(local_tz)

        # Determine current cycle and counter-reversal
        current_cycle = "UP" if latest_type == 'dip' else "DOWN"
        counter_reversal = 'top' if latest_type == 'dip' else 'dip'
        tf_reversals = next(r for tf, r in mtf_reversals if tf == latest_tf)
        incoming_target = tf_reversals['top']['price'] if current_cycle == "UP" else tf_reversals['dip']['price']

        # MTF range and symmetry
        mtf_high = ath_price
        mtf_low = atl_price
        mtf_amplitude = (mtf_high - mtf_low) / 4
        mtf_middle = (mtf_high + mtf_low) / 2

        # Granulation ratios (1:3:5) and harmonic periods
        tf_weights = {"1m": 1, "3m": 3, "5m": 5}
        reversal_times = sorted([t for _, _, t in confirmed_tops + confirmed_dips])
        if len(reversal_times) > 1:
            avg_period_hours = np.mean(np.diff(reversal_times)) / 3600
        else:
            avg_period_hours = 24

        # Harmonic ratios
        harmonic_ratios = {
            'octave': 2, '3rd': 4/3, '5th': 3/2, '7th': 7/4, '8th': 2, '9th': 9/4, '12th': 3
        }

        # Stationary harmonic wave with inner harmonics
        current_time = datetime.now(pytz.utc).astimezone(local_tz)
        time_since_start = (current_time - latest_date_local).total_seconds() / 3600
        forecast_hours = 24
        time_future = np.linspace(0, forecast_hours, 100)
        
        # Base wave and harmonics
        degrees = (360 / avg_period_hours) * time_future
        mtf_osc = mtf_middle + mtf_amplitude * np.sin(np.radians(degrees))
        for harmonic, ratio in harmonic_ratios.items():
            mtf_osc += (mtf_amplitude / ratio) * np.sin(np.radians(degrees * ratio))

        # Quadrant mapping
        phase = np.mod(degrees, 360)
        quadrants = np.where(phase < 90, "Q2",
                            np.where(phase < 180, "Q3",
                                    np.where(phase < 270, "Q4", "Q1")))
        
        # Q1 (dip) and Q4 (top) with volume confirmation
        q1_idx = np.where(quadrants == "Q1")[0][0] if "Q1" in quadrants else None
        q4_idx = np.where(quadrants == "Q4")[0][0] if "Q4" in quadrants else None
        q1_price = mtf_osc[q1_idx] if q1_idx is not None else mtf_low
        q4_price = mtf_osc[q4_idx] if q4_idx is not None else mtf_high
        
        # Adjust support/resistance with volume confirmation
        support_offset = (mtf_high - mtf_low) / 4
        mtf_support = np.full_like(time_future, q1_price - support_offset) if q1_idx is not None else np.full_like(time_future, mtf_low)
        mtf_top = np.full_like(time_future, q4_price + support_offset) if q4_idx is not None else np.full_like(time_future, mtf_high)
        mtf_middle_wave = np.full_like(time_future, mtf_middle)

        # Thresholds
        min_threshold = q1_price - support_offset if q1_idx is not None else mtf_low
        max_threshold = q4_price + support_offset if q4_idx is not None else mtf_high
        middle_threshold = mtf_middle

        # Print statements
        print(f"Most Recent Major Reversal: {latest_type.capitalize()} at {latest_price:.2f} on {latest_tf} ({latest_date_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
        print(f"Current Cycle: {current_cycle}")
        print(f"Incoming Target: {counter_reversal.capitalize()} at {incoming_target:.2f} (from {latest_tf})")
        print(f"MTF Price Range: Min (Q1 Support) = {min_threshold:.2f}, Middle = {middle_threshold:.2f}, Max (Q4 Resistance) = {max_threshold:.2f}")
        print(f"Average Cycle Period: {avg_period_hours:.2f} hours")
        print(f"Confirmed Tops: {len(confirmed_tops)} across TFs")
        print(f"Confirmed Dips: {len(confirmed_dips)} across TFs")
        print(f"Current Close (1m): {current_close:.2f}")
        print(f"Q1 (Dip) Price: {q1_price:.2f}{' [Volume Confirmed]' if latest_type == 'dip' and tf_reversals['dip']['volume_confirmed'] else ''}")
        print(f"Q4 (Top) Price: {q4_price:.2f}{' [Volume Confirmed]' if latest_type == 'top' and tf_reversals['top']['volume_confirmed'] else ''}")

        # Plot MTF harmonic oscillator
        plt.figure(figsize=(12, 6))
        plt.plot(time_future, mtf_osc, label='MTF Harmonic Oscillator', color='blue', linewidth=2)
        plt.plot(time_future, mtf_middle_wave, label=f'Middle Wave ({middle_threshold:.2f})', color='gray', linestyle='--')
        plt.plot(time_future, mtf_support, label=f'Support (Q1: {min_threshold:.2f})', color='green', linestyle='--')
        plt.plot(time_future, mtf_top, label=f'Top (Q4: {max_threshold:.2f})', color='red', linestyle='--')
        plt.axvline(time_since_start, color='black', linestyle='--', label='Current Time')
        plt.scatter(time_since_start, current_close, color='purple', label=f'Current Close: {current_close:.2f}', zorder=5)
        if q1_idx is not None:
            plt.scatter(time_future[q1_idx], q1_price, color='green', label=f'Q1 Dip: {q1_price:.2f}', zorder=5)
        if q4_idx is not None:
            plt.scatter(time_future[q4_idx], q4_price, color='red', label=f'Q4 Top: {q4_price:.2f}', zorder=5)
        plt.title("Multi-Timeframe Harmonic Oscillator with Quadrants")
        plt.xlabel("Time (hours into forecast)")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    analyze_timeframes()
