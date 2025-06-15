import numpy as np
import requests
from binance.client import Client as BinanceClient
import matplotlib.pyplot as plt
from scipy.fft import fft
import talib

# ==== Configuration ====
API_KEY = ""      # Put your Binance API key here if you want authenticated endpoints (not used here)
API_SECRET = ""   # Put your Binance API secret here

TRADE_SYMBOL = "BTCUSDC"  # Symbol on Binance (adjust if needed)
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '6h', '8h', '12h', '1d']

# ==== Binance Client Setup ====
def get_binance_client():
    # For public endpoints, API key not mandatory. Add keys if needed for private endpoints.
    return BinanceClient(API_KEY, API_SECRET)

client = get_binance_client()

# ==== Fetch Historical Candles for all timeframes ====
def fetch_candles(symbol, timeframes, limit=1200):
    """
    Fetch candles for each timeframe and store in a dict.
    """
    candle_map = {}
    for tf in timeframes:
        raw_klines = client.get_klines(symbol=symbol, interval=tf, limit=limit)
        candles = []
        for k in raw_klines:
            candles.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6]
            })
        candle_map[tf] = candles
    return candle_map

candle_map = fetch_candles(TRADE_SYMBOL, TIMEFRAMES, limit=1200)

# ==== Utility: Get Close Prices Array ====
def get_close_prices(candles):
    return np.array([c['close'] for c in candles])

# ==== Utility: Get Volume Array ====
def get_volumes(candles):
    return np.array([c['volume'] for c in candles])

# ==== Detect Major Reversals (Lowest Low and Highest High) ====
def detect_major_reversals(closes):
    """
    Returns indices and prices of the lowest low (support candidate) and highest high (resistance candidate).
    """
    lowest_index = np.argmin(closes)
    highest_index = np.argmax(closes)
    return lowest_index, closes[lowest_index], highest_index, closes[highest_index]

# ==== Calculate Buy and Sell Volumes Based on Candle Direction ====
def calculate_buy_sell_volume(candles):
    buy_volume = 0
    sell_volume = 0
    for c in candles:
        if c['close'] > c['open']:
            buy_volume += c['volume']
        elif c['close'] < c['open']:
            sell_volume += c['volume']
    return buy_volume, sell_volume

# ==== Calculate Dynamic Thresholds for Support, Resistance, Middle ====
def calculate_thresholds(low_price, high_price):
    """
    Enforce symmetrical support (-5), resistance (+5), and middle (0) levels.
    The resistance is set 5 points above middle, support 5 points below middle.
    """
    support = low_price
    resistance = high_price
    middle = (support + resistance) / 2
    return support, resistance, middle

# ==== Normalize Wave Close Prices to Range [-5, +5] ====
def normalize_wave(closes, support, resistance):
    """
    Normalize price so support maps to -5, resistance maps to +5,
    and middle maps to 0 linearly.
    """
    closes = np.array(closes)
    middle = (support + resistance) / 2
    # Normalize prices linearly: support -> -5, resistance -> +5
    norm_wave = -5 + 10 * (closes - support) / (resistance - support)
    # Clip to exactly -5 and +5 to ensure wave touches extremes
    norm_wave = np.clip(norm_wave, -5, 5)
    return norm_wave, middle

# ==== FFT Forecasting ====
def fft_forecast(close_prices):
    n = len(close_prices)
    freq_components = fft(close_prices)
    magnitudes = np.abs(freq_components[:n//2])
    total_power = np.sum(magnitudes)
    dominant_idx = np.argmax(magnitudes)
    dominant_power = magnitudes[dominant_idx]
    positive_ratio = dominant_power / total_power * 100 if total_power > 0 else 0
    negative_ratio = 100 - positive_ratio
    return {
        'dominant_frequency_index': dominant_idx,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio
    }

# ==== Plotting function ====
def plot_wave(norm_wave, support, resistance, middle, timeframe):
    plt.figure(figsize=(14, 6))
    plt.plot(norm_wave, label="Normalized Price Wave", color='blue')
    plt.axhline(5, color='red', linestyle='--', label='Resistance (+5)')
    plt.axhline(0, color='gray', linestyle='--', label='Middle (0)')
    plt.axhline(-5, color='green', linestyle='--', label='Support (-5)')
    plt.title(f"Wave Model Normalized to Support/Resistance for {TRADE_SYMBOL} {timeframe}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== Main Analysis Loop ====
def main_analysis():
    print(f"\n--- Starting analysis for {TRADE_SYMBOL} ---")
    summary = []

    for tf in TIMEFRAMES:
        candles = candle_map[tf]
        closes = get_close_prices(candles)
        volumes = get_volumes(candles)

        # Detect major reversals for support/resistance candidates
        low_idx, low_price, high_idx, high_price = detect_major_reversals(closes)

        # Calculate support, resistance, middle threshold
        support, resistance, middle = calculate_thresholds(low_price, high_price)

        # Buy and sell volume
        buy_volume, sell_volume = calculate_buy_sell_volume(candles)
        total_volume = buy_volume + sell_volume

        # Normalize wave
        norm_wave, _ = normalize_wave(closes, support, resistance)

        # Ensure wave touches exactly -5 at lowest low and +5 at highest high
        assert abs(norm_wave[low_idx] + 5) < 1e-6, "Wave does not touch support at -5"
        assert abs(norm_wave[high_idx] - 5) < 1e-6, "Wave does not touch resistance at +5"

        # FFT Forecast
        fft_result = fft_forecast(closes)

        # Current price info
        current_price = closes[-1]

        # Distances to thresholds in percentage terms
        dist_to_support_pct = (current_price - support) / (resistance - support) * 100
        dist_to_resistance_pct = (resistance - current_price) / (resistance - support) * 100

        # Cycle direction guess by proximity to last major reversal
        dist_to_low = abs(current_price - low_price)
        dist_to_high = abs(current_price - high_price)
        if dist_to_low < dist_to_high:
            cycle_direction = "UP (from Support)"
        else:
            cycle_direction = "DOWN (from Resistance)"

        # Forecast price target (midway between current and extreme threshold)
        if cycle_direction.startswith("UP"):
            price_forecast_target = current_price + (resistance - current_price) * 0.5
        else:
            price_forecast_target = current_price - (current_price - support) * 0.5

        # Print timeframe summary
        print(f"\nTimeframe: {tf}")
        print(f"Current Price: {current_price:.4f}")
        print(f"Support (Lowest Low): {support:.4f} at index {low_idx}")
        print(f"Resistance (Highest High): {resistance:.4f} at index {high_idx}")
        print(f"Middle Threshold: {middle:.4f}")
        print(f"Buy Volume: {buy_volume:.4f}, Sell Volume: {sell_volume:.4f}, Total Volume: {total_volume:.4f}")
        print(f"Bullish Volume Ratio: {(buy_volume/total_volume*100) if total_volume>0 else 0:.2f}%")
        print(f"Bearish Volume Ratio: {(sell_volume/total_volume*100) if total_volume>0 else 0:.2f}%")
        print(f"Cycle Direction: {cycle_direction}")
        print(f"Distance to Support (%): {dist_to_support_pct:.2f}%")
        print(f"Distance to Resistance (%): {dist_to_resistance_pct:.2f}%")
        print(f"FFT Dominant Frequency Index: {fft_result['dominant_frequency_index']}")
        print(f"FFT Positive Ratio: {fft_result['positive_ratio']:.2f}%")
        print(f"FFT Negative Ratio: {fft_result['negative_ratio']:.2f}%")
        print(f"Forecast Price Target: {price_forecast_target:.4f}")

        # Plot normalized wave with support/resistance/middle lines
        plot_wave(norm_wave, support, resistance, middle, tf)

        summary.append({
            'Timeframe': tf,
            'Current Price': current_price,
            'Support': support,
            'Resistance': resistance,
            'Middle': middle,
            'Buy Volume': buy_volume,
            'Sell Volume': sell_volume,
            'Total Volume': total_volume,
            'Bullish Ratio %': (buy_volume/total_volume*100) if total_volume > 0 else 0,
            'Bearish Ratio %': (sell_volume/total_volume*100) if total_volume > 0 else 0,
            'Cycle Direction': cycle_direction,
            'Distance to Support %': dist_to_support_pct,
            'Distance to Resistance %': dist_to_resistance_pct,
            'FFT Dominant Frequency Index': fft_result['dominant_frequency_index'],
            'FFT Positive Ratio %': fft_result['positive_ratio'],
            'FFT Negative Ratio %': fft_result['negative_ratio'],
            'Price Forecast Target': price_forecast_target
        })

    # Print Summary Table
    print("\n" + "="*140)
    print(f"{'Timeframe':<10} {'Current Price':<14} {'Support':<10} {'Resistance':<12} {'Middle':<10} {'Buy Vol':<10} {'Sell Vol':<10} {'Bullish %':<10} {'Bearish %':<10} {'Cycle Dir':<18} {'FFT Dom Freq':<12} {'FFT Pos %':<10} {'Forecast Target':<15}")
    print("="*140)
    for s in summary:
        print(f"{s['Timeframe']:<10} {s['Current Price']:<14.4f} {s['Support']:<10.4f} {s['Resistance']:<12.4f} {s['Middle']:<10.4f} {s['Buy Volume']:<10.4f} {s['Sell Volume']:<10.4f} {s['Bullish Ratio %']:<10.2f} {s['Bearish Ratio %']:<10.2f} {s['Cycle Direction']:<18} {s['FFT Dominant Frequency Index']:<12} {s['FFT Positive Ratio %']:<10.2f} {s['Price Forecast Target']:<15.4f}")

if __name__ == "__main__":
    main_analysis()
