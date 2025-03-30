import requests
import numpy as np
import talib  # Ensure TA-Lib is installed
from binance.client import Client as BinanceClient
import matplotlib.pyplot as plt  # For visualization
from scipy.fft import fft, ifft

def get_binance_client():
    """Instantiate Binance client using API credentials."""
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    return BinanceClient(api_key, api_secret)

# Initialize the Binance client
client = get_binance_client()

TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    """Fetch candlestick data from Binance for given timeframes."""
    candles = []
    for timeframe in timeframes:
        limit = 1000
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
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

candles = get_candles(TRADE_SYMBOL, timeframes)
candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]
    candle_map.setdefault(timeframe, []).append(candle)

def get_price(symbol):
    """Get the current price of the specified trading symbol."""
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("Price key not found in API response")

def get_close(timeframe):
    """Fetch closing prices for a given timeframe."""
    closes = []
    candles = candle_map[timeframe]
    for c in candles:
        close = c['close']
        if not np.isnan(close):
            closes.append(close)
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)  # Append current price
    return closes

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    """Calculate min and max thresholds for the closing prices."""
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    avg_mtf = np.nanmean(close_prices)
    return min_threshold, max_threshold, avg_mtf

def calculate_enforced_support_resistance(candles):
    """Calculate enforced support and resistance levels based on weighted SMA."""
    closes = np.array([candle['close'] for candle in candles])
    volumes = np.array([candle['volume'] for candle in candles])

    weighted_sma = talib.SMA(closes * volumes, timeperiod=20) / talib.SMA(volumes, timeperiod=20)
    support = np.nanmin(weighted_sma)
    resistance = np.nanmax(weighted_sma)

    return support, resistance

def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in candle_map:
        buy_volume[timeframe] = 0
        sell_volume[timeframe] = 0
        for candle in candle_map[timeframe]:
            if candle["close"] > candle["open"]:  # Bullish candle
                buy_volume[timeframe] += candle["volume"]
            elif candle["close"] < candle["open"]:  # Bearish candle
                sell_volume[timeframe] += candle["volume"]
    return buy_volume, sell_volume

def calculate_volume_ratio(buy_volume, sell_volume):
    volume_ratio = {}
    for timeframe in buy_volume.keys():
        total_volume = buy_volume[timeframe] + sell_volume[timeframe]
        if total_volume > 0:
            ratio = (buy_volume[timeframe] / total_volume) * 100
            volume_ratio[timeframe] = {
                "buy_ratio": ratio,
                "sell_ratio": 100 - ratio,
                "status": "Bullish" if ratio > 50 else "Bearish" if ratio < 50 else "Neutral"
            }
        else:
            volume_ratio[timeframe] = {
                "buy_ratio": 0,
                "sell_ratio": 0,
                "status": "No Activity"
            }
    return volume_ratio

def forecast_fft(close_prices):
    """Perform FFT and return significant frequencies along with the dominant index value."""
    n = len(close_prices)
    freq_components = fft(close_prices)
    pos_freq = np.abs(freq_components[:n // 2])

    total_power = np.sum(pos_freq)

    dominant_freq_index = np.argmax(pos_freq)

    positive_ratio = pos_freq[dominant_freq_index] / total_power * 100
    negative_ratio = (total_power - pos_freq[dominant_freq_index]) / total_power * 100

    return {
        "dominant_index": dominant_freq_index,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio
    }, pos_freq[dominant_freq_index]

def inverse_fft(frequencies, n):
    """Convert frequencies back into price using IFFT."""
    if isinstance(frequencies, int):
        frequencies = np.array([frequencies] * n)
    elif len(frequencies) < n:
        pad_length = n - len(frequencies)
        frequencies = np.pad(frequencies, (0, pad_length), 'constant')
    price_forecast = ifft(np.fft.ifftshift(frequencies)).real
    return price_forecast

def plot_market_trend(close_prices, min_threshold, max_threshold, support, resistance, timeframe, last_reversal_index_dip, last_reversal_index_top, price_forecast_target, current_cycle_direction):
    """Plot the market trend along with thresholds."""
    plt.figure(figsize=(10, 5))
    plt.plot(close_prices, label=f"Close Prices - {timeframe}")
    plt.axhline(y=min_threshold, color='green', linestyle='--', label='Min Threshold')
    plt.axhline(y=max_threshold, color='red', linestyle='--', label='Max Threshold')
    plt.axhline(y=support, color='blue', linestyle='--', label='Support')
    plt.axhline(y=resistance, color='blue', linestyle='--', label='Resistance')
    plt.scatter(last_reversal_index_dip, close_prices[last_reversal_index_dip], marker='o', color='green', label='Last Dip Reversal')
    plt.scatter(last_reversal_index_top, close_prices[last_reversal_index_top], marker='o', color='red', label='Last Top Reversal')
    plt.scatter(len(close_prices)-1, price_forecast_target, marker='x', color='black', label='Forecast Target')
    plt.title(f'Market Trend for {timeframe}')
    plt.legend()
    plt.show()

# Main loop for analysis
overall_forecasts = []
overall_volumes = []
summary_data = []

# Print current price
current_prices = {}

for timeframe in timeframes:
    close = get_close(timeframe)

    # Store the current price for printing later
    current_prices[timeframe] = close[-1]

    # Calculate last reversal indices
    last_reversal_index_dip = np.argmin(close)
    last_reversal_index_top = np.argmax(close)

    last_reversal_price_dip = close[last_reversal_index_dip]
    last_reversal_price_top = close[last_reversal_index_top]

    # Calculate volume for each timeframe
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)

    # Calculate total volume
    total_volume = buy_volume[timeframe] + sell_volume[timeframe]

    # Calculate percentage ratios
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

    # Get volume info
    bullish_volume = buy_volume[timeframe]
    bearish_volume = sell_volume[timeframe]

    # Avoid division by zero
    total_timeframe_volume = bullish_volume + bearish_volume
    if total_timeframe_volume > 0:
        bullish_ratio_percentage = (bullish_volume / total_timeframe_volume) * 100
        bearish_ratio_percentage = (bearish_volume / total_timeframe_volume) * 100
    else:
        bullish_ratio_percentage = 0.0
        bearish_ratio_percentage = 0.0

    # Calculate thresholds
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close)

    # Calculate support and resistance
    support, resistance = calculate_enforced_support_resistance(candle_map[timeframe])

    # FFT analysis
    fft_analysis, forecast_fft_value = forecast_fft(close)

    # Calculate the inverse FFT to derive prices from the FFT frequencies
    forecasted_prices_from_fft = inverse_fft(np.array([forecast_fft_value]), len(close))[:10]  # Use the dominant frequency for inverse FFT
    price_forecast_target = forecasted_prices_from_fft[0]  # Get the first price as the forecast target

    # Determine current cycle direction based on proximity to last reversals
    current_close = close[-1]
    distance_to_dip = abs(current_close - last_reversal_price_dip)
    distance_to_top = abs(current_close - last_reversal_price_top)

    if distance_to_dip < distance_to_top:  # Closer to last dip
        current_cycle_direction = "UP"
    elif distance_to_top < distance_to_dip:  # Closer to last top
        current_cycle_direction = "DOWN"
    else:
        current_cycle_direction = "NEUTRAL"

    # Print last major reversal found
    if current_cycle_direction == "UP":
        print(f"\nTimeframe: {timeframe} | Last Major Reversal: DIP at {last_reversal_price_dip:.2f}, Cycle Direction: UP")
    elif current_cycle_direction == "DOWN":
        print(f"\nTimeframe: {timeframe} | Last Major Reversal: TOP at {last_reversal_price_top:.2f}, Cycle Direction: DOWN")

    # Adjust the FFT target price according to the cycle direction
    if current_cycle_direction == "UP":
        price_forecast_target = current_close + (max_threshold - current_close) * 0.5  # Push target above current close
    elif current_cycle_direction == "DOWN":
        price_forecast_target = current_close - (current_close - min_threshold) * 0.5  # Pull target below current close

    # Determine overall volume trend (Bullish or Bearish)
    overall_trend = "Bullish" if bullish_volume >= bearish_volume else "Bearish"

    # Store data for summary
    summary_data.append({
        "Timeframe": timeframe,
        "Last Reversal Price": last_reversal_price_dip if current_cycle_direction == "UP" else last_reversal_price_top,
        "Last Reversal Type": "DIP" if current_cycle_direction == "UP" else "TOP",
        "Cycle Direction": current_cycle_direction,  # Added cycle direction information
        "FFT Forecast Price Target": price_forecast_target,
        "Min Threshold": min_threshold,
        "Max Threshold": max_threshold,
        "Support": support,
        "Resistance": resistance,
        "Bullish Volume": bullish_volume,
        "Bearish Volume": bearish_volume,
        "Total Volume": total_volume,
        "Bullish Ratio %": bullish_ratio_percentage,
        "Bearish Ratio %": bearish_ratio_percentage,
        "Volume Trend": overall_trend,
    })

    # Print detailed information for each timeframe
    print("=" * 30)
    print(f"Timeframe: {timeframe} | Cycle Direction: {current_cycle_direction}")
    print(f"FFT Forecast Price Target: {price_forecast_target:.2f}")
    print(f"Min Threshold: {min_threshold:.2f} | Max Threshold: {max_threshold:.2f} | Support: {support:.2f} | Resistance: {resistance:.2f}")
    print(f"Bullish Volume: {bullish_volume:.2f} | Bearish Volume: {bearish_volume:.2f} | Total Volume: {total_volume:.2f}")
    print(f"Bullish Volume Ratio: {bullish_ratio_percentage:.2f}% | Bearish Volume Ratio: {bearish_ratio_percentage:.2f}%")
    print(f"Overall Volume Trend: {overall_trend}")

# Create Summary Table
print("\nSummary of All Timeframes:")
print("=" * 90)
print(f"{'Timeframe':<10} {'Last Reversal':<15} {'Reversal Price':<15} {'Cycle Direction':<15} {'FFT Target':<15} {'Min Threshold':<15} {'Max Threshold':<15} {'Support':<15} {'Resistance':<15} {'Bullish Volume':<15} {'Bearish Volume':<15} {'Total Volume':<15} {'Volume Trend':<15}")
print("=" * 90)
for data in summary_data:
    print(f"{data['Timeframe']:<10} {data['Last Reversal Type']:<15} {data['Last Reversal Price']:<15.2f} {data['Cycle Direction']:<15} {data['FFT Forecast Price Target']:<15.2f} {data['Min Threshold']:<15.2f} {data['Max Threshold']:<15.2f} {data['Support']:<15.2f} {data['Resistance']:<15.2f} {data['Bullish Volume']:<15.2f} {data['Bearish Volume']:<15.2f} {data['Total Volume']:<15.2f} {data['Volume Trend']:<15}")

# MTF Analysis - averaging across timeframes
avg_bullish_volume = np.mean([data["Bullish Volume"] for data in summary_data])
avg_bearish_volume = np.mean([data["Bearish Volume"] for data in summary_data])

# Adjust the overall volume calculation
total_mtf_volume = avg_bullish_volume + avg_bearish_volume

if total_mtf_volume > 0:
    avg_bullish_volume_scaled = (avg_bullish_volume / total_mtf_volume) * 100
    avg_bearish_volume_scaled = (avg_bearish_volume / total_mtf_volume) * 100
else:
    avg_bullish_volume_scaled = 0.0
    avg_bearish_volume_scaled = 0.0

print("\nOverall Multi-Timeframe Analysis:")
print("=" * 60)
print(f"Average Bullish Volume: {avg_bullish_volume_scaled:.2f}%")
print(f"Average Bearish Volume: {avg_bearish_volume_scaled:.2f}%")
print(f"Total Volume: {total_mtf_volume:.2f}")  # Raw total volume still printed for completeness

# MTF FFT Analysis
combined_close = np.concatenate([get_close(tf) for tf in timeframes])
mtf_fft_analysis, mtf_forecast_fft_value = forecast_fft(combined_close)
mtf_forecast_price_target = inverse_fft(np.array([mtf_forecast_fft_value]), len(combined_close))  # Convert MTF FFT value back to price

print(f"\nMTF Forecast FFT:\nPositive Ratio: {mtf_fft_analysis['positive_ratio']:.2f}% - Negative Ratio: {mtf_fft_analysis['negative_ratio']:.2f}%")
print(f"Significant MTF FFT Value: {mtf_forecast_fft_value:.2f} -> Forecast Price Target: {mtf_forecast_price_target[0]:.2f}")  # Print MTF target price

print("=" * 30)
