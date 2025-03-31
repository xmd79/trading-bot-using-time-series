import requests
import numpy as np
import talib  # Ensure TA-Lib is installed
from binance.client import Client as BinanceClient
import matplotlib.pyplot as plt  # For visualization
from scipy.fft import fft, ifft
import pandas as pd  # For handling datetime
import pywt  # Ensure PyWavelets is installed

def get_binance_client():
    """Instantiate Binance client using API credentials."""
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    return BinanceClient(api_key, api_secret)

# Initialize the Binance client
client = get_binance_client()

TRADE_SYMBOL = "BTCUSDT"  # Adjusted symbol for trading on Binance
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    """Fetch candlestick data from Binance for given timeframes."""
    candles = []
    for timeframe in timeframes:
        limit = 1000
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        for k in klines:
            candle = {
                "time": k[0] / 1000,  # Convert to seconds
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
    """Fetch closing prices for a given timeframe and the current price."""
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
    """Calculate dynamic incoming min and max thresholds for the closing prices."""
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100
    min_threshold = min_close - (max_close - min_close) * min_percentage_custom
    max_threshold = max_close + (max_close - min_close) * max_percentage_custom
    middle_threshold = (min_threshold + max_threshold) / 2  # Middle threshold

    return min_threshold, max_threshold, middle_threshold

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

def forecast_fft(close_prices):
    """Perform FFT and return dominant frequencies along with their respective ratios."""
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

# Main loop for analysis
summary_data = []

# Print current price
current_prices = {}

for timeframe in timeframes:
    close = get_close(timeframe)
    
    # Store the current price for printing later
    current_prices[timeframe] = close[-1]

    # Find significant dip and top using argmin and argmax
    last_reversal_index_dip = np.argmin(close)   # index of most significant DIP 
    last_reversal_index_top = np.argmax(close)    # index of most significant TOP 

    last_reversal_price_dip = close[last_reversal_index_dip]
    last_reversal_price_top = close[last_reversal_index_top]

    # Calculate volume for each timeframe
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)

    # Calculate total volume
    total_volume = buy_volume[timeframe] + sell_volume[timeframe]

    # Calculate incoming thresholds
    min_threshold, max_threshold, middle_threshold = calculate_thresholds(close)

    # Calculate support and resistance
    support, resistance = calculate_enforced_support_resistance(candle_map[timeframe])

    # FFT analysis
    fft_analysis, forecast_fft_value = forecast_fft(close)

    # Current close price
    current_close = current_prices[timeframe]

    # Calculate distances to thresholds
    distance_to_min_threshold = (current_close - min_threshold) / (max_threshold - min_threshold) * 100
    distance_to_max_threshold = (max_threshold - current_close) / (max_threshold - min_threshold) * 100

    # Compare the most recent significant dip and top to the current close
    distance_to_dip = abs(current_close - last_reversal_price_dip)
    distance_to_top = abs(current_close - last_reversal_price_top)

    # Determine which is closer
    if distance_to_dip < distance_to_top:
        last_major_reversal_price = last_reversal_price_dip
        last_major_reversal_type = "DIP"
        current_cycle_direction = "UP"
        incoming_max_threshold = max_threshold  # Forecasting the maximum threshold
    else:
        last_major_reversal_price = last_reversal_price_top
        last_major_reversal_type = "TOP"
        current_cycle_direction = "DOWN"
        incoming_max_threshold = min_threshold  # Forecasting the minimum threshold

    # Set price forecast target based on the last major reversal
    if current_cycle_direction == "UP":
        price_forecast_target = current_close + (max_threshold - current_close) * 0.5
    else:
        price_forecast_target = current_close - (current_close - min_threshold) * 0.5

    # Print the results for the current timeframe
    print(f"\nTimeframe: {timeframe}")
    print(f"Current Close: {current_close:.2f}")
    print(f"Most Significant Major Reversal: {last_major_reversal_type} at {last_major_reversal_price:.2f}, Cycle Direction: {current_cycle_direction}")
    print(f"Incoming Threshold: {incoming_max_threshold:.2f}")
    print(f"Forecast Price Target: {price_forecast_target:.2f}")
    
    # Volume ratios and threshold percentages
    bullish_ratio = (buy_volume[timeframe] / total_volume * 100) if total_volume > 0 else 0
    bearish_ratio = (sell_volume[timeframe] / total_volume * 100) if total_volume > 0 else 0

    print(f"Buy Volume: {buy_volume[timeframe]:.2f}, Sell Volume: {sell_volume[timeframe]:.2f}, Total Volume: {total_volume:.2f}")
    print(f"Bullish Ratio: {bullish_ratio:.2f}%, Bearish Ratio: {bearish_ratio:.2f}%")
    
    # Print thresholds
    print(f"Min Threshold: {min_threshold:.2f}")
    print(f"Max Threshold: {max_threshold:.2f}")
    print(f"Middle Threshold: {middle_threshold:.2f}")
    
    # Determine if current close price is above or below the middle threshold
    if current_close > middle_threshold:
        print(f"Current Close is above the middle threshold.")
    else:
        print(f"Current Close is below the middle threshold.")

    # Print distances to min and max thresholds
    print(f"Distance to Min Threshold: {distance_to_min_threshold:.2f}%")
    print(f"Distance to Max Threshold: {distance_to_max_threshold:.2f}%")

    # Store data for summary
    summary_data.append({
        "Timeframe": timeframe,
        "Last Major Reversal Price": last_major_reversal_price,
        "Last Major Reversal Type": last_major_reversal_type,
        "Cycle Direction": current_cycle_direction,
        "Incoming Threshold": incoming_max_threshold,
        "FFT Forecast Price Target": price_forecast_target,
        "Min Threshold": min_threshold,
        "Max Threshold": max_threshold,
        "Middle Threshold": middle_threshold,
        "Support": support,
        "Resistance": resistance,
        "Bullish Volume": buy_volume[timeframe],
        "Bearish Volume": sell_volume[timeframe],
        "Total Volume": total_volume,
        "Bullish Ratio %": bullish_ratio,
        "Bearish Ratio %": bearish_ratio,
        "Distance to Min Threshold %": distance_to_min_threshold,
        "Distance to Max Threshold %": distance_to_max_threshold,
    })

# Create Summary Table
print("\nSummary of All Timeframes:")
print("=" * 120)
print(f"{'Timeframe':<10} {'Last Reversal':<15} {'Reversal Price':<15} {'Cycle Direction':<15} {'Incoming Threshold':<15} {'FFT Target':<15} {'Min Threshold':<15} {'Max Threshold':<15} {'Middle Threshold':<15} {'Support':<15} {'Resistance':<15} {'Bullish Volume':<15} {'Bearish Volume':<15} {'Total Volume':<15} {'Bullish Ratio %':<15} {'Bearish Ratio %':<15} {'Dist to Min %':<15} {'Dist to Max %':<15}")
print("=" * 120)
for data in summary_data:
    print(f"{data['Timeframe']:10} {data['Last Major Reversal Type']:15} {data['Last Major Reversal Price']:15.2f} {data['Cycle Direction']:15} {data['Incoming Threshold']:15.2f} {data['FFT Forecast Price Target']:15.2f} {data['Min Threshold']:15.2f} {data['Max Threshold']:15.2f} {data['Middle Threshold']:15.2f} {data['Support']:15.2f} {data['Resistance']:15.2f} {data['Bullish Volume']:15.2f} {data['Bearish Volume']:15.2f} {data['Total Volume']:15.2f} {data['Bullish Ratio %']:15.2f} {data['Bearish Ratio %']:15.2f} {data['Distance to Min Threshold %']:15.2f} {data['Distance to Max Threshold %']:15.2f}")

plt.show()  # Display any plots