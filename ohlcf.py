import requests
import numpy as np
import talib  # Ensure TA-Lib is installed
from binance.client import Client as BinanceClient
import datetime
import matplotlib.pyplot as plt  # For visualization

# Define Binance client by reading API key and secret from a local file
def get_binance_client():
    with open("credentials.txt", "r") as f:   
        lines = f.readlines()
        api_key = lines[0].strip()  
        api_secret = lines[1].strip()  
    return BinanceClient(api_key, api_secret)

client = get_binance_client()

TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
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
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("price key not found in API response")

def get_close(timeframe):
    closes = []
    candles = candle_map[timeframe]
    for c in candles:
        close = c['close']
        if not np.isnan(close):
            closes.append(close)
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)
    return closes

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])
    
    avg_mtf = np.nanmean(close_prices)
    return min_threshold, max_threshold, avg_mtf

def detect_reversal_and_sma(candles):
    closes = np.array([candle['close'] for candle in candles])
    sma7 = talib.SMA(closes, timeperiod=7)
    sma12 = talib.SMA(closes, timeperiod=12)
    sma26 = talib.SMA(closes, timeperiod=26)
    sma45 = talib.SMA(closes, timeperiod=45)

    smas = {
        'SMA7': sma7[-1],
        'SMA12': sma12[-1],
        'SMA26': sma26[-1],
        'SMA45': sma45[-1]
    }

    current_close = closes[-1]
    position_messages = []
    
    for label, value in smas.items():
        if current_close < value:
            position_messages.append(f"Current close is below {label}: {value:.2f}")
        elif current_close > value:
            position_messages.append(f"Current close is above {label}: {value:.2f}")
        else:
            position_messages.append(f"Current close is equal to {label}: {value:.2f}")

    return position_messages

def calculate_enforced_support_resistance(candles):
    closes = np.array([candle['close'] for candle in candles])
    volumes = np.array([candle['volume'] for candle in candles])

    weighted_sma = talib.SMA(closes * volumes, timeperiod=20) / talib.SMA(volumes, timeperiod=20)
    support = np.nanmin(weighted_sma)
    resistance = np.nanmax(weighted_sma)
    
    return support, resistance

def scale_to_sine(thresholds, last_reversal, cycles=5):  
    min_threshold, max_threshold, avg_threshold = thresholds
    num_points = 100
    t = np.linspace(0, 2 * np.pi * cycles, num_points)

    amplitude = (max_threshold - min_threshold) / 2

    baseline = min_threshold + amplitude if last_reversal == "dip" else max_threshold - amplitude
    sine_wave = baseline + amplitude * np.sin(t) if last_reversal == "dip" else baseline - amplitude * np.sin(t)

    return sine_wave

def plot_market_trend(close_prices, min_threshold, max_threshold, support, resistance, timeframe):
    plt.figure(figsize=(10, 5))
    plt.plot(close_prices, label='Close Prices', color='blue')
    plt.axhline(y=min_threshold, label='Min Threshold (Support)', color='green', linestyle='--')
    plt.axhline(y=max_threshold, label='Max Threshold (Resistance)', color='red', linestyle='--')
    plt.axhline(y=support, label='SMA Volume Enforced Support', color='purple', linestyle='-')
    plt.axhline(y=resistance, label='SMA Volume Enforced Resistance', color='orange', linestyle='-')
    
    plt.title(f'Market Trend and Thresholds - Timeframe: {timeframe}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

for timeframe in timeframes:
    close = get_close(timeframe)
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close, period=14, minimum_percentage=2, maximum_percentage=2)

    current_close = close[-1]
    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - current_close))
    last_reversal = "dip" if "Dip" in (dip_signal := detect_reversal_and_sma(candle_map[timeframe]))[0] else "top"

    # Calculate enforced support and resistance
    support, resistance = calculate_enforced_support_resistance(candle_map[timeframe])

    # Analyzing signals
    plot_market_trend(close, min_threshold, max_threshold, support, resistance, timeframe)

    print("=" * 30)
    print(f"Timeframe: {timeframe}")
    print(f"Current Price: {current_close:.2f}")
    print("Minimum threshold:", min_threshold)
    print("Maximum threshold:", max_threshold)
    print("Average MTF:", avg_mtf)
    
    if closest_threshold == min_threshold:
        print("The last minimum value is closest to the current close:", closest_threshold)
    elif closest_threshold == max_threshold:
        print("The last maximum value is closest to the current close:", closest_threshold)

    print("Current close position relative to SMAs:")
    for message in dip_signal:
        print(message)
    
    next_reversal_target = "N/A"  # Initialize
    if closest_threshold == max_threshold:
        next_reversal_target = np.min(scale_to_sine((min_threshold, max_threshold, avg_mtf), last_reversal))
        print(f"Expected next dip price: {next_reversal_target:.2f}")
    elif closest_threshold == min_threshold:
        next_reversal_target = np.max(scale_to_sine((min_threshold, max_threshold, avg_mtf), last_reversal))
        print(f"Expected next top price: {next_reversal_target:.2f}")

    print("=" * 30)