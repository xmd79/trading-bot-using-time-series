import requests
import numpy as np
import talib  # Ensure TA-Lib is installed
from binance.client import Client as BinanceClient
import matplotlib.pyplot as plt  # For visualization
from scipy.fft import fft

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
    """Get the current price for the specified trading symbol."""
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
    closes.append(current_price)
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

def detect_reversal_and_sma(candles):
    """Detect reversal signals and calculate Simple Moving Averages (SMA)."""
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
    """Calculate enforced support and resistance levels based on SMA."""
    closes = np.array([candle['close'] for candle in candles])
    volumes = np.array([candle['volume'] for candle in candles])

    weighted_sma = talib.SMA(closes * volumes, timeperiod=20) / talib.SMA(volumes, timeperiod=20)
    support = np.nanmin(weighted_sma)
    resistance = np.nanmax(weighted_sma)
    
    return support, resistance

def forecast_fft(close_prices):
    """Forecast future prices using FFT and return significant frequencies."""
    n = len(close_prices)
    freq_components = fft(close_prices)
    
    # Use the positive frequencies only
    pos_freq = np.abs(freq_components[:n // 2])
    
    # Determine if frequencies are mostly positive or negative
    is_positive = np.mean(pos_freq) >= 0  # Consider average frequency
    
    # Return index and average indication
    dominant_freq_index = np.argmax(pos_freq)
    return dominant_freq_index, is_positive

def create_zigzag(min_threshold, max_threshold, num_points=100):
    """Create a zigzag pattern between min and max thresholds."""
    middle_threshold = (min_threshold + max_threshold) / 2
    zigzag = np.zeros(num_points)
    
    # Create a zigzag pattern
    for i in range(num_points):
        if i % 2 == 0:
            zigzag[i] = max_threshold
        else:
            zigzag[i] = min_threshold
            
    return zigzag, middle_threshold

def plot_market_trend(close_prices, min_threshold, max_threshold, support, resistance, zigzag, middle_threshold, timeframe, last_reversal_index_dip, last_reversal_index_top):
    """Plot the market trend along with thresholds and zigzag."""
    plt.figure(figsize=(10, 5))
    
    plt.plot(close_prices, label='Close Prices', color='blue')
    
    # Highlight key reversal points
    last_reversal_price_dip = close_prices[last_reversal_index_dip]
    plt.scatter(last_reversal_index_dip, last_reversal_price_dip, color='green', label='Reversal Dip')
    plt.text(last_reversal_index_dip, last_reversal_price_dip, f'{last_reversal_price_dip:.2f}', fontsize=9, ha='right')
    
    last_reversal_price_top = close_prices[last_reversal_index_top]
    plt.scatter(last_reversal_index_top, last_reversal_price_top, color='red', label='Reversal Top')
    plt.text(last_reversal_index_top, last_reversal_price_top, f'{last_reversal_price_top:.2f}', fontsize=9, ha='right')

    plt.axhline(y=min_threshold, label='Min Threshold (Support)', color='green', linestyle='--')
    plt.axhline(y=max_threshold, label='Max Threshold (Resistance)', color='red', linestyle='--')
    plt.axhline(y=support, label='SMA Volume Enforced Support', color='purple', linestyle='-')
    plt.axhline(y=resistance, label='SMA Volume Enforced Resistance', color='orange', linestyle='-')
    plt.axhline(y=middle_threshold, label='Middle Threshold', color='blue', linestyle='--')

    # Plot the zigzag
    plt.plot(zigzag, label='Zigzag Pattern', color='orange', linestyle='--', alpha=0.5)

    plt.title(f'Market Trend and Zigzag - Timeframe: {timeframe}')
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

    # Forecast using FFT to identify the dominant frequency
    dominant_freq, is_positive_freq = forecast_fft(close)

    if is_positive_freq:
        frequency_status = "Positive"
    else:
        frequency_status = "Negative"
        
    print(f"Dominant frequency index: {dominant_freq} - Most dominant frequency is: {frequency_status}")

    # Find best dip and top positions
    last_reversal_index_dip = np.argmin(close)
    last_reversal_index_top = np.argmax(close)

    # Create zigzag pattern
    zigzag, middle_threshold = create_zigzag(min_threshold, max_threshold, num_points=len(close))

    # Plot all components
    plot_market_trend(close, min_threshold, max_threshold, support, resistance,
                      zigzag, middle_threshold, timeframe, last_reversal_index_dip, last_reversal_index_top)

    print("=" * 30)
    print(f"Timeframe: {timeframe}")
    print(f"Current Price: {current_close:.2f}")
    print("Minimum threshold:", min_threshold)
    print("Maximum threshold:", max_threshold)
    print("Middle threshold:", middle_threshold)
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
        print(f"Expected next dip price: {min_threshold:.2f}")  # Placeholder logic for simplicity
    elif closest_threshold == min_threshold:
        print(f"Expected next top price: {max_threshold:.2f}")  # Placeholder logic for simplicity

    print("=" * 30)