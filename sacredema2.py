import numpy as np
import talib
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

symbol = "BTCUSDC"
timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
candle_map = {}

# Define a function to get candles
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

# Fetch candles for all timeframes
for timeframe in timeframes:
    candle_map[timeframe] = get_candles(symbol, timeframe)

# Helper function to remove NaNs and zeros from arrays
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Define a function to calculate EMAs for given lengths
def calculate_emas(candles, lengths):
    close_prices = np.array([c["close"] for c in candles])
    close_prices, = remove_nans_and_zeros(close_prices)
    emas = {}
    for length in lengths:
        ema = talib.EMA(close_prices, timeperiod=length)
        emas[length] = ema[-1] if len(ema) > 0 and not np.isnan(ema[-1]) and ema[-1] != 0 else np.nan
    return emas

# Define the function to evaluate EMA patterns
def evaluate_ema_pattern(candles):
    ema_lengths = [9, 12, 21, 27, 45, 56, 100, 150, 200, 369]
    
    # Calculate EMAs
    emas = calculate_emas(candles, ema_lengths)
    
    # Get the current close price
    current_close = candles[-1]["close"]
    
    # Count how many EMAs the current close is below
    below_count = sum(1 for length in ema_lengths if current_close < emas[length])
    
    # Calculate intensity
    max_emas = len(ema_lengths)
    intensity = (below_count / max_emas) * 100

    # Determine if it's a dip or a top
    is_dip = False
    is_top = False

    if current_close < emas[9] and all(emas[ema_lengths[i]] > emas[ema_lengths[i + 1]] for i in range(len(ema_lengths) - 1)):
        is_dip = True
    elif current_close > emas[369] and all(emas[ema_lengths[i]] < emas[ema_lengths[i + 1]] for i in range(len(ema_lengths) - 1)):
        is_top = True

    return current_close, emas, is_dip, is_top, intensity

# Define function for generating a detailed report with symmetrical fear and greed analysis
def generate_report(timeframe, candles):
    print(f"\nTimeframe: {timeframe}")
    
    # EMA Pattern Evaluation
    current_close, emas, is_dip, is_top, intensity = evaluate_ema_pattern(candles)
    
    print(f"Current Close Price: {current_close:.2f}")
    print(f"{'EMA Length':<10} {'EMA Value':<20} {'Close < EMA':<15}")
    
    for length in sorted(emas.keys()):
        ema_value = emas[length]
        close_below = current_close < ema_value
        print(f"{length:<10} {ema_value:<20.2f} {close_below:<15}")
    
    # Determine the position on the fear-greed spectrum
    neutral_threshold = 50.00
    if intensity <= neutral_threshold:
        greed_percentage = 100 - intensity
        fear_percentage = intensity
    else:
        fear_percentage = 100 - intensity
        greed_percentage = intensity
    
    # Print key points for most fear and most greed
    most_fear_ema = min(emas, key=lambda x: emas[x])
    most_greed_ema = max(emas, key=lambda x: emas[x])
    most_fear_value = emas[most_fear_ema]
    most_greed_value = emas[most_greed_ema]
    
    print(f"\nMost Fear EMA Length: {most_fear_ema}, Value: {most_fear_value:.2f}")
    print(f"Most Greed EMA Length: {most_greed_ema}, Value: {most_greed_value:.2f}")
    
    # Print symmetrical fear and greed analysis
    print(f"\nFear Percentage: {fear_percentage:.2f}%")
    print(f"Greed Percentage: {greed_percentage:.2f}%")
    
    print(f"\nEMA Pattern: {'Dip' if is_dip else 'Top' if is_top else 'Neutral'}")
    print(f"Intensity: {intensity:.2f}%")
    print(f"Fear: {fear_percentage:.2f}%")
    print(f"Greed: {greed_percentage:.2f}%")

# Generate reports for all timeframes
for timeframe, candles in candle_map.items():
    generate_report(timeframe, candles)
