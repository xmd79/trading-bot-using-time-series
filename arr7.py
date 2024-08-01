import numpy as np
import talib
import requests
import json
from datetime import datetime
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Define a function to get the account balance in USDT
def get_account_balance():
    account = client.get_asset_balance(asset='USDC')
    return float(account['free']) if account else 0.0

# Get the USDT balance
bUSD_balance = get_account_balance()
print("USDC Spot balance:", bUSD_balance)

# Initialize variables for tracking trade state
TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Define a function to get candles
def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 1000  # default limit
        tf_value = int(timeframe[:-1])  # extract numeric value of timeframe
        if tf_value >= 4:  # check if timeframe is 4h or above
            limit = 2000  # increase limit for 4h timeframe and above
        klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
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

# Get candles and organize by timeframe
candles = get_candles(TRADE_SYMBOL, timeframes)
candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]
    candle_map.setdefault(timeframe, []).append(candle)

# Define a function to get the latest candle
def get_latest_candle(symbol, interval, start_time=None):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=1)
    candle = {
        "time": klines[0][0],
        "open": float(klines[0][1]),
        "high": float(klines[0][2]),
        "low": float(klines[0][3]),
        "close": float(klines[0][4]),
        "volume": float(klines[0][5]),
        "timeframe": interval
    }
    return candle

# Print latest candle for each timeframe
for interval in timeframes:
    latest_candle = get_latest_candle(TRADE_SYMBOL, interval)
    print(f"Latest Candle ({interval}):")
    print(f"Time: {latest_candle['time']}")
    print(f"Open: {latest_candle['open']}")
    print(f"High: {latest_candle['high']}")
    print(f"Low: {latest_candle['low']}")
    print(f"Close: {latest_candle['close']}")
    print(f"Volume: {latest_candle['volume']}")
    print(f"Timeframe: {latest_candle['timeframe']}")
    print("\n" + "="*30 + "\n")

# Get close prices for a specific timeframe
def get_close_prices(timeframe):
    closes = [c['close'] for c in candle_map.get(timeframe, []) if not np.isnan(c['close'])]
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)
    return closes

# Get price from Binance
def get_price(symbol):
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        data = response.json()
        return float(data["price"]) if "price" in data else 0.0
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return 0.0

# Calculate momentum for each timeframe
def get_momentum(timeframe):
    """Calculate momentum for a single timeframe"""
    # Get candle data               
    candles = candle_map[timeframe][-100:]  
    # Calculate momentum using talib MOM
    close_prices = np.array([c["close"] for c in candles])
    momentum = talib.MOM(close_prices, timeperiod=14)
    return momentum[-1]

# Print momentum values for each timeframe
momentum_values = {}
for timeframe in timeframes:
    momentum = get_momentum(timeframe)
    momentum_values[timeframe] = momentum
    print(f"Momentum for {timeframe}: {momentum}")

# Normalize momentum values
def normalize_momentum(momentum_values):
    """Normalize momentum values to a 0-100 scale based on observed min and max."""
    if not momentum_values:
        return {}
    
    min_momentum = min(momentum_values.values())
    max_momentum = max(momentum_values.values())
    
    normalized = {}
    for timeframe, momentum in momentum_values.items():
        # Avoid division by zero
        if max_momentum == min_momentum:
            normalized[timeframe] = 50.0  # Arbitrary value if no range exists
        else:
            normalized[timeframe] = (momentum - min_momentum) / (max_momentum - min_momentum) * 100
    return normalized

normalized_momentum = normalize_momentum(momentum_values)

# Print normalized momentum values
for timeframe, norm_momentum in normalized_momentum.items():
    print(f"Normalized Momentum for {timeframe}: {norm_momentum:.2f}%")

# Calculate dominant ratio
positive_count = sum(1 for value in normalized_momentum.values() if value > 50)
negative_count = len(normalized_momentum) - positive_count

print(f"Positive momentum timeframes: {positive_count}/{len(normalized_momentum)}")
print(f"Negative momentum timeframes: {negative_count}/{len(normalized_momentum)}")

if positive_count > negative_count:
    print("Overall dominant momentum: Positive")
elif positive_count < negative_count:
    print("Overall dominant momentum: Negative")
else:
    print("Overall dominant momentum: Balanced")
