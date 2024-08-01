# Import modules
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

# Define a function to get the account balance in BUSD
def get_account_balance():
    account = client.get_asset_balance(asset='USDT')
    return float(account['free']) if account else 0.0

# Get the USDT balance
bUSD_balance = get_account_balance()
print("USDT Spot balance:", bUSD_balance)

# Define Binance client reading api key and secret from local file
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

# Initialize variables for tracking trade state
TRADE_SYMBOL = "BTCUSDT"
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

# Scale current close price to sine wave
def scale_to_sine(timeframe):
    close_prices = np.array(get_close_prices(timeframe))
    if len(close_prices) < 2:
        return None, None, None

    # Calculate HT_SINE values
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)  # Replace NaNs with 0
    sine_wave = -sine_wave

    # Get current sine and min/max values
    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate distances
    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if sine_wave_max != sine_wave_min else 0
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if sine_wave_max != sine_wave_min else 0

    return dist_min, dist_max, current_sine

# Print results for each timeframe
def print_results(timeframes):
    for timeframe in timeframes:
        dist_min, dist_max, current_sine = scale_to_sine(timeframe)
        if dist_min is not None:
            print(f"For {timeframe} timeframe:")
            print(f"Distance to min: {dist_min:.2f}%")
            print(f"Distance to max: {dist_max:.2f}%")
            print(f"Current Sine value: {current_sine}\n")

print_results(timeframes)

# Calculate overall percentages for different bands of timeframes
def calculate_band_averages(timeframe_groups):
    band_averages = {}
    for band_name, timeframes in timeframe_groups.items():
        distances_min = []
        distances_max = []
        for timeframe in timeframes:
            dist_min, dist_max, _ = scale_to_sine(timeframe)
            if dist_min is not None:
                distances_min.append(dist_min)
                distances_max.append(dist_max)
        
        if distances_min:
            band_averages[band_name] = {
                'average_min': np.mean(distances_min),
                'average_max': np.mean(distances_max)
            }
        else:
            band_averages[band_name] = {
                'average_min': None,
                'average_max': None
            }
    
    return band_averages

# Define timeframe bands
timeframe_bands = {
    'Small Band': ['1m', '3m', '5m'],
    'Medium Band': ['15m', '30m', '1h'],
    'Major Band': ['2h', '4h', '6h'],
    'Big Band': ['8h', '12h', '1d']
}

# Calculate and print band averages
band_averages = calculate_band_averages(timeframe_bands)

for band_name, averages in band_averages.items():
    print(f"{band_name} Averages:")
    print(f"Average Distance to Min: {averages['average_min']:.2f}%" if averages['average_min'] is not None else "No Data")
    print(f"Average Distance to Max: {averages['average_max']:.2f}%" if averages['average_max'] is not None else "No Data")
    print("\n" + "="*30 + "\n")
