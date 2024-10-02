import requests
import numpy as np
import talib  # Make sure to have TA-Lib installed
from binance.client import Client as BinanceClient
import datetime
from scipy.fft import fft, ifft

##################################################
# Define Binance client by reading API key and secret from local file:

def get_binance_client():
    with open("credentials.txt", "r") as f:   
        lines = f.readlines()
        api_key = lines[0].strip()  
        api_secret = lines[1].strip()  
    client = BinanceClient(api_key, api_secret)
    return client

client = get_binance_client()
##################################################
TRADE_SYMBOL = "BTCUSDT"
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

##################################################
def get_latest_candle(symbol, interval):
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

for interval in timeframes:
    latest_candle = get_latest_candle(TRADE_SYMBOL, interval)
    print(f"Latest Candle ({interval}):")
    print(f"Time: {latest_candle['time']}, Open: {latest_candle['open']}, High: {latest_candle['high']}, Low: {latest_candle['low']}, Close: {latest_candle['close']}, Volume: {latest_candle['volume']}, Timeframe: {latest_candle['timeframe']}")
    print("\n" + "=" * 30 + "\n")

##################################################
def get_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("price key not found in API response")

price = get_price(TRADE_SYMBOL)
print(f"Current Price: {price}\n")

##################################################
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

##################################################
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = np.nanmin(momentum)   
    max_momentum = np.nanmax(momentum)
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])
    avg_mtf = np.nanmean(close_prices)
    return min_threshold, max_threshold, avg_mtf

for timeframe in timeframes:
    close = get_close(timeframe)
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

    print(f"Timeframe: {timeframe}")
    print("Minimum threshold:", min_threshold)
    print("Maximum threshold:", max_threshold)
    print("Average MTF:", avg_mtf)
    
    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close[-1]))
    if closest_threshold == min_threshold:
        print("The last minimum value is closest to the current close.")
    elif closest_threshold == max_threshold:
        print("The last maximum value is closest to the current close.")
    else:
        print("No threshold value found.")

    print()  # Add a newline for better readability

##################################################
current_time = datetime.datetime.now()
current_close = close[-1]

print("Current local Time is now at: ", current_time)
print("Current close price is at: ", current_close)

print()

##################################################
def detect_reversal(candles):
    closes = np.array([candle['close'] for candle in candles])
    sma7 = talib.SMA(closes, timeperiod=7)
    sma12 = talib.SMA(closes, timeperiod=12)
    sma26 = talib.SMA(closes, timeperiod=26)

    local_dip_signal = "No Local Dip"
    local_top_signal = "No Local Top"

    if closes[-1] < sma7[-1] and sma7[-1] < sma12[-1] and sma12[-1] < sma26[-1]:
        local_dip_signal = f"Local Dip detected at price {closes[-1]}"

    if closes[-1] > sma7[-1] and sma7[-1] > sma12[-1] and sma12[-1] > sma26[-1]:
        local_top_signal = f"Local Top detected at price {closes[-1]}"

    return local_dip_signal, local_top_signal

##################################################
def scale_to_sine(thresholds, last_reversal, cycles=5):  
    min_threshold, max_threshold, avg_threshold = thresholds
    num_points = 100
    t = np.linspace(0, 2 * np.pi * cycles, num_points)

    # Establish the amplitude based on max and min thresholds
    amplitude = (max_threshold - min_threshold) / 2

    if last_reversal == "dip":
        baseline = min_threshold + amplitude
        sine_wave = baseline + amplitude * np.sin(t)
    else:
        baseline = max_threshold - amplitude
        sine_wave = baseline - amplitude * np.sin(t)

    return sine_wave

def find_next_reversal_target(sine_wave, dt):
    # Return the first maximum or minimum point of the sine wave
    next_target = sine_wave[0] + (sine_wave[-1] - sine_wave[0]) * 0.5  # Midpoint as the target for reversal
    return next_target

for timeframe in timeframes:
    candles = candle_map[timeframe]
    dip_signal, top_signal = detect_reversal(candles)

    # Fetch the last close prices and thresholds
    close_prices = get_close(timeframe)
    thresholds = calculate_thresholds(close_prices)

    # Generate harmonic sine wave based on threshold values and last reversal
    sine_wave = scale_to_sine(thresholds, "dip" if "Dip" in dip_signal else "top")

    # Find the next target reversal price based on sine wave
    next_reversal_target = find_next_reversal_target(sine_wave, 1)  # Assuming next target on the next minute/cycle

    # Print results for each timeframe
    print(f"Timeframe: {timeframe}")
    print(dip_signal)
    print(top_signal)
    print(f"Next target reversal price expected: {next_reversal_target}")
    print("\n" + "=" * 30 + "\n")