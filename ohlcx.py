import requests
import numpy as np
import talib  # Make sure to have TA-Lib installed
from binance.client import Client as BinanceClient
import datetime
from sklearn.linear_model import LinearRegression  # Importing LinearRegression for forecasting

# Define Binance client by reading API key and secret from local file
def get_binance_client():
    with open("credentials.txt", "r") as f:   
        lines = f.readlines()
        api_key = lines[0].strip()  
        api_secret = lines[1].strip()  
    client = BinanceClient(api_key, api_secret)
    return client

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

price = get_price(TRADE_SYMBOL)
print(f"Current Price: {price}\n")

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

for timeframe in timeframes:
    close = get_close(timeframe)
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close, period=14, minimum_percentage=2, maximum_percentage=2)

    print(f"Timeframe: {timeframe}")
    print("Minimum threshold:", min_threshold)
    print("Maximum threshold:", max_threshold)
    print("Average MTF:", avg_mtf)
    
    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close[-1]))
    if closest_threshold == min_threshold:
        print("The last minimum value is closest to the current close.")
        print("The last minimum value is", closest_threshold)
    elif closest_threshold == max_threshold:
        print("The last maximum value is closest to the current close.")
        print("The last maximum value is", closest_threshold)
    else:
        print("No threshold value found.")

    print()  # Add a newline for better readability

current_time = datetime.datetime.now()
current_close = close[-1]

print("Current local Time is now at: ", current_time)
print("Current close price is at: ", current_close)

print()

def detect_reversal(candles):
    closes = np.array([candle['close'] for candle in candles])
    sma7 = talib.SMA(closes, timeperiod=7)
    sma12 = talib.SMA(closes, timeperiod=12)
    sma26 = talib.SMA(closes, timeperiod=26)
    sma45 = talib.SMA(closes, timeperiod=45)

    local_dip_signal = "No Local close vs SMA Dip"
    local_top_signal = "No Local close vs SMA Top"

    if closes[-1] < sma7[-1] and sma7[-1] < sma12[-1] and sma12[-1] < sma26[-1] and sma26[-1] < sma45[-1]:
        local_dip_signal = f"Local Dip detected at price {closes[-1]}"

    if closes[-1] > sma7[-1] and sma7[-1] > sma12[-1] and sma12[-1] > sma26[-1] and sma26[-1] > sma45[-1]:
        local_top_signal = f"Local Top detected at price {closes[-1]}"

    return local_dip_signal, local_top_signal

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

def find_next_reversal_target(sine_wave, last_reversal):  
    if last_reversal == "dip":
        return np.min(sine_wave)  # Next target for a dip, so find the smallest point
    else:
        return np.max(sine_wave)  # Next target for a top, so find the largest point

# New Logic to Check Open and Close Prices for Reversals
def check_open_close_reversals(candle):
    daily_open = candle['open']
    current_close = candle['close']

    # Get support and resistance levels safely
    timeframe = candle['timeframe']
    lows = [c['low'] for c in candle_map[timeframe]]
    highs = [c['high'] for c in candle_map[timeframe]]
    
    # Check if we have enough data for support and resistance
    if len(lows) == 0 or len(highs) == 0:
        return "Insufficient data for support and resistance levels."

    support_level = np.min(lows)  # Approximate support level
    resistance_level = np.max(highs)  # Approximate resistance level

    if current_close < daily_open and current_close > support_level:
        return "Below daily open"
    elif current_close > daily_open and current_close < resistance_level:
        return "Above daily open"
    
    return "No reversal signal detected."

# Updated market mood analysis to align with last maximum and minimum conditions
def analyze_market_mood(sine_wave, min_threshold, max_threshold, current_close):
    last_max = max_threshold
    last_min = min_threshold

    # Determine mood based on proximity to the last maximum and minimum values
    if abs(current_close - last_max) < abs(current_close - last_min):
        market_mood = "Bearish"  # Current close is closer to the last maximum
    else:
        market_mood = "Bullish"  # Current close is closer to the last minimum

    print(f"Market Mood based on the sine wave: {market_mood}")
    print(f"Last Maximum Value: {last_max}")
    print(f"Last Minimum Value: {last_min}")

# New function to forecast price using linear regression
def forecast_price(candle_map, timeframe, periods_ahead=1):
    candles = candle_map[timeframe]
    closes = np.array([candle['close'] for candle in candles])

    # Prepare the data for linear regression
    X = np.arange(len(closes)).reshape(-1, 1)  # Time points (0, 1, 2, ...)
    y = closes.reshape(-1, 1)  # Closing prices

    # Create a linear regression model and fit it
    model = LinearRegression()
    model.fit(X, y)

    # Make prediction for the next 'periods_ahead' time steps
    future_X = np.array([len(closes) + i for i in range(1, periods_ahead + 1)]).reshape(-1, 1)
    forecasted_prices = model.predict(future_X)

    return forecasted_prices.flatten()

for timeframe in timeframes:
    candles = candle_map[timeframe]
    dip_signal, top_signal = detect_reversal(candles)

    # Fetch the last close prices and thresholds
    close_prices = get_close(timeframe)
    thresholds = calculate_thresholds(close_prices)

    # Generate harmonic sine wave based on threshold values and last reversal
    last_reversal = "dip" if "Dip" in dip_signal else "top"
    sine_wave = scale_to_sine(thresholds, last_reversal)

    # Analyze market mood
    analyze_market_mood(sine_wave, thresholds[0], thresholds[1], close_prices[-1])

    # Determine which threshold is closest to current close
    current_close = close_prices[-1]
    closest_threshold = min(thresholds[0], thresholds[1], key=lambda x: abs(x - current_close))

    # Check if last maximum or minimum value is closest to the current close
    if closest_threshold == thresholds[1]:  # Last maximum is closest
        next_reversal_target = find_next_reversal_target(sine_wave, "dip")
        print(f"Starting a down cycle from last maximum price: {thresholds[1]:.2f}, expected next dip: {next_reversal_target:.2f}")
    else:  # Last minimum is closest
        next_reversal_target = find_next_reversal_target(sine_wave, "top")
        print(f"Starting an up cycle from last minimum price: {thresholds[0]:.2f}, expected next top: {next_reversal_target:.2f}")

    # Check for reversal based on daily open
    last_candle = candles[-1]
    reversal_signal = check_open_close_reversals(last_candle)
    print(reversal_signal)

    # New Forecasting Step
    forecasted_prices = forecast_price(candle_map, timeframe, periods_ahead=3)  # Forecast for the next 3 periods
    print(f"Forecasted Prices for {timeframe}: {forecasted_prices}")

    # Print results for each timeframe
    print(f"Timeframe: {timeframe}")
    print(dip_signal)
    print(top_signal)
    print(f"Next target reversal price expected: {next_reversal_target}")
    print("\n" + "=" * 30 + "\n")
