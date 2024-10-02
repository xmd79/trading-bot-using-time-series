import requests
import numpy as np
import talib  # Make sure to have TA-Lib installed
from binance.client import Client as BinanceClient
import datetime

##################################################
# Define Binance client by reading API key and secret from local file:

def get_binance_client():
    # Read credentials from file    
    with open("credentials.txt", "r") as f:   
        lines = f.readlines()
        api_key = lines[0].strip()  
        api_secret = lines[1].strip()  
          
    # Instantiate client        
    client = BinanceClient(api_key, api_secret)
          
    return client

# Call the function to get the client  
client = get_binance_client()

##################################################
# Initialize variables for tracking trade state:

TRADE_SYMBOL = "BTCUSDT"

##################################################
# Define timeframes and get candles:

timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 1000  # default limit for the spot API

        # Fetch klines (candlestick data)
        klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )

        # Convert klines to candle dict
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

# Get candles  
candles = get_candles(TRADE_SYMBOL, timeframes)

# Organize candles by timeframe        
candle_map = {}  
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

##################################################
def get_latest_candle(symbol, interval):
    """Retrieve the latest candle for a given symbol and interval using Spot API"""
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
    print("\n" + "=" * 30 + "\n")

##################################################
# Get current price

def get_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"  # Use Spot API endpoint
    params = {
        "symbol": symbol 
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("price key not found in API response")

# Fetch and print current price
price = get_price(TRADE_SYMBOL)
print(f"Current Price: {price}\n")

##################################################
# Get entire list of close prices as <class 'list'> type

def get_close(timeframe):
    closes = []
    candles = candle_map[timeframe]

    for c in candles:
        close = c['close']
        if not np.isnan(close):
            closes.append(close)

    # Append current price to the list of closing prices
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)

    return closes

##################################################
# Calculate thresholds and momentum signal
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate thresholds and averages based on min and max percentages. 
    """
    # Convert close_prices to a numpy array
    close_prices = np.array(close_prices)

    # Get min/max close    
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=period)

    # Get min/max momentum    
    min_momentum = np.nanmin(momentum)   
    max_momentum = np.nanmax(momentum)

    # Calculate custom percentages 
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    # Calculate thresholds       
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    # Calculate range of prices within a certain distance from the current close price
    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)

    # Filter close prices
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)

    # Calculate avg    
    avg_mtf = np.nanmean(filtered_close)

    # Get current momentum       
    current_momentum = momentum[-1]

    # Calculate % to min/max momentum    
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) /   
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               

        percent_to_max_momentum = ((current_momentum - min_momentum) / 
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
 
    # Calculate combined percentages              
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    # Combined momentum signal     
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal

# Iterate over each timeframe and calculate thresholds
for timeframe in timeframes:
    close = get_close(timeframe)

    # Calculate thresholds for current timeframe
    min_threshold, max_threshold, avg_mtf, momentum_signal = calculate_thresholds(
        close, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
    )

    # Print calculated values for each timeframe
    print(f"Timeframe: {timeframe}")
    print("Momentum signal:", momentum_signal)
    print("Minimum threshold:", min_threshold)
    print("Maximum threshold:", max_threshold)
    print("Average MTF:", avg_mtf)

    # Determine which threshold is closest to the current close
    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close[-1]))

    if closest_threshold == min_threshold:
        print("The last minimum value is closest to the current close.")
    elif closest_threshold == max_threshold:
        print("The last maximum value is closest to the current close.")
    else:
        print("No threshold value found.")

    print()  # Add a newline for better readability

##################################################
# Define the current time and close price
current_time = datetime.datetime.now()
current_close = close[-1]

print("Current local Time is now at: ", current_time)
print("Current close price is at : ", current_close)

print()

##################################################
# New Function to Detect Last Reversal using Three SMAs
def detect_reversal(candles):
    """
    Detect reversals based on three SMA ribbons.
    - Local Dip: Current close < SMA(7) and SMA(7) < SMA(12) < SMA(26)
    - Local Top: Current close > SMA(7) and SMA(7) > SMA(12) > SMA(26)
    """
    closes = np.array([candle['close'] for candle in candles])

    # Calculate the SMAs
    sma7 = talib.SMA(closes, timeperiod=7)
    sma12 = talib.SMA(closes, timeperiod=12)
    sma26 = talib.SMA(closes, timeperiod=26)

    # Signals
    local_dip_signal = "No Local Dip"
    local_top_signal = "No Local Top"

    # Check for local dip conditions
    if closes[-1] < sma7[-1] and sma7[-1] < sma12[-1] and sma12[-1] < sma26[-1]:
        local_dip_signal = f"Local Dip detected at price {closes[-1]}"

    # Check for local top conditions
    if closes[-1] > sma7[-1] and sma7[-1] > sma12[-1] and sma12[-1] > sma26[-1]:
        local_top_signal = f"Local Top detected at price {closes[-1]}"

    return local_dip_signal, local_top_signal

##################################################

# Scale current close price to sine wave       
def scale_to_sine(timeframe):  
  
    close_prices = np.array(get_close(timeframe))
  
    # Get last close price 
    current_close = close_prices[-1]      
        
    # Calculate sine wave        
    sine_wave, leadsine = talib.HT_SINE(close_prices)
            
    # Replace NaN values with 0        
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave
        
    # Get the sine value for last close      
    current_sine = sine_wave[-1]
            
    # Calculate the min and max sine           
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # Calculate % distances            
    dist_min, dist_max = [], []
 
    for close in close_prices:    
        # Calculate distances as percentages        
        dist_from_close_to_min = ((current_sine - sine_wave_min) /  
                                   (sine_wave_max - sine_wave_min)) * 100            
        dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                                   (sine_wave_max - sine_wave_min)) * 100
                
        dist_min.append(dist_from_close_to_min)       
        dist_max.append(dist_from_close_to_max)

    return dist_from_close_to_min, dist_from_close_to_max, current_sine
     
 
# Iterate over each timeframe and call the scale_to_sine function
for timeframe in timeframes:
    dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)
    
    # Print the results for each timeframe
    print(f"For {timeframe} timeframe:")
    print(f"Distance to min: {dist_from_close_to_min:.2f}%")
    print(f"Distance to max: {dist_from_close_to_max:.2f}%")
    print(f"Current Sine value: {current_sine}\n")

##################################################

# Iterate over each timeframe and check for reversals using the new SMA method
for timeframe in timeframes:
    candles = candle_map[timeframe]
    dip_signal, top_signal = detect_reversal(candles)
    print(f"Timeframe: {timeframe}")
    print(dip_signal)
    print(top_signal)
    print("\n" + "=" * 30 + "\n")


##################################################
