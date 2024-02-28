##################################################
##################################################

# Start code:

print()

##################################################
##################################################

print("Init code: ")
print()

##################################################
##################################################

print("Test")
print()

##################################################
##################################################

# Import modules:

import math
import time
import numpy as np
import hashlib
import requests
import hmac
import talib
import json
import datetime
from datetime import timedelta
from decimal import Decimal
import decimal
import random
import statistics
from statistics import mean
import scipy.fftpack as fftpack
import gc

from discord_webhook import DiscordWebhook

##################################################
##################################################

# binance module imports
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *

##################################################
##################################################

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

##################################################
##################################################

# Define a function to get the account balance in BUSD

def get_account_balance():
    accounts = client.futures_account_balance()
    for account in accounts:
        if account['asset'] == 'USDT':
            bUSD_balance = float(account['balance'])
            return bUSD_balance

# Get the USDT balance of the futures account
bUSD_balance = float(get_account_balance())

# Print account balance
print("USDT Futures balance:", bUSD_balance)
print()

##################################################
##################################################

# Define Binance client reading api key and secret from local file:

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
##################################################

# Initialize variables for tracking trade state:

TRADE_SYMBOL = "BTCUSDT"

##################################################
##################################################

##################################################
##################################################

# Define timeframes and get candles:

timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

def get_candles(TRADE_SYMBOL, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 10000  # default limit
        tf_value = int(timeframe[:-1])  # extract numeric value of timeframe
        if tf_value >= 4:  # check if timeframe is 4h or above
            limit = 20000  # increase limit for 4h timeframe and above
        klines = client.get_klines(
            symbol=TRADE_SYMBOL,
            interval=timeframe,
            limit=limit
        )
        # Convert klines to candle dict
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

# Get candles  
candles = get_candles(TRADE_SYMBOL, timeframes) 
#print(candles)

# Organize candles by timeframe        
candle_map = {}  
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

#print(candle_map)

##################################################
##################################################

def get_latest_candle(TRADE_SYMBOL, interval, start_time=None):
    """Retrieve the latest candle for a given symbol and interval"""
    if start_time is None:
        klines = client.futures_klines(symbol=TRADE_SYMBOL, interval=interval, limit=1)
    else:
        klines = client.futures_klines(symbol=TRADE_SYMBOL, interval=interval, startTime=start_time, limit=1)
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
    print(f"Time: {latest_candle['time']}")
    print(f"Open: {latest_candle['open']}")
    print(f"High: {latest_candle['high']}")
    print(f"Low: {latest_candle['low']}")
    print(f"Close: {latest_candle['close']}")
    print(f"Volume: {latest_candle['volume']}")
    print(f"Timeframe: {latest_candle['timeframe']}")
    print("\n" + "="*30 + "\n")

##################################################
##################################################

# Get current price as <class 'float'>

def get_price(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
        params = {
            "symbol": symbol 
            }
        response = requests.get(url, params=params)
        data = response.json()
        if "price" in data:
            price = float(data["price"])
        else:
            raise KeyError("price key not found in API response")
        return price      
    except (BinanceAPIException, KeyError) as e:
        print(f"Error fetching price for {symbol}: {e}")
        return 0

price = get_price("BTCUSDT")

print(price)

print()

##################################################
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

close = get_close('1m')
#print(close)

##################################################
##################################################

# Get entire list of close prices as <class 'list'> type

def get_closes(timeframe):
    closes = []
    candles = candle_map[timeframe]
    
    for c in candles:
        close = c['close']
        if not np.isnan(close):     
            closes.append(close)
            
    return closes

closes = get_closes('1m')

#print(closes)

print()

##################################################
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
#for timeframe in timeframes:
    #dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)
    
    # Print the results for each timeframe
    #print(f"For {timeframe} timeframe:")
    #print(f"Distance to min: {dist_from_close_to_min:.2f}%")
    #print(f"Distance to max: {dist_from_close_to_max:.2f}%")
    #print(f"Current Sine value: {current_sine}\n")

print()

##################################################
##################################################

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate thresholds and averages based on min and max percentages. 
    """
  
    # Get min/max close    
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    # Convert close_prices to numpy array
    close_prices = np.array(close_prices)
    
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

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price


# Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

print("Momentum signal:", momentum_signal)
print()

print("Minimum threshold:", min_threshold)
print("Maximum threshold:", max_threshold)
print("Average MTF:", avg_mtf)

#print("Range of prices within distance from current close price:")
#print(range_price[-1])

# Determine which threshold is closest to the current close
closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close[-1]))

if closest_threshold == min_threshold:
    print("The last minimum value is closest to the current close.")
elif closest_threshold == max_threshold:
    print("The last maximum value is closest to the current close.")
else:
    print("No threshold value found.")

print()


##################################################
##################################################

def get_momentum(timeframe):
    """Calculate momentum for a single timeframe"""
    # Get candle data               
    candles = candle_map[timeframe][-100:]  
    # Calculate momentum using talib MOM
    momentum = talib.MOM(np.array([c["close"] for c in candles]), timeperiod=14)
    return momentum[-1]

# Calculate momentum for each timeframe
momentum_values = {}
for timeframe in timeframes:
    momentum = get_momentum(timeframe)
    momentum_values[timeframe] = momentum
    print(f"Momentum for {timeframe}: {momentum}")

# Convert momentum to a normalized scale and determine if it's positive or negative
normalized_momentum = {}
for timeframe, momentum in momentum_values.items():
    normalized_value = (momentum + 100) / 2  # Normalize to a scale between 0 and 100
    normalized_momentum[timeframe] = normalized_value
    print(f"Normalized Momentum for {timeframe}: {normalized_value:.2f}%")

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

print()

##################################################
##################################################

# Define the current time and close price
current_time = datetime.datetime.now()
current_close = closes[-1]

print("Current local Time is now at: ", current_time)
print("Current close price is at : ", current_close)

print()


##################################################
##################################################

def get_closes_last_n_minutes(interval, n):
    """Generate mock closing prices for the last n minutes"""
    closes = []
    for i in range(n):
        closes.append(random.uniform(0, 100))
    return closes

print()

##################################################
##################################################

import numpy as np
import scipy.fftpack as fftpack
import datetime

def get_target(closes, n_components, target_distance=0.01):
    # Calculate FFT of closing prices
    fft = fftpack.rfft(closes) 
    frequencies = fftpack.rfftfreq(len(closes))
    
    # Sort frequencies by magnitude and keep only the top n_components 
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]
    
    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    
    # Calculate the target price as the next value after the last closing price, plus a small constant
    current_close = closes[-1]
    target_price = filtered_signal[-1] + target_distance
    
    # Get the current time           
    current_time = datetime.datetime.now()
    
    # Calculate the market mood based on the predicted target price and the current close price
    diff = target_price - current_close
    if diff > 0:           
        market_mood = "Bullish"
    else:
        market_mood = "Bearish"
    
    # Calculate fast cycle targets
    fastest_target = current_close + target_distance / 2
    fast_target1 = current_close + target_distance / 4
    fast_target2 = current_close + target_distance / 8
    fast_target3 = current_close + target_distance / 16
    fast_target4 = current_close + target_distance / 32
    
    # Calculate other targets
    target1 = target_price + np.std(closes) / 16
    target2 = target_price + np.std(closes) / 8
    target3 = target_price + np.std(closes) / 4
    target4 = target_price + np.std(closes) / 2
    target5 = target_price + np.std(closes)
    
    # Calculate the stop loss and target levels
    entry_price = closes[-1]    
    stop_loss = entry_price - 3 * np.std(closes)   
    target6 = target_price + np.std(closes)
    target7 = target_price + 2 * np.std(closes)
    target8 = target_price + 3 * np.std(closes)
    target9 = target_price + 4 * np.std(closes)
    target10 = target_price + 5 * np.std(closes)
    
    return current_time, entry_price, stop_loss, fastest_target, market_mood

closes = get_closes("1m")     
n_components = 5

current_time, entry_price, stop_loss, fastest_target, market_mood = get_target(closes, n_components, target_distance=56)

print("Current local Time is now at:", current_time)
print("Market mood is:", market_mood)

print()

current_close = closes[-1]
print("Current close price is at:", current_close)

print()

print("Fastest target is:", fastest_target)

print()

##################################################
##################################################

def get_current_price():
    url = "https://fapi.binance.com/fapi/v1/ticker/price"
    params = {
        "symbol": "BTCUSDT" 
    }
    response = requests.get(url, params=params)
    data = response.json()
    price = float(data["price"])
    return price

# Get the current price
price = get_current_price()

print()

##################################################
##################################################

from sklearn.linear_model import LinearRegression

def price_regression(close):
    # Convert 'close' to a numpy array
    close_data = np.array(close)

    # Create timestamps based on the index (assuming each close price corresponds to a single time unit)
    timestamps = np.arange(len(close_data))

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(timestamps.reshape(-1, 1), close_data)

    # Predict future prices using the regression model
    num_targets = 1
    future_timestamps = np.arange(len(close_data), len(close_data) + num_targets)
    future_prices = model.predict(future_timestamps.reshape(-1, 1))

    return future_timestamps, future_prices

##################################################
##################################################

def calculate_reversal_and_forecast(close):
    # Initialize variables
    current_reversal = None
    next_reversal = None
    last_reversal = None
    forecast_dip = None
    forecast_top = None
    
    # Calculate minimum and maximum values
    min_value = np.min(close)
    max_value = np.max(close)
    
    # Calculate forecast direction and price using FFT
    fft = fftpack.rfft(close)
    frequencies = fftpack.rfftfreq(len(close))
    idx = np.argsort(np.abs(fft))[::-1][:10]
    top_frequencies = frequencies[idx]
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    
    if len(close) > 1:
        if filtered_signal[-1] > filtered_signal[-2]:
            forecast_direction = "Up"
            forecast_price_fft = filtered_signal[-1] + (filtered_signal[-1] - filtered_signal[-2]) * 0.5
        else:
            forecast_direction = "Down"
            forecast_price_fft = filtered_signal[-1] - (filtered_signal[-2] - filtered_signal[-1]) * 0.5
    else:
        forecast_direction = "Neutral"
        forecast_price_fft = close[-1]
    
    # Check the relationship between the last value and min/max
    last_value = close[-1]
    if min_value <= last_value <= max_value:
        if last_value == min_value:
            current_reversal = "DIP"
            next_reversal = "TOP"
        elif last_value == max_value:
            current_reversal = "TOP"
            next_reversal = "DIP"
    else:
        forecast_direction = "Up" if close[-1] > close[-2] else "Down"
        forecast_price_fft = price_regression(close)
    
    # Initialize variables for last reversal and distance
    distance = None
    last_reversal = None
    
    # Calculate the distance between the last reversal and the last value
    reversal_idx = None
    for i in range(len(close) - 2, -1, -1):
        if current_reversal == "DIP" and close[i] == min_value:
            reversal_idx = i
            break
        elif current_reversal == "TOP" and close[i] == max_value:
            reversal_idx = i
            break
    
    if reversal_idx is not None:
        distance = len(close) - 1 - reversal_idx
        if current_reversal == "DIP":
            last_reversal = "DIP"
        elif current_reversal == "TOP":
            last_reversal = "TOP"
    
    # Calculate forecast DIP and TOP
    if last_reversal == "DIP":
        forecast_dip = close[-1] - (distance * 0.1)
        forecast_top = forecast_dip + (forecast_dip - close[-1]) * 2
    elif last_reversal == "TOP":
        forecast_top = close[-1] + (distance * 0.1)
        forecast_dip = forecast_top - (close[-1] - forecast_top) * 2
    
    future_price_regression = price_regression(close)
    
    return current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression, last_reversal, forecast_dip, forecast_top



# Call the calculate_reversal_and_forecast function with the example data
(current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression, last_reversal, forecast_dip, forecast_top) = calculate_reversal_and_forecast(close)

print()

##################################################
##################################################

def calculate_elements():
    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177
    # Calculate the Brun constant from the phi ratio and sqrt(5)
    brun_constant = math.sqrt(PHI * math.sqrt(5))
    # Define PI constant with 15 decimals
    PI = 3.1415926535897932384626433832795028842
    # Define e constant with 15 decimals
    e = 2.718281828459045235360287471352662498
    # Calculate sacred frequency
    sacred_freq = (432 * PHI ** 2) / 360
    # Calculate Alpha and Omega ratios
    alpha_ratio = PHI / PI
    omega_ratio = PI / PHI
    # Calculate Alpha and Omega spiral angle rates
    alpha_spiral = (2 * math.pi * sacred_freq) / alpha_ratio
    omega_spiral = (2 * math.pi * sacred_freq) / omega_ratio
    # Calculate inverse powers of PHI and fractional reciprocals
    inverse_phi = 1 / PHI
    inverse_phi_squared = 1 / (PHI ** 2)
    inverse_phi_cubed = 1 / (PHI ** 3)
    reciprocal_phi = PHI ** -1
    reciprocal_phi_squared = PHI ** -2
    reciprocal_phi_cubed = PHI ** -3

    # Calculate unit circle degrees for each quadrant, including dip reversal up and top reversal down cycles
    unit_circle_degrees = {
        1: {'angle': 135, 'polarity': ('-', '-'), 'cycle': 'dip_to_top'},  # Quadrant 1 (Dip to Top)
        2: {'angle': 45, 'polarity': ('+', '+'), 'cycle': 'top_to_dip'},   # Quadrant 2 (Top to Dip)
        3: {'angle': 315, 'polarity': ('+', '-'), 'cycle': 'dip_to_top'},  # Quadrant 3 (Dip to Top)
        4: {'angle': 225, 'polarity': ('-', '+'), 'cycle': 'top_to_dip'},   # Quadrant 4 (Top to Dip)
    }

    # Calculate ratios up to 12 ratio degrees
    ratios = [math.atan(math.radians(degrees)) for degrees in range(1, 13)]

    # Calculate arctanh values
    arctanh_values = {
        0: 0,
        1: float('inf'),
        -1: float('-inf')
    }

    # Calculate imaginary number
    imaginary_number = 1j

    return PHI, sacred_freq, unit_circle_degrees, ratios, arctanh_values, imaginary_number, brun_constant, PI, e, alpha_ratio, omega_ratio, inverse_phi, inverse_phi_squared, inverse_phi_cubed, reciprocal_phi, reciprocal_phi_squared, reciprocal_phi_cubed  

print()


def forecast_sma_targets(price):
    (PHI, sacred_freq, unit_circle_degrees, ratios, arctanh_values, imaginary_number, brun_constant, PI, e, alpha_ratio, omega_ratio, inverse_phi, inverse_phi_squared, inverse_phi_cubed, reciprocal_phi, reciprocal_phi_squared, reciprocal_phi_cubed) = calculate_elements()

    output_data = []
    target_variables = {}  # Dictionary to store target and target_45 for each quadrant
    
    output_data.append(f"Given Close Price (Center of Unit Circle): {price}\n")
    
    for quadrant, _ in unit_circle_degrees.items():
        # Calculate the forecast price using sacred_freq and square of 9 for the quadrant
        target = price + (sacred_freq * math.pow(9, (quadrant * 0.25)))
        
        # Adjust the target price with a 45-degree angle (using trigonometry)
        angle_adjustment = sacred_freq * math.cos(math.radians(45)) * (quadrant * 0.25)
        target_45 = price + angle_adjustment

        distance = ((target - price) / price) * 100

        output_data.append(f"Quadrant: {quadrant}")
        output_data.append(f"Forecasted Target_Quad_{quadrant}: Price - {target:.2f}, Distance Percentage - {distance:.2f}%")
        output_data.append(f"Forecasted 45Degree_Target_Quad_{quadrant}: Price - {target_45:.2f}")
        output_data.append("-" * 50)

        # Store target and target_45 in the dictionary
        target_variables[f"target_quad_{quadrant}"] = target
        target_variables[f"target_45_quad_{quadrant}"] = target_45

    return output_data, target_variables

# Capture the targets when calling the function
results, targets = forecast_sma_targets(price)

# Print each output string separately
for result in results:
    print(result)

# Assign each target_45 value to a separate variable
target_45_quad_1 = targets['target_45_quad_1']
target_45_quad_2 = targets['target_45_quad_2']
target_45_quad_3 = targets['target_45_quad_3']
target_45_quad_4 = targets['target_45_quad_4']

# Print each variable
print(f"Target_45 for Quadrant 1: {target_45_quad_1:.2f}")
print(f"Target_45 for Quadrant 2: {target_45_quad_2:.2f}")
print(f"Target_45 for Quadrant 3: {target_45_quad_3:.2f}")
print(f"Target_45 for Quadrant 4: {target_45_quad_4:.2f}")
print("-" * 50)

print()

##################################################
##################################################

def entry_long(symbol):
    try:     
        # Get balance and leverage     
        account_balance = get_account_balance()   
        trade_leverage = 20
    
        # Get symbol price    
        symbol_price = client.futures_symbol_ticker(symbol=symbol)['price']
        
        # Get step size from exchange info
        info = client.futures_exchange_info()
        filters = [f for f in info['symbols'] if f['symbol'] == symbol][0]['filters']
        step_size = [f['stepSize'] for f in filters if f['filterType']=='LOT_SIZE'][0]
                    
        # Calculate max quantity based on balance, leverage, and price
        max_qty = int(account_balance * trade_leverage / float(symbol_price) / float(step_size)) * float(step_size)  
                    
        # Create buy market order    
        order = client.futures_create_order(
            symbol=symbol,        
            side='BUY',           
            type='MARKET',         
            quantity=max_qty)          
                    
        if 'orderId' in order:
            return True
          
        else: 
            print("Error creating long order.")  
            return False
            
    except BinanceAPIException as e:
        print(f"Error creating long order: {e}")
        return False

def entry_short(symbol):
    try:     
        # Get balance and leverage     
        account_balance = get_account_balance()   
        trade_leverage = 20
    
        # Get symbol price    
        symbol_price = client.futures_symbol_ticker(symbol=symbol)['price']
        
        # Get step size from exchange info
        info = client.futures_exchange_info()
        filters = [f for f in info['symbols'] if f['symbol'] == symbol][0]['filters']
        step_size = [f['stepSize'] for f in filters if f['filterType']=='LOT_SIZE'][0]
                    
        # Calculate max quantity based on balance, leverage, and price
        max_qty = int(account_balance * trade_leverage / float(symbol_price) / float(step_size)) * float(step_size)  
                    
        # Create sell market order    
        order = client.futures_create_order(
            symbol=symbol,        
            side='SELL',           
            type='MARKET',         
            quantity=max_qty)          
                    
        if 'orderId' in order:
            return True
          
        else: 
            print("Error creating short order.")  
            return False
            
    except BinanceAPIException as e:
        print(f"Error creating short order: {e}")
        return False

def exit_trade():
    try:
        # Get account information including available margin
        account_info = client.futures_account()

        # Check if 'availableBalance' is present in the response
        if 'availableBalance' in account_info:
            available_margin = float(account_info['availableBalance'])

            # Check available margin before proceeding
            if available_margin < 0:
                print("Insufficient available margin to exit trades.")
                return

            # Get all open positions
            positions = client.futures_position_information()

            # Loop through each position
            for position in positions:
                symbol = position['symbol']
                position_amount = float(position['positionAmt'])

                # Determine order side
                if position_amount > 0:
                    order_side = 'SELL'
                elif position_amount < 0:
                    order_side = 'BUY'
                else:
                    continue  # Skip positions with zero amount

                # Place order to exit position      
                order = client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type='MARKET',
                    quantity=abs(position_amount))

                print(f"{order_side} order created to exit {abs(position_amount)} {symbol}.")

            print("All positions exited!")
        else:
            print("Error: 'availableBalance' not found in account_info.")
    except BinanceAPIException as e:
        print(f"Error exiting trade: {e}")

print()

##################################################
##################################################

print()

##################################################
##################################################

def calculate_normalized_distance(price, close):
    min_price = np.min(close)
    max_price = np.max(close)
    
    distance_to_min = price - min_price
    distance_to_max = max_price - price
    
    normalized_distance_to_min = distance_to_min / (distance_to_min + distance_to_max) * 100
    normalized_distance_to_max = distance_to_max / (distance_to_min + distance_to_max) * 100
    
    return normalized_distance_to_min, normalized_distance_to_max

def calculate_price_distance_and_wave(price, close):
    normalized_distance_to_min, normalized_distance_to_max = calculate_normalized_distance(price, close)
    
    # Calculate HT_SINE using talib
    ht_sine, _ = talib.HT_SINE(close) 
    #ht_sine = -ht_sine

    # Initialize market_mood
    market_mood = None
    
    # Determine market mood based on HT_SINE crossings and closest reversal
    closest_to_min = np.abs(close - np.min(close)).argmin()
    closest_to_max = np.abs(close - np.max(close)).argmin()
    
    # Check if any of the elements in the close array up to the last value is the minimum or maximum
    if np.any(close[:len(close)-1] == np.min(close[:len(close)-1])):
        market_mood = "Uptrend"
    elif np.any(close[:len(close)-1] == np.max(close[:len(close)-1])):
        market_mood = "Downtrend"

    result = {
        "price": price,
        "ht_sine_value": ht_sine[-1],
        "normalized_distance_to_min": normalized_distance_to_min,
        "normalized_distance_to_max": normalized_distance_to_max,
        "min_price": np.min(close),
        "max_price": np.max(close),
        "market_mood": market_mood
    }
    
    return result

# Example close prices
close_prices = np.array(close)  # Insert your actual close prices here

result = calculate_price_distance_and_wave(price, close_prices)

# Print the detailed information for the given price
print(f"Price: {result['price']:.2f}")
print(f"HT_SINE Value: {result['ht_sine_value']:.2f}")
print(f"Normalized Distance to Min: {result['normalized_distance_to_min']:.2f}%")
print(f"Normalized Distance to Max: {result['normalized_distance_to_max']:.2f}%")
print(f"Min Price: {result['min_price']:.2f}")
print(f"Max Price: {result['max_price']:.2f}")
print(f"Market Mood: {result['market_mood']}")

print()

##################################################
##################################################

def calculate_sine_wave_and_forecast(closes, min_threshold, max_threshold):
    """
    Determine market mood and forecast price movement based on sine wave reversals.
    """
    
    # Create a sequence of close prices centered around the thresholds to compute the sine wave
    sequence_min = np.linspace(min_threshold - 10, min_threshold + 10, 100)  # Adjust range and number as needed
    sequence_max = np.linspace(max_threshold - 10, max_threshold + 10, 100)  # Adjust range and number as needed
    
    # Compute HT_SINE for both sequences
    ht_sine_min, _ = talib.HT_SINE(sequence_min)
    ht_sine_max, _ = talib.HT_SINE(sequence_max)

    market_mood = None
    forecast_price = None

    # Check for uptrend based on the last reversal being the min_threshold
    if ht_sine_min[-1] < ht_sine_min[-2] and closes[-1] > min_threshold:
        market_mood = "Uptrend"
        forecast_price = min_threshold + (ht_sine_min[-1] - np.min(ht_sine_min[-10:]))
    
    # Check for downtrend based on the last reversal being the max_threshold
    elif ht_sine_max[-1] > ht_sine_max[-2] and closes[-1] < max_threshold:
        market_mood = "Downtrend"
        forecast_price = max_threshold - (np.max(ht_sine_max[-10:]) - ht_sine_max[-1])

    return market_mood, forecast_price

min_threshold, max_threshold, avg_mtf, momentum_signal, _ = calculate_thresholds(close, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

print("Momentum signal:", momentum_signal)
print("Minimum threshold:", min_threshold)
print("Maximum threshold:", max_threshold)
print("Average MTF:", avg_mtf)

market_mood, forecast_price = calculate_sine_wave_and_forecast(closes, min_threshold, max_threshold)
if market_mood:
    print(f"Forecast price: {forecast_price}")

closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - closes[-1]))
if closest_threshold == min_threshold:
    print("The last minimum value is closest to the current close.")
elif closest_threshold == max_threshold:
    print("The last maximum value is closest to the current close.")
else:
    print("No threshold value found.")

print()

##################################################
##################################################

# Calculate the 45-degree angle (simple linear regression)
x = np.arange(len(close))
slope, intercept = np.polyfit(x, close, 1)

# Calculate the expected trend line value for the last close price
expected_price = slope * len(close) + intercept

# Display the expected price on the 45-degree angle trend
print(f"Expected price on the 45-degree angle trend: {expected_price}")

# Get the last close price from the list for forecasting
last_close_price = price

# Define a function to forecast based on the 45-degree angle
def forecast_45_degree_angle(close_price, expected_price):
    if close_price < expected_price:
        return "Bullish Market Mood: Close below 45-degree angle moving towards it."
    elif close_price > expected_price:
        return "Bearish Market Mood: Close above 45-degree angle moving towards it."
    else:
        return "Neutral: Close at the 45-degree angle."

# Generate forecast based on the 45-degree angle
forecast_result = forecast_45_degree_angle(last_close_price, expected_price)

# Display the forecast result
print(forecast_result)

print()

##################################################
##################################################

import numpy as np
import talib  # Assuming you have the TA-Lib library installed and imported

# Time Series Decomposition
def decompose_time_series(close):
    padding = len(close) - len(np.convolve(close, np.ones(10)/10, mode='valid'))
    trend = np.convolve(close, np.ones(10)/10, mode='valid')
    trend = np.pad(trend, (padding, 0), mode='constant', constant_values=(trend[0], trend[-1]))

    seasonal = close - trend
    residual = close - (trend + seasonal)
    return trend, seasonal, residual

# Fast Fourier Transform (FFT)
def apply_fft(close):
    fft_values = np.fft.fft(close)
    frequencies = np.fft.fftfreq(len(fft_values))
    dominant_idx = np.argmax(np.abs(fft_values[:25]))
    dominant_frequency = frequencies[dominant_idx]
    return dominant_frequency

# Identify Reversals
def identify_reversals(close):
    trend, seasonal, residual = decompose_time_series(close)
    trend_direction = "up" if trend[-1] < trend[-2] else "down"
    dominant_frequency = apply_fft(close)
    
    if dominant_frequency < 0:
        market_mood = "Negative"
    else:
        market_mood = "Positive"
        
    forecasted_price = 4264490  # Placeholder value
    print(f"Forecasted Price: {forecasted_price // 100}.{forecasted_price % 100:02}")
    
    return forecasted_price

# Toroidal Group Symmetry Analysis
def toroidal_group_symmetry_analysis(close):
    fft_values = np.fft.fft(close)
    frequencies = np.fft.fftfreq(len(fft_values))

    # Get the indices of the 25 dominant frequencies
    dominant_indices = np.argsort(np.abs(fft_values))[-25:]

    # Get the last 5 dominant frequencies
    last_five_dominant_frequencies = frequencies[dominant_indices][-5:]

    # Count how many of the last 5 dominant frequencies are negative or positive
    negative_count = np.sum(last_five_dominant_frequencies < 0)
    positive_count = np.sum(last_five_dominant_frequencies > 0)

    if negative_count > positive_count:
        trend_behavior = "up"
    elif positive_count > negative_count:
        trend_behavior = "down"
    else:
        trend_behavior = "neutral"

    if abs(np.mean(last_five_dominant_frequencies)) < 0.5:
        print("The sinewave stationary circuit is stable based on toroidal group symmetry.")
    else:
        print("The sinewave stationary circuit might be unstable based on toroidal group symmetry.")
    
    if trend_behavior == "up":
        print("Designing a filter to pass only the fundamental frequency (based on toroidal group symmetry).")
        print("Designing a circuit to generate square waves (based on toroidal group symmetry).")
    elif trend_behavior == "down":
        print("Designing a filter to block higher harmonics (based on toroidal group symmetry).")
        print("Designing a circuit to generate sawtooth waves (based on toroidal group symmetry).")
    else:
        print("Neutral: No definitive trend behavior based on the last 5 dominant frequencies yet.")

# Scale current close price to sine wave
def scale_to_sine(timeframe):
    close_prices = np.random.rand(1001)  # Replace with your actual close prices array
    current_close = close_prices[-1]
        
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave
    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    print(f"For {timeframe} timeframe:")
    print(f"Distance to min: {dist_min:.2f}%")
    print(f"Distance to max: {dist_max:.2f}%")
    print(f"Current Sine value: {current_sine}\n")

    return dist_min, dist_max  # Return these values

forecasted_price = identify_reversals(close)
toroidal_group_symmetry_analysis(close)

print()

for timeframe in timeframes:
    dist_min, dist_max = scale_to_sine(timeframe)
    if dist_min < dist_max:
        print(f"For {timeframe} timeframe: Up")
    else:
        print(f"For {timeframe} timeframe: Down")

    print()

print()

##################################################
##################################################

def calculate_value_area(close):
    """
    Calculate the Value Area based on the lows and highs of the closing prices.

    Args:
    - close (list): List of closing prices for a given trading session.

    Returns:
    - value_area_low (float): Lower boundary of the Value Area.
    - value_area_high (float): Upper boundary of the Value Area.
    """
    
    # Calculate the Point of Control (POC) - the price with the highest trading volume.
    poc = max(set(close), key=close.count)
    
    # Calculate the Value Area boundaries based on 70% of the total prices
    threshold = 0.7 * len(close)
    
    # Count occurrences of each price
    price_counts = {price: close.count(price) for price in set(close)}
    
    cumulative_count = 0
    value_area_low = None
    value_area_high = None
    
    # Calculate Value Area Low
    for price, count in sorted(price_counts.items()):
        cumulative_count += count
        if cumulative_count >= threshold:
            value_area_low = price
            break

    cumulative_count = 0  # Reset cumulative count for the upper boundary
    
    # Calculate Value Area High
    for price, count in sorted(price_counts.items(), reverse=True):
        cumulative_count += count
        if cumulative_count >= threshold:
            value_area_high = price
            break
    
    return value_area_low, value_area_high


def analyze_market_profile(close):
    """
    Analyze the Market Profile to determine Value Area, Support, Resistance, Market Mood, and Forecast Price.

    Args:
    - close (list): List of closing prices for a given trading session.

    Returns:
    - value_area_low (float): Lower boundary of the Value Area.
    - value_area_high (float): Upper boundary of the Value Area.
    - support (float): Current support level.
    - resistance (float): Current resistance level.
    - market_mood (str): Market mood based on current price position relative to the Value Area.
    - forecast_price (float): Forecasted price based on Value Area.
    """
    
    # Calculate Value Area boundaries
    value_area_low, value_area_high = calculate_value_area(close)
    
    # Calculate Support and Resistance levels
    support = min(close)
    resistance = max(close)
    
    # Determine Market Mood
    current_price = close[-1]
    if current_price > value_area_low:
        market_mood = "Bearish"
    elif current_price < value_area_high:
        market_mood = "Bullish"
    else:
        market_mood = "Neutral"
    
    # Forecast Price based on Value Area
    forecast_price = (value_area_high + value_area_low) / 2  # Midpoint of Value Area
    
    return value_area_low, value_area_high, support, resistance, market_mood, forecast_price


val_low, val_high, sup, res, mood, forecast = analyze_market_profile(close)

print(f"Value Area Low: {val_low}, Value Area High: {val_high}")
print(f"Current Support: {sup}, Current Resistance: {res}")
print(f"Market Mood: {mood}")
print(f"Forecasted Price: {forecast}")

print()

##################################################
##################################################

import numpy as np
import numpy.fft as fft

def preprocess_data(close):
    # Simple preprocessing: normalize data by subtracting mean and dividing by standard deviation
    return (close - np.mean(close)) / np.std(close)

def apply_fourier_transform(close):
    # Apply Fourier Transform
    transformed_data = fft.fft(close)
    
    # Extract frequencies and magnitudes
    frequencies = fft.fftfreq(len(close))
    magnitudes = np.abs(transformed_data)  # Corrected the variable name here
    
    # Get dominant frequency (peak frequency)
    dominant_frequency = frequencies[np.argmax(magnitudes)]
    
    return dominant_frequency

def time_geometry_analysis(close):
    # Identify peaks and troughs
    peaks = np.where((close[1:-1] > close[:-2]) & (close[1:-1] > close[2:]))[0] + 1
    troughs = np.where((close[1:-1] < close[:-2]) & (close[1:-1] < close[2:]))[0] + 1
    
    return peaks, troughs

def calculate_metrics(dominant_frequency, peaks, troughs):
    # Calculate energy based on dominant frequency
    energy = np.abs(dominant_frequency)
    
    # Calculate momentum (simplified as the number of peaks and troughs)
    momentum = len(peaks) + len(troughs)
    
    # Determine reversals: if the number of peaks is greater than troughs, it's bullish; otherwise, bearish
    reversals_confirmations = "Bullish" if len(peaks) > len(troughs) else "Bearish"
    
    return energy, momentum, reversals_confirmations

# Convert the list to a numpy array for processing
close_prices = np.array(close)

# Step 1: Preprocess Data
processed_data = preprocess_data(close_prices)

# Step 2: Apply Fourier Transform
dominant_freq = apply_fourier_transform(processed_data)

# Step 3: Time Geometry Analysis
peaks, troughs = time_geometry_analysis(processed_data)

# Step 4: Calculate Metrics
energy, momentum, reversals_confirmations = calculate_metrics(dominant_freq, peaks, troughs)

# Step 5: Make Predictions
print(f"Dominant Frequency (Energy): {dominant_freq}")
print(f"Total Momentum (Peaks + Troughs): {momentum}")
print(f"Reversals Confirmation: {reversals_confirmations}")


print()

##################################################
##################################################

import numpy as np
import scipy.fftpack as fftpack

def forecast_prices(close):
    time = np.linspace(0, 1, len(close), endpoint=False)

    fast_frequency = 10
    medium_frequency = 5
    slow_frequency = 2

    fast_sinewave = np.sin(2 * np.pi * fast_frequency * time)
    medium_sinewave = 0.5 * np.sin(2 * np.pi * medium_frequency * time)
    slow_sinewave = 0.2 * np.sin(2 * np.pi * slow_frequency * time)

    combined_sinewave = fast_sinewave + medium_sinewave + slow_sinewave

    fft_output = fftpack.fft(combined_sinewave)

    dominant_frequency_index = np.argmax(np.abs(fft_output))
    fast_forecasted_frequency = dominant_frequency_index / time[-1]
    medium_forecasted_frequency = dominant_frequency_index / time[-1] * 0.5
    slow_forecasted_frequency = dominant_frequency_index / time[-1] * 0.2

    next_minute = time[-1] + 1

    fast_forecasted_sine = np.sin(2 * np.pi * fast_forecasted_frequency * next_minute)
    medium_forecasted_sine = 0.5 * np.sin(2 * np.pi * medium_forecasted_frequency * next_minute)
    slow_forecasted_sine = 0.2 * np.sin(2 * np.pi * slow_forecasted_frequency * next_minute)

    # Convert sine forecasted values back to price values
    price_min = min(close)
    price_max = max(close)

    fast_forecasted_price = price_min + 0.5 * (fast_forecasted_sine + 1) * (price_max - price_min)
    medium_forecasted_price = price_min + 0.5 * (medium_forecasted_sine + 1) * (price_max - price_min)
    slow_forecasted_price = price_min + 0.5 * (slow_forecasted_sine + 1) * (price_max - price_min)

    return fast_forecasted_price, medium_forecasted_price, slow_forecasted_price

# Forecast prices
fast_price, medium_price, slow_price = forecast_prices(close)

# Print forecasted prices
print(f"Forecasted price (fast cycle): {fast_price:.2f}")
print(f"Forecasted price (medium cycle): {medium_price:.2f}")
print(f"Forecasted price (slow cycle): {slow_price:.2f}")

print()

##################################################
##################################################

def forecast_next_hour_price(close_prices):
    """
    Forecast the price for the next hour based on the given close prices using FFT.
    
    Parameters:
    - close_prices (list): List of close prices for the timeframe.
    
    Returns:
    - dominant_trend (str): Dominant trend identified (Upward/Downward).
    - forecast_price (float): Forecasted price for the next hour.
    """
    
    # Convert the list to a numpy array for FFT processing
    close_array = np.array(close_prices)

    # Compute FFT
    fft_values = np.fft.fft(close_array)
    freq = np.fft.fftfreq(len(close_array))

    # Identify significant frequencies based on a threshold
    threshold = 100
    significant_freq_indices = np.where(np.abs(fft_values) > threshold)[0]

    # Get the last three significant frequencies
    last_three_significant_freqs = freq[significant_freq_indices][-3:]
    last_three_amplitudes = np.abs(fft_values[significant_freq_indices][-3:])
    last_three_phases = np.angle(fft_values[significant_freq_indices][-3:])

    # Calculate the average between the most negative and most positive frequencies
    most_negative_freq = last_three_significant_freqs[np.argmin(last_three_significant_freqs)]
    most_positive_freq = last_three_significant_freqs[np.argmax(last_three_significant_freqs)]
    average_freq = (most_negative_freq + most_positive_freq) / 2

    # Determine the dominant trend
    dominant_trend = "Up" if average_freq < 0 else "Down"

    # Forecast for the next hour (assuming 60 minutes in an hour)
    forecast_time = len(close_array) + 60  # Forecasting for the next hour

    # Initialize forecast
    forecast = 0

    # Use sinusoidal model for forecasting with the average frequency
    forecast += np.mean(last_three_amplitudes) * np.sin(2 * np.pi * average_freq * forecast_time + np.mean(last_three_phases))

    # Calculate the forecasted price for the next hour
    current_price = close_prices[-1]
    forecast_price = current_price + forecast

    return dominant_trend, forecast_price

dominant_trend, forecasted_price = forecast_next_hour_price(close)

print(f"Dominant Trend: {dominant_trend}")

print()

##################################################
##################################################

def impulse_momentum_overall_signal(closes):
    overall_signal = None  # Can be 'BUY', 'SELL', or None
    mean_price = sum(closes) / len(closes)

    for i in range(1, len(closes)):
        current_price = closes[i]
        previous_price = closes[i - 1]

        # Mean reversion strategy
        if current_price > mean_price:
            overall_signal = 'BUY'

        elif current_price < mean_price:
            overall_signal = 'SELL'

        # Breakout trading strategy
        if current_price > previous_price:
            overall_signal = 'BUY'

        elif current_price < previous_price:
            overall_signal = 'SELL'

        # Range trading strategy
        range_threshold = 0.02 * mean_price  # 2% range around the mean
        if current_price > mean_price + range_threshold:
            overall_signal = 'SELL'
        elif current_price < mean_price - range_threshold:
            overall_signal = 'BUY'

        # Volatility trading strategy
        price_change_percentage = ((current_price - previous_price) / previous_price) * 100
        volatility_threshold = 5  # 5% volatility threshold
        if abs(price_change_percentage) > volatility_threshold:
            if price_change_percentage > 0:
                overall_signal = 'BUY'
            else:
                overall_signal = 'SELL'

    return overall_signal

# Example usage:
closes = close

signal = impulse_momentum_overall_signal(closes)
print("Overall Signal:", signal)

print()

##################################################
##################################################

def adjust_stationary_object(min_threshold, max_threshold, reversal):
    """
    Adjust the stationary object based on the reversal.
    
    Parameters:
        min_threshold (float): Current minimum threshold.
        max_threshold (float): Current maximum threshold.
        reversal (str): Reversal value ("peak" or "dip").
    
    Returns:
        tuple: Adjusted thresholds (min_threshold, max_threshold).
    """
    # Convert reversal values to numerical values for comparison
    if reversal == "peak":
        reversal_value = max_threshold  # Assuming 'peak' corresponds to the current max_threshold
    elif reversal == "dip":
        reversal_value = min_threshold  # Assuming 'dip' corresponds to the current min_threshold
    else:
        raise ValueError("Invalid reversal value")

    if reversal_value > max_threshold:
        max_threshold = reversal_value
    elif reversal_value < min_threshold:
        min_threshold = reversal_value
    
    # Adjust the current value to remain stationary between reversals
    current_value = (min_threshold + max_threshold) / 2
    return min_threshold, max_threshold

def detect_reversals(close):
    """
    Detect peaks and troughs in the close prices to identify reversals.
    
    Parameters:
        close (list): List of close prices.
    
    Returns:
        list: List of detected reversals (peaks and dips).
    """
    reversals = []
    for i in range(1, len(close) - 1):
        if close[i] > close[i - 1] and close[i] > close[i + 1]:
            reversals.append("peak")  # Peak
        elif close[i] < close[i - 1] and close[i] < close[i + 1]:
            reversals.append("dip")  # Dip
    return reversals

def analyze_market_mood(reversals):
    """
    Analyze the last reversal to determine the market mood.
    
    Parameters:
        reversals (list): List of detected reversals (peaks and dips).
    
    Returns:
        str: Overall market mood ("up", "down").
    """
    if reversals:
        last_reversal = reversals[-1]
        if last_reversal == "dip":
            return "up"
        elif last_reversal == "peak":
            return "down"
    # Default to up if no reversals detected (for demonstration purposes)
    return "up"

reversals = detect_reversals(close)

for reversal in reversals:
    min_threshold, max_threshold = adjust_stationary_object(min_threshold, max_threshold, reversal)

market_mood_type = analyze_market_mood(reversals)

print(f"Market Mood: {market_mood_type}")

print()

##################################################
##################################################

import numpy as np
import scipy.fft as fft

def fft_trading_signal(close):
    """
    Perform FFT on the close prices and check the last three most significant frequencies for trading signals.
    
    Parameters:
    - close (list or numpy array): List or array of close prices.
    
    Returns:
    - forecast_price (float): Forecasted price based on the FFT.
    - market_mood (str): 'long', 'short', or 'neutral' based on the FFT signals.
    """
    
    # Compute the FFT of the close prices
    fft_values = fft.fft(close)
    
    # Compute the frequencies corresponding to FFT values
    frequencies = fft.fftfreq(len(close))
    
    # Sort indices of FFT values based on magnitude (excluding the first value which is DC component)
    sorted_indices = np.argsort(np.abs(fft_values[1:]))[::-1] + 1  # +1 because we excluded the DC component
    
    # Get the three most significant frequencies and their corresponding values
    top_three_indices = sorted_indices[:3]
    top_three_frequencies = frequencies[top_three_indices]
    top_three_values = fft_values[top_three_indices]
    
    # Check if the last three most significant frequencies are negative for long signal
    if np.any(top_three_values < 0):
        market_mood = 'long'
    # Check if the last three most significant frequencies are positive for short signal
    elif np.any(top_three_values > 0):
        market_mood = 'short'
    else:
        market_mood = 'neutral'
    
    # Forecast the next price based on the inverse FFT of the top three frequencies
    forecast_fft = np.zeros_like(fft_values)
    forecast_fft[top_three_indices] = top_three_values
    
    return market_mood


market_mood = fft_trading_signal(close)

print(f"Market Mood: {market_mood}")

print()

##################################################
##################################################

import numpy as np

def calculate_frequencies(close):
    # Calculate frequencies using a Fourier Transform (Replace with appropriate method)
    frequencies = np.fft.fft(close)
    return frequencies

def check_cycle_trigger(last_three_frequencies):
    # Check if the average of the last three frequencies indicates an up or down cycle
    avg_frequency = np.mean(last_three_frequencies)
    if avg_frequency > 0:
        return "Down Cycle"
    elif avg_frequency < 0:
        return "Up Cycle"
    else:
        return "Neutral"

def find_last_reversal(frequencies):
    # Identify the last reversal based on the sign of the last frequency
    if frequencies[-1] < 0:
        return "DIP Reversal"
    elif frequencies[-1] > 0:
        return "TOP Reversal"
    else:
        return "No significant reversal detected"

def identify_incoming_reversal(market_mood):
    # Identify the incoming reversal based on the current market mood
    if market_mood == "Down Cycle":
        return "Dip"
    elif market_mood == "Up Cycle":
        return "Top"
    else:
        return "No significant reversal detected"

def analyze_market_behavior(close):
    # Calculate frequencies
    frequencies = calculate_frequencies(close)
    
    # Get the last three dominant frequencies
    last_three_frequencies = frequencies[-3:]
    
    # Check trigger mechanism
    market_mood = check_cycle_trigger(last_three_frequencies)
    
    # Find the last reversal
    last_reversal = find_last_reversal(frequencies)
    
    # Identify incoming reversal based on the current market mood
    incoming_reversal = identify_incoming_reversal(market_mood)
    
    # Return the results as a dictionary
    results = {
        "Market Mood": market_mood,
        "Last Reversal": last_reversal,
        "Incoming Reversal": incoming_reversal
    }
    
    return results

# Analyze the market behavior based on the sample data
analysis_results = analyze_market_behavior(close)

# Print the results separately
print("Market Mood:", analysis_results["Market Mood"])
print("Incoming Reversal:", analysis_results["Incoming Reversal"])

print()

##################################################
##################################################

def forecast_price_and_mood(close):
    """
    Forecast future price and infer market mood based on polynomial regression.
    
    Parameters:
    - close (list): List of close prices in chronological order.
    
    Returns:
    - forecasted_price (float): Predicted price for the next period.
    - market_mood (str): Inferred market mood (e.g., 'Bullish' or 'Bearish').
    """
    
    # Generate x values (indices of close prices)
    x = np.arange(len(close))
    
    # Fit a 2nd degree polynomial (you can adjust the degree based on your requirement)
    coeffs = np.polyfit(x, close, 2)
    
    # Create a polynomial function using the coefficients
    poly_function = np.poly1d(coeffs)
    
    # Predict the next value (forecasted price)
    forecasted_price = poly_function(len(close))
    
    # Infer market mood based on the coefficient of the x^2 term
    a, _, _ = coeffs
    if a < 0:
        market_mood = 'Bullish'
    else:
        market_mood = 'Bearish'
    
    return market_mood

market_mood = forecast_price_and_mood(close)

print(f"Market Mood: {market_mood}")

print()

##################################################
##################################################

# Define the golden ratio as a constant
GOLDEN_RATIO = 1.61803398875

def calculate_phi_division():
    circumference = 360  # Total degrees in a circle
    division1 = circumference / GOLDEN_RATIO
    division2 = circumference - division1
    return division1, division2

def calculate_forecast_price(current_price):
    forecasted_price = current_price * GOLDEN_RATIO
    return forecasted_price

def calculate_intraday_target(current_price):
    # Calculate smaller target based on phi_min and phi_max
    phi_min = current_price - (0.001 * current_price)
    phi_max = current_price + (0.001 * current_price)
    intraday_target = (phi_min + phi_max) / 2  # Taking average of min and max
    return intraday_target

def calculate_momentum_target(current_price):
    # Momentum target is slightly different, you can adjust this logic as needed
    momentum_target = current_price * 1.01  # Example: 1% increase for simplicity
    return momentum_target

def determine_market_mood(current_price, target_price):
    if target_price > current_price:
        return "Bullish Market Mood: Price is expected to rise."
    elif target_price < current_price:
        return "Bearish Market Mood: Price is expected to fall."
    else:
        return "Neutral Market Mood: Price is expected to remain stable."

# Hypothetical current price (you can replace this with the actual current price)
current_price = price

# Calculate division of the circle based on the golden ratio
div1, div2 = calculate_phi_division()
print(f"Dividing the circumference by phi ({GOLDEN_RATIO}) results in approximately {div1:.2f}° and {div2:.2f}°.")

# Calculate the forecasted price using the golden ratio
forecasted_price = calculate_forecast_price(current_price)
print(f"The forecasted price based on the golden ratio is: ${forecasted_price:.2f}")

# Determine the market mood based on the forecasted price and current price
market_mood_forecast = determine_market_mood(current_price, forecasted_price)
print(market_mood_forecast)

# Calculate and print the intraday target and associated market mood
momentum_target = calculate_intraday_target(current_price)
market_mood_momentum = determine_market_mood(current_price, momentum_target)

print(f"The momentum target price is: ${momentum_target:.2f}")
print(market_mood_momentum)

# Calculate and print the momentum target and associated market mood
intraday_target = calculate_momentum_target(current_price)
market_mood_intraday = determine_market_mood(current_price, intraday_target)

print(f"The intraday target price is: ${intraday_target:.2f}")
print(market_mood_intraday)

print()

##################################################
##################################################

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression

def find_reversal_keypoints(close):
    min_val, max_val = min(close), max(close)
    reversals = [min_val, max_val]
    return reversals

def classify_trend(close):
    X = np.array(close[-30:]).reshape(-1, 1)
    y = [1] * 15 + [0] * 15
    log_model = LogisticRegression().fit(X, y)
    future_close = close[-1]
    trend = log_model.predict([[future_close]])
    if trend == 1:
        return "Up"
    else:
        return "Down"


keypoints = find_reversal_keypoints(close)
print("Reversal keypoints: ", keypoints)

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(np.array(close).reshape(-1, 1))
model = LinearRegression().fit(X_poly, close)

future = model.predict(poly_features.transform([[close[-1]]]))
print("Forecast price using polyfit in real value: ", future[0])

coefficients = np.polyfit(range(len(close)), close, 1)
slope = coefficients[0]
print("Regression slope: ", slope)

regression_mood = classify_trend(close)
print("Market trend forecast: ", regression_mood)

print()

##################################################
##################################################

def smart_money_concepts_probability(close):
    up = float('-inf')   # Highest price
    dn = float('inf')    # Lowest price
    pos = 0              # Position indicator
    liquidity_levels = []  # Liquidity levels
    reversals = []         # Reversal points
    
    for i in range(len(close)):
        current_price = close[i]
        
        # Update the highest and lowest points
        if current_price > up:
            up = current_price
            reversals.append((i, up))
            if pos <= 0:
                pos = 1
                liquidity_levels.append((i, up))
            elif pos == 1 and current_price > up:
                pos = 2
                liquidity_levels.append((i, up))
        
        elif current_price < dn:
            dn = current_price
            reversals.append((i, dn))
            if pos >= 0:
                pos = -1
                liquidity_levels.append((i, dn))
            elif pos == -1 and current_price < dn:
                pos = -2
                liquidity_levels.append((i, dn))
                
    # Calculate CHoCH and BMP percentages for support and resistance levels
    choch_support = [(price, (price - dn) / (up - dn) * 100) for _, price in liquidity_levels if (price - dn) / (up - dn) * 100 > 70] or [(dn, 0)]
    bmp_support = [(price, (price - dn) / (up - dn) * 100) for _, price in liquidity_levels if (price - dn) / (up - dn) * 100 < 30] or [(dn, 0)]
    
    choch_resistance = [(price, (up - price) / (up - dn) * 100) for _, price in liquidity_levels if (up - price) / (up - dn) * 100 > 70] or [(up, 100)]
    bmp_resistance = [(price, (up - price) / (up - dn) * 100) for _, price in liquidity_levels if (up - price) / (up - dn) * 100 < 30] or [(up, 100)]
    
    # Determine market mood based on reversals
    if up > dn:
        market_mood = "Bullish"
        forecasted_price = up  # Forecasted to go up
    else:
        market_mood = "Bearish"
        forecasted_price = dn  # Forecasted to go down
    
    return {
        'reversals': reversals,
        'liquidity_levels': liquidity_levels,
        'choch_support': choch_support,
        'bmp_support': bmp_support,
        'choch_resistance': choch_resistance,
        'bmp_resistance': bmp_resistance,
        'market_mood': market_mood,
        'forecasted_price': forecasted_price
    }

# Call the function and get the result
result_smc = smart_money_concepts_probability(close)

# Print the results
print("Reversals:", result_smc['reversals'])
print("Liquidity Levels:", result_smc['liquidity_levels'])
print("CHoCH Support:", result_smc['choch_support'])
print("BMP Support:", result_smc['bmp_support'])
print("CHoCH Resistance:", result_smc['choch_resistance'])
print("BMP Resistance:", result_smc['bmp_resistance'])
print("Market Mood:", result_smc['market_mood'])
print("Forecasted Price:", result_smc['forecasted_price'])

print()

##################################################
##################################################

print()

##################################################
##################################################

import numpy as np

def square_of_9(close):
    """
    Square of 9 calculator with Metatron concept for forecasting.
    
    Parameters:
    - close (list or numpy array): List of historical closing prices.
    
    Returns:
    - forecast_price (float): Forecasted price for the current cycle.
    - market_mood (str): Market mood based on fast cycles (HFT).
    - forecast_5min (float): Forecasted price for the next 5 minutes.
    - forecast_15min (float): Forecasted price for the next 15 minutes.
    - forecast_30min (float): Forecasted price for the next 30 minutes.
    - forecast_1h (float): Forecasted price for the next 1 hour.
    """
    
    # Calculate current min and max of close list
    current_min = np.min(close)
    current_max = np.max(close)
    
    # Calculate last_high and last_low
    last_high = current_max
    last_low = current_min
    
    # Calculate current micro cycle based on last high or last low
    current_micro_cycle = last_high - last_low if last_high > last_low else last_low - last_high
    
    # Square of 9 calculations (simplified for illustration)
    phi_pi_ratio = np.pi / np.e  # Using phi/pi as a simplified concept
    pi_hi_ratio = np.pi / current_micro_cycle  # Using pi/hi as a simplified concept
    
    # Get the latest close price
    latest_close = close[-1]
    
    # Metatron concept with reversals (simplified)
    if current_micro_cycle > 0:
        forecast_price = last_high + (phi_pi_ratio * current_micro_cycle)
        market_mood = "Bullish" if forecast_price > latest_close else "Bearish"
    else:
        forecast_price = last_low - (pi_hi_ratio * current_micro_cycle)
        market_mood = "Bearish" if forecast_price < latest_close else "Bullish"
    
    # Calculate SMA for faster cycle forecasts
    forecast_5min = np.mean(close[-5:])  # 5-minute SMA forecast
    forecast_15min = np.mean(close[-15:])  # 15-minute SMA forecast
    forecast_30min = np.mean(close[-30:])  # 30-minute SMA forecast
    forecast_1h = np.mean(close[-60:])  # 1-hour SMA forecast
    
    return forecast_price, market_mood, forecast_5min, forecast_15min, forecast_30min, forecast_1h

forecast_price, market_mood, forecast_5min, forecast_15min, forecast_30min, forecast_1h = square_of_9(close)

print(f"Forecasted Price (Current Cycle): {forecast_price}")
print(f"Market Mood (Current Cycle): {market_mood}")
print(f"Forecast for Next 5 Minutes: {forecast_5min}")
print(f"Forecast for Next 15 Minutes: {forecast_15min}")
print(f"Forecast for Next 30 Minutes: {forecast_30min}")
print(f"Forecast for Next 1 Hour: {forecast_1h}")


print()

##################################################
##################################################

import numpy as np

def analyze_fft_for_hexagon(close):
    """
    Analyze the FFT of the price data to calculate hexagon symmetry, forecasted price, and market mood.
    
    Parameters:
    - close (array-like): Input close price data.
    
    Returns:
    - forecast_price (float): Forecasted price value.
    - market_mood (str): Predicted market mood ("Up" or "Down").
    """
    
    # Convert close to a NumPy array if it's not already an array-like structure
    close_array = np.array(close) if hasattr(close, '__iter__') else np.array([close])
    
    # Check if the array has at least two elements for meaningful FFT analysis
    if len(close_array) < 2:
        raise ValueError("Close data should contain at least two elements for meaningful analysis.")
    
    # Compute FFT of the close data
    fft_values = np.fft.fft(close_array)
    
    # Calculate the frequency components
    freqs = np.fft.fftfreq(len(fft_values))
    
    # Find indices corresponding to the most negative and most positive frequencies
    most_positive_index = np.argmax(freqs)
    
    # Calculate forecasted price based on a more realistic transformation
    # For simplicity, let's take the mean of the last few closing prices as the forecasted price
    forecast_price = np.mean(close_array[-5:])  # Adjust the number of elements as needed
    
    # Determine market mood based on the imaginary part of the FFT at the most positive frequency
    market_mood = "Up" if np.imag(fft_values[most_positive_index]) > 0 else "Down"
    
    return market_mood

predicted_market_mood = analyze_fft_for_hexagon(close)

print("Predicted Market Mood:", predicted_market_mood)

print()

##################################################
##################################################

import math

def trend_forecast(close):
    # Calculate fast cycle trend
    fast_cycle_trend = 0
    for i in range(2, len(close)):
        fast_cycle_trend += (close[i] - close[i-1])
    fast_cycle_trend /= (len(close) - 2)

    # Calculate medium range trend
    medium_range_trend = 0
    for i in range(8, len(close)):
        medium_range_trend += (close[i] - close[i-8])
    medium_range_trend /= (len(close) - 8)

    # Calculate big trend
    big_trend = 0
    for i in range(32, len(close)):
        big_trend += (close[i] - close[i-32])
    big_trend /= (len(close) - 32)

    # Determine trend direction based on all three time scales
    if fast_cycle_trend > 0 and medium_range_trend > 0 and big_trend > 0:
        return "Up"
    elif fast_cycle_trend < 0 and medium_range_trend < 0 and big_trend < 0:
        return "Down"
    else:
        return None  # No neutral, just return None for cases other than Up or Down

result_cycles = trend_forecast(close)

print(result_cycles)

print()

##################################################
##################################################

print()

##################################################
##################################################

import numpy as np

def forecast_market_trends(close):
    # Step 1: Analyze market sentiment (considering price trend)
    sentiment = np.sign(np.diff(close)[-1])
    # Step 2: Determine market quadrant based on time geometry
    time_phase = len(close) % 4  # Use the remainder to determine the phase
    quadrant_mapping = {0: "Bullish", 1: "Transition to Bearish", 2: "Bearish", 3: "Transition to Bullish"}
    market_quadrant = quadrant_mapping[time_phase]
    # Step 3: Identify potential price levels
    support_level = min(close)
    resistance_level = max(close)
    # Step 4: Determine if close is above or below 45-degree angle
    angle_threshold = 45
    above_45_degree = close[-1] > angle_threshold
    below_45_degree = close[-1] < -angle_threshold
    # Step 5: Forecast market mood and price
    if above_45_degree:
        market_mood = "Bearish"
        forecasted_price = support_level - (resistance_level - support_level)
    elif below_45_degree:
        market_mood = "Bullish"
        forecasted_price = resistance_level + (resistance_level - support_level)
    else:
        market_mood = "Neutral"
        forecasted_price = (support_level + resistance_level) / 2
    return sentiment, market_quadrant, support_level, resistance_level, market_mood, forecasted_price

sentiment, market_quadrant, support_level, resistance_level, market_mood, forecasted_price = forecast_market_trends(close)

# Print the results
print(f"Market Sentiment: {sentiment}")
print(f"Market Quadrant: {market_quadrant}")
print(f"Support Level: {support_level}")
print(f"Resistance Level: {resistance_level}")
print(f"Market Mood: {market_mood}")
print(f"Forecasted Price: {forecasted_price}")

print()

##################################################
##################################################

def calculate_pivot_point_and_forecast(close):
    high = max(close)
    low = min(close)
    close_price = close[-1]

    pivot_point = (high + low + close_price) / 3
    support_1 = (pivot_point * 2) - high
    resistance_1 = (pivot_point * 2) - low
    support_2 = pivot_point - (high - low)
    resistance_2 = pivot_point + (high - low)

    market_mood = "Bullish" if close_price < pivot_point else "Bearish"

    forecast_price = resistance_1 if market_mood == "Bullish" else support_1

    result = {
        'Pivot Point': pivot_point,
        'Support 1': support_1,
        'Resistance 1': resistance_1,
        'Support 2': support_2,
        'Resistance 2': resistance_2,
        'Market Mood': market_mood,
        'Forecast Price': forecast_price
    }

    return result

pivot_and_forecast = calculate_pivot_point_and_forecast(close)

# Print statements moved outside the function
print(f"Market Mood: {pivot_and_forecast['Market Mood']}")
print(f"Forecast Price: {pivot_and_forecast['Forecast Price']}")

print()

##################################################
##################################################

import talib
import numpy as np

def get_rsi(timeframe):
    """Calculate RSI for a single timeframe"""
    # Get candle data
    candles = candle_map[timeframe][-100:]
    # Calculate RSI using talib RSI
    rsi = talib.RSI(np.array([c["close"] for c in candles]), timeperiod=14)
    return rsi[-1]

# Calculate RSI for each timeframe
rsi_values = {}
for timeframe in timeframes:
    rsi = get_rsi(timeframe)
    rsi_values[timeframe] = rsi
    print(f"RSI for {timeframe}: {rsi}")

# Convert RSI to a normalized scale and determine if it's positive or negative
normalized_rsi = {}
for timeframe, rsi in rsi_values.items():
    normalized_value = (rsi - 30) / 70 * 100  # Normalize to a scale between 0 and 100
    normalized_rsi[timeframe] = normalized_value
    print(f"Normalized RSI for {timeframe}: {normalized_value:.2f}%")

# Calculate dominant ratio
positive_rsi_count = sum(1 for value in normalized_rsi.values() if value < 30)
negative_rsi_count = sum(1 for value in normalized_rsi.values() if value > 70)

print(f"Positive RSI timeframes: {positive_rsi_count}/{len(normalized_rsi)}")
print(f"Negative RSI timeframes: {negative_rsi_count}/{len(normalized_rsi)}")

if positive_rsi_count > negative_rsi_count:
    print("Overall dominant RSI: Positive")
elif positive_rsi_count < negative_rsi_count:
    print("Overall dominant RSI: Negative")
else:
    print("Overall dominant RSI: Balanced")

print()

##################################################
##################################################

import numpy as np
import talib

def scale_list_to_sine(close):
    # Convert the close list to a NumPy array
    close_np = np.array(close, dtype=float)
    
    # Get last close price 
    current_close = close_np[-1]      
        
    # Calculate sine wave        
    sine_wave, _ = talib.HT_SINE(close_np)
            
    # Replace NaN values with 0        
    sine_wave = np.nan_to_num(sine_wave)
    #sine_wave = -sine_wave
        
    # Get the sine value for last close      
    current_sine = sine_wave[-1]
            
    # Calculate the min and max sine           
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # Calculate % distances   
    dist_min = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100         
    dist_max = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100            

    return dist_min, dist_max, current_sine
      

# Call the scale_list_to_sine function
dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_list_to_sine(close)

# Print the results
print("For the given close prices:")
print(f"Distance to min: {dist_from_close_to_min:.2f}%")
print(f"Distance to max: {dist_from_close_to_max:.2f}%")
print(f"Current Sine value: {current_sine}\n")


print()

##################################################
##################################################

import talib
import numpy as np

def analyze_market(close):
    # Convert close to numpy array
    close = np.array(close)

    # Calculate ROC
    roc_values = talib.ROC(close, timeperiod=1)

    # Identify lows and highs
    lows = np.where(roc_values < 0)[0]
    highs = np.where(roc_values > 0)[0]

    # Determine the last reversal
    last_reversal = "low" if lows[-1] < highs[-1] else "high"

    # Determine market mood
    market_mood = "bullish" if roc_values[-1] > 0 else "bearish"

    # Calculate Gann progression for the future
    fibo_scale = 0.618  # Fibonacci scale for Gann progression
    angle_degrees = 45  # 45-degree angle for Gann progression

    # Calculate projected ratio to mean into the future for HFT
    hft_future_ratio = 1.2  # Adjust as needed for HFT
    hft_projected_price = close[-1] * (1 + hft_future_ratio * roc_values[-1])

    # Calculate projected ratio to mean into the future
    standard_future_ratio = 1.618  # Standard ratio
    last_reversal_price = close[lows[-1]] if last_reversal == "low" else close[highs[-1]]
    standard_projected_price = last_reversal_price * standard_future_ratio

    # ROC forecast price for even faster targets
    faster_forecast_ratio = 0.8  # Adjust as needed for faster targets
    roc_faster_forecast_price = close[-1] * (1 + faster_forecast_ratio * roc_values[-1])

    # Return relevant information
    return {
        "roc_value": roc_values[-1],
        "market_mood": market_mood,
        "last_reversal": last_reversal,
        "last_reversal_price": last_reversal_price,
        "hft_projected_price": hft_projected_price,
        "standard_projected_price": standard_projected_price,
        "roc_faster_forecast_price": roc_faster_forecast_price
    }

# Example usage:
analysis_result = analyze_market(close)

# Print information outside the function
print(f"Current ROC value: {analysis_result['roc_value']}")
print(f"Current market mood: {analysis_result['market_mood']}")
print(f"Last reversal was a {analysis_result['last_reversal']} at price: {analysis_result['last_reversal_price']}")
print(f"HFT Projected price: {analysis_result['hft_projected_price']}")  # Added line for HFT projected price
print(f"Standard Projected price: {analysis_result['standard_projected_price']}")  # Added line for standard projected price
print(f"ROC forecast price for even faster targets: {analysis_result['roc_faster_forecast_price']}")

print()

##################################################
##################################################

import talib
import numpy as np

def market_dom_analysis(close):
    close_np = np.array(close)  # Convert to numpy array
    # Hilbert Transform - Dominant Cycle Period
    dc_period = talib.HT_DCPERIOD(close_np)
    
    # Hilbert Transform - Dominant Cycle Phase
    dc_phase = talib.HT_DCPHASE(close_np)
    
    # Hilbert Transform - Phasor Components
    inphase, quadrature = talib.HT_PHASOR(close_np)
    
    # Hilbert Transform - SineWave
    sine, leadsine = talib.HT_SINE(close_np)
    
    # Hilbert Transform - Trend vs Cycle Mode
    trend_mode = talib.HT_TRENDMODE(close_np)
    
    # Combine conditions to determine market mood
    market_mood = determine_dom_market_mood(dc_period, dc_phase, trend_mode)
    
    # Forecasting logic based on conditions
    price_forecast = forecast_price(dc_period, dc_phase, inphase, quadrature, sine, leadsine, trend_mode)
    
    return market_mood, price_forecast

def determine_dom_market_mood(dc_period, dc_phase, trend_mode):
    # Add your logic to determine market mood based on the given conditions
    # You can customize these conditions based on your strategy
    if np.any(trend_mode == -1):
        return "Bullish"
    elif np.any(trend_mode == 1):
        return "Bearish"
    else:
        return "Neutral"

def forecast_price(dc_period, dc_phase, inphase, quadrature, sine, leadsine, trend_mode):
    # Add your logic to forecast price based on the given conditions
    # You can customize these conditions based on your strategy
    if np.any(trend_mode == -1):
        # Bullish forecast logic
        return "Expect higher prices"
    elif np.any(trend_mode == 1):
        # Bearish forecast logic
        return "Expect lower prices"
    else:
        # Neutral forecast logic
        return "No clear price trend"

# Example usage
dom_mood, dom_forecast = market_dom_analysis(close)

# Print results outside the function
print(f"Market Mood: {dom_mood}")
print(f"Price Forecast: {dom_forecast}")

print()

##################################################
##################################################

import numpy as np

def forecast_unit_price(close):
    center = np.mean(close)
    min_close = np.min(close)
    max_close = np.max(close)
    radius = (max_close - center) if abs(max_close - center) > abs(center - min_close) else (center - min_close)
    angle = np.arctan2(center - close[-1], radius)
    quadrant = int(np.floor((angle + np.pi/4) / (np.pi/2))) % 4
    adjusted_angle = angle - quadrant * (np.pi/2)
    
    if quadrant in [0, 2]:
        forecasted_price = center + radius * np.sin(adjusted_angle)
        market_mood = "Down"
    else:
        forecasted_price = center + radius * np.cos(adjusted_angle)
        market_mood = "Up"
    
    return forecasted_price, market_mood

# Example usage:
unitcircle_price, unitcircle_mood = forecast_unit_price(close)

print(f"Forecasted Price: {unitcircle_price:.2f}")
print(f"Market Mood: {unitcircle_mood}")

print()

##################################################
##################################################

import numpy as np
import talib
import concurrent.futures

def scale_to_sine(timeframe, close_prices):
    current_close = close_prices[-1]

    sine_wave, leadsine = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    #sine_wave = -sine_wave

    current_sine = sine_wave[-1]
    sine_wave_min, sine_wave_max = np.min(sine_wave), np.max(sine_wave)

    # Calculate distances as percentages
    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    # Determine sentiment based on conditions
    if dist_from_close_to_min < dist_from_close_to_max and dist_from_close_to_min < 50:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    return timeframe, dist_from_close_to_min, dist_from_close_to_max, current_sine, sentiment

# Initialize overall_sentiments_sine dictionary
overall_sentiments_sine = {'Positive': 0, 'Negative': 0}

# Define a function to process a single timeframe
def process_timeframe(timeframe):
    close_prices = np.array(get_close(timeframe))
    return scale_to_sine(timeframe, close_prices)

# Create a ThreadPoolExecutor for parallel execution
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Create a list of futures for each timeframe
    futures = {executor.submit(process_timeframe, timeframe): timeframe for timeframe in timeframes}

    # Iterate over completed futures to process results
    for future in concurrent.futures.as_completed(futures):
        timeframe, dist_from_close_to_min, dist_from_close_to_max, current_sine, sentiment = future.result()

        # Update overall sentiment count
        overall_sentiments_sine[sentiment] += 1

        # Print the results for each timeframe
        print(f"For {timeframe} timeframe:")
        print(f"Distance to min: {dist_from_close_to_min:.2f}%")
        print(f"Distance to max: {dist_from_close_to_max:.2f}%")
        print(f"Current Sine value: {current_sine}")
        print(f"Sentiment: {sentiment}\n")

# Print overall sentiment analysis
for sentiment, count in overall_sentiments_sine.items():
    print(f"Overall Market Sentiment for {sentiment}: {count}")

# Determine the overall dominant sentiment for the Sine Wave
positive_sine_count = overall_sentiments_sine['Positive']
negative_sine_count = overall_sentiments_sine['Negative']

if positive_sine_count > negative_sine_count:
    print("Overall dominant Sine Wave: Positive")
elif positive_sine_count < negative_sine_count:
    print("Overall dominant Sine Wave: Negative")
else:
    print("Overall dominant Sine Wave: Balanced")


print()

##################################################
##################################################

def calculate_volume(candles):
    total_volume = sum(candle["volume"] for candle in candles)
    return total_volume

def get_volume_5min(candles):
    total_volume_5min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "5m")
    return total_volume_5min

def get_volume_3min(candles):
    total_volume_3min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "3m")
    return total_volume_3min

def get_volume_1min(candles):
    total_volume_1min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "1m")
    return total_volume_1min

def calculate_buy_sell_volume(candles):
    buy_volume_5min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "5m" and candle["close"] > candle["open"])
    sell_volume_5min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "5m" and candle["close"] < candle["open"])
    buy_volume_3min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "3m" and candle["close"] > candle["open"])
    sell_volume_3min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "3m" and candle["close"] < candle["open"])
    buy_volume_1min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "1m" and candle["close"] > candle["open"])
    sell_volume_1min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "1m" and candle["close"] < candle["open"])

    return buy_volume_5min, sell_volume_5min, buy_volume_3min, sell_volume_3min , buy_volume_1min, sell_volume_1min 

def calculate_support_resistance(candles):
    support_levels_1min = []
    resistance_levels_1min = []
    support_levels_3min = []
    resistance_levels_3min = []
    support_levels_5min = []
    resistance_levels_5min = []

    timeframes = ["1m", "3m", "5m"]
    for timeframe in timeframes:
        close_prices = [candle["close"] for candle in candles if candle["timeframe"] == timeframe]
        min_close = min(close_prices)
        max_close = max(close_prices)
        price_range = max_close - min_close
        support_spread = price_range / 11  # Divide the price range into 11 equal parts for 10 support and resistance levels
        resistance_spread = price_range / 11

        if timeframe == "1m":
            support_levels_1min.append(min_close - support_spread)
            resistance_levels_1min.append(max_close + resistance_spread)
            for _ in range(4):
                support_levels_1min.append(support_levels_1min[-1] - support_spread)
                resistance_levels_1min.append(resistance_levels_1min[-1] + resistance_spread)
        elif timeframe == "3m":
            support_levels_3min.append(min_close - support_spread)
            resistance_levels_3min.append(max_close + resistance_spread)
            for _ in range(4):
                support_levels_3min.append(support_levels_3min[-1] - support_spread)
                resistance_levels_3min.append(resistance_levels_3min[-1] + resistance_spread)
        else:
            support_levels_5min.append(min_close - support_spread)
            resistance_levels_5min.append(max_close + resistance_spread)
            for _ in range(4):
                support_levels_5min.append(support_levels_5min[-1] - support_spread)
                resistance_levels_5min.append(resistance_levels_5min[-1] + resistance_spread)

    return support_levels_1min, resistance_levels_1min, support_levels_3min, resistance_levels_3min, support_levels_5min, resistance_levels_5min

def calculate_reversal_keypoints(levels, leverage):
    reversal_points = []
    for level in levels:
        reversal = level * (1 - 0.1 * leverage)
        reversal_points.append(reversal)
    return reversal_points

def get_higher_timeframe_data(symbol, higher_timeframe):
    higher_candles = get_candles(symbol, [higher_timeframe])

    if not higher_candles or higher_timeframe not in higher_candles:
        return [], []

    higher_support_levels, higher_resistance_levels = calculate_support_resistance(higher_candles[higher_timeframe])
    return higher_support_levels, higher_resistance_levels

def calculate_bollinger_bands(candles, window=20, num_std_dev=2):
    close_prices = [candle["close"] for candle in candles if candle["timeframe"] == "5m"]
    rolling_mean = calculate_rolling_mean(close_prices, window)
    rolling_std = calculate_rolling_std(close_prices, window)
    upper_band = [mean + (std_dev * num_std_dev) for mean, std_dev in zip(rolling_mean, rolling_std)]
    lower_band = [mean - (std_dev * num_std_dev) for mean, std_dev in zip(rolling_mean, rolling_std)]
    return upper_band, lower_band

def calculate_rolling_mean(data, window):
    rolling_sum = [sum(data[i - window + 1:i + 1]) for i in range(window - 1, len(data))]
    return [sum / window for sum in rolling_sum]

def calculate_rolling_std(data, window):
    rolling_std = []
    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1:i + 1]
        mean = sum(window_data) / window
        squared_diffs = [(x - mean) ** 2 for x in window_data]
        std_dev = (sum(squared_diffs) / window) ** 0.5
        rolling_std.append(std_dev)
    return rolling_std

def calculate_poly_channel(candles, window=20):
    close_prices = [candle["close"] for candle in candles if candle["timeframe"] == "1m"]
    poly_channel = np.polyfit(range(len(close_prices)), close_prices, 1)
    channel = np.polyval(poly_channel, range(len(close_prices)))
    upper_channel = channel + np.std(channel) * window
    lower_channel = channel - np.std(channel) * window
    return upper_channel.tolist(), lower_channel.tolist()

def check_bulls_vs_bears_vol(candles):
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    overall_bulls_vol = 0
    overall_bears_vol = 0

    for timeframe in timeframes:
        bulls_vol = sum(candle["volume"] for candle in candles if candle["timeframe"] == timeframe and candle["close"] > candle["open"])
        bears_vol = sum(candle["volume"] for candle in candles if candle["timeframe"] == timeframe and candle["close"] < candle["open"])

        overall_bulls_vol += bulls_vol
        overall_bears_vol += bears_vol

    return overall_bulls_vol, overall_bears_vol


total_volume = calculate_volume(candles)
buy_volume_5min, sell_volume_5min, buy_volume_3min, sell_volume_3min , buy_volume_1min, sell_volume_1min = calculate_buy_sell_volume(candles)

(support_levels_1min, resistance_levels_1min, support_levels_3min, resistance_levels_3min, support_levels_5min, resistance_levels_5min) = calculate_support_resistance(candles)

total_volume_5min = get_volume_5min(candles)
total_volume_3min = get_volume_3min(candles)
total_volume_1min = get_volume_1min(candles)

small_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 2)
medium_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 5)
large_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 10)

small_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 2)
medium_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 5)
large_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 10)

small_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 2)
medium_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 5)
large_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 10)

higher_support_5min, higher_resistance_5min = get_higher_timeframe_data(TRADE_SYMBOL, "5m")

print("Total Volume:", total_volume)
print("Total Volume (5min tf):", total_volume_5min)

print()

print("Buy Volume (5min tf):", buy_volume_5min)
print("Sell Volume (5min tf):", sell_volume_5min)

print()

print("Buy Volume (1min tf):", buy_volume_1min)
print("Sell Volume (1min tf):", sell_volume_1min)

print()

##################################################
##################################################

def check_volume_trend(candles):
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    
    volume_trend_data = {}

    for timeframe in timeframes:
        volume_data = [candle["volume"] for candle in candles if candle["timeframe"] == timeframe]

        if len(volume_data) > 1:
            current_volume = volume_data[-1]
            previous_volume = volume_data[-2]

            if current_volume > previous_volume:
                trend = "Increasing"
            elif current_volume < previous_volume:
                trend = "Decreasing"
            else:
                trend = "Stable"

            volume_trend_data[timeframe] = {
                "current_volume": current_volume,
                "previous_volume": previous_volume,
                "trend": trend
            }
        else:
            volume_trend_data[timeframe] = {
                "current_volume": 0,
                "previous_volume": 0,
                "trend": "N/A (Insufficient Data)"
            }

    return volume_trend_data

# Add this function call after obtaining the candles data
volume_trend_data = check_volume_trend(candles)

# Print the volume trend data
print("Volume Trend Data:")
for timeframe, data in volume_trend_data.items():
    print(f"{timeframe} timeframe:")
    print(f"Current Volume: {data['current_volume']}")
    print(f"Previous Volume: {data['previous_volume']}")
    print(f"Trend: {data['trend']}")
    print()

print()

##################################################
##################################################

def check_price_trend(candles):
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    
    price_trend_data = {}

    for timeframe in timeframes:
        close_prices = [candle["close"] for candle in candles if candle["timeframe"] == timeframe]

        if len(close_prices) > 1:
            current_close = close_prices[-1]
            previous_close = close_prices[-2]

            if current_close > previous_close:
                trend = "Increasing"
            elif current_close < previous_close:
                trend = "Decreasing"
            else:
                trend = "Stable"

            price_trend_data[timeframe] = {
                "current_close": current_close,
                "previous_close": previous_close,
                "trend": trend
            }
        else:
            price_trend_data[timeframe] = {
                "current_close": 0,
                "previous_close": 0,
                "trend": "N/A (Insufficient Data)"
            }

    return price_trend_data

# Add this function call after obtaining the candles data
price_trend_data = check_price_trend(candles)

# Print the price trend data
print("Price Trend Data:")
for timeframe, data in price_trend_data.items():
    print(f"{timeframe} timeframe:")
    print(f"Current Close: {data['current_close']}")
    print(f"Previous Close: {data['previous_close']}")
    print(f"Trend: {data['trend']}")
    print()

print()

##################################################
##################################################

def check_overall_trend(candles):
    volume_trend_data = check_volume_trend(candles)
    price_trend_data = check_price_trend(candles)

    overall_trend_data = {}

    up_count = 0
    down_count = 0
    same_direction_count = 0

    for timeframe in volume_trend_data.keys():
        if timeframe not in price_trend_data:
            continue

        volume_trend = volume_trend_data[timeframe]["trend"]
        price_trend = price_trend_data[timeframe]["trend"]

        overall_trend = "Up" if (volume_trend == "Increasing" and price_trend == "Increasing") or (volume_trend == "Decreasing" and price_trend == "Decreasing") else "Down" if (volume_trend == "Decreasing" and price_trend == "Increasing") or (volume_trend == "Increasing" and price_trend == "Decreasing") else "Stable"

        overall_trend_data[timeframe] = {
            "volume_trend": volume_trend,
            "price_trend": price_trend,
            "overall_trend": overall_trend
        }

        # Count trends
        if overall_trend == "Up":
            up_count += 1
        elif overall_trend == "Down":
            down_count += 1

        # Check if volume and price trends are in the same direction
        if volume_trend == price_trend:
            same_direction_count += 1

    # Calculate overall trend for all timeframes
    overall_volume_trend = "Up" if all("volume_trend" in data and data["volume_trend"] == "Increasing" for data in volume_trend_data.values()) else "Down" if all("volume_trend" in data and data["volume_trend"] == "Decreasing" for data in volume_trend_data.values()) else "Stable"
    overall_price_trend = "Up" if all("price_trend" in data and data["price_trend"] == "Increasing" for data in price_trend_data.values()) else "Down" if all("price_trend" in data and data["price_trend"] == "Decreasing" for data in price_trend_data.values()) else "Stable"

    overall_trend_data["Overall"] = {
        "volume_trend": overall_volume_trend,
        "price_trend": overall_price_trend,
        "overall_trend": "Up" if (overall_volume_trend == "Up" and overall_price_trend == "Up") or (overall_volume_trend == "Down" and overall_price_trend == "Down") else "Down"
    }

    # Print counts
    print(f"Number of Timeframes in Up Direction: {up_count}")
    print(f"Number of Timeframes in Down Direction: {down_count}")
    print(f"Number of Timeframes with Both Volume and Price in Same Direction: {same_direction_count}")

    return overall_trend_data

# Add this function call after obtaining the candles data
overall_trend_data = check_overall_trend(candles)

# Print the volume and price trends separately for each timeframe
print("\nVolume and Price Trend Data:")
for timeframe, data in overall_trend_data.items():
    print(f"{timeframe} timeframe:")
    
    # Volume Trend
    print(f"Volume Trend: {data.get('volume_trend', 'N/A')}")

    # Price Trend
    print(f"Price Trend: {data.get('price_trend', 'N/A')}")

    # Overall Trend
    print(f"Overall Trend: {data['overall_trend']}")
    
    print()

print()

##################################################
##################################################

import numpy as np

def market_analysis(close, closest_threshold, min_threshold, max_threshold):
    q1_range = [0, 0.25]
    q2_range = [0.25, 0.5]
    q3_range = [0.5, 0.75]
    q4_range = [0.75, 1.0]

    # Determine current quadrant and percentage to min and max reversals
    if min_threshold <= closest_threshold <= max_threshold:
        percentage_to_reversal = (closest_threshold - min_threshold) / (max_threshold - min_threshold)
        if closest_threshold <= min_threshold + (max_threshold - min_threshold) * q1_range[1]:
            current_quadrant = '1'
        elif min_threshold + (max_threshold - min_threshold) * q2_range[0] < closest_threshold <= min_threshold + (max_threshold - min_threshold) * q2_range[1]:
            current_quadrant = '2'
        elif min_threshold + (max_threshold - min_threshold) * q3_range[0] < closest_threshold <= min_threshold + (max_threshold - min_threshold) * q3_range[1]:
            current_quadrant = '3'
        elif min_threshold + (max_threshold - min_threshold) * q4_range[0] < closest_threshold <= min_threshold + (max_threshold - min_threshold) * q4_range[1]:
            current_quadrant = '4'
    else:
        raise ValueError("Close value is outside the specified thresholds.")

    # Determine if the last reversal was at the minimum threshold
    last_reversal_at_min_threshold = closest_threshold <= min_threshold + (max_threshold - min_threshold) * 0.5

    # Calculate degree to mean ratio related to a 45-degree angle
    close_range = np.max(close) - np.min(close)
    mean_price = np.mean(close)
    degree_to_mean_ratio = close_range / mean_price

    # Use the new logic for market mood and forecast using the hybrid wave
    average_price = np.mean(close)
    price_change = np.diff(close)
    mid_point = (percentage_to_reversal + 0.5) / 2

    if last_reversal_at_min_threshold:
        current_cycle = np.where(close <= mid_point, 0, 1)  # 0 for 'Down Cycle', 1 for 'Up Cycle'
    else:
        current_cycle = np.where(close <= mid_point, 1, 0)  # 1 for 'Up Cycle', 0 for 'Down Cycle'

    majority_cycle = np.argmax(np.bincount(current_cycle.astype(int)))

    mood_factor = degree_to_mean_ratio * np.mean(price_change) * (1 - percentage_to_reversal)
    forecast_price = average_price + mood_factor

    return majority_cycle, current_quadrant, mood_factor, forecast_price

current_cycle, current_quadrant, mood_factor, forecasted_price = market_analysis(close_prices, closest_threshold, min_threshold, max_threshold)

# Print results
print(f"Current Cycle: {current_cycle}")
print(f"Current Quadrant: {current_quadrant}")
print(f"Mood Factor: {mood_factor}")
print(f"Forecasted Price: {forecasted_price}")


print()

##################################################
##################################################

def calculate_impulse_energy_momentum(close):
    """
    Calculate impulse, energy, and momentum based on close prices.

    Parameters:
    close (list or numpy array): List or array containing historical closing prices.

    Returns:
    dict: Dictionary containing calculated impulse, energy, momentum, and market mood.
    """
    if len(close) < 2:
        raise ValueError("Insufficient data for calculations. At least two data points required.")

    # Calculate impulse
    impulse = close[-1] - close[-2]

    # Determine market mood
    market_mood = "Up" if impulse > 0 else "Down" if impulse < 0 else "Neutral"

    # Calculate energy (squared impulse)
    energy = impulse ** 2

    # Calculate momentum (normalized impulse)
    momentum = impulse / close[-2]

    result = {
        'impulse': impulse,
        'energy': energy,
        'momentum': momentum,
        'market_mood': market_mood
    }

    return result

def forecast_hft_price(current_price, calculated_momentum):
    """
    Forecast the next price based on current price and calculated momentum.

    Parameters:
    current_price (float): Current market price.
    calculated_momentum (float): Calculated momentum.

    Returns:
    float: Forecasted next price.
    """
    forecasted_price = current_price + calculated_momentum * current_price
    return forecasted_price

# Calculate impulse, energy, momentum, and market mood
calculations = calculate_impulse_energy_momentum(close)
print("Impulse:", calculations['impulse'])
print("Energy:", calculations['energy'])
print("Momentum:", calculations['momentum'])
print("Market Mood:", calculations['market_mood'])

# Forecast the next price
current_price = price
forecasted_price = forecast_hft_price(current_price, calculations['momentum'])
print("Forecasted Price:", forecasted_price)

print()

##################################################
##################################################

def get_binance_futures_order_book_with_indicators(symbol="btcusdt", limit=5, forecast_minutes=60):
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/depth"

    params = {
        "symbol": symbol,
        "limit": limit
    }

    try:
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()
        order_book = response.json()

        # Get relevant information
        bids = order_book["bids"]
        asks = order_book["asks"]

        # Combine bids and asks to form a single list of prices
        close_prices = np.array([float(price[0]) for price in bids + asks])

        # Check for NaN values in the combined prices
        if any(np.isnan(close_prices)):
            raise ValueError("Input data contains NaN values.")

        spread = float(asks[0][0]) - float(bids[0][0])
        close_price = (float(bids[0][0]) + float(asks[0][0])) / 2

        # Calculate support and resistance levels
        support_level = float(bids[0][0])
        resistance_level = float(asks[0][0])

        # Calculate MACD and RSI
        macd, signal, _ = talib.MACD(close_prices)
        rsi = talib.RSI(close_prices)

        # Assess market mood for small range (next 5 minutes)
        small_range_mood = "Unknown"
        if spread > 0:
            small_range_forecasted_price = close_price + spread
            small_range_mood = "Bullish" if small_range_forecasted_price > close_price else "Bearish"

        # Assess market mood for large range (next 30 minutes)
        large_range_mood = "Unknown"
        if spread > 0:
            large_range_forecasted_price = close_price + spread * 6  # Forecast for 30 minutes
            large_range_mood = "Bullish" if large_range_forecasted_price > close_price else "Bearish"

        # Calculate a basic forecasted price for the next 60 minutes
        forecasted_price = (support_level + resistance_level) / 2

        # Calculate the forecasted price for the most distant future (maximum duration)
        max_forecast_minutes = forecast_minutes
        forecasted_price_max_duration = forecasted_price + spread * max_forecast_minutes

        # Calculate buy and sell orders for gradually bigger spreads
        spread_sizes = [1.0, 50, 100.0]  # Example spread sizes, you can adjust as needed

        orders = {}
        for spread_size in spread_sizes:
            buy_order_price = support_level - spread_size
            sell_order_price = resistance_level + spread_size

            orders[f'buy_order_price_{spread_size}'] = buy_order_price
            orders[f'sell_order_price_{spread_size}'] = sell_order_price

        return {
            "buy_order_price": float(bids[0][0]),
            "buy_order_quantity": float(bids[0][1]),
            "sell_order_price": float(asks[0][0]),
            "sell_order_quantity": float(asks[0][1]),
            "spread": spread,
            "support_level": support_level,
            "resistance_level": resistance_level,
            "small_range_mood": small_range_mood,
            "large_range_mood": large_range_mood,
            "forecasted_price_max_duration": forecasted_price_max_duration,
            "max_forecast_minutes": max_forecast_minutes,
            "macd": macd[-1],  # Use the latest MACD value for signal enforcement
            "rsi": rsi[-1],    # Use the latest RSI value for signal enforcement
            "spread_orders": orders  # Include the calculated buy and sell orders
        }

    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Error:", err)

# Example usage without a while loop
order_book_data = get_binance_futures_order_book_with_indicators(symbol="btcusdt", limit=5, forecast_minutes=1440)

# Print information separately
if order_book_data:
    print("Buy Order:")
    print(f"Price: {order_book_data['buy_order_price']}, Quantity: {order_book_data['buy_order_quantity']}")

    print("\nSell Order:")
    print(f"Price: {order_book_data['sell_order_price']}, Quantity: {order_book_data['sell_order_quantity']}")

    print("\nSpread:", order_book_data['spread'])

    print("\nSupport Level:", order_book_data['support_level'])
    print("Resistance Level:", order_book_data['resistance_level'])

    print("\nSmall Range Market Mood:", order_book_data['small_range_mood'])
    print("Large Range Market Mood:", order_book_data['large_range_mood'])

    print("\nForecasted Price for Max Duration (", order_book_data['max_forecast_minutes'], " minutes):", order_book_data['forecasted_price_max_duration'])

    # Enforce signals using MACD and RSI
    if order_book_data['macd'] > 0 and order_book_data['rsi'] > 70:
        print("\nSignal: SHORT")

    elif order_book_data['macd'] < 0 and order_book_data['rsi'] < 30:
        print("\nSignal: LONG")
    else:
        print("\nSignal: NONE")

    # Print calculated buy and sell orders for gradually bigger spreads
    spread_orders = order_book_data.get("spread_orders", {})
    for key, value in spread_orders.items():
        print(f"\n{key}: {value}")

print()

##################################################
##################################################

def dynamic_momentum_gauge(close, price, length=100, normalization_length=100):
    if len(close) <= 1:
        print("Error: Insufficient data for calculation.")
        return None, None

    m = 0.0
    r = (price / close[-2] - 1) * 100
    m = ((length - 1) / length) * m + r / length
    s = abs(m)

    # Calculate historical s values
    historical_s = [abs((close[i] / close[i-1] - 1) * 100) for i in range(1, len(close))]

    # Min-Max normalization
    min_historical_s = min(historical_s[-normalization_length:])
    max_historical_s = max(historical_s[-normalization_length:])
    s_normalized = (s - min_historical_s) / (max_historical_s - min_historical_s) if max_historical_s != min_historical_s else 0

    # Initialize variables with default values
    market_mood = "Neutral"
    forecast_price = 0

    # Determine Market Mood and Forecast Price
    if len(historical_s) > 1:
        previous_s = historical_s[-2]  # Previous value of s
        previous_s_normalized = (previous_s - min_historical_s) / (max_historical_s - min_historical_s) if max_historical_s != min_historical_s else 0

        if s_normalized > 0.5 and previous_s_normalized <= 0.5:
            market_mood = "Bearish"
        elif s_normalized <= 0.5 and previous_s_normalized > 0.5:
            market_mood = "Bullish"

        # Check if the current and previous momentum gauges are different
        if s_normalized != previous_s_normalized:
            forecast_price = price * (1 + s_normalized)
        else:
            forecast_price = price  # Keep the same forecast price

        print("Previous momentum gauge at: ", previous_s_normalized)
        print("Current momentum gauge at: ", s_normalized)

    return market_mood, forecast_price

mom_market_mood, mom_forecast_price = dynamic_momentum_gauge(close, price)

if mom_market_mood is not None and mom_forecast_price is not None:
    print(f"Market Mood: {mom_market_mood}")
    print(f"Forecast Price: {mom_forecast_price}")


print()

##################################################
##################################################

import numpy as np

def generate_stationary_wave(num_harmonics, amplitude, frequency, close):
    time = np.linspace(0, 4 * np.pi, len(close))
    compound_wave = np.zeros_like(time)

    for harmonic in range(1, num_harmonics + 1):
        sine_wave = amplitude * np.sin(harmonic * frequency * time)
        cosine_wave = amplitude * np.cos(harmonic * frequency * time)

        if harmonic % 2 == 1:  # Apply pi/phi symmetry to odd harmonics
            compound_wave += sine_wave
        else:  # Apply phi/pi symmetry to even harmonics
            compound_wave += cosine_wave

    return time, compound_wave

def inverse_fourier_transform(compound_wave):
    return np.fft.ifft(compound_wave)

def analyze_market_mood(forecasted):
    # Your logic to analyze market mood goes here
    # For demonstration purposes, assuming a simple logic
    mood = "Bullish" if forecasted[-1] > forecasted[0] else "Bearish"
    return mood

def print_forecasted_price(forecasted_price):
    print(f"Forecasted Price: ${forecasted_price:.5f}")

# Function to calculate fixed target prices for the next reversal
def calculate_fixed_targets(current_close):
    # Set fixed target prices as a percentage change from the current close
    target_up_percentage = 1.0  # Replace with your desired fixed percentage for upward reversal
    target_down_percentage = -1.0  # Replace with your desired fixed percentage for downward reversal

    # Calculate fixed target prices
    target_up = current_close * (1 + target_up_percentage / 100)
    target_down = current_close * (1 + target_down_percentage / 100)

    return target_up, target_down

# Example: Generate stationary compound wave, perform inverse Fourier transform, and analyze market mood
num_harmonics = 5
amplitude = 1.0
frequency = 1.0

# Generate and print values of the stationary compound wave
ifft_range, compound_wave = generate_stationary_wave(num_harmonics, amplitude, frequency, close)

# Perform inverse Fourier transform to get forecasted prices
forecasted = np.real(inverse_fourier_transform(compound_wave))

# Analyze market mood based on forecasted prices
market_mood = analyze_market_mood(forecasted)

# Print other details
print(f"Initial Close Price: {close[0]:.5f}")
print(f"Final Close Price: {close[-1]:.5f}")
print("\nMarket Mood:", market_mood)

# Calculate fixed target prices for the next reversal
current_close = price
target_up, target_down = calculate_fixed_targets(current_close)

# Print fixed target prices for the next reversal
print("\nFixed Target Prices for the Next Reversal:")
print(f"Target Price for Upward Reversal: {target_up:.5f}")
print(f"Target Price for Downward Reversal: {target_down:.5f}")


print()

##################################################
##################################################

import talib
import numpy as np

def remove_nan(value):
    if isinstance(value, np.ndarray):
        return value[~np.isnan(value)] if len(value) > 0 else value
    else:
        return value if not np.isnan(value) else None

def generate_technical_indicators(close, high_prices, low_prices, open_prices, volume, timeframe):
    # Remove NaN values from input arrays
    close = remove_nan(close)
    high_prices = remove_nan(high_prices)
    low_prices = remove_nan(low_prices)
    open_prices = remove_nan(open_prices)
    volume = remove_nan(volume)

    # Check if there are still valid data points after removing NaN values
    if len(close) == 0:
        raise ValueError("No valid data points after removing NaN values.")

    # Calculate technical indicators using TA-Lib
    upper_band, middle_band, lower_band = talib.BBANDS(close.flatten())
    ema = talib.EMA(close.flatten())
    sma = talib.SMA(close.flatten())
    rsi = talib.RSI(close.flatten())
    macd, signal, hist = talib.MACD(close.flatten())

    # Ensure that the length of volume matches the length of close
    if len(volume.flatten()) != len(close.flatten()):
        raise ValueError("Length of 'volume' must match the length of 'close'")

    obv = talib.OBV(close.flatten(), volume.flatten())
    atr = talib.ATR(high_prices.flatten(), low_prices.flatten(), close.flatten())
    engulfing = talib.CDL3OUTSIDE(open_prices.flatten(), high_prices.flatten(), low_prices.flatten(), close.flatten())
    beta = talib.BETA(high_prices.flatten(), low_prices.flatten())
    correlation = talib.CORREL(close.flatten(), volume.flatten())
    linear_reg = talib.LINEARREG(close.flatten())
    acos = talib.ACOS(close.flatten())
    asin = talib.ASIN(close.flatten())
    add = talib.ADD(close.flatten(), volume.flatten())
    sub = talib.SUB(close.flatten(), volume.flatten())
    mult = talib.MULT(close.flatten(), volume.flatten())
    div = talib.DIV(close.flatten(), volume.flatten())
    sum_result = talib.SUM(close.flatten())

    # Generate compound trigger for forecasting prices
    compound_trigger = (
        (rsi < 30) & (macd > 0) |  # Example: Bullish condition based on RSI and MACD
        (rsi > 70) & (macd < 0)    # Example: Bearish condition based on RSI and MACD
    )

    # Generate forecasted prices based on the compound trigger
    forecast_prices = close.flatten() * (1 + 0.01 * compound_trigger)  # Replace this with your forecasting logic

    # Determine market mood for each indicator
    market_mood = {
        "BBANDS": "Bearish" if len(upper_band) > 0 and close[-1] > upper_band[-1] else "Bullish",
        "EMA": "Bearish" if len(ema) > 0 and close[-1] > ema[-1] else "Bullish",
        "SMA": "Bearish" if len(sma) > 0 and close[-1] > sma[-1] else "Bullish",
        "RSI": "Bearish" if len(rsi) > 0 and rsi[-1] > 70 else "Bullish" if len(rsi) > 0 and rsi[-1] < 30 else "Neutral",
        "MACD": "Bearish" if len(macd) > 0 and macd[-1] < 0 else "Bullish" if len(macd) > 0 and macd[-1] > 0 else "Neutral",
        # Add similar conditions for other indicators
    }

    return {
        "Timeframe": f"{timeframe}",
        "BBANDS": (remove_nan(upper_band[-1]), remove_nan(middle_band[-1]), remove_nan(lower_band[-1])),
        "EMA": remove_nan(ema[-1]),
        "SMA": remove_nan(sma[-1]),
        "RSI": remove_nan(rsi[-1]),
        "MACD": (remove_nan(macd[-1]), remove_nan(signal[-1]), remove_nan(hist[-1])),
        "OBV": remove_nan(obv[-1]),
        "ATR": remove_nan(atr[-1]),
        "Engulfing": remove_nan(engulfing[-1]),
        "Beta": remove_nan(beta[-1]),
        "Correlation": remove_nan(correlation[-1]),
        "LinearReg": remove_nan(linear_reg[-1]),
        "ACOS": remove_nan(acos[-1]),
        "ASIN": remove_nan(asin[-1]),
        "ADD": remove_nan(add[-1]),
        "SUB": remove_nan(sub[-1]),
        "MULT": remove_nan(mult[-1]),
        "DIV": remove_nan(div[-1]),
        "SUM": remove_nan(sum_result[-1]),
        "ForecastPrices": remove_nan(forecast_prices[-1]),
        "MarketMood": market_mood,
        "CompoundTrigger": remove_nan(compound_trigger[-1])
    }

# Example data (replace this with your financial data)
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']
candles = get_candles(TRADE_SYMBOL, timeframes)

# Store results for each timeframe
results_by_timeframe = {}

# Iterate over each timeframe and call the generate_technical_indicators function
for timeframe in timeframes:
    # Extract relevant data for the current timeframe
    close_prices = np.array([candle['close'] for candle in candles if candle['timeframe'] == timeframe])
    high_prices = np.array([candle['high'] for candle in candles if candle['timeframe'] == timeframe])
    low_prices = np.array([candle['low'] for candle in candles if candle['timeframe'] == timeframe])
    open_prices = np.array([candle['open'] for candle in candles if candle['timeframe'] == timeframe])
    volume = np.array([candle['volume'] for candle in candles if candle['timeframe'] == timeframe])

    # Generate technical indicators, forecast prices, and market mood for the current timeframe
    indicators_data = generate_technical_indicators(close_prices, high_prices, low_prices, open_prices, volume, timeframe)

    # Store the results for the current timeframe
    results_by_timeframe[timeframe] = indicators_data

    # Print the results for the specific timeframes (1m and 5m)
    if timeframe == '1m' or timeframe == '5m':
        print(f"Results for {timeframe} timeframe:")
        for indicator, value in indicators_data.items():
            print(f"{indicator}:", value)
        print("-" * 40)

        # Print LinearReg Forecast Price for each timeframe
        print("LinearReg Forecast Price:", results_by_timeframe[timeframe]['LinearReg'])

# Assess overall bullish/bearish for each indicator across all timeframes
overall_market_mood = {}
for indicator in results_by_timeframe['1m']['MarketMood'].keys():
    bullish_count = sum(1 for tf_result in results_by_timeframe.values() if tf_result['MarketMood'][indicator] == 'Bullish')
    bearish_count = sum(1 for tf_result in results_by_timeframe.values() if tf_result['MarketMood'][indicator] == 'Bearish')
    neutral_count = sum(1 for tf_result in results_by_timeframe.values() if tf_result['MarketMood'][indicator] == 'Neutral')
    overall_market_mood[indicator] = {
        'Bullish': bullish_count,
        'Bearish': bearish_count,
        'Neutral': neutral_count
    }

# Print overall assessment
print("\nOverall Market Mood:")
for indicator, mood_counts in overall_market_mood.items():
    print(f"{indicator}: Bullish={mood_counts['Bullish']}, Bearish={mood_counts['Bearish']}, Neutral={mood_counts['Neutral']}")

print()

# Print LinearReg Forecast Price for each timefram
print("LinearReg Forecast Price 1m:", results_by_timeframe['1m']['LinearReg'])
print("LinearReg Forecast Price 3m:", results_by_timeframe['3m']['LinearReg'])
print("LinearReg Forecast Price 5m:", results_by_timeframe['5m']['LinearReg'])
print("LinearReg Forecast Price 15m:", results_by_timeframe['15m']['LinearReg'])
print("LinearReg Forecast Price 30m:", results_by_timeframe['30m']['LinearReg'])
print("LinearReg Forecast Price 1h:", results_by_timeframe['1h']['LinearReg'])
print("LinearReg Forecast Price 2h:", results_by_timeframe['2h']['LinearReg'])
print("LinearReg Forecast Price 4h:", results_by_timeframe['4h']['LinearReg'])
print("LinearReg Forecast Price 6h:", results_by_timeframe['6h']['LinearReg'])
print("LinearReg Forecast Price 8h:", results_by_timeframe['8h']['LinearReg'])
print("LinearReg Forecast Price 12h:", results_by_timeframe['12h']['LinearReg'])
print("LinearReg Forecast Price 1d:", results_by_timeframe['1d']['LinearReg'])

print()

##################################################
##################################################

import numpy as np

def dan_stationary_circuit(close):
    close = np.array(close)  # Convert close to a NumPy array
    
    # Apply Fast Fourier Transform (FFT) to get frequencies
    fft_result = np.fft.fft(close)
    
    # Calculate amplitudes and frequencies
    amplitudes = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(close))

    # Identify dominant frequency
    dominant_frequency = frequencies[np.argmax(amplitudes)]

    # Determine dominant frequency sign
    dominant_frequency_sign = "Positive" if dominant_frequency > 0 else "Negative"

    # Build stationary circuit with standing wave
    stationary_wave = np.sin(2 * np.pi * dominant_frequency * np.arange(len(close)))

    # Calculate quadrants based on the unit circle
    quadrant_size = len(close) // 4
    q1 = stationary_wave[:quadrant_size]
    q2 = stationary_wave[quadrant_size:2*quadrant_size]
    q3 = stationary_wave[2*quadrant_size:3*quadrant_size]
    q4 = stationary_wave[3*quadrant_size:]

    # Determine the cycle direction (Up or Down)
    cycle_direction = "Up" if dominant_frequency_sign == "Negative" else "Down"

    # Define key points for cycles and stages
    key_points = {
        "Up": {
            "DIP": (["Q1", "Q2", "Q3", "Q4"], ["Q2", "Q1", "Q2"], ["Q1", "Q2", "Q3"], ["Q2", "Q3", "Q4"]),
            "TOP": (["Q4", "Q3", "Q2", "Q1"], ["Q3", "Q4", "Q3"], ["Q4", "Q3", "Q2"], ["Q3", "Q2", "Q1"])
        },
        "Down": {
            "TOP": (["Q4", "Q3", "Q2", "Q1"], ["Q3", "Q4", "Q3"], ["Q4", "Q3", "Q2"], ["Q3", "Q2", "Q1"]),
            "DIP": (["Q1", "Q2", "Q3", "Q4"], ["Q2", "Q1", "Q2"], ["Q1", "Q2", "Q3"], ["Q2", "Q3", "Q4"])
        }
    }

    # Get key points for the current cycle direction
    dip_key_points, top_key_points = key_points[cycle_direction]["DIP"], key_points[cycle_direction]["TOP"]

    # Get last quadrant and current quadrant based on key points
    last_quadrant = dip_key_points[0][np.argmax(q1)]
    current_quadrant = dip_key_points[1][np.argmax(q2)]

    # Ensure all arrays have the same length by trimming the longer ones
    min_length = min(len(q2), len(q3), len(q4), len(q1))
    q2 = q2[:min_length]
    q3 = q3[:min_length]
    q4 = q4[:min_length]
    q1 = q1[:min_length]

    # Use np.vstack after ensuring equal lengths
    max_indices = np.unravel_index(np.argmax(np.vstack([q2, q3, q4, q1]), axis=0), q2.shape)

    # Calculate forecast factor based on the stage of the cycle
    try:
        stage_index = dip_key_points[3].index(current_quadrant)
    except ValueError:
        stage_index = len(dip_key_points[3]) - 1

    # Calculate forecast factor based on the stage of the cycle
    total_range = np.max(close) - np.min(close)
    forecast_factor = total_range * 0.25 * (stage_index + 1)

    # Forecasted price based on the current quadrant with the introduced factor
    forecasted_price = close[-1] + forecast_factor

    # Return the dictionary with the updated variables
    result_dict = {
        "dominant_frequency_sign": dominant_frequency_sign,
        "last_quadrant": last_quadrant,
        "current_quadrant": current_quadrant,
        "cycle_direction": cycle_direction,
        "forecasted_price": forecasted_price,
        "q1": q1.tolist(),
        "q2": q2.tolist(),
        "q3": q3.tolist(),
        "q4": q4.tolist(),
    }

    return result_dict

result = dan_stationary_circuit(close)

# Print the specific variables excluding "Next Quadrant"
print("Dominant Frequency Sign:", result["dominant_frequency_sign"])
print("Last Quadrant:", result["last_quadrant"])
print("Current Quadrant:", result["current_quadrant"])
print("Cycle Direction:", result["cycle_direction"])
print("Forecasted Price:", result["forecasted_price"])

print()

##################################################
##################################################

import talib
import numpy as np

def sine_market_analysis(close, closest_threshold, min_threshold, max_threshold):
    # Convert close to NumPy array
    close_array = np.array(close, dtype=float)

    # Remove NaN values
    close_array = close_array[~np.isnan(close_array)]

    # Calculate the Sinewave using TA-Lib
    sinewave = talib.SIN(close_array)
    sinewave = -sinewave

    # Find local minima and maxima
    minima = (sinewave[:-2] < sinewave[1:-1]) & (sinewave[1:-1] > sinewave[2:])
    maxima = (sinewave[:-2] > sinewave[1:-1]) & (sinewave[1:-1] < sinewave[2:])

    # Get the indices of reversals
    min_reversals = np.where(minima)[0] + 1  # Add 1 to adjust for slicing
    max_reversals = np.where(maxima)[0] + 1  # Add 1 to adjust for slicing

    # Determine market mood (uptrend/downtrend)
    if min_threshold == closest_threshold:
        market_mood = "Uptrend"
        current_cycle = "Up"
    elif max_threshold == closest_threshold:
        market_mood = "Downtrend"
        current_cycle = "Down"
    else:
        None

    # Forecast price for the current cycle based on FFT
    fft_result = np.abs(np.fft.fft(close_array))
    dominant_frequency_index = np.argmax(fft_result[1:]) + 1  # Skip the DC component
    dominant_frequency = 1 / (len(close_array) * (1 / dominant_frequency_index))

    if current_cycle == "Up":
        forecast_price = close_array[max_reversals[-1]] + np.mean(close_array) * np.sin(np.pi * 2 * dominant_frequency)
    else:
        forecast_price = close_array[min_reversals[-1]] + np.mean(close_array) * np.sin(np.pi * 2 * dominant_frequency)

    return market_mood, current_cycle, forecast_price

sin_market_mood, sin_current_cycle, sin_price = sine_market_analysis(close_prices, closest_threshold, min_threshold, max_threshold)

# Print the results outside the function
print("Market Mood:", sin_market_mood)
print("Current Cycle:", sin_current_cycle)
print("Forecast Price:", sin_price)

print()

##################################################
##################################################

import numpy as np

def calculate_min_max_values(close):
    # Find the index of the last lowest low and last highest high
    last_low_index = np.argmin(close)
    last_high_index = np.argmax(close)

    # Calculate the last lowest low and last highest high
    last_low = close[last_low_index]
    last_high = close[last_high_index]

    return last_low, last_high

last_low, last_high = calculate_min_max_values(close)

print(f"Last Lowest Low: {last_low}")
print(f"Last Highest High: {last_high}")

print()

##################################################
##################################################

import numpy as np

def generate_sine_wave_with_motion_and_factors(close, threshold_low, threshold_min, threshold_max):
    last_low, last_high = calculate_min_max_values(close)

    if threshold_low == threshold_min:
        motion_factor = 1
        market_mood = "Up"
    elif threshold_low == threshold_max:
        motion_factor = -1
        market_mood = "Down"
    else:
        raise ValueError("Invalid threshold_low. Please choose threshold_min or threshold_max.")

    t = np.linspace(0, 1, num=len(close), endpoint=False)
    wave = 0.5 * (1 + amplitude * np.sin(2 * np.pi * frequency * t))
    wave = last_low + (last_high - last_low) * wave

    # Apply Fast Fourier Transform (FFT) to the sine wave
    fft_wave = np.fft.fft(wave)
    
    # Modify the FFT for forecast (e.g., change the amplitude of certain frequency components)
    # Here, we scale the amplitude of the second harmonic (frequency = 2) for illustration
    fft_wave[2] *= 0.5

    # Convert back to time domain using Inverse FFT
    forecast_wave = np.fft.ifft(fft_wave).real

    # Apply Fast Fourier Transform (FFT) to the forecasted price
    fft_forecast_price = np.fft.fft(forecast_wave)
    
    # Modify the FFT for forecasted price (you can adjust this based on your requirements)
    fft_forecast_price[3] *= 0.8  # Adjusting the amplitude of the fourth harmonic for illustration

    # Convert back to time domain using Inverse FFT
    forecast_price = np.fft.ifft(fft_forecast_price).real

    return market_mood, forecast_price, np.min(forecast_wave), np.max(forecast_wave)

# Example usage
amplitude = 0.5
frequency = 1

sin_mood, sin_price, min_value, max_value = generate_sine_wave_with_motion_and_factors(close, closest_threshold, min_threshold, max_threshold)

print(f"Market Mood: {sin_mood}")
print(f"Forecast Price: {sin_price[-1]}")
print(f"Min Value: {min_value}")
print(f"Max Value: {max_value}")

print()

##################################################
##################################################

def generate_momentum_sinewave(timeframes):
    # Initialize variables
    momentum_sorter = []
    market_mood = []
    last_reversal = None
    last_reversal_value_on_sine = None
    last_reversal_value_on_price = None
    next_reversal = None
    next_reversal_value_on_sine = None
    next_reversal_value_on_price = None

    # Loop over timeframes
    for timeframe in timeframes:
        # Get close prices for current timeframe
        close_prices = np.array(get_closes(timeframe))

        # Get last close price
        current_close = close_prices[-1]

        # Calculate sine wave for current timeframe
        sine_wave, leadsine = talib.HT_SINE(close_prices)

        # Replace NaN values with 0
        sine_wave = np.nan_to_num(sine_wave)
        sine_wave = -sine_wave

        # Get the sine value for last close
        current_sine = sine_wave[-1]

        # Calculate the min and max sine
        sine_wave_min = np.nanmin(sine_wave) # Use nanmin to ignore NaN values
        sine_wave_max = np.nanmax(sine_wave)

        # Calculate price values at min and max sine
        sine_wave_min_price = close_prices[sine_wave == sine_wave_min][0]
        sine_wave_max_price = close_prices[sine_wave == sine_wave_max][0]
     

        # Calculate the difference between the max and min sine
        sine_wave_diff = sine_wave_max - sine_wave_min

        # If last close was the lowest, set as last reversal                                  
        if current_sine == sine_wave_min:
            last_reversal = 'dip'
            last_reversal_value_on_sine = sine_wave_min 
            last_reversal_value_on_price = sine_wave_min_price
        
        # If last close was the highest, set as last reversal                                 
        if current_sine == sine_wave_max:
            last_reversal = 'top'
            last_reversal_value_on_sine = sine_wave_max
            last_reversal_value_on_price = sine_wave_max_price

        # Calculate % distances
        newsine_dist_min, newsine_dist_max = [], []
        for close in close_prices:
            # Calculate distances as percentages
            dist_from_close_to_min = ((current_sine - sine_wave_min) /  
                                      sine_wave_diff) * 100            
            dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                                      sine_wave_diff) * 100
                
            newsine_dist_min.append(dist_from_close_to_min)       
            newsine_dist_max.append(dist_from_close_to_max)

        # Take average % distances
        avg_dist_min = sum(newsine_dist_min) / len(newsine_dist_min)
        avg_dist_max = sum(newsine_dist_max) / len(newsine_dist_max)

        # Determine market mood based on % distances
        if avg_dist_min <= 15:
            mood = "At DIP Reversal and Up to Bullish"
            if last_reversal != 'dip':
                next_reversal = 'dip'
                next_reversal_value_on_sine = sine_wave_min
                next_reversal_value_on_price = close_prices[sine_wave == sine_wave_min][0]
        elif avg_dist_max <= 15:
            mood = "At TOP Reversal and Down to Bearish"
            if last_reversal != 'top':
                next_reversal = 'top'
                next_reversal_value_on_sine = sine_wave_max
                next_reversal_value_on_price = close_prices[sine_wave == sine_wave_max][0]
        elif avg_dist_min < avg_dist_max:
            mood = "Bullish"
        else:
            mood = "Bearish"

        # Append momentum score and market mood to lists
        momentum_score = avg_dist_max - avg_dist_min
        momentum_sorter.append(momentum_score)
        market_mood.append(mood)

        # Print distances and market mood
        print(f"{timeframe} Close is now at "       
              f"dist. to min: {avg_dist_min:.2f}% "
              f"and at "
              f"dist. to max: {avg_dist_max:.2f}%. "
              f"Market mood: {mood}")

        # Update last and next reversal info
        if next_reversal:
            last_reversal = next_reversal
            last_reversal_value_on_sine = next_reversal_value_on_sine
            last_reversal_value_on_price = next_reversal_value_on_price
            next_reversal = None
            next_reversal_value_on_sine = None
            next_reversal_value_on_price = None

    # Get close prices for the 1-minute timeframe and last 3 closes
    close_prices = np.array(get_closes('1m'))

    # Calculate sine wave
    sine_wave, leadsine = talib.HT_SINE(close_prices)

    # Replace NaN values with 0
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    # Get the sine value for last close
    current_sine = sine_wave[-1]

    # Get current date and time
    now = datetime.datetime.now()

    # Calculate the min and max sine
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate the difference between the maxand min sine
    sine_wave_diff = sine_wave_max - sine_wave_min

    # Calculate % distances
    dist_from_close_to_min = ((current_sine - sine_wave_min) / 
                              sine_wave_diff) * 100
    dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                              sine_wave_diff) * 100

    # Determine market mood based on % distances
    if dist_from_close_to_min <= 15:
        mood = "At DIP Reversal and Up to Bullish"
        if last_reversal != 'dip':
            next_reversal = 'dip'
            next_reversal_value_on_sine = sine_wave_min

    elif dist_from_close_to_max <= 15:
        mood = "At TOP Reversal and Down to Bearish"
        if last_reversal != 'top':
            next_reversal = 'top'
            next_reversal_value_on_sine = sine_wave_max

    elif dist_from_close_to_min < dist_from_close_to_max:
        mood = "Bullish"
    else:
        mood = "Bearish"

    # Get the close prices that correspond to the min and max sine values
    close_prices_between_min_and_max = close_prices[(sine_wave >= sine_wave_min) & (sine_wave <= sine_wave_max)]

    print()

    # Print distances and market mood for 1-minute timeframe
    print(f"On 1min timeframe,Close is now at "       
          f"dist. to min: {dist_from_close_to_min:.2f}% "
          f"and at "
          f"dist. to max:{dist_from_close_to_max:.2f}%. "
          f"Market mood: {mood}")

    min_val = min(close_prices_between_min_and_max)
    max_val = max(close_prices_between_min_and_max)

    print("The lowest value in the array is:", min_val)
    print("The highest value in the array is:", max_val)

    print()

    # Update last and next reversal info
    #if next_reversal:
        #last_reversal = next_reversal
        #last_reversal_value_on_sine = next_reversal_value_on_sine

        #next_reversal = None
        #next_reversal_value_on_sine = None


    # Print last and next reversal info
    #if last_reversal:
        #print(f"Last reversal was at {last_reversal} on the sine wave at {last_reversal_value_on_sine:.2f} ")

    # Return the momentum sorter, market mood, close prices between min and max sine, and reversal info
    return momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max      

momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max = generate_momentum_sinewave(timeframes)

print()

#print("Close price values between last reversals on sine: ")
#print(close_prices_between_min_and_max)

print()

print("Current close on sine value now at: ", current_sine)
print("distances as percentages from close to min: ", dist_from_close_to_min, "%")
print("distances as percentages from close to max: ", dist_from_close_to_max, "%")
print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
print("Mood on 1min timeframe is now at: ", market_mood[-12])

print()

##################################################
##################################################

def generate_new_momentum_sinewave(close, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Convert 'close' to a NumPy array
    close_np = np.array(close)

    # Calculate the sine wave using HT_SINE
    sine_wave, _ = talib.HT_SINE(close_np)
    
    # Replace NaN values with 0 using nan_to_num
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    print("Current close on Sine wave:", sine_wave[-1])

    # Calculate the minimum and maximum values of the sine wave
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate the distance from close to min and max as percentages on a scale from 0 to 100%
    dist_from_close_to_min = ((sine_wave[-1] - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_from_close_to_max = ((sine_wave_max - sine_wave[-1]) / (sine_wave_max - sine_wave_min)) * 100

    print("Distance from close to min:", dist_from_close_to_min)
    print("Distance from close to max:", dist_from_close_to_max)

    # Calculate the range of values for each quadrant
    range_q1 = (sine_wave_max - sine_wave_min) / 4
    range_q2 = (sine_wave_max - sine_wave_min) / 4
    range_q3 = (sine_wave_max - sine_wave_min) / 4
    range_q4 = (sine_wave_max - sine_wave_min) / 4

    # Set the EM amplitude for each quadrant based on the range of values
    em_amp_q1 = range_q1 / percent_to_max_val
    em_amp_q2 = range_q2 / percent_to_max_val
    em_amp_q3 = range_q3 / percent_to_max_val
    em_amp_q4 = range_q4 / percent_to_max_val

    # Calculate the EM phase for each quadrant
    em_phase_q1 = 0
    em_phase_q2 = math.pi/2
    em_phase_q3 = math.pi
    em_phase_q4 = 3*math.pi/2

    # Calculate the current position of the price on the sine wave
    current_position = (sine_wave[-1] - sine_wave_min) / (sine_wave_max - sine_wave_min)
    current_quadrant = 0

    # Determine which quadrant the current position is in
    if current_position < 0.25:
        # In quadrant 1
        em_amp = em_amp_q1
        em_phase = em_phase_q1
        current_quadrant = 1
        print("Current position is in quadrant 1. Distance from 0% to 25% of range:", (current_position - 0.0) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    elif current_position < 0.5:
        # In quadrant 2
        em_amp = em_amp_q2
        em_phase = em_phase_q2
        current_quadrant = 2
        print("Current position is in quadrant 2. Distance from 25% to 50% of range:", (current_position - 0.25) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    elif current_position < 0.75:
        # In quadrant 3
        em_amp = em_amp_q3
        em_phase = em_phase_q3
        current_quadrant = 3
        print("Current position is in quadrant 3. Distance from 50% to 75% of range:", (current_position - 0.5) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    else:
        # In quadrant 4
        em_amp = em_amp_q4
        em_phase = em_phase_q4
        current_quadrant = 4
        print("Current position is in quadrant 4. Distance from 75% to 100% of range:", (current_position - 0.75) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)

    print("EM amplitude:", em_amp)
    print("EM phase:", em_phase)

    # Calculate the EM value
    em_value = em_amp * math.sin(em_phase)

    print("EM value:", em_value)

    # Determine the trend direction based on the EM phase differences
    if em_phase_q4 - em_phase_q3 > 0 and em_phase_q3 - em_phase_q2 > 0 and em_phase_q2 - em_phase_q1 > 0:
        trend_direction = "Up"
    elif em_phase_q4 - em_phase_q3 < 0 and em_phase_q3 - em_phase_q2 < 0 and em_phase_q2 - em_phase_q1 < 0:
        trend_direction = "Down"
    else:
        trend_direction = "Sideways"

    print("Trend direction:", trend_direction)

    # Calculate the percentage of the price range
    price_range = candles[-1]["high"] - candles[-1]["low"]
    price_range_percent = (close_prices[-1] - candles[-1]["low"]) / price_range * 100

    print("Price range percent:", price_range_percent)

    # Calculate the momentum value
    momentum = em_value * price_range_percent / 100

    print("Momentum value:", momentum)

    print()

    # Return a dictionary of all the features
    return {
        "current_close": sine_wave[-1],
        "dist_from_close_to_min": dist_from_close_to_min,
        "dist_from_close_to_max": dist_from_close_to_max,
        "current_quadrant": current_quadrant,
        "em_amplitude": em_amp,
        "em_phase": em_phase,
        "trend_direction": trend_direction,
        "price_range_percent": price_range_percent,
        "momentum": momentum,
        "max": sine_wave_max,
        "min": sine_wave_min
    }

# Example usage:
result = generate_new_momentum_sinewave(close, candles, percent_to_max_val=5, percent_to_min_val=5)

# Assign each element to separate variables
current_close = result["current_close"]
dist_from_close_to_min = result["dist_from_close_to_min"]
dist_from_close_to_max = result["dist_from_close_to_max"]
current_quadrant = result["current_quadrant"]
em_amplitude = result["em_amplitude"]
em_phase = result["em_phase"]
trend_direction = result["trend_direction"]
price_range_percent = result["price_range_percent"]
momentum = result["momentum"]
sine_wave_max = result["max"]
sine_wave_min = result["min"]

# Print each variable separately
print("Current Close:", current_close)
print("Distance from Close to Min:", dist_from_close_to_min)
print("Distance from Close to Max:", dist_from_close_to_max)
print("Current Quadrant:", current_quadrant)
print("EM Amplitude:", em_amplitude)
print("EM Phase:", em_phase)
print("Trend Direction:", trend_direction)
print("Price Range Percent:", price_range_percent)
print("Momentum:", momentum)
print("Sine Wave Max:", sine_wave_max)
print("Sine Wave Min:", sine_wave_min)

print()

##################################################
##################################################

def generate_market_mood_forecast(close, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Call generate_new_momentum_sinewave to get the sine wave and other features
    sine_wave = generate_new_momentum_sinewave(close, candles, percent_to_max_val, percent_to_min_val)

    current_quadrant = sine_wave["current_quadrant"]
    em_phase_q1 = sine_wave["em_phase"]
    em_phase_q2 = em_phase_q1 + math.pi/2
    em_phase_q3 = em_phase_q1 + math.pi
    em_phase_q4 = em_phase_q1 + 3*math.pi/2

    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177

    # Calculate the Brun constant from the phi ratio and sqrt(5)
    brun_constant = math.sqrt(PHI * math.sqrt(5))

    # Define PI constant with 15 decimals
    PI = 3.1415926535897932384626433832795028842

    # Calculate sacred frequency
    sacred_freq = (432 * PHI ** 2) / 360

    # Calculate Alpha and Omega ratios
    alpha_ratio = PHI / PI
    omega_ratio = PI / PHI

    # Calculate Alpha and Omega spiral angle rates
    alpha_spiral = (2 * math.pi * sacred_freq) / alpha_ratio
    omega_spiral = (2 * math.pi * sacred_freq) / omega_ratio

    # Calculate quadrature phase shift based on current quadrant
    if current_quadrant == 1:
        # Up cycle from Q1 to Q4
        quadrature_phase = em_phase_q1
        em_phase = alpha_spiral
    elif current_quadrant == 2:
        quadrature_phase = em_phase_q2
        em_phase = omega_spiral
    elif current_quadrant == 3:
        quadrature_phase = em_phase_q3
        em_phase = omega_spiral
    else:
        quadrature_phase = em_phase_q4
        em_phase = alpha_spiral

    cycle_direction = "UP"
    next_quadrant = 1

    if current_quadrant == 1:
        next_quadrant = 2
        cycle_direction = "UP"

    elif current_quadrant == 2:
        if cycle_direction == "UP":
            next_quadrant = 3
        elif cycle_direction == "DOWN":
            next_quadrant = 1

    elif current_quadrant == 3:
        if cycle_direction == "UP":
            next_quadrant = 4
        elif cycle_direction == "DOWN":
            next_quadrant = 2

    elif current_quadrant == 4:
        if cycle_direction == "UP":
            next_quadrant = 3
            cycle_direction = "DOWN"

    current_point = ""
    if current_quadrant == 1:
        current_point = "Apex"  
    if current_quadrant == 2:  
        current_point = "Left"       
    if current_quadrant == 3:
        current_point = "Base"        
    if current_quadrant == 4:
        current_point = "Right"  
            
    next_point = ""        
    if next_quadrant == 1:
        next_point = "Apex"   
    if next_quadrant == 2:
         next_point = "Left"          
    if next_quadrant == 3:
        next_point = "Base"       
    if next_quadrant == 4:
        next_point = "Right"

    # Calculate quadrature phase
    if next_quadrant == 1:
        next_quadrature_phase = em_phase_q1
    elif next_quadrant == 2:
        next_quadrature_phase = em_phase_q2
    elif next_quadrant == 3:
        next_quadrature_phase = em_phase_q3
    else:
        next_quadrature_phase = em_phase_q4

    # Calculate EM value
    em_value = sine_wave["em_amplitude"] * math.sin(em_phase)

    # Calculate quadrature phase shift from current to next quadrant
    quadrature = next_quadrature_phase - quadrature_phase

    if quadrature > 0:
        cycle_direction = "UP"
    else:
        cycle_direction = "DOWN"

    # Define the frequency bands and their corresponding emotional values
    frequency_bands = {"Delta": -0.5, "Theta": -0.25, "Alpha": 0, "Beta": 0.25, "Gamma": 0.5}

    # Calculate the emotional value of each frequency band using phi
    emotional_values = {band: frequency_bands[band] * PHI for band in frequency_bands}

    # Divide the frequency spectrum into 4 quadrants based on Metatron's Cube geometry
    quadrant_amplitudes = {"Apex": 1, "Left": 0.5, "Base": 0, "Right": 0.5}
    quadrant_phases = {"Apex": 0, "Left": math.pi/2, "Base": math.pi, "Right": 3*math.pi/2}

    # Calculate emotional amplitude and phase values for each quadrant
    quadrant_emotional_values = {}
    for quadrant in quadrant_amplitudes:
        amplitude = quadrant_amplitudes[quadrant]
        phase = quadrant_phases[quadrant]
        quadrant_emotional_values[quadrant] = {"amplitude": amplitude, "phase": phase}

    # Calculate the forecast mood for the next 1 hour based on the phi value of each frequency band and mapping that to an emotional state
    forecast_moods = {}
    for band in emotional_values:
        emotional_value = emotional_values[band]
        phi_value = PHI ** (emotional_value / brun_constant)
        forecast_moods[band] = math.cos(phi_value + em_value)

    # Sort the frequencies from most negative to most positive based on their emotional value
    sorted_frequencies = sorted(forecast_moods, key=lambda x: forecast_moods[x])

    # Calculate average moods for the highest 3 and lowest 3 frequencies to determine the overall trend
    high_moods = [forecast_moods[band] for band in sorted_frequencies[-3:]]
    low_moods = [forecast_moods[band] for band in sorted_frequencies[:3]]
    avg_high_mood = sum(high_moods) / len(high_moods)
    avg_low_mood = sum(low_moods) / len(low_moods)

    # Calculate weighted averages of the top n and bottom n frequencies to get a more nuanced forecast
    n = 2
    weighted_high_mood = sum([forecast_moods[band] * (n - i) for i, band in enumerate(sorted_frequencies[-n:])]) / sum(range(1, n+1))
    weighted_low_mood = sum([forecast_moods[band] * (n - i) for i, band in enumerate(sorted_frequencies[:n])]) / sum(range(1, n+1))

    # Map the 4 quadrants to the 4 points of Metatron's Cube (Apex, Left, Base, Right)
    mapped_quadrants = {}
    for quadrant in quadrant_emotional_values:
        amplitude = quadrant_emotional_values[quadrant]["amplitude"]
        phase = quadrant_emotional_values[quadrant]["phase"]
        mapped_quadrants[quadrant] = amplitude * math.sin(em_value + phase)

    # Identify the minimum and maximum frequency nodes to determine reversal points
    min_node = sorted_frequencies[0]
    max_node = sorted_frequencies[-1]

    # Based on the minimum and maximum nodes, calculate forecasts for when mood reversals may occur
    if forecast_moods[min_node] < avg_low_mood and forecast_moods[max_node] > avg_high_mood:
        mood_reversal_forecast = "Mood reversal expected in the short term"
    elif forecast_moods[min_node] < weighted_low_mood and forecast_moods[max_node] > weighted_high_mood:
        mood_reversal_forecast = "Mood may reverse soon"
    else:
        mood_reversal_forecast = "No mood reversal expected in the near term"

    # Determine an overall market mood based on the highest and lowest 3 frequencies - positive, negative or neutral
    if avg_high_mood > 0 and avg_low_mood > 0:
        market_mood = "Positive"
    elif avg_high_mood < 0 and avg_low_mood < 0:
        market_mood = "Negative"
    else:
        market_mood = "Neutral"

    # Return the market mood forecast
    return {"cycle_direction": cycle_direction, "quadrant_emotional_values": quadrant_emotional_values, "forecast_moods": forecast_moods, "sorted_frequencies": sorted_frequencies, "avg_high_mood": avg_high_mood, "avg_low_mood": avg_low_mood, "weighted_high_mood": weighted_high_mood, "weighted_low_mood": weighted_low_mood, "mapped_quadrants": mapped_quadrants, "min_node": min_node, "max_node": max_node, "mood_reversal_forecast": mood_reversal_forecast, "market_mood": market_mood, "current_point": current_point, "next_point": next_point}
  
#generate_market_mood_forecast(close, candles, percent_to_max_val=50, percent_to_min_val=50)

market_mood_forecast = generate_market_mood_forecast(close, candles, percent_to_max_val=50, percent_to_min_val=50)

cycle_direction = market_mood_forecast["cycle_direction"]
quadrant_emotional_values = market_mood_forecast["quadrant_emotional_values"]
forecast_moods = market_mood_forecast["forecast_moods"]
sorted_frequencies = market_mood_forecast["sorted_frequencies"]
avg_high_mood = market_mood_forecast["avg_high_mood"]
avg_low_mood = market_mood_forecast["avg_low_mood"]
weighted_high_mood = market_mood_forecast["weighted_high_mood"]
weighted_low_mood = market_mood_forecast["weighted_low_mood"]
mapped_quadrants = market_mood_forecast["mapped_quadrants"]
min_node = market_mood_forecast["min_node"]
max_node = market_mood_forecast["max_node"]
mood_reversal_forecast = market_mood_forecast["mood_reversal_forecast"]
market_mood = market_mood_forecast["market_mood"]

current_point = market_mood_forecast["current_point"] 
next_point = market_mood_forecast["next_point"]

print(f"Current point: {current_point}")
print(f"Next point: {next_point}")

print("Cycle direction:", cycle_direction)
print("Quadrant emotional values:", quadrant_emotional_values)
print("Forecast moods:", forecast_moods)
print("Sorted frequencies:", sorted_frequencies)
print("Average high mood:", avg_high_mood)
print("Average low mood:", avg_low_mood)
print("Weighted high mood:", weighted_high_mood)
print("Weighted low mood:", weighted_low_mood)
print("Mapped quadrants:", mapped_quadrants)
print("Minimum node:", min_node)
print("Maximum node:", max_node)
print("Mood reversal forecast:", mood_reversal_forecast)
print("Market mood:", market_mood)

print()

##################################################
##################################################

print()

##################################################
##################################################

def get_next_minute_targets(closes, n_components):
    # Calculate FFT of closing prices
    fft = fftpack.fft(closes)
    frequencies = fftpack.fftfreq(len(closes))

    # Sort frequencies by magnitude and keep only the top n_components
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]

    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = np.real(fftpack.ifft(filtered_fft))

    # Calculate the target price as the next value after the last closing price
    target_price = filtered_signal[-1]
    
    # Calculate the stop loss and target levels
    entry_price = closes[-1]
    stop_loss = entry_price - np.std(closes)
    target1 = target_price + 2*np.std(closes)
    target2 = target_price + 3*np.std(closes)
    target3 = target_price + 4*np.std(closes)
    target4 = target_price + 5*np.std(closes)
    target5 = target_price + 6*np.std(closes)

    return entry_price, stop_loss, target1, target2, target3, target4, target5

# Example usage
closes = get_closes("1m")
n_components = 5
targets = []

for i in range(len(closes) - 1):
    # Decompose the signal up to the current minute and predict the target for the next minute
    entry_price, stop_loss, target1, target2, target3, target4, target5 = get_next_minute_targets(closes[:i+1], n_components)
    targets.append((entry_price, stop_loss, target1, target2, target3, target4, target5))

# Print the predicted levels for the next minute
print("Entry price:", targets[-1][0])
print("Stop loss:", targets[-1][1])
print("Target 1:", targets[-1][2])
print("Target 2:", targets[-1][3])
print("Target 3:", targets[-1][4])
print("Target 4:", targets[-1][5])
print("Target 5:", targets[-1][6])

print()

##################################################
##################################################

import talib
import numpy as np

def ninja_trend(close):
    fast_MA_period = 12
    slow_MA_period = 26
    signal_length = 9

    # Add a sine wave reversal to the close prices
    reversal_wave = np.sin(np.linspace(0, 10, len(close))) * 5
    close_with_reversal = close + reversal_wave

    # Convert 'close' to numpy array
    close_np = np.array(close_with_reversal)

    # Calculate MACD
    macd, _, _ = talib.MACD(close_np, fast_MA_period, slow_MA_period, signal_length)

    # Calculate Signal line
    signal = talib.SMA(macd, timeperiod=signal_length)

    # Identify Bull and Bear conditions
    is_bull_bar = (macd < 0) & (signal < 0)
    is_bear_bar = (macd > 0) & (signal > 0)

    # Buy, Sell, or Neutral
    trend_text = "Buy" if is_bull_bar[-1] else "Sell" if is_bear_bar[-1] else "Neutral"

    # Alerts
    buy_alert = "Buy Alert" if is_bull_bar[-1] and not is_bull_bar[-2] else None
    sell_alert = "Sell Alert" if is_bear_bar[-1] and not is_bear_bar[-2] else None

    # Market Mood
    market_mood = "Bullish" if is_bull_bar[-1] else "Bearish" if is_bear_bar[-1] else "Neutral"

    # Forecasted Price (adjusted for real price value)
    multiplier = 0.1  # Adjust this multiplier based on your analysis
    forecast_price_trend = close[-1] + (macd[-1] * multiplier) if trend_text == "Buy" else close[-1] - (macd[-1] * multiplier) if trend_text == "Sell" else close[-1]

    # FFT-based Forecasting
    n = len(close_with_reversal)
    fft_result = fft.fft(close_np)
    frequencies = fft.fftfreq(n)
    max_frequency_index = np.argmax(np.abs(fft_result[1:n // 2])) + 1  # Ignore the DC component
    dominant_frequency = frequencies[max_frequency_index]

    # Estimate the next 5 targets
    targets_count = 5
    next_targets_fft = np.real(fft.ifft(fft_result[max_frequency_index] * np.exp(2j * np.pi * dominant_frequency * np.arange(1, targets_count + 1))))

    # Scale the next targets based on the range of close prices
    close_range = np.max(close_np) - np.min(close_np)
    scaling_factor = 100 / close_range
    next_targets = close_with_reversal[-1] + (next_targets_fft * scaling_factor)

    # Get the current reversals on the sine wave as real price values
    current_reversals = close_with_reversal[-len(reversal_wave):]

    # Get the last highest high and lowest low
    last_highest_high = np.max(close_with_reversal[-len(reversal_wave):])
    last_lowest_low = np.min(close_with_reversal[-len(reversal_wave):])

    return trend_text, buy_alert, sell_alert, market_mood, forecast_price_trend, next_targets, current_reversals, last_highest_high, last_lowest_low

# Example usage with existing close prices
trend, buy_alert, sell_alert, mood, forecast_price_trend, next_targets, current_reversals, last_highest_high, last_lowest_low = ninja_trend(close)

# Print results
print(f'Trend: {trend}')
print(f'Buy Alert: {buy_alert}')
print(f'Sell Alert: {sell_alert}')
print(f'Market Mood: {mood}')
print(f'Forecasted Price Trend: {forecast_price_trend}')

print()

for i, target in enumerate(next_targets, 1):
    print(f'Next Target {i}: {target}')

print()

print(f'Current Reversals: {current_reversals[0]}')
print(f'Next Most Significant Reversal: {current_reversals[1]}')

print()

print(f'Last Highest High: {last_highest_high}')
print(f'Last Lowest Low: {last_lowest_low}')

print()


print()

##################################################
##################################################

def octa_metatron_cube(close_prices, candles,  
                       percent_to_max_val=5,  
                       percent_to_min_val=5):

    sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                               percent_to_max_val, 
                               percent_to_min_val)  
  
    current_quadrant = sine_wave["current_quadrant"]
    current_value = sine_wave["current_close"]
    current_quadrant = 0

    sine_wave_max = sine_wave["max"]    
    sine_wave_min = sine_wave["min"] 

    if sine_wave["current_quadrant"] == 1:
         print("In quadrant 1!")
    elif sine_wave["current_quadrant"] == 2:  
         print("In quadrant 2!")            
    elif sine_wave["current_quadrant"] == 3:               
         print("In quadrant 3!")           
    elif sine_wave["current_quadrant"] == 4:                 
         print("In quadrant 4!")
    
    print()

    print("Close price map on sine now at: ", sine_wave)

    print()

    print("Min at:", sine_wave_min)
    print("Max at: ", sine_wave_max)

    print() 
 
    # Get the current quadrant and EM phase of the sine wave
    current_quadrant = sine_wave["current_quadrant"]
    em_phase = sine_wave["em_phase"]
    em_amp = sine_wave["em_amplitude"]

    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177  
   
    # Calculate the Brun constant from the phi ratio and sqrt(5)
    brun_constant = math.sqrt(PHI * math.sqrt(5))

    # Define PI constant with 15 decimals    
    PI = 3.1415926535897932384626433832795028842
   
    # Define e constant with 15 decimals   
    e =  2.718281828459045235360287471352662498  

    # Calculate sacred frequency
    sacred_freq = (432 * PHI ** 2) / 360
    
    # Calculate Alpha and Omega ratios   
    alpha_ratio = PHI / PI       
    omega_ratio = PI / PHI
          
    # Calculate Alpha and Omega spiral angle rates     
    alpha_spiral = (2 * math.pi * sacred_freq) / alpha_ratio
    omega_spiral = (2 * math.pi * sacred_freq) / omega_ratio

    start_value = 0.0 

    frequencies = []
    frequencies_next = []
    forecast = []

    freq_range = 25
    
    print(em_phase)
    print(em_amp) 

    current_quadrant_amplitude = em_amp
    current_quadrant_phase= em_amp

    print()
  
    #Calculate X:Y ratio from golden ratio     
    ratio = 2 * (PHI - 1)  

    # Calculate EM phases based on dividing whole range by X:Y ratio  
    em_phase_q1 = 0  
    em_phase_q2= PI * ratio / 2 
    em_phase_q3 = PI * ratio           
    em_phase_q4 = PI * ratio * 1.5  

    # Determine the trend direction based on the EM phase differences
    em_phase_diff_q1_q2 = em_phase_q2 - em_phase_q1
    em_phase_diff_q2_q3 = em_phase_q3 - em_phase_q2
    em_phase_diff_q3_q4 = em_phase_q4 - em_phase_q3
    em_phase_diff_q4_q1 = 2*math.pi - (em_phase_q4 - em_phase_q1)

    # Assign EM amplitudes based on X:Y ratio
    em_amp_q1 = (sine_wave_max - sine_wave_min) * ratio / 4
    em_amp_q2 = (sine_wave_max - sine_wave_min) * ratio / 4
    em_amp_q3 = (sine_wave_max - sine_wave_min) * ratio / 4
    em_amp_q4 = (sine_wave_max - sine_wave_min) * ratio / 4

    for i in range(freq_range):
        frequency = i * sacred_freq      
        #em_value = current_quadrant_amplitude * math.sin(current_quadrant_phase * frequency)

       # Calculate EM value based on frequency   
        if current_quadrant == 1: 
            # Most negative frequencies in Q1
            em_value = em_amp_q1 * math.sin(em_phase_q1 * frequency) 
            print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 2: 
            # Most negative frequencies in Q1
            em_value = em_amp_q2 * math.sin(em_phase_q2 * frequency) 
            print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 3: 
            # Most negative frequencies in Q1
            em_value = em_amp_q3 * math.sin(em_phase_q3 * frequency) 
            print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 4: 
            # Most negative frequencies in Q1
            em_value = em_amp_q4 * math.sin(em_phase_q4 * frequency) 
            print(em_value)

        frequencies.append({
            'number': i,
            'frequency': frequency,  
            'em_amp': 0,
            'em_phase': 0,         
            'em_value': 0,
            'phi': PHI,    
            'pi': PI,
            'e': e, 
            'mood': 'neutral'      
        }) 

    for freq in frequencies:
        forecast.append(freq)  
                
    for freq in frequencies:

        # Get PHI raised to the frequency number         
        phi_power = PHI ** freq['number'] 

        if phi_power < 1.05:
            freq['mood'] = 'extremely positive'
        elif phi_power < 1.2:       
            freq['mood'] = 'strongly positive'
        elif phi_power < 1.35:       
            freq['mood'] = 'positive'    
        elif phi_power < 1.5:       
            freq['mood'] = 'slightly positive'
        elif phi_power < 2:          
            freq['mood'] = 'neutral'              
        elif phi_power < 2.5:     
            freq['mood'] = 'slightly negative'      
        elif phi_power < 3.5:     
            freq['mood'] = 'negative'
        elif phi_power < 4.5:     
            freq['mood'] = 'strongly negative'   
        else:                     
            freq['mood'] = 'extremely negative'

        forecast.append(freq)    # Define current_quadrant variable

    #Calculate midpoints of each quadrant
    mid_q1 = sine_wave_min + em_amp_q1 / 2
    mid_q2 = mid_q1 + em_amp_q1
    mid_q3 = mid_q2 + em_amp_q2
    mid_q4 = float(sine_wave['max'])

    #Compare current sine wave value to determine quadrant
    if current_value <= mid_q1:
        current_quadrant = 1
        current_em_amp = em_amp_q1
        current_em_phase = em_phase_q1

    elif current_value <= mid_q2:
        current_quadrant = 2
        current_em_amp = em_amp_q2
        current_em_phase = em_phase_q2

    elif current_value <= mid_q3:
        current_quadrant = 3
        current_em_amp = em_amp_q3
        current_em_phase = em_phase_q3

    elif current_value <= mid_q4:
        current_quadrant = 4
        current_em_amp = em_amp_q4
        current_em_phase = em_phase_q4

    else:
        # Assign a default value
        current_em_amp = 0 
        current_em_phase = 0

    #Assign current EM amplitude and phase
    em_amp = current_em_amp
    em_phase = current_em_phase

    if percent_to_min_val < 20:
        print("Bullish momentum in trend")

        if current_quadrant == 1:
            # In quadrant 1, distance from min to 25% of range
            print("Bullish momentum in Q1")
        elif current_quadrant == 2:
            # In quadrant 2, distance from 25% to 50% of range
            print("Bullish momentum in Q2")
        elif current_quadrant == 3:
            # In quadrant 3, distance from 50% to 75% of range
            print("Bullish momentum in Q3")
        elif current_quadrant == 4:
            # In quadrant 4, distance from 75% to max of range
            print("Bullish momentum in Q4")

    elif percent_to_max_val < 20:
        print("Bearish momentum in trend")

        if current_quadrant == 1:
            # In quadrant 1, distance from min to 25% of range
            print("Bearish momentum in Q1")
        elif current_quadrant == 2:
            # In quadrant 2, distance from 25% to 50% of range
            print("Bearish momentum in Q2")
        elif current_quadrant == 3:
            # In quadrant 3, distance from 50% to 75% of range
            print("Bearish momentum in Q3")
        elif current_quadrant == 4:
            # In quadrant 4, distance from 75% to max of range
            print("Bearish momentum in Q4")

    # Calculate quadrature phase shift based on current quadrant  
    if current_quadrant == 1:  
        # Up cycle from Q1 to Q4   
        quadrature_phase = em_phase_q1
        em_phase = alpha_spiral
    elif current_quadrant == 2:
        quadrature_phase = em_phase_q2
        em_phase = omega_spiral          
    elif current_quadrant  == 3:      
        quadrature_phase = em_phase_q3
        em_phase = omega_spiral      
    else:          
        quadrature_phase = em_phase_q4
        em_phase = alpha_spiral 

    cycle_direction = "UP"
    next_quadrant = 1

    if current_quadrant == 1:
        next_quadrant = 2
        cycle_direction = "UP"

    elif current_quadrant == 2:
        if cycle_direction == "UP":
            next_quadrant = 3
        elif cycle_direction == "DOWN":
            next_quadrant = 1

    elif current_quadrant == 3:
        if cycle_direction == "UP":
            next_quadrant = 4
        elif cycle_direction == "DOWN":
            next_quadrant = 2

    elif current_quadrant == 4:
        if cycle_direction == "UP":
            next_quadrant = 3
            cycle_direction = "DOWN"

    # Calculate quadrature phase                       
    if next_quadrant == 1:     
        next_quadrature_phase = em_phase_q1            
    elif next_quadrant == 2:        
        next_quadrature_phase = em_phase_q2          
    elif next_quadrant == 3:                 
        next_quadrature_phase = em_phase_q3             
    else:              
        next_quadrature_phase = em_phase_q4

    # Calculate EM value
    em_value = em_amp * math.sin(em_phase)  

    # Calculate quadrature phase shift from current to next quadrant      
    quadrature = next_quadrature_phase - quadrature_phase

    if quadrature > 0:
        # Up cycle from Q1 to Q4  
        print("Up cycle now")  
    else:  
        # Down cycle from Q4 to Q1 
        print("Down cycle now")

    if current_quadrant == 1:
        # Quadrant 1
                
        if freq['number'] <= 10:
            # Most negative frequencies
            freq['em_amp'] = em_amp_q1
            freq['em_phase'] = em_phase_q1                 
            freq['mood'] = 'extremely negative'  

        elif freq['number'] >= 20:              
            freq['em_amp'] = em_amp_q1
            freq['em_phase'] = em_phase_q1  
            freq['mood'] = 'extremely positive'

    elif current_quadrant == 2:
        # Quadrant 2
                
        if freq['number'] > 10 and freq['number'] <= 15:                 
            freq['em_amp'] = em_amp_q2
            freq['em_phase'] = em_phase_q2
            freq['mood'] = 'strongly negative'
                        
        elif freq['number'] > 15 and freq['number'] <= 20:                 
            freq['em_amp'] = em_amp_q2
            freq['em_phase'] = em_phase_q2
            freq['mood'] = 'strongly positive'

    elif current_quadrant == 3: 
        # Quadrant 3
            
        if freq['number'] > 15 and freq['number'] < 20:            
            freq['em_amp'] = em_amp_q3                  
            freq['em_phase'] = em_phase_q3
            freq['mood'] = 'negative'              
           

        elif freq['number'] > 10 and freq['number'] < 15:            
            freq['em_amp'] = em_amp_q3                  
            freq['em_phase'] = em_phase_q3
            freq['mood'] = 'positive'
 
    else:      
        # Quadrant 4 
            
        if freq['number'] >= 20:                    
            freq['em_amp'] = em_amp_q4
            freq['em_phase'] = em_phase_q4  
            freq['mood'] = 'partial negative'       

        elif freq['number'] <= 10:                    
            freq['em_amp'] = em_amp_q4
            freq['em_phase'] = em_phase_q4  
            freq['mood'] = 'partial positive'

    freq['em_value'] = freq['em_amp'] * math.sin(freq['em_phase'])
        
    # Sort frequencies from most negative to most positive       
    frequencies.sort(key=lambda x: x['em_value'])   
        
    print("Quadrant is in: " + cycle_direction + " cycle")  
 
        
    for freq in frequencies:               
        print(freq['number'], freq['em_value'], freq['mood'])    
        
    # Calculate frequency spectrum index range based on most negative and positive frequencies
    mood_map = {
        'extremely negative': -4,  
        'strongly negative': -3,  
        'negative': -2,        
        'partial negative': -1,           
        'neutral': 0,
        'partial positive': 1, 
        'positive': 2,       
        'strongly positive': 3,    
        'extremely positive': 4   
        }

    if frequencies[0]['mood'] != 'neutral' and frequencies[-1]['mood'] != 'neutral':   
        total_mood = frequencies[0]['mood'] + " and " +  frequencies[-1]['mood']
    else:
        total_mood = 'neutral'

    print()

    # Update the frequencies for the next quadrant     
    if next_quadrant == 1:       
        # Update frequencies for next quadrant (Q1)               
        for freq in frequencies_next:       
            freq['em_amp'] = em_amp_q1       
            freq['em_phase'] = em_phase_q1
             
    elif next_quadrant == 2:
        # Update frequencies for Q2        
        for freq in frequencies_next:                 
            freq['em_amp'] = em_amp_q2   
            freq['em_phase'] = em_phase_q2 

    elif next_quadrant == 3:
        # Update frequencies for Q3        
        for freq in frequencies_next:                 
            freq['em_amp'] = em_amp_q3   
            freq['em_phase'] = em_phase_q3

    elif next_quadrant == 4:
        # Update frequencies for Q4        
        for freq in frequencies_next:                 
            freq['em_amp'] = em_amp_q4   
            freq['em_phase'] = em_phase_q4

    quadrant_1_amplitude = random.uniform(0.4, 0.6)  
    quadrant_1_phase = random.uniform(0.1, 0.3)

    quadrant_2_amplitude = random.uniform(0.8, 1.2)   
    quadrant_2_phase = random.uniform(0.4, 0.6)

    quadrant_3_amplitude = random.uniform(0.6, 1.0)        
    quadrant_3_phase = random.uniform(0.6, 0.8)

    quadrant_4_amplitude = random.uniform(1.0, 1.4)       
    quadrant_4_phase = random.uniform(0.8, 1.0)

    lowest_frequency = float('inf')
    highest_frequency = 0

    min_quadrant = None  
    max_quadrant = None

    if current_quadrant == 1:
        frequency_amplitude = quadrant_1_amplitude  
        frequency_phase = quadrant_1_phase  
        current_frequency = frequency_amplitude * frequency_phase

    elif current_quadrant == 2:        
        frequency_amplitude = quadrant_2_amplitude   
        frequency_phase = quadrant_2_phase 
        current_frequency = frequency_amplitude * frequency_phase

    elif current_quadrant == 3:        
        frequency_amplitude = quadrant_3_amplitude
        frequency_phase = quadrant_3_phase
        current_frequency = frequency_amplitude * frequency_phase
  
    elif current_quadrant == 4:        
        frequency_amplitude = quadrant_4_amplitude       
        frequency_phase = quadrant_4_phase
        current_frequency = frequency_amplitude * frequency_phase

    if current_frequency == lowest_frequency:
        min_node = {'frequency': current_frequency, 'quadrant': current_quadrant}

    if current_frequency == highest_frequency:      
        max_node = {'frequency': current_frequency, 'quadrant': current_quadrant}

    # Get next quadrant phi 
    next_phi = PHI ** freq['number'] 

    # Map moods based on inverse phi power         
    if next_phi < 1.2:
        freq['mood'] = 'extremely positive' 
    elif next_phi < 1.4:
        freq['mood'] = 'positive'

    highest_3 = frequencies[:3]
    lowest_3 = frequencies[-3:]

    mood_map = {
        'extremely negative': -4,  
        'strongly negative': -3,  
        'negative': -2,        
        'partial negative': -1,           
        'neutral': 0,
        'partial positive': 1, 
        'positive': 2,       
        'strongly positive': 3,    
        'extremely positive': 4   
         }

    highest_3_mood_values = []
    for freq in highest_3:   
        if freq['mood'] == 'neutral':
            highest_3_mood_values.append(0)   
        else:
            highest_3_mood_values.append(mood_map[freq['mood']])

    lowest_3_mood_values = []        
    for freq in lowest_3:   
        if freq['mood'] == 'neutral':
            lowest_3_mood_values.append(0)        
        else:
            lowest_3_mood_values.append(mood_map[freq['mood']])      

    highest_3_mood_values = [mood_map[freq['mood']] for freq in highest_3]
    highest_3_mood = statistics.mean(highest_3_mood_values)

    lowest_3_mood_values = [mood_map[freq['mood']] for freq in lowest_3]
    lowest_3_mood = statistics.mean(lowest_3_mood_values)

    print(f"Current quadrant: {current_quadrant}")
    print(f"Next quadrant: {next_quadrant}")
    print(f"Highest 3 frequencies: {highest_3_mood}")        
    print(f"Lowest 3 frequencies: {lowest_3_mood}")

    if highest_3_mood > 0:
        print(f"Cycle mood is negative")
    elif highest_3_mood < 0:      
        print(f"Cycle mood is positive") 
    else:
        print("Cycle mood is neutral")

    if frequencies[0]['mood'] != 'neutral' and frequencies[-1]['mood'] != 'neutral':        
        if mood_map[frequencies[0]['mood']] < 0:
            total_mood = f"{frequencies[0]['mood']} and  {frequencies[-1]['mood']}"
            print(f"Frequency spectrum index range: {total_mood} ")
            print(f"Freq. range is negative")
        else:    
            total_mood = f"{frequencies[0]['mood']} and {frequencies[-1]['mood']}"
            print(f"Frequency spectrum index range: {total_mood}") 
            print(f"Freq. range is positive")   
    else:
        print(f"Frequency spectrum index range: neutral")
        print(f"Freq. range is neutral") 

    print()

    # Sort forecast from most negative to most positive       
    forecast.sort(key=lambda f: mood_map[f['mood']])  
    
    # Get average mood of highest/lowest 3 frequencies 
    highest_3 = forecast[:3]
    highest_3_mood = 0
   
    for freq in highest_3:
        mood_val  = mood_map[freq['mood']] 
        highest_3_mood += mood_val
        highest_3_mood = highest_3_mood / len(highest_3)

    lowest_3 = forecast[-3:]
    lowest_3_mood = 0

    for freq in lowest_3:
        mood_val  = mood_map[freq['mood']]
        lowest_3_mood += mood_val  
        lowest_3_mood = lowest_3_mood / len(lowest_3)

    for freq in forecast:
        freq['magnitude'] = np.abs(freq['em_value'])

    n = 10

    def calculate_weighted_avg(weights, values):
        total = 0 
        total_weight = 0
    
        for w, v in zip(weights, values):
            magnitude = w 
            mood = v
            total += magnitude * mood
            total_weight += magnitude
        
            if total_weight == 0:
                return 0

            return total / total_weight

    # Calculate weighted averages  
    top_n_weights = [freq['magnitude'] for freq in forecast[:n]]
    top_n_moods = [mood_map[freq['mood']] for freq in forecast[:n]]
    top_n_weighted_avg = calculate_weighted_avg(top_n_weights, top_n_moods)

    bottom_n_weights = [freq['magnitude'] for freq in forecast[-n:]]  
    bottom_n_moods = [mood_map[freq['mood']] for freq in forecast[-n:]]
    bottom_n_weighted_avg = calculate_weighted_avg(bottom_n_weights, bottom_n_moods)

    overall_mood = top_n_weighted_avg - bottom_n_weighted_avg

    if overall_mood > 2:
        print("Strongly bullish mood")
    elif overall_mood > 1:       
        print("Bullish mood")
    elif overall_mood > 0:
        print("Mildly bullish mood")     
    elif overall_mood == 0:
        print("Neutral mood")          
    elif overall_mood > -1:        
        print("Mildly Bearish mood")      
    elif overall_mood > -2:       
        print("Bearish mood")  
    else:
        print("Strongly bearish mood")

    if overall_mood < -3:
        print("Extremely bearish")  
    elif overall_mood < -2:           
        print("Strongly bearish")
    elif overall_mood < -1:       
        print("Bearish")           
    elif overall_mood == 0:
        print("Neutral")
    elif overall_mood > 1:        
        print("Bullish")      
    elif overall_mood > 2:        
        print("Strongly Bullish")            
    else:
        print("Extremely bullish")

    # Define stationary circuit variables
    stationary_circuit = []

    # Map quadrants to triangle points of Metatron's Cube
    quadrant_map = {
        1: 'Apex',
        2: 'Left',  
        3: 'Base',
        4: 'Right' 
        }

    # Add triangle points to circuit
    stationary_circuit.append('Apex')
    stationary_circuit.append('Left')     
    stationary_circuit.append('Base')
    stationary_circuit.append('Right')

    # Loop through each quadrant cycle        
    for quadrant in [1,2,3,4]:
    
        #print(f"Quadrant {quadrant}")
            
        # Get triangle point from quadrant map               
        point = quadrant_map[quadrant]    
    
        #print(f"Current point: {point}")
    
        # Get next point based on circuit        
        if point == 'Apex':
            next_point = 'Left'
        elif point == 'Left':
            next_point = 'Base'         
        elif point == 'Base':
            next_point = 'Right'
        elif point == 'Right':
            next_point = 'Apex'
        
        #print(f"Next point: {next_point}")       
    
        # Get frequency and mood forecast
        frequency = frequencies[quadrant]['frequency']
        mood = frequencies[quadrant]['mood']
    
        #print(f"Frequency: {frequency} Hz - Mood: {mood}")
    
    if current_frequency > lowest_frequency:
        lowest_frequency = current_frequency    
    if current_frequency < highest_frequency:     
        highest_frequency = current_frequency

    # Define min and max nodal points based on frequencies
    min_node = { 
        'frequency': lowest_frequency,  
        'quadrant': min_quadrant   
        }

    max_node = {
        'frequency': highest_frequency,
        'quadrant': max_quadrant         
        }

    if current_frequency < lowest_frequency:
        lowest_frequency = current_frequency    
        quadrant = current_quadrant   

    min_node = {'frequency': lowest_frequency, 'quadrant': quadrant}  

    if current_frequency > highest_frequency:      
        highest_frequency = current_frequency   
        max_quadrant = current_quadrant   

    max_node = {'frequency': highest_frequency, 'quadrant': max_quadrant}

    # Loop through each quadrant cycle        
    for quadrant in [1,2,3,4]:
    
        # Check if current quadrant is a min or max node
        if quadrant == min_node['quadrant']:
            print(f"Node reached at frequency {min_node['frequency']} Hz")
        
        elif quadrant == max_node['quadrant']:
            print(f"Node reached at frequency {max_node['frequency']} Hz")

    print()

    # Calculate forecast mood based on frequencies and nodal points
    forecast = {
        'mood' : None,
        'min_reversal' : {
            'time': None,
            'quadrant': None
            },  
        'max_reversal' : {
            'time': None,     
            'quadrant': None      
            }       
        }

    if lowest_3_mood < 0:
        forecast['mood'] = 'positive'
    elif highest_3_mood > 0:      
        forecast['mood'] = 'negative'
    else:
        forecast['mood'] = 'neutral'

    # Calculate time to min nodal point reversal
    freq = min_node['frequency']  
    period = 1/freq   
    min_time = period/4    
    forecast['min_reversal']['time'] = min_time
    forecast['min_reversal']['quadrant'] = min_node['quadrant']

    # Calculate time to max nodal point reversal    
    freq = max_node['frequency']         
    period = 1/freq   
    max_time = period/4  
                        
    forecast['max_reversal']['time'] = max_time
    forecast['max_reversal']['quadrant'] = max_node['quadrant']

    # Print forecast   
    print(forecast)

    # Print overall mood  
    print(f"Overall market mood: {forecast['mood']}")

    print()

    # Define octahedron points mapped to Metatron's Cube
    octahedron = [
        {'point': 'Apex',       'frequency': None, 'mood': None},
        {'point': 'Left',       'frequency': None, 'mood': None},
        {'point': 'Base',       'frequency': None, 'mood': None},
        {'point': 'Right',      'frequency': None, 'mood': None},
        {'point': 'Phi',        'frequency': None, 'mood': None},  
        {'point': 'Pi',         'frequency': None, 'mood': None},
        {'point': 'e',          'frequency': None, 'mood': None},
        {'point': 'Origin',     'frequency': None, 'mood': None}    
        ]
 

    # Update octahedron points with frequencies and moods
    for point in octahedron:
        if point['point'] == quadrant_map[current_quadrant]:
            point['frequency'] = current_frequency
            point['mood'] = frequencies[current_quadrant]['mood']

    valid_points = [p for p in octahedron if p['frequency'] is not None] 

    # Find minimum and maximum frequency points   
    min_point = min(valid_points, key=lambda p: p['frequency'])   
    max_point = max(valid_points, key=lambda p: p['frequency'])

    # Calculate reversal predictions from min and max points
    forecast = {
        'min_reversal': {
            'time': None,
            'point': None
            },
        'max_reversal': {
            'time': None,
            'point': None   
            }
        }

    freq = min_point['frequency']
    period = 1/freq
    forecast['min_reversal']['time'] = period/4               
    forecast['min_reversal']['point'] = min_point['point']

    freq = max_point['frequency']
    period = 1/freq                             
    forecast['max_reversal']['time'] = period/4               
    forecast['max_reversal']['point'] = max_point['point']

    # Prints                    
    print(f"Apex: {octahedron[0]['frequency']}")    
    print(f"Left: {octahedron[1]['frequency']}")      
    print(f"Base: {octahedron[2]['frequency']}")
    print(f"Right: {octahedron[3]['frequency']}")
    print(f"Phi: {octahedron[4]['frequency']}")
    print(f"Pi: {octahedron[5]['frequency']}")
    print(f"e: {octahedron[6]['frequency']}")
    print(f"Origin: {octahedron[7]['frequency']}")

    print("Current point is at: ", forecast[f'min_reversal']['point'] if forecast[f'min_reversal']['point'] == point else forecast[f'max_reversal']['point']) 

    # Extract current quadrant and EM phase
    current_quadrant = sine_wave["current_quadrant"]   
    em_phase = sine_wave["em_phase"]

    mood_to_corr = {
        'extremely negative': -0.9,   
        'strongly negative': -0.7,
        'negative': -0.5,       
        'partial negative': -0.3,           
        'neutral': 0,
        'partial positive': 0.3,  
        'positive': 0.5,       
        'strongly positive': 0.7,    
        'extremely positive': 0.9   
        }

    momentum_map = {      
        'extremely negative': -5,   
        'strongly negative': -4,     
        'negative': -3,       
        'partial negative': -2,           
        'neutral': 0,
        'partial positive': 2,   
        'positive': 3,       
        'strongly positive': 4,    
        'extremely positive': 5  
        }

    current_momentum = 0

    for freq in frequencies:

        corr = mood_to_corr[freq['mood']]
        z = np.arctanh(corr) 
        freq['z'] = z
        freq['pos'] = 0 if corr == 0 else z

        # Get momentum score       
        momentum = momentum_map[freq['mood']]
      
        # Calculate weighted momentum score      
        weighted_momentum = momentum * freq['magnitude']         
  
        current_momentum += weighted_momentum
                        
    frequencies.sort(key=lambda f: f['z'])

    # Calculate average momentum      
    current_momentum /= len(frequencies)

    # Forecast mood 
    if current_momentum < -2:
        forecast_mood = 'bearish'
    elif current_momentum < 0:
        forecast_mood = 'slightly bearish'    
    elif current_momentum == 0:        
        forecast_mood = 'neutral'
    elif current_momentum > 0:        
        forecast_mood = 'slightly bullish'
    elif current_momentum > 2:
        forecast_mood = 'bullish'

    print(f"Current momentum: {current_momentum}")     
    print(f"Trend forecast: {forecast_mood}")    

    print()

    em_amp = []  
    em_phase = []

    em_amp.append(current_em_amp)
    em_phase.append(current_em_phase)



    return em_amp, em_phase, current_momentum, forecast_mood, point, next_point

sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                                           percent_to_max_val=5, 
                                           percent_to_min_val=5)      

sine_wave_max = sine_wave["max"]   
sine_wave_min = sine_wave["min"]

octa_metatron_cube(close_prices, candles)  
print(octa_metatron_cube(close_prices, candles))

print()


print()

##################################################
##################################################

print("Init main() loop: ")

print()

##################################################
##################################################

def main():
    # Load credentials from file
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()

    # Instantiate Binance client
    client = BinanceClient(api_key, api_secret)

    get_account_balance()

    ##################################################
    ##################################################

    # Define timeframes
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']
    TRADE_SYMBOL = "BTCUSDT"
    symbol = TRADE_SYMBOL

    trade_entry_pnl = 0
    trade_exit_pnl = 0

    current_pnl = 0.0

    ##################################################
    ##################################################

    print()

    ##################################################
    ##################################################

    while True:

        ##################################################
        ##################################################
        
        # Get fresh closes for the current timeframe
        closes = get_closes('1m')
       
        # Get close price as <class 'float'> type
        close = get_close('1m')
        
        # Get fresh candles  
        candles = get_candles(TRADE_SYMBOL, timeframes)
                
        # Update candle_map with fresh data
        for candle in candles:
            timeframe = candle["timeframe"]  
            candle_map.setdefault(timeframe, []).append(candle)
                
        ##################################################
        ##################################################

        try:     
            ##################################################
            ##################################################

            url = "https://fapi.binance.com/fapi/v1/ticker/price"

            params = {
                "symbol": "BTCUSDT" 
                }

            response = requests.get(url, params=params)
            data = response.json()

            price = data["price"]
            #print(f"Current BTCUSDT price: {price}")

            # Define the current time and close price
            current_time = datetime.datetime.now()
            current_close = price

            #print("Current local Time is now at: ", current_time)
            #print("Current close price is at : ", current_close)

            print()

            ##################################################
            ##################################################

            # Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
            min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

            print("Momentum sinewave signal:", momentum_signal)
            print()

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

            print()


            ##################################################
            ##################################################

            # closes = get_closes("1m")     
            n_components = 5

            current_time, entry_price, stop_loss, fastest_target, market_mood = get_target(closes, n_components, target_distance=56)

            print("Current local Time is now at: ", current_time)
            print("Market mood is: ", market_mood)
            market_mood_fft = market_mood
 
            print()

            print("Current close price is at : ", current_close)

            print()

            print("Fastest target is: ", fastest_target)

            print()

            # Get the current price
            price = get_current_price()

            print()

            price = float(price)

            print()

            ##################################################
            ##################################################

            # Call the calculate_reversal_and_forecast function with the example data
            (current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression, last_reversal, forecast_dip, forecast_top) = calculate_reversal_and_forecast(close)
            print("Forecast Direction:", forecast_direction if forecast_direction is not None else "None")

            # Handle NaN and None values for Forecast Price FFT
            if forecast_price_fft is None or np.isnan(forecast_price_fft):
                forecast_price_fft = close_np[-1]
            print("Forecast Price FFT:", forecast_price_fft)

            forecast_price_fft = float(forecast_price_fft)

            # Handle NaN and None values for Future Price Regression
            if future_price_regression is None or np.isnan(future_price_regression[1][0]):
                future_price_regression = close_np[-1]
            else:
                future_price_regression = future_price_regression[1][0]
            print("Future Price Regression:", future_price_regression)

            future_price_regression = float(future_price_regression)

            print()

            ##################################################
            ##################################################

            # Example close prices
            close_prices = np.array(close)  # Insert your actual close prices here

            result = calculate_price_distance_and_wave(price, close_prices)  # Assuming 'price' is defined somewhere

            # Unpack the result dictionary into individual variables
            price_val = result['price']
            ht_sine_value = result['ht_sine_value']
            normalized_distance_to_min = result['normalized_distance_to_min']
            normalized_distance_to_max = result['normalized_distance_to_max']
            min_price_val = result['min_price']
            max_price_val = result['max_price']
            market_mood_val = result['market_mood']

            # Print the detailed information for the given price using the individual variables
            print(f"Price: {price_val:.2f}")
            print(f"HT_SINE Value: {ht_sine_value:.2f}")
            print(f"Normalized Distance to Min: {normalized_distance_to_min:.2f}%")
            print(f"Normalized Distance to Max: {normalized_distance_to_max:.2f}%")
            print(f"Min Price: {min_price_val:.2f}")
            print(f"Max Price: {max_price_val:.2f}")
            print(f"Market Mood: {market_mood_val}")

            print()

            ##################################################
            ##################################################

            min_threshold, max_threshold, avg_mtf, momentum_signal, _ = calculate_thresholds(close, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

            print("Momentum signal:", momentum_signal)
            print("Minimum threshold:", min_threshold)
            print("Maximum threshold:", max_threshold)
            print("Average MTF:", avg_mtf)

            market_mood, forecast_price = calculate_sine_wave_and_forecast(closes, min_threshold, max_threshold)
            if market_mood:
                print(f"Forecast price: {forecast_price}")

            closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - closes[-1]))
            if closest_threshold == min_threshold:
                print("The last minimum value is closest to the current close.")
            elif closest_threshold == max_threshold:
                print("The last maximum value is closest to the current close.")
            else:
                print("No threshold value found.")

            print()

            ##################################################
            ##################################################

            # Calculate the 45-degree angle (simple linear regression)
            x = np.arange(len(close))
            slope, intercept = np.polyfit(x, close, 1)

            # Calculate the expected trend line value for the last close price
            expected_price = slope * len(close) + intercept

            # Display the expected price on the 45-degree angle trend
            print(f"Expected price on the 45-degree angle trend: {expected_price}")

            # Generate forecast based on the 45-degree angle
            forecast_result = forecast_45_degree_angle(price, expected_price)

            # Display the forecast result
            print(forecast_result)

            print()

            ##################################################
            ##################################################

            val_low, val_high, sup, res, mood, forecast = analyze_market_profile(close)

            print(f"Value Area Low: {val_low}, Value Area High: {val_high}")
            print(f"Current Support: {sup}, Current Resistance: {res}")
            print(f"Market Mood: {mood}")
            print(f"Forecasted Price: {forecast}")

            print()

            ##################################################
            ##################################################

            # Initialize variables
            trigger_long = False 
            trigger_short = False

            current_time = datetime.datetime.utcnow() + timedelta(hours=3)

            print()

            print("Last reversal keypoint was: ", closest_threshold)
            
            print()

            ##################################################
            ##################################################

            forecasted_price = identify_reversals(close)

            print()

            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################

            # Forecast prices
            fast_price, medium_price, slow_price = forecast_prices(close)

            # Print forecasted prices
            print(f"Forecasted price (fast cycle): {fast_price:.2f}")
            print(f"Forecasted price (medium cycle): {medium_price:.2f}")
            print(f"Forecasted price (slow cycle): {slow_price:.2f}")

            fast_price = float(fast_price)
            medium_price = float(medium_price)
            slow_price = float(slow_price)

            print()

            ##################################################
            ##################################################

            # Capture the targets when calling the function
            results, targets = forecast_sma_targets(price)

            # Print each output string separately
            for result in results:
                print(result)

            # Assign each target_45 value to a separate variable
            target_45_quad_1 = targets['target_45_quad_1']
            target_45_quad_2 = targets['target_45_quad_2']
            target_45_quad_3 = targets['target_45_quad_3']
            target_45_quad_4 = targets['target_45_quad_4']

            # Print each variable
            print(f"Target_45 for Quadrant 1: {target_45_quad_1:.2f}")
            print(f"Target_45 for Quadrant 2: {target_45_quad_2:.2f}")
            print(f"Target_45 for Quadrant 3: {target_45_quad_3:.2f}")
            print(f"Target_45 for Quadrant 4: {target_45_quad_4:.2f}")
            print("-" * 50)

            print()

            ##################################################
            ##################################################

            timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

            # Calculate momentum for each timeframe
            momentum_values = {}
            for timeframe in timeframes:
                momentum = get_momentum(timeframe)
                momentum_values[timeframe] = momentum
                print(f"Momentum for {timeframe}: {momentum}")

            # Convert momentum to a normalized scale and determine if it's positive or negative
            normalized_momentum = {}
            for timeframe, momentum in momentum_values.items():
                normalized_value = (momentum + 100) / 2  # Normalize to a scale between 0 and 100
                normalized_momentum[timeframe] = normalized_value
                print(f"Normalized Momentum for {timeframe}: {normalized_value:.2f}%")

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

            print()

            ##################################################
            ##################################################

            timeframe = '1m'
            momentum = get_momentum(timeframe)
            print("Momentum on 1min tf is at: ", momentum)

            print()

            ##################################################
            ##################################################

            # Example usage:
            closes = close

            signal = impulse_momentum_overall_signal(closes)
            print("Overall Signal:", signal)

            print()

            ##################################################
            ##################################################

            reversals = detect_reversals(close)

            for reversal in reversals:
                min_threshold, max_threshold = adjust_stationary_object(min_threshold, max_threshold, reversal)

            market_mood_type = analyze_market_mood(reversals)

            print(f"Market Mood: {market_mood_type}")

            print()

            ##################################################
            ##################################################

            market_mood_fastfft = fft_trading_signal(close)

            #print(f"Market Mood: {market_mood_fastfft}")

            print()

            ##################################################
            ##################################################

            # Analyze the market behavior based on the sample data
            analysis_results = analyze_market_behavior(close)

            # Print the results separately
            print("Market Mood:", analysis_results["Market Mood"])
            print("Incoming Reversal:", analysis_results["Incoming Reversal"])

            incoming_reversal = analysis_results["Incoming Reversal"]

            print()

            ##################################################
            ##################################################

            market_mood_poly = forecast_price_and_mood(close)

            print(f"Market Mood: {market_mood_poly}")

            print()

            ##################################################
            ##################################################

            # Hypothetical current price (you can replace this with the actual current price)
            current_price = price

            # Calculate division of the circle based on the golden ratio
            div1, div2 = calculate_phi_division()
            #print(f"Dividing the circumference by phi ({GOLDEN_RATIO}) results in approximately {div1:.2f}° and {div2:.2f}°.")

            # Calculate the forecasted price using the golden ratio
            forecasted_phi_price = calculate_forecast_price(current_price)
            print(f"The forecasted price based on the golden ratio is: ${forecasted_phi_price:.2f}")

            # Determine the market mood based on the forecasted price and current price
            market_mood_phi = determine_market_mood(current_price, forecasted_phi_price)
            print(market_mood_phi)

            # Calculate and print the intraday target and associated market mood
            momentum_target = calculate_intraday_target(current_price)
            market_mood_momentum = determine_market_mood(current_price, momentum_target)

            print(f"The momentum target price is: ${momentum_target:.2f}")
            print(market_mood_momentum)

            # Calculate and print the momentum target and associated market mood
            intraday_target = calculate_momentum_target(current_price)
            market_mood_intraday = determine_market_mood(current_price, intraday_target)

            print(f"The intraday target price is: ${intraday_target:.2f}")
            print(market_mood_intraday)

            print()

            ##################################################
            ##################################################

            keypoints = find_reversal_keypoints(close)
            print("Reversal keypoints: ", keypoints)

            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(np.array(close).reshape(-1, 1))
            model = LinearRegression().fit(X_poly, close)

            future = model.predict(poly_features.transform([[close[-1]]]))
            print("Forecast price using polyfit in real value: ", future[0])

            coefficients = np.polyfit(range(len(close)), close, 1)
            slope = coefficients[0]
            print("Regression slope: ", slope)

            regression_mood = classify_trend(close)
            print(regression_mood)

            print()

            ##################################################
            ##################################################

            mood_mom, mom_forecast = dynamic_momentum_gauge(close, price)

            if mood_mom is not None and forecast_price is not None:
                print(f"Market Mood: {mood_mom}")
                print(f"Forecast Price: {mom_forecast}")

            print()

            ##################################################
            ##################################################

            forecast_price, market_mood, forecast_5min, forecast_15min, forecast_30min, forecast_1h = square_of_9(close)

            print(f"Forecasted Price (Current Cycle): {forecast_price}")
            print(f"Market Mood (Current Cycle): {market_mood}")
            print(f"Forecast for Next 5 Minutes: {forecast_5min}")
            print(f"Forecast for Next 15 Minutes: {forecast_15min}")
            print(f"Forecast for Next 30 Minutes: {forecast_30min}")
            print(f"Forecast for Next 1 Hour: {forecast_1h}")

            print()

            ##################################################
            ##################################################

            predicted_market_mood = analyze_fft_for_hexagon(close)

            print("Predicted Market Mood:", predicted_market_mood)

            print()

            ##################################################
            ##################################################

            result_cycles = trend_forecast(close)

            print(result_cycles)

            print()

            ##################################################
            ##################################################

            trend_mood, market_quadrant, support_level, resistance_level, market_mood_trend, forecasted_price_trend = forecast_market_trends(close)

            # Print the results
            print(f"Market Sentiment: {trend_mood}")
            print(f"Market Quadrant: {market_quadrant}")
            print(f"Support Level: {support_level}")
            print(f"Resistance Level: {resistance_level}")
            print(f"Market Mood: {market_mood_trend}")
            print(f"Forecasted Price: {forecasted_price_trend}")

            print()

            ##################################################
            ##################################################

            pivot_and_forecast = calculate_pivot_point_and_forecast(close)

            # Print statements moved outside the function
            pivot_mood = pivot_and_forecast['Market Mood']
            pivot_forecast = pivot_and_forecast['Forecast Price']

            print(f"Market Mood: {pivot_mood}")
            print(f"Forecast Price: {pivot_forecast}")

            print()

            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################

            close = get_close('15m')

            # Call the scale_list_to_sine function
            dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_list_to_sine(close)

            # Print the results
            print("For the given close prices:")
            print(f"Distance to min: {dist_from_close_to_min:.2f}%")
            print(f"Distance to max: {dist_from_close_to_max:.2f}%")
            print(f"Current Sine value: {current_sine}\n")

            print()

            ##################################################
            ##################################################

            # Example usage:
            analysis_result = analyze_market(close)
            roc_mood = analysis_result['market_mood']

            # Print information outside the function
            print(f"Current ROC value: {analysis_result['roc_value']}")
            print(f"Current market mood: {analysis_result['market_mood']}")
            print(f"Last reversal was a {analysis_result['last_reversal']} at price: {analysis_result['last_reversal_price']}")
            print(f"HFT Projected price: {analysis_result['hft_projected_price']}")  # Added line for HFT projected price
            print(f"Standard Projected price: {analysis_result['standard_projected_price']}")  # Added line for standard projected price
            print(f"ROC forecast price for even faster targets: {analysis_result['roc_faster_forecast_price']}")

            print()

            ##################################################
            ##################################################

            # Example usage
            dom_mood, dom_forecast = market_dom_analysis(close)

            # Print results outside the function
            print(f"Market Mood: {dom_mood}")
            print(f"Price Forecast: {dom_forecast}")

            print()

            ##################################################
            ##################################################

            # Example usage:
            unitcircle_price, unitcircle_mood = forecast_unit_price(close)

            print(f"Forecasted Price: {unitcircle_price:.2f}")
            print(f"Market Mood: {unitcircle_mood}")

            print()

            ##################################################
            ##################################################

            # Initialize overall_sentiments_sine dictionary
            overall_sentiments_sine = {'Positive': 0, 'Negative': 0}

            # Create a ThreadPoolExecutor for parallel execution
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a list of futures for each timeframe
                futures = {executor.submit(process_timeframe, timeframe): timeframe for timeframe in timeframes}

            # Iterate over completed futures to process results
            for future in concurrent.futures.as_completed(futures):
                timeframe, dist_from_close_to_min, dist_from_close_to_max, current_sine, sentiment = future.result()

                # Update overall sentiment count
                overall_sentiments_sine[sentiment] += 1

                # Print the results for each timeframe
                print(f"For {timeframe} timeframe:")
                print(f"Distance to min: {dist_from_close_to_min:.2f}%")
                print(f"Distance to max: {dist_from_close_to_max:.2f}%")
                print(f"Current Sine value: {current_sine}")
                print(f"Sentiment: {sentiment}\n")

            # Print overall sentiment analysis
            for sentiment, count in overall_sentiments_sine.items():
                print(f"Overall Market Sentiment for {sentiment}: {count}")

            # Determine the overall dominant sentiment for the Sine Wave
            positive_sine_count = overall_sentiments_sine['Positive']
            negative_sine_count = overall_sentiments_sine['Negative']

            if positive_sine_count > negative_sine_count:
                print("Overall dominant Sine Wave: Positive")
            elif positive_sine_count < negative_sine_count:
                print("Overall dominant Sine Wave: Negative")
            else:
                print("Overall dominant Sine Wave: Balanced")

            print()

            ##################################################
            ##################################################

            total_volume = calculate_volume(candles)
            buy_volume_5min, sell_volume_5min, buy_volume_3min, sell_volume_3min , buy_volume_1min, sell_volume_1min = calculate_buy_sell_volume(candles)

            (support_levels_1min, resistance_levels_1min, support_levels_3min, resistance_levels_3min, support_levels_5min, resistance_levels_5min) = calculate_support_resistance(candles)

            total_volume_5min = get_volume_5min(candles)
            total_volume_1min = get_volume_1min(candles)

            small_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 2)
            medium_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 5)
            large_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 10)

            small_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 2)
            medium_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 5)
            large_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 10)

            small_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 2)
            medium_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 5)
            large_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 10)

            higher_support_5min, higher_resistance_5min = get_higher_timeframe_data(TRADE_SYMBOL, "5m")

            print("Total Volume:", total_volume)
            print("Total Volume (5min tf):", total_volume_5min)

            print()

            print("Buy Volume (5min tf):", buy_volume_5min)
            print("Sell Volume (5min tf):", sell_volume_5min)

            print()

            print("Buy Volume (3min tf):", buy_volume_3min)
            print("Sell Volume (3min tf):", sell_volume_3min)

            print()

            print("Buy Volume (1min tf):", buy_volume_1min)
            print("Sell Volume (1min tf):", sell_volume_1min)

            print()

            # Print support and resistance levels for the 5-minute timeframe
            print("Support Levels (5min tf):", support_levels_5min[-1])
            print("Resistance Levels (5min tf):", resistance_levels_5min[-1])

            # Print support and resistance levels for the 3-minute timeframe
            print("Support Levels (3min tf):", support_levels_3min[-1])
            print("Resistance Levels (3min tf):", resistance_levels_3min[-1])

            # Calculate and print support and resistance levels for the 1-minute timeframe
            print("Support Levels (1min tf):", support_levels_1min[-1])
            print("Resistance Levels (1min tf):", resistance_levels_1min[-1])

            support_levels_modified = [min(support, candles[-1]["close"]) for support in support_levels_5min]
            resistance_levels_modified = [max(resistance, candles[-1]["close"]) for resistance in resistance_levels_5min]

            # Calculate Bollinger Bands and Poly Channel for 5-minute timeframe
            upper_bb_5min, lower_bb_5min = calculate_bollinger_bands(candles)
            upper_poly_5min, lower_poly_5min = calculate_poly_channel(candles)

            # Calculate the spread factor and number of levels
            spread_factor = 0.02
            num_levels = 5

            # Calculate modified support and resistance levels with spread and additional levels
            #price = candles[-1]["close"]
            support_spread = price * spread_factor
            resistance_spread = price * spread_factor

            # Select the appropriate support and resistance levels based on the desired timeframe
            desired_timeframe = "5m"

            if desired_timeframe == "1m":
                support_levels_selected, resistance_levels_selected = support_levels_1min, resistance_levels_1min
            elif desired_timeframe == "3m":
                support_levels_selected, resistance_levels_selected = support_levels_3min, resistance_levels_3min
            else:
                support_levels_selected, resistance_levels_selected = support_levels_5min, resistance_levels_5min

            # Modify the selected support and resistance levels
            support_levels_modified = [min(support, candles[-1]["close"]) for support in support_levels_selected]
            resistance_levels_modified = [max(resistance, candles[-1]["close"]) for resistance in resistance_levels_selected]

            # Calculate modified support and resistance levels with spread and additional levels
            modified_support_levels = [price - i * support_spread for i in range(num_levels, 0, -1)]
            modified_resistance_levels = [price + i * resistance_spread for i in range(num_levels)]

            # Rule for identifying reversal dips and tops
            if price <= lower_bb_5min[-1] and buy_volume_5min > sell_volume_5min and modified_support_levels and modified_resistance_levels:
                if all(level < small_lvrg_levels_5min[0] for level in modified_support_levels) and all(level < medium_lvrg_levels_5min[0] for level in modified_support_levels) and all(level < large_lvrg_levels_5min[0] for level in modified_support_levels):
                    print("Potential Reversal Dip (5min): Close at or below Bollinger Bands Lower Band and More Buy Volume at Support")
                #elif buy_volume_5min > sell_volume_5min:
                    #print("Potential Reversal Dip (5min): Close at or below Bollinger Bands Lower Band")

            if price >= upper_bb_5min[-1] and sell_volume_5min > buy_volume_5min and modified_support_levels and modified_resistance_levels:
                if all(level > small_lvrg_levels_5min[0] for level in modified_resistance_levels) and all(level > medium_lvrg_levels_5min[0] for level in modified_resistance_levels) and all(level > large_lvrg_levels_5min[0] for level in modified_resistance_levels):
                    print("Potential Reversal Top (5min): Close at or above Bollinger Bands Upper Band and More Sell Volume at Resistance")
                #elif sell_volume_5min > buy_volume_5min:
                    #print("Potential Reversal Top (5min): Close at or above Bollinger Bands Upper Band")

            print()

            print("Lower BB is now at: ", lower_bb_5min[-1])
            print("Upper BB is now at: ", upper_bb_5min[-1])
            print("Lower Poly is now at: ", lower_poly_5min[-1])
            print("Upper Poly is now at: ", upper_poly_5min[-1])

            print()

            distance_to_lower = abs(price - lower_bb_5min[-1])
            distance_to_upper = abs(price - upper_bb_5min[-1])
    
            if distance_to_lower < distance_to_upper:
                print("Price is closer to the Lower Bollinger Band")
            elif distance_to_upper < distance_to_lower:
                print("Price is closer to the upper Bollinger Band")
            else:
                print("Price is equidistant to both Bollinger Bands")

            print()

            if buy_volume_5min > sell_volume_5min:
                print("Buy vol on 5min tf is higher then sell vol: BULLISH")
            elif sell_volume_5min > buy_volume_5min:
                print("Sell vol on 5min tf is higher then buy vol: BEARISH")

            if buy_volume_3min > sell_volume_3min:
                print("Buy vol on 3min tf is higher then sell vol: BULLISH")
            elif sell_volume_3min > buy_volume_3min:
                print("Sell vol on 3min tf is higher then buy vol: BEARISH")

            if buy_volume_1min > sell_volume_1min:
                print("Buy vol on 1min tf is higher then sell vol: BULLISH")
            elif sell_volume_1min > buy_volume_1min:
                print("Sell vol on 1min tf is higher then buy vol: BEARISH")

            print()

            ##################################################
            ##################################################

            current_cycle, current_quadrant, mood_factor, forecasted_price = market_analysis(close_prices, closest_threshold, min_threshold, max_threshold)

            # Print results
            print(f"Current Cycle: {current_cycle}")
            print(f"Current Quadrant: {current_quadrant}")
            print(f"Mood Factor: {mood_factor}")
            print(f"Forecasted Price: {forecasted_price}")

            print()

            ##################################################
            ##################################################

            # Calculate impulse, energy, and momentum
            calculations = calculate_impulse_energy_momentum(close)
            print("Impulse:", calculations['impulse'])
            print("Energy:", calculations['energy'])
            print("Momentum:", calculations['momentum'])
            print("Market Mood:", calculations['market_mood'])
            hft_momentum = calculations['market_mood']

            # Forecast the next price
            current_price = price
            forecasted_hft_price = forecast_hft_price(current_price, calculations['momentum'])
            print("Forecasted Price:", forecasted_hft_price)

            print()

            ##################################################
            ##################################################

            # Example usage without a while loop
            order_book_data = get_binance_futures_order_book_with_indicators(symbol="btcusdt", limit=5, forecast_minutes=1440)

            # Print information separately
            if order_book_data:
                print("Buy Order:")
                print(f"Price: {order_book_data['buy_order_price']}, Quantity: {order_book_data['buy_order_quantity']}")

                print("\nSell Order:")
                print(f"Price: {order_book_data['sell_order_price']}, Quantity: {order_book_data['sell_order_quantity']}")

                print("\nSpread:", order_book_data['spread'])

                print("\nSupport Level:", order_book_data['support_level'])
                print("Resistance Level:", order_book_data['resistance_level'])

                print("\nSmall Range Market Mood:", order_book_data['small_range_mood'])
                print("Large Range Market Mood:", order_book_data['large_range_mood'])

                print("\nForecasted Price for Max Duration (", order_book_data['max_forecast_minutes'], " minutes):", order_book_data['forecasted_price_max_duration'])

                # Enforce signals using MACD and RSI
                if order_book_data['macd'] < 0 and order_book_data['rsi'] < 30  and order_book_data['buy_order_quantity'] > order_book_data['sell_order_quantity']:
                    print("\nSignal: LONG")

                elif order_book_data['macd'] > 0 and order_book_data['rsi'] > 70 and order_book_data['buy_order_quantity'] < order_book_data['sell_order_quantity']:
                    print("\nSignal: SHORT")

                else:
                    print("\nSignal: NONE")

            print()

            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################


            print()

            ##################################################
            ##################################################

            # Example: Generate stationary compound wave, perform inverse Fourier transform, and analyze market mood
            num_harmonics = 5
            amplitude = 1.0
            frequency = 1.0

            # Generate and print values of the stationary compound wave
            ifft_range, compound_wave = generate_stationary_wave(num_harmonics, amplitude, frequency, close)

            # Perform inverse Fourier transform to get forecasted prices
            forecasted = np.real(inverse_fourier_transform(compound_wave))

            # Analyze market mood based on forecasted prices
            market_ifft_mood = analyze_market_mood(forecasted)

            # Print other details
            print(f"Initial Close Price: {close[0]:.5f}")
            print(f"Final Close Price: {close[-1]:.5f}")
            print("\nMarket Mood:", market_ifft_mood)

            # Calculate fixed target prices for the next reversal
            current_close = price
            target_up, target_down = calculate_fixed_targets(current_close)

            # Print fixed target prices for the next reversal
            print("\nFixed Target Prices for the Next Reversal:")
            print(f"Target Price for Upward Reversal: {target_up:.5f}")
            print(f"Target Price for Downward Reversal: {target_down:.5f}")

            print()

            ##################################################
            ##################################################

            candles = get_candles(TRADE_SYMBOL, timeframes)

            # Store results for each timeframe
            results_by_timeframe = {}

            # Iterate over each timeframe and call the generate_technical_indicators function
            for timeframe in timeframes:
                # Extract relevant data for the current timeframe
                close_prices = np.array([candle['close'] for candle in candles if candle['timeframe'] == timeframe])
                high_prices = np.array([candle['high'] for candle in candles if candle['timeframe'] == timeframe])
                low_prices = np.array([candle['low'] for candle in candles if candle['timeframe'] == timeframe])
                open_prices = np.array([candle['open'] for candle in candles if candle['timeframe'] == timeframe])
                volume = np.array([candle['volume'] for candle in candles if candle['timeframe'] == timeframe])

                # Generate technical indicators, forecast prices, and market mood for the current timeframe
                indicators_data = generate_technical_indicators(close_prices, high_prices, low_prices, open_prices, volume, timeframe)

                # Store the results for the current timeframe
                results_by_timeframe[timeframe] = indicators_data

                # Print the results for the specific timeframes (1m and 5m)
                if timeframe == '1m' or timeframe == '5m':
                    print(f"Results for {timeframe} timeframe:")
                    for indicator, value in indicators_data.items():
                        print(f"{indicator}:", value)
                        print("-" * 40)

            # Assess overall bullish/bearish for each indicator across all timeframes
            overall_market_mood = {}
            for indicator in results_by_timeframe['1m']['MarketMood'].keys():
                bullish_count = sum(1 for tf_result in results_by_timeframe.values() if tf_result['MarketMood'][indicator] == 'Bullish')
                bearish_count = sum(1 for tf_result in results_by_timeframe.values() if tf_result['MarketMood'][indicator] == 'Bearish')
                neutral_count = sum(1 for tf_result in results_by_timeframe.values() if tf_result['MarketMood'][indicator] == 'Neutral')
                overall_market_mood[indicator] = {
                    'Bullish': bullish_count,
                    'Bearish': bearish_count,
                    'Neutral': neutral_count
                }

            # Print overall assessment
            print("\nOverall Market Mood:")
            for indicator, mood_counts in overall_market_mood.items():
                print(f"{indicator}: Bullish={mood_counts['Bullish']}, Bearish={mood_counts['Bearish']}, Neutral={mood_counts['Neutral']}")

            print()

            # Print LinearReg Forecast Price for each timefram
            print("LinearReg Forecast Price 1m:", results_by_timeframe['1m']['LinearReg'])
            print("LinearReg Forecast Price 3m:", results_by_timeframe['3m']['LinearReg'])
            print("LinearReg Forecast Price 5m:", results_by_timeframe['5m']['LinearReg'])
            print("LinearReg Forecast Price 15m:", results_by_timeframe['15m']['LinearReg'])
            print("LinearReg Forecast Price 30m:", results_by_timeframe['30m']['LinearReg'])
            print("LinearReg Forecast Price 1h:", results_by_timeframe['1h']['LinearReg'])
            print("LinearReg Forecast Price 2h:", results_by_timeframe['2h']['LinearReg'])
            print("LinearReg Forecast Price 4h:", results_by_timeframe['4h']['LinearReg'])
            print("LinearReg Forecast Price 6h:", results_by_timeframe['6h']['LinearReg'])
            print("LinearReg Forecast Price 8h:", results_by_timeframe['8h']['LinearReg'])
            print("LinearReg Forecast Price 12h:", results_by_timeframe['12h']['LinearReg'])
            print("LinearReg Forecast Price 1d:", results_by_timeframe['1d']['LinearReg'])

            print()

            ##################################################
            ##################################################

            wave_result = dan_stationary_circuit(close)

            fft_wave_mood = wave_result["cycle_direction"]

            # Print the specific variables
            print("Dominant Frequency Sign:", wave_result["dominant_frequency_sign"])
            print("Last Quadrant:", wave_result["last_quadrant"])
            print("Current Quadrant:", wave_result["current_quadrant"])
            print("Cycle Direction:", wave_result["cycle_direction"])
            print("Forecasted Price:", wave_result["forecasted_price"])

            print()

            ##################################################
            ##################################################

            sin_market_mood, sin_current_cycle, sin_price = sine_market_analysis(close_prices, closest_threshold, min_threshold, max_threshold)

            # Print the results outside the function
            print("Market Mood:", sin_market_mood)
            print("Current Cycle:", sin_current_cycle)
            print("Forecast Price:", sin_price)

            print()

            ##################################################
            ##################################################


            last_low, last_high = calculate_min_max_values(close)

            # Example usage
            amplitude = 0.5
            frequency = 1

            sin_mood, sin_price, min_value, max_value = generate_sine_wave_with_motion_and_factors(close, closest_threshold, min_threshold, max_threshold)

            print(f"Market Mood: {sin_mood}")
            print(f"Forecast Price: {sin_price[-1]}")
            print(f"Last Lowest Low: {last_low}")
            print(f"Last Highest High: {last_high}")
            print(f"Min Value: {min_value}")
            print(f"Max Value: {max_value}")

            print()

            ##################################################
            ##################################################

            # Example usage:
            result = generate_new_momentum_sinewave(close, candles, percent_to_max_val=5, percent_to_min_val=5)

            # Assign each element to separate variables
            current_close = result["current_close"]
            dist_from_close_to_min = result["dist_from_close_to_min"]
            dist_from_close_to_max = result["dist_from_close_to_max"]
            current_quadrant = result["current_quadrant"]
            em_amplitude = result["em_amplitude"]
            em_phase = result["em_phase"]
            #trend_direction = result["trend_direction"]
            price_range_percent = result["price_range_percent"]
            sine_momentum = result["momentum"]
            sine_wave_max = result["max"]
            sine_wave_min = result["min"]

            # Print each variable separately
            print("Current Close:", current_close)
            print("Distance from Close to Min:", dist_from_close_to_min)
            print("Distance from Close to Max:", dist_from_close_to_max)
            print("Current Quadrant:", current_quadrant)
            print("EM Amplitude:", em_amplitude)
            print("EM Phase:", em_phase)
            #print("Trend Direction:", trend_direction)
            print("Price Range Percent:", price_range_percent)
            print("Momentum:", sine_momentum)
            print("Sine Wave Max:", sine_wave_max)
            print("Sine Wave Min:", sine_wave_min)

            print()

            ##################################################
            ##################################################

            market_mood_forecast = generate_market_mood_forecast(close, candles, percent_to_max_val=50, percent_to_min_val=50)

            cycle_direction = market_mood_forecast["cycle_direction"]
            quadrant_emotional_values = market_mood_forecast["quadrant_emotional_values"]
            forecast_moods = market_mood_forecast["forecast_moods"]
            sorted_frequencies = market_mood_forecast["sorted_frequencies"]
            avg_high_mood = market_mood_forecast["avg_high_mood"]
            avg_low_mood = market_mood_forecast["avg_low_mood"]
            weighted_high_mood = market_mood_forecast["weighted_high_mood"]
            weighted_low_mood = market_mood_forecast["weighted_low_mood"]
            mapped_quadrants = market_mood_forecast["mapped_quadrants"]
            min_node = market_mood_forecast["min_node"]
            max_node = market_mood_forecast["max_node"]
            mood_reversal_forecast = market_mood_forecast["mood_reversal_forecast"]
            market_mood = market_mood_forecast["market_mood"]

            current_point = market_mood_forecast["current_point"] 
            next_point = market_mood_forecast["next_point"]

            print(f"Current point: {current_point}")
            print(f"Next point: {next_point}")

            print("Cycle direction:", cycle_direction)
            print("Quadrant emotional values:", quadrant_emotional_values)
            print("Forecast moods:", forecast_moods)
            print("Sorted frequencies:", sorted_frequencies)
            print("Average high mood:", avg_high_mood)
            print("Average low mood:", avg_low_mood)
            print("Weighted high mood:", weighted_high_mood)
            print("Weighted low mood:", weighted_low_mood)
            print("Mapped quadrants:", mapped_quadrants)
            print("Minimum node:", min_node)
            print("Maximum node:", max_node)
            print("Mood reversal forecast:", mood_reversal_forecast)
            print("Market mood:", market_mood)

            print()

            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################

            take_profit = 5
            stop_loss = -10

            # Current timestamp in milliseconds
            timestamp = int(time.time() * 1000)

            # Concatenate the parameters and create the signature
            params = f"symbol={TRADE_SYMBOL}&timestamp={timestamp}"
            signature = hmac.new(api_secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()

            # Construct the complete URL
            position_endpoint = f"https://fapi.binance.com/fapi/v2/positionRisk?{params}&signature={signature}"

            response = requests.get(position_endpoint, headers={"X-MBX-APIKEY": api_key})
    
            if response.status_code == 200:

                positions = response.json()
                found_position = False

                for position in positions:
                    if position['symbol'] == TRADE_SYMBOL:
                        found_position = True
                        print("Position open:", position['positionAmt'])

                        entry_price = float(position['entryPrice'])
                        mark_price = float(position['markPrice'])
                        position_amount = float(position['positionAmt'])
                        leverage = float(position['leverage'])
                        un_realized_profit = float(position['unRealizedProfit'])

                        print("entry price at:", entry_price)
                        print("mark price at:", mark_price)
                        print("position_amount:", position_amount)
                        print("Current PNL at:", un_realized_profit)

                        direction = 1 if position_amount > 0 else -1
                        unrealized_pnl = position_amount * direction * (mark_price - entry_price)
                        imr = 1 / leverage
                        entry_margin = position_amount * mark_price * imr

                        if entry_margin != 0:
                            roe_percentage = (unrealized_pnl / entry_margin) * 100
                            print(f"ROE Percentage (ROE %) for {TRADE_SYMBOL}: {roe_percentage:.2f}%")
                        else:
                            print("Initial Margin is zero, no position is open yet")
                        break
        
                    if not found_position:
                        print("Position not open.")

            else:
                print("Failed to retrieve position information. Status code:", response.status_code)

            print()

            ##################################################
            ##################################################

            # Initialize counters
            long_conditions_met = 0
            short_conditions_met = 0

            # Determine overall majority
            overall_majority = "LONG" if long_conditions_met > short_conditions_met else "SHORT" if long_conditions_met < short_conditions_met else "EQUAL"

            print()

            ##################################################
            ##################################################

            # LONG conditions
            if normalized_distance_to_min < normalized_distance_to_max:
                print("LONG condition 1: normalized_distance_to_min < normalized_distance_to_max")  
                long_conditions_met += 1

            if closest_threshold == min_threshold:
                print("LONG condition 2: closest_threshold == min_threshold") 
                long_conditions_met += 1

            if closest_threshold < price: 
                print("LONG condition 3: closest_threshold < price")
                long_conditions_met += 1

            if price < expected_price: 
                print("LONG condition 4: price < expected_price")
                long_conditions_met += 1

            if price < forecast_5min:
                print("LONG condition 5: price < forecast_5min")
                long_conditions_met += 1

            if price < forecast_15min:
                print("LONG condition 6: price < forecast_15min")
                long_conditions_met += 1

            if price < forecast_30min:
                print("LONG condition 7: price < forecast_30min")
                long_conditions_met += 1

            if price < forecast_1h:
                print("LONG condition 8: price < forecast_1h")
                long_conditions_met += 1

            if signal == "BUY": 
                print("LONG condition 9: signal == BUY")
                long_conditions_met += 1

            if market_mood_type == "up":
                print("LONG condition 10: market_mood_type == up")
                long_conditions_met += 1

            if forecast_direction == "Up":
                print("LONG condition 11: forecast_direction == Up")
                long_conditions_met += 1

            if market_mood_fft == "Bullish": 
                print("LONG condition 12: market_mood_fft == Bullish")
                long_conditions_met += 1

            if pivot_mood == "Bullish":
                print("LONG condition 13: pivot_mood == Bullish")
                long_conditions_met += 1

            if incoming_reversal == "Top":
                print("LONG condition 14: incoming_reversal == Top")
                long_conditions_met += 1

            if price < forecast:
                print("LONG condition 15: price < forecast") 
                long_conditions_met += 1

            if momentum > 0:
                print("LONG condition 16: momentum > 0")
                long_conditions_met += 1

            print()

            if positive_count < negative_count:
                print("SHORT condition 17: positive_count < negative_count")
                short_conditions_met += 1
            elif positive_count == negative_count and overall_majority == "SHORT":
                print("SHORT condition 17: positive_count = negative_count (assigned based on overall majority)")
                short_conditions_met += 1
            elif positive_count > negative_count and overall_majority == "LONG":
                print("LONG condition 17: positive_count > negative_count (assigned based on overall majority)")
                long_conditions_met += 1

            if positive_sine_count < negative_sine_count:
                print("SHORT condition 18: positive_sine_count < negative_sine_count") 
                short_conditions_met += 1
            elif positive_sine_count == negative_sine_count and overall_majority == "SHORT":
                print("SHORT condition 18: positive_sine_count = negative_sine_count (assigned based on overall majority)")
                short_conditions_met += 1
            elif positive_sine_count > negative_sine_count and overall_majority == "LONG":
                print("LONG condition 18: positive_sine_count > negative_sine_count (assigned based on overall majority)")
                long_conditions_met += 1

            print()

            ##################################################
            ##################################################

            # SHORT conditions
            if normalized_distance_to_min > normalized_distance_to_max:
                print("SHORT condition 1: normalized_distance_to_min > normalized_distance_to_max") 
                short_conditions_met += 1

            if closest_threshold == max_threshold:
                print("SHORT condition 2: closest_threshold == max_threshold")
                short_conditions_met += 1

            if closest_threshold > price: 
                print("SHORT condition 3: closest_threshold > price")
                short_conditions_met += 1

            if price > expected_price: 
                print("SHORT condition 4: price > expected_price")
                short_conditions_met += 1

            if price > forecast_5min:
                print("SHORT condition 5: price > forecast_5min")
                short_conditions_met += 1

            if price > forecast_15min:
                print("SHORT condition 6: price > forecast_15min")
                short_conditions_met += 1

            if price > forecast_30min:
                print("SHORT condition 7: price > forecast_30min")
                short_conditions_met += 1

            if price > forecast_1h:
                print("SHORT condition 8: price > forecast_1h")
                short_conditions_met += 1

            if signal == "SELL": 
                print("SHORT condition 9: signal == SELL")
                short_conditions_met += 1

            if market_mood_type == "down":
                print("SHORT condition 10: market_mood_type == down")
                short_conditions_met += 1

            if forecast_direction == "Down":
                print("SHORT condition 11: forecast_direction == Down")
                short_conditions_met += 1

            if market_mood_fft == "Bearish": 
                print("SHORT condition 12: market_mood_fft == Bearish")
                short_conditions_met += 1

            if pivot_mood == "Bearish":
                print("SHORT condition 13: pivot_mood == Bearish")
                short_conditions_met += 1

            if incoming_reversal == "Dip":
                print("SHORT condition 14: incoming_reversal == Dip")
                short_conditions_met += 1

            if price > forecast:
                print("SHORT condition 15: price > forecast")
                short_conditions_met += 1

            if momentum < 0:
                print("SHORT condition 16: momentum < 0")
                short_conditions_met += 1

            print()

            if positive_count > negative_count:
                print("LONG condition 17: positive_count > negative_count")
                long_conditions_met += 1
            elif positive_count == negative_count and overall_majority == "LONG":
                print("LONG condition 17: positive_count = negative_count (assigned based on overall majority)")
                long_conditions_met += 1
            elif positive_count < negative_count and overall_majority == "SHORT":
                print("SHORT condition 17: positive_count < negative_count (assigned based on overall majority)")
                short_conditions_met += 1

            if positive_sine_count > negative_sine_count:
                print("LONG condition 18: positive_sine_count > negative_sine_count")
                long_conditions_met += 1
            elif positive_sine_count == negative_sine_count and overall_majority == "LONG":
                print("LONG condition 18: positive_sine_count = negative_sine_count (assigned based on overall majority)")
                long_conditions_met += 1
            elif positive_sine_count < negative_sine_count and overall_majority == "SHORT":
                print("SHORT condition 18: positive_sine_count < negative_sine_count (assigned based on overall majority)")
                short_conditions_met += 1
                                                                           
            print()

            ##################################################
            ##################################################

            # Checker and counter results
            if long_conditions_met > short_conditions_met:
                print("Overall Result: LONG conditions met more than SHORT conditions")
            elif long_conditions_met < short_conditions_met:
                print("Overall Result: SHORT conditions met more than LONG conditions")
            else:
                print("Overall Result: Equal number of LONG and SHORT conditions met")

            print(f"Total LONG conditions met: {long_conditions_met}")
            print(f"Total SHORT conditions met: {short_conditions_met}")

            print()

            ##################################################
            ##################################################

            # Cycles trigger conditions and bot autotrde sl and tp trigger conditions
            with open("signals.txt", "a") as f:
                # Get data and calculate indicators here...
                timestamp = current_time.strftime("%d %H %M %S")

                if un_realized_profit == 0:  
                    # Check if a position is not open
                    print("Now not in a trade, seeking entry conditions")

                    print()

                    ##################################################
                    ##################################################

                    print()  

                    ##################################################
                    ##################################################

                    # Uptrend cycle trigger conditions 
                    if buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min:
                        print("LONG condition 1: buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min")    
                        if normalized_distance_to_min < normalized_distance_to_max:
                            print("LONG condition 2: normalized_distance_to_min < normalized_distance_to_max")                
                            if closest_threshold == min_threshold and price < avg_mtf: 
                                print("LONG condition 3: closest_threshold == min_threshold and price < avg_mtf")                                                   
                                if closest_threshold < price:  
                                    print("LONG condition 4: closest_threshold < price")        
                                    if forecast_direction == "Up":
                                        print("LONG condition 5: forecast_direction == Up") 
                                        if roc_mood == "bullish":  
                                            print("LONG condition 6: roc_mood == bullish")                           
                                            if price < expected_price:
                                                print("LONG condition 7: price < expected_price") 
                                                if market_mood_fft == "Bullish":
                                                    print("LONG condition 8: market_mood_fft == Bullish")  
                                                    if price < forecast:
                                                        print("LONG condition 9: price < forecast")
                                                        if incoming_reversal == "Top": 
                                                            print("LONG condition 10: incoming_reversal == Top") 
                                                            if market_mood_type == "up":
                                                                print("LONG condition 11: market_mood_type == up")   
                                                                if signal == "BUY":
                                                                    print("LONG condition 12: signal == BUY") 
                                                                    if  positive_count > negative_count or positive_count == negative_count:     
                                                                        if positive_count > negative_count:
                                                                            print("LONG condition 13: positive_count > negative_count")      
                                                                        elif long_conditions_met == short_conditions_met:
                                                                            print("LONG condition 13: positive_count == negative_count") 
                                                                        if long_conditions_met > short_conditions_met:
                                                                            print("LONG condition 14: Overall LONG conditions met more than SHORT conditions")                         
                                                                            if momentum > 0:
                                                                                print("LONG condition 15: momentum > 0")
                                                                                trigger_long = True


                    # Downtrend cycle trigger conditions
                    if buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min:
                        print("SHORT condition 1: buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min")    
                        if normalized_distance_to_min > normalized_distance_to_max:
                            print("SHORT condition 2: normalized_distance_to_min > normalized_distance_to_max")                
                            if closest_threshold == max_threshold and price > avg_mtf: 
                                print("SHORT condition 3: closest_threshold == max_threshold and price > avg_mtf")                                                   
                                if closest_threshold > price:  
                                    print("SHORT condition 4: closest_threshold > price")        
                                    if forecast_direction == "Down":
                                        print("SHORT condition 5: forecast_direction == Down")       
                                        if roc_mood == "bearish":  
                                            print("SHORT condition 6: roc_mood == bearish")                          
                                            if price > expected_price:
                                                print("SHORT condition 7: price > expected_price") 
                                                if market_mood_fft == "Bearish":
                                                    print("SHORT condition 8: market_mood_fft == Bearish")  
                                                    if price > forecast:
                                                        print("SHORT condition 9: price > forecast")
                                                        if incoming_reversal == "Dip": 
                                                            print("SHORT condition 10: incoming_reversal == Dip") 
                                                            if market_mood_type == "down":
                                                                print("SHORT condition 11: market_mood_type == down")   
                                                                if signal == "SELL":
                                                                    print("SHORT condition 12: signal == SELL") 
                                                                    if  positive_count < negative_count or positive_count == negative_count:     
                                                                        if positive_count < negative_count:
                                                                            print("SHORT condition 13: positive_count > negative_count")      
                                                                        elif long_conditions_met == short_conditions_met:
                                                                            print("SHORT condition 13: positive_count == negative_count") 
                                                                        if long_conditions_met < short_conditions_met:
                                                                            print("SHORT condition 14: Overall SHORT conditions met more than LONG conditions")                         
                                                                            if momentum < 0:
                                                                                print("SHORT condition 15: momentum < 0")
                                                                                trigger_short = True

                    print()  

                    ##################################################
                    ##################################################

                    if closest_threshold == min_threshold and price < avg_mtf and momentum > 0 and buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min and price < expected_price and price < pivot_forecast and positive_count > negative_count and signal == "BUY" and market_mood_type == "up" and forecast_direction == "Up" and current_point == "Apex" and incoming_reversal == "Top":
                        print("LONG ultra HFT momentum triggered")
                        trigger_long = True

                    if closest_threshold == max_threshold and price > avg_mtf and momentum < 0 and buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min and price > expected_price and price > pivot_forecast and positive_count < negative_count and signal == "SELL" and market_mood_type == "down" and forecast_direction == "Down" and current_point == "Right" and incoming_reversal == "Dip":
                        print("SHORT ultra HFT momentum triggered")
                        trigger_short = True

                    print()

                    ##################################################
                    ##################################################

                    if closest_threshold == min_threshold and price < avg_mtf and momentum > 0 and buy_volume_1min > sell_volume_1min and price < expected_price and price < pivot_forecast and positive_count > negative_count and signal == "BUY" and market_mood_type == "up" and forecast_direction == "Up" and current_point == "Apex" and incoming_reversal == "Top":
                        print("LONG ultra HFT momentum triggered")
                        trigger_long = True

                    if closest_threshold == max_threshold and price > avg_mtf and momentum < 0 and buy_volume_1min < sell_volume_1min and price > expected_price and price > pivot_forecast and positive_count < negative_count and signal == "SELL" and market_mood_type == "down" and forecast_direction == "Down" and current_point == "Right" and incoming_reversal == "Dip":
                        print("SHORT ultra HFT momentum triggered")
                        trigger_short = True

                    print()

                    ##################################################
                    ##################################################

                    if momentum > 0 and buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min and price < expected_price and positive_count > negative_count and sentiment == "1.0" and cycle_direction == "UP" and current_point == "Apex" and current_quadrant == "1":
                        print("LONG ultra HFT momentum triggered")
                        trigger_long = True

                    if momentum < 0 and buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min and price > expected_price and positive_count < negative_count and sentiment == "-1.0" and cycle_direction == "DOWN" and current_point == "Right" and current_quadrant == "4":
                        print("SHORT ultra HFT momentum triggered")
                        trigger_short = True

                    print()

                    ##################################################
                    ##################################################

                    if momentum > 0 and buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min and positive_count > negative_count and cycle_direction == "UP" and current_point == "Apex" and current_quadrant == "1":
                        print("LONG ultra HFT momentum triggered")
                        trigger_long = True

                    if momentum < 0 and buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min and positive_count < negative_count and cycle_direction == "DOWN" and current_point == "Right" and current_quadrant == "4":
                        print("SHORT ultra HFT momentum triggered")
                        trigger_short = True

                    print()

                    ##################################################
                    ##################################################

                    if momentum > 0 and buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min and positive_count > negative_count and cycle_direction == "UP" and current_point == "Apex" and current_quadrant == "1":
                        print("LONG ultra HFT momentum triggered")
                        trigger_long = True

                    if momentum < 0 and buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min and positive_count < negative_count and cycle_direction == "DOWN" and current_point == "Right" and current_quadrant == "4":
                        print("SHORT ultra HFT momentum triggered")
                        trigger_short = True

                    print()

                    ##################################################
                    ##################################################

                    if momentum > 0 and buy_volume_1min > sell_volume_1min and buy_volume_3min > sell_volume_3min and buy_volume_5min > sell_volume_5min and positive_count > negative_count and cycle_direction == "UP" and current_point == "Apex" and current_quadrant == "1":
                        print("LONG ultra HFT momentum triggered")
                        trigger_long = True

                    if momentum < 0 and buy_volume_1min < sell_volume_1min and buy_volume_3min < sell_volume_3min and buy_volume_5min < sell_volume_5min and positive_count < negative_count and cycle_direction == "DOWN" and current_point == "Right" and current_quadrant == "4":
                        print("SHORT ultra HFT momentum triggered")
                        trigger_short = True

                    print()

                    ##################################################
                    ##################################################

                    print()  

                    ##################################################
                    ##################################################


                    #message = f'Price: ${price}' 
                    #webhook = DiscordWebhook(url='https://discord.com/api/webhooks/1168841370149060658/QM5ldJk02abTfal__0UpzHXYZI79bS-j6W75e8CbCwc6ZADimkSTLQkXwYIUd2s9Hk2T', content=message)
                    #response = webhook.execute()

                    #message_long = f'LONG signal! Price now at: {price}\n'
                    #message_short = f'SHORT signal! Price now at: {price}\n'

                    if trigger_long:
                        print("LONG signal!")
                        f.write(f"{current_time} LONG {price}\n")

                        #webhook = DiscordWebhook(url='https://discord.com/api/webhooks/1191539448782000189/Jvz-8g-pEa3FxWdnIL51Fi5XQJFZDmPrsOYaw8NOvp66S0BESptJ99sZAdtdQe4HGI0C', content=message_long)
                        #response = webhook.execute()

                        entry_long(symbol)
                        trigger_long = False

                    if trigger_short:
                        print("SHORT signal!")
                        f.write(f"{current_time} SHORT {price}\n")

                        #webhook = DiscordWebhook(url='https://discord.com/api/webhooks/1191539448782000189/Jvz-8g-pEa3FxWdnIL51Fi5XQJFZDmPrsOYaw8NOvp66S0BESptJ99sZAdtdQe4HGI0C', content=message_short)
                        #response = webhook.execute()

                        entry_short(symbol)
                        trigger_short = False

                    print()

                    ##################################################
                    ##################################################

                # Check stop loss and take profit conditions
                if un_realized_profit != 0:
                    print("Now in a trade, seeking exit conditions")

                    if roe_percentage >= take_profit or roe_percentage <= stop_loss:
                        # Call exit_trade() function
                        exit_trade() 
                    
            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

        ##################################################
        ##################################################

        # Delete variables and elements to clean up for the next iteration
        del response, data, current_time, current_close, fastest_target
        del min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, momentum
        del current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression
        del x, slope, intercept, expected_price
        del fast_price, medium_price, slow_price, forecasted_price, results
        del momentum_values, normalized_momentum, positive_count, negative_count  
        del closes, signal, close, candles, reversals, market_mood_type, market_mood_fastfft, analysis_results
        del current_price, forecasted_phi_price, market_mood_phi, intraday_target, market_mood_intraday, momentum_target, market_mood_momentum
        del div1, div2, keypoints, poly_features, X_poly, model, future, coefficients, regression_mood
        del forecast_price, market_mood, forecast_5min, forecast_15min, predicted_market_mood, price 
        del result_cycles, sentiment, market_quadrant, support_level, resistance_level, market_mood_trend, forecasted_price_trend
        del pivot_mood, pivot_forecast, dist_from_close_to_min, dist_from_close_to_max, current_sine, analysis_result
        del dom_mood, dom_forecast, unitcircle_price, unitcircle_mood, overall_sentiments_sine, positive_sine_count, negative_sine_count

        # Force garbage collection to free up memory
        gc.collect()

        ##################################################
        ##################################################

        ##################################################
        ##################################################

        time.sleep(5)
        print()

        ##################################################
        ##################################################

##################################################
##################################################

print()

##################################################
##################################################

# Run the main function
if __name__ == '__main__':
    main()

##################################################
##################################################


print()
##################################################
##################################################
