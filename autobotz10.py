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

# Define timeframes and get candles:

timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 10000  # default limit
        tf_value = int(timeframe[:-1])  # extract numeric value of timeframe
        if tf_value >= 4:  # check if timeframe is 4h or above
            limit = 20000  # increase limit for 4h timeframe and above
        klines = client.get_klines(
            symbol=symbol,
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

def get_latest_candle(symbol, interval, start_time=None):
    """Retrieve the latest candle for a given symbol and interval"""
    if start_time is None:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=1)
    else:
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, limit=1)
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
for timeframe in timeframes:
    dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)
    
    # Print the results for each timeframe
    print(f"For {timeframe} timeframe:")
    print(f"Distance to min: {dist_from_close_to_min:.2f}%")
    print(f"Distance to max: {dist_from_close_to_max:.2f}%")
    print(f"Current Sine value: {current_sine}\n")

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

##################################################
##################################################

# Calculate momentum for each timeframe
for timeframe in timeframes:
    momentum = get_momentum(timeframe)
    print(f"Momentum for {timeframe}: {momentum}")

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
    
    return current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood

closes = get_closes("1m")     
n_components = 5

current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood = get_target(closes, n_components, target_distance=56)

print("Current local Time is now at:", current_time)
print("Market mood is:", market_mood)

print()

current_close = closes[-1]
print("Current close price is at:", current_close)

print()

print("Fast target 1 is:", fast_target4)
print("Fast target 2 is:", fast_target3)
print("Fast target 3 is:", fast_target2)
print("Fast target 4 is:", fast_target1)

print()

print("Fastest target is:", fastest_target)

print()

print("Target 1 is:", target1)
print("Target 2 is:", target2)
print("Target 3 is:", target3)
print("Target 4 is:", target4)
print("Target 5 is:", target5)

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

def generate_gann_magic_square():
    square = np.zeros((3, 3), dtype=int)
    square[1, 1] = 1
    
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                square[i, j] = (i * 3 + j + 1) ** 2
    return square

def calculate_forecasts(square, close):
    magic_number = np.sum(square[1, :])
    
    support_level = magic_number - np.min(close)
    resistance_level = magic_number + np.max(close)
    
    fast_wave_cycle_forecast = f"Support Level (Fast): {abs(support_level):.2f}"
    large_wave_cycle_forecast = f"Resistance Level (Large): {abs(resistance_level):.2f}"
    
    return fast_wave_cycle_forecast, large_wave_cycle_forecast, magic_number

# Generate the Gann Magic Square
gann_square = generate_gann_magic_square()

# Calculate forecasts based on the Gann Magic Square and close prices
fast_wave_cycle_forecast, large_wave_cycle_forecast, magic_number = calculate_forecasts(gann_square, close)

# Determine market mood for Fast and Large Wave Cycles
market_mood_fast = "Negative" if abs(magic_number - np.min(close)) < 0 else "Positive"
market_mood_large = "Positive" if abs(magic_number + np.max(close)) >= 0 else "Negative"

# Print forecasts for Fast and Large Wave Cycles
print("\nMarket Mood Forecast:")
print("Fast Wave Cycle:", fast_wave_cycle_forecast)
print("Large Wave Cycle:", large_wave_cycle_forecast)

# Print market mood type for Fast and Large Wave Cycles
print("\nMarket Mood Type:")
print("Fast Wave Cycle:", market_mood_fast)
print("Large Wave Cycle:", market_mood_large)


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

            # Calculate fresh sine wave  
            close_prices = np.array(closes)
            sine, leadsine = talib.HT_SINE(close_prices)
            sine = -sine
    
            # Call scale_to_sine() function   
            #dist_from_close_to_min, dist_from_close_to_max = scale_to_sine('1m')

            #for timeframe in timeframes:
                #dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)

                # Print results        
                #print(f"On {timeframe} Close price value on sine is now at: {current_sine})")
                #print(f"On {timeframe} Distance from close to min perc. is now at: {dist_from_close_to_min})")
                #print(f"On {timeframe} Distance from close to max perc. is now at: {dist_from_close_to_max})")

            ##################################################
            ##################################################

            print()

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

            timeframe = '1m'
            momentum = get_momentum(timeframe)
            print("Momentum on 1min tf is at: ", momentum)

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

            current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood = get_target(closes, n_components, target_distance=56)

            print("Current local Time is now at: ", current_time)
            print("Market mood is: ", market_mood)
            market_mood_fft = market_mood
 
            print()

            print("Current close price is at : ", current_close)

            print()

            print("Fast target 1 is: ", fast_target4)
            print("Fast target 2 is: ", fast_target3)
            print("Fast target 3 is: ", fast_target2)
            print("Fast target 4 is: ", fast_target1)

            print()

            print("Fastest target is: ", fastest_target)

            print()

            print("Target 1 is: ", target1)
            print("Target 2 is: ", target2)
            print("Target 3 is: ", target3)
            print("Target 4 is: ", target4)
            print("Target 5 is: ", target5)

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

            # Get the last close price from the list for forecasting
            last_close_price = price

            # Generate forecast based on the 45-degree angle
            forecast_result = forecast_45_degree_angle(last_close_price, expected_price)

            # Display the forecast result
            print(forecast_result)

            print()

            ##################################################
            ##################################################

            # Generate the Gann Magic Square
            gann_square = generate_gann_magic_square()

            # Calculate forecasts based on the Gann Magic Square and close prices
            fast_wave_cycle_forecast, large_wave_cycle_forecast, magic_number = calculate_forecasts(gann_square, close)

            # Determine market mood for Fast and Large Wave Cycles
            market_mood_fast = "Negative" if abs(magic_number - np.min(close)) < 0 else "Positive"
            market_mood_large = "Positive" if abs(magic_number + np.max(close)) >= 0 else "Negative"

            # Print forecasts for Fast and Large Wave Cycles
            print("\nMarket Mood Forecast:")
            print("Fast Wave Cycle:", fast_wave_cycle_forecast)
            print("Large Wave Cycle:", large_wave_cycle_forecast)

            # Print market mood type for Fast and Large Wave Cycles
            print("\nMarket Mood Type:")
            print("Fast Wave Cycle:", market_mood_fast)
            print("Large Wave Cycle:", market_mood_large)

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
            toroidal_group_symmetry_analysis(close)

            print()

            ##################################################
            ##################################################

            fast_wave_cycle_forecast, large_wave_cycle_forecast, magic_number = calculate_forecasts(gann_square, close)
        
            # Determine market mood for Fast and Large Wave Cycles
            market_mood_fast = "Negative" if abs(magic_number - np.min(close)) < 0 else "Positive"
            market_mood_large = "Positive" if abs(magic_number + np.max(close)) >= 0 else "Negative"
        
            # Print forecasts for Fast and Large Wave Cycles
            print("\nMarket Mood Forecast:")
            print("Fast Wave Cycle:", fast_wave_cycle_forecast)
            print("Large Wave Cycle:", large_wave_cycle_forecast)

            # Print market mood type for Fast and Large Wave Cycles
            print("\nMarket Mood Type:")
            print("Fast Wave Cycle:", market_mood_fast)
            print("Large Wave Cycle:", market_mood_large)

            print()

            ##################################################
            ##################################################

            take_profit = 5.00
            stop_loss = -10.00

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
            # Cycles trigger conditions and bot autotrde sl and tp trigger conditions
            with open("signals.txt", "a") as f:
                # Get data and calculate indicators here...
                timestamp = current_time.strftime("%d %H %M %S")

                if un_realized_profit == 0:  
                    # Check if a position is not open
                    print("Now not in a trade, seeking entry conditions")

                    print()

                    # Uptrend cycle trigger conditions 
                    if normalized_distance_to_min < normalized_distance_to_max and normalized_distance_to_min < 15:
                        print("LONG condition 1: normalized_distance_to_min < normalized_distance_to_max and < 15")                
                        if closest_threshold == min_threshold and price < avg_mtf: 
                            print("LONG condition 2: closest_threshold == min_threshold and price < avg_mtf")                                                   
                            if closest_threshold < price:  
                                print("LONG condition 3: closest_threshold < price")            
                                if price < fastest_target:
                                    print("LONG condition 4: price < fastest_target") 
                                    if forecast_direction == "Up":
                                        print("LONG condition 5: forecast_direction == Up")                             
                                        if future_price_regression > price:
                                            print("LONG condition 6: future_price_regression > price")
                                            if forecast_price_fft > price:
                                                print("LONG condition 7: forecast_price_fft > price")
                                                if price < expected_price:
                                                    print("LONG condition 8: price < expected_price") 
                                                    if market_mood_fft == "Bullish":
                                                        print("LONG condition 9: market_mood_fft == Bullish")  
                                                        if mood == "Bullish" and price < forecast:
                                                            print("LONG condition 10: mood == Bullish and price < forecast")                                      
                                                            if momentum > 0:
                                                                print("LONG condition 11: momentum > 0")
                                                                trigger_long = True

                    # Downtrend cycle trigger conditions
                    if normalized_distance_to_max < normalized_distance_to_min and normalized_distance_to_max < 15:
                        print("SHORT condition 1: normalized_distance_to_max < normalized_distance_to_min and < 15") 
                        if closest_threshold == max_threshold and price > avg_mtf:
                            print("SHORT condition 2: closest_threshold == max_threshold and price > avg_mtf")  
                            if closest_threshold > price:
                                print("SHORT condition 3: closest_threshold > price")   
                                if price > fastest_target:
                                    print("SHORT condition 4: price > fastest_target") 
                                    if forecast_direction == "Down":
                                        print("SHORT condition 5: forecast_direction == Down") 
                                        if future_price_regression < price:
                                            print("SHORT condition 6: future_price_regression < price")
                                            if forecast_price_fft < price:
                                                print("SHORT condition 7: forecast_price_fft < price")
                                                if price > expected_price:
                                                    print("SHORT condition 8: price > expected_price") 
                                                    if market_mood_fft == "Bearish":
                                                        print("SHORT condition 9: market_mood_fft == Bearish")
                                                        if mood == "Bearish" and price > forecast:
                                                            print("SHORT condition 10: mood == Bearish and price > forecast")                                                  
                                                            if momentum < 0:
                                                                print("SHORT condition 11: momentum < 0")
                                                                trigger_short = True
                    print()

                    #message = f'Price: ${price}' 
                    #webhook = DiscordWebhook(url='https://discord.com/api/webhooks/1168841370149060658/QM5ldJk02abTfal__0UpzHXYZI79bS-j6W75e8CbCwc6ZADimkSTLQkXwYIUd2s9Hk2T', content=message)
                    #response = webhook.execute()

                    message_long = f'LONG signal! Price now at: {price}\n'
                    message_short = f'SHORT signal! Price now at: {price}\n'

                    if trigger_long:
                        print("LONG signal!")
                        f.write(f"{current_time} LONG {price}\n")

                        webhook = DiscordWebhook(url='https://discord.com/api/webhooks/1191539448782000189/Jvz-8g-pEa3FxWdnIL51Fi5XQJFZDmPrsOYaw8NOvp66S0BESptJ99sZAdtdQe4HGI0C', content=message_long)
                        response = webhook.execute()

                        entry_long(symbol)
                        trigger_long = False

                    if trigger_short:
                        print("SHORT signal!")
                        f.write(f"{current_time} SHORT {price}\n")

                        webhook = DiscordWebhook(url='https://discord.com/api/webhooks/1191539448782000189/Jvz-8g-pEa3FxWdnIL51Fi5XQJFZDmPrsOYaw8NOvp66S0BESptJ99sZAdtdQe4HGI0C', content=message_short)
                        response = webhook.execute()

                        entry_short(symbol)
                        trigger_short = False

                    print()

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
        del closes, close, candles, sine, leadsine
        del response, data, price, current_time, current_close, momentum
        del min_threshold, max_threshold, avg_mtf, momentum_signal, range_price
        del current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression

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
