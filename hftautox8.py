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

    return output_data

results = forecast_sma_targets(price)

# Print each output string separately
for result in results:
    print(result)

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
    sine_wave = -sine_wave

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

    market_sine_mood = None
    forecast_sine_price = None

    # Check for uptrend based on the last reversal being the min_threshold
    if ht_sine_min[-1] < ht_sine_min[-2] and closes[-1] > min_threshold:
        market_sine_mood = "Downtrend"
        forecast_sine_price = min_threshold + (ht_sine_min[-1] - np.min(ht_sine_min[-10:]))

    # Check for downtrend based on the last reversal being the max_threshold
    elif ht_sine_max[-1] > ht_sine_max[-2] and closes[-1] < max_threshold:
        market_sine_mood = "Uptrend"
        forecast_sine_price = max_threshold - (np.max(ht_sine_max[-10:]) - ht_sine_max[-1]) 

    return market_sine_mood, forecast_sine_price

# Use the calculate_sine_wave_and_forecast function with the calculated thresholds
market_sine_mood, forecast_sine_price = calculate_sine_wave_and_forecast(closes, min_threshold, max_threshold)

print(f"Market Mood: {market_sine_mood}")
print(f"Forecast price: {forecast_sine_price}")

print()

##################################################
##################################################


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

            results = forecast_sma_targets(price)

            # Print each output string separately
            for result in results:
                print(result)

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

            sentiment, market_quadrant, support_level, resistance_level, market_mood_trend, forecasted_price_trend = forecast_market_trends(close)

            # Print the results
            print(f"Market Sentiment: {sentiment}")
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

            close = get_close('1h')

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

            # Use the calculate_sine_wave_and_forecast function with the calculated thresholds
            market_sine_mood, forecast_sine_price = calculate_sine_wave_and_forecast(closes, min_threshold, max_threshold)

            print(f"Market Mood: {market_sine_mood}")
            print(f"Forecast price: {forecast_sine_price}")

            print()

            ##################################################
            ##################################################

            take_profit = 5.00
            stop_loss = -25.00

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

                    # Uptrend cycle trigger conditions 
                    if normalized_distance_to_min < normalized_distance_to_max:
                        print("LONG condition 1: normalized_distance_to_min < normalized_distance_to_max")               
                        if closest_threshold == min_threshold:
                            print("LONG condition 2: closest_threshold == min_threshold") 
                            if closest_threshold < price: 
                                print("LONG condition 3: closest_threshold < price")                                                  
                                if price < expected_price: 
                                    print("LONG condition 4: price < expected_price") 
                                    if price < forecast_5min:
                                        print("LONG condition 5: price < forecast_5min")  
                                        if price < forecast_15min: 
                                            print("LONG condition 6: price < forecast_15min")  
                                            if price < forecast_30min: 
                                                print("LONG condition 7: price < forecast_30min") 
                                                if price < forecast_1h:  
                                                    print("LONG condition 8: price < forecast_1h") 
                                                    if signal == "BUY": 
                                                        print("LONG condition 9: signal == BUY")
                                                        if market_mood_type == "up":
                                                            print("LONG condition 10: market_mood_type == up")
                                                            if forecast_direction == "Up":
                                                                print("LONG condition 11: forecast_direction == Up")
                                                                if market_mood_fft == "Bullish": 
                                                                    print("LONG condition 12: market_mood_fft == Bullish")
                                                                    if pivot_mood == "Bullish":
                                                                        print("LONG condition 13: pivot_mood == Bullish")
                                                                        if incoming_reversal == "Top":
                                                                            print("LONG condition 14: incoming_reversal == Top") 
                                                                            if price < forecast:
                                                                                print("LONG condition 15: price < forecast") 
                                                                                if positive_count > negative_count or positive_count == negative_count:
                                                                                    if positive_count > negative_count:
                                                                                        print("LONG condition 16: positive_count > negative_count")     
                                                                                    elif positive_count == negative_count:
                                                                                        print("LONG condition 16: positive_count = negative_count")   
                                                                                    if positive_sine_count > negative_sine_count or positive_sine_count == negative_sine_count:
                                                                                        if positive_sine_count > negative_sine_count:
                                                                                            print("LONG condition 17: positive_sine_count > negative_sine_count")     
                                                                                        elif positive_sine_count == negative_sine_count:
                                                                                            print("LONG condition 17: positive_sine_count == negative_sine_count")                                                                               
                                                                                        if momentum > 0:
                                                                                            print("LONG condition 18: momentum > 0")
                                                                                            trigger_long = True

                    ##################################################
                    ##################################################

                    # Downtrend cycle trigger conditions
                    if normalized_distance_to_min > normalized_distance_to_max:
                        print("SHORT condition 1: normalized_distance_to_min > normalized_distance_to_max")               
                        if closest_threshold == max_threshold:
                            print("SHORT condition 2: closest_threshold == max_threshold") 
                            if closest_threshold > price: 
                                print("SHORT condition 3: closest_threshold > price")                                                  
                                if price > expected_price: 
                                    print("SHORT condition 4: price > expected_price") 
                                    if price > forecast_5min:
                                        print("SHORT condition 5: price > forecast_5min")  
                                        if price > forecast_15min: 
                                            print("SHORT condition 6: price > forecast_15min")  
                                            if price > forecast_30min: 
                                                print("SHORT condition 7: price > forecast_30min") 
                                                if price > forecast_1h:  
                                                    print("SHORT condition 8: price > forecast_1h") 
                                                    if signal == "SELL": 
                                                        print("SHORT condition 9: signal == SELL")
                                                        if market_mood_type == "down":
                                                            print("SHORT condition 10: market_mood_type == down")
                                                            if forecast_direction == "Down":
                                                                print("SHORT condition 11: forecast_direction == Down")
                                                                if market_mood_fft == "Bearish": 
                                                                    print("SHORT condition 12: market_mood_fft == Bearish")
                                                                    if pivot_mood == "Bearish":
                                                                        print("SHORT condition 13: pivot_mood == Bearish")
                                                                        if incoming_reversal == "Dip":
                                                                            print("SHORT condition 14: incoming_reversal == Dip") 
                                                                            if price > forecast:
                                                                                print("SHORT condition 15: price > forecast") 
                                                                                if positive_count < negative_count or positive_count == negative_count:
                                                                                    if positive_count < negative_count:
                                                                                        print("SHORT condition 16: positive_count < negative_count")     
                                                                                    elif positive_count == negative_count:
                                                                                        print("SHORT condition 16: positive_count = negative_count")   
                                                                                    if positive_sine_count < negative_sine_count or positive_sine_count == negative_sine_count:
                                                                                        if positive_sine_count < negative_sine_count:
                                                                                            print("SHORT condition 17: positive_sine_count < negative_sine_count")     
                                                                                        elif positive_sine_count == negative_sine_count:
                                                                                            print("SHORT condition 17: positive_sine_count == negative_sine_count")                                                                               
                                                                                        if momentum < 0:
                                                                                            print("SHORT condition 18: momentum < 0")
                                                                                            trigger_short = True

                    print()  

                    ##################################################
                    ##################################################

                    if normalized_distance_to_min < normalized_distance_to_max and closest_threshold == min_threshold and price < avg_mtf and forecast_direction == "Up" and market_mood_fft == "Bullish" and price < fastest_target and signal == "BUY" and market_mood_type == "up" and price < forecast and price < expected_price and pivot_mood == "Bullish" and momentum > 0 and positive_count > negative_count and long_conditions_met > short_conditions_met:
                        print("HFT LONG signal triggered!") 
                        trigger_long = True

                    elif normalized_distance_to_min > normalized_distance_to_max and closest_threshold == max_threshold and price > avg_mtf and forecast_direction == "Down" and  market_mood_fft == "Bearish" and price > fastest_target and signal == "SELL" and market_mood_type == "down" and price > forecast and price > expected_price and pivot_mood == "Bearish" and momentum < 0 and positive_count < negative_count and long_conditions_met < short_conditions_met:
                        print("HFT SHORT signal triggered!")  
                        trigger_short = True

                    if normalized_distance_to_min < normalized_distance_to_max and closest_threshold == min_threshold and price < avg_mtf and forecast_direction == "Up" and market_mood_fft == "Bullish" and price < fastest_target  and signal == "BUY" and market_mood_type == "up" and price < forecast and price < expected_price and pivot_mood == "Bullish" and momentum > 0 and positive_count == negative_count and long_conditions_met > short_conditions_met:
                        print("HFT LONG signal triggered!") 
                        trigger_long = True

                    elif normalized_distance_to_min > normalized_distance_to_max and closest_threshold == max_threshold and price > avg_mtf and forecast_direction == "Down" and  market_mood_fft == "Bearish" and price > fastest_target  and signal == "SELL" and market_mood_type == "down" and price > forecast and price > expected_price and pivot_mood == "Bearish" and momentum < 0 and positive_count == negative_count and long_conditions_met < short_conditions_met:
                        print("HFT SHORT signal triggered!")  
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
        del  market_sine_mood, forecast_sine_price 

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
