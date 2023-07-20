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
import requests
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

def get_close(timeframe):
    # Get candles for this specific timeframe
    candles = candle_map[timeframe]
    
    # Get close of last candle    
    close = candles[-1]['close']
    
    return close

##################################################
##################################################

# Get close price as <class 'float'> type

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

##################################################
##################################################

# Scale current close price to sine wave       
def scale_to_sine(timeframe):  
  
    close_prices = np.array(get_closes(timeframe))
  
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

    # Take average % distances    
    avg_dist_min = sum(dist_min) / len(dist_min)
    avg_dist_max = sum(dist_max) / len(dist_max) 

    print(f"{timeframe} Close is now at "       
          f"dist. to min: {dist_from_close_to_min:.2f}% "
          f"and at "
          f"dist. to max: {dist_from_close_to_max:.2f}%")

    return dist_from_close_to_min, dist_from_close_to_max

# Call function           
#for timeframe in timeframes:        
    #scale_to_sine(timeframe)

print()

##################################################
##################################################

def collect_results():
    results = []
    
    for timeframe in timeframes:
        # Call existing function 
        dist_to_min, dist_to_max = scale_to_sine(timeframe)  
        
        # Append result tuple
        results.append((dist_to_min, dist_to_max)) 
        
    # Calculate overall percentages      
    overall_dist_min = sum([r[0] for r in results]) / len(results)    
    overall_dist_max = sum([r[1] for r in results]) / len(results)
    
    return overall_dist_min, overall_dist_max, results

# Call function      
overall_dist_min, overall_dist_max, results = collect_results()
print()

# Fast range - 1, 3, 5 mins
fast_range = results[:3]
fast_dist_min = sum([r[0] for r in fast_range]) / len(fast_range)  
fast_dist_max = sum([r[1] for r in fast_range]) / len(fast_range)  

# Medium range - 15min, 30min, 1hr       
medium_range = results[3:6]     
medium_dist_min = sum([r[0] for r in medium_range]) / len(medium_range)
medium_dist_max = sum([r[1] for r in medium_range]) / len(medium_range)

# Long range - 2hr to 1 day       
long_range = results[6:]    
long_dist_min = sum([r[0] for r in long_range]) / len(long_range)  
long_dist_max = sum([r[1] for r in long_range]) / len(long_range)

print("Overall distances:")
print(f"  To minimum: {overall_dist_min:.2f}%")  
print(f"  To maximum: {overall_dist_max:.2f}%")

print()

print("Fast range averages:")
print(f" To minimum: {fast_dist_min:.2f}%")   
print(f" To maximum: {fast_dist_max:.2f}%")

print()

print("Medium range averages:")       
print(f" To minimum: {medium_dist_min:.2f}%")  
print(f" To maximum: {medium_dist_max:.2f}%")

print()

print("Long range averages:")                 
print(f" To minimum: {long_dist_min:.2f}%")
print(f" To maximum: {long_dist_max:.2f}%")

print()

##################################################
##################################################

EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200

EMA_THRESHOLD = 1

# Function to get EMAs  
def get_emas(close_prices):
    close_array = np.array(close_prices)
    
    # Check length     
    if len(close_array) < EMA_FAST_PERIOD:
       return 0, 0
    
    # Replace NaN    
    close_array = np.nan_to_num(close_array, nan=0.0)
    
    # Calculate EMAs      
    ema_slow = talib.EMA(close_array, timeperiod=EMA_SLOW_PERIOD)[-1]   
    ema_fast = talib.EMA(close_array, timeperiod=EMA_FAST_PERIOD)[-1]   
   
    return ema_slow, ema_fast

##################################################
##################################################

# Checking EMA crosees:

def check_cross(ema_slow, ema_fast):  
    "Check if EMAs have crossed"
    return ema_slow < ema_fast  

##################################################
##################################################

def calculate_emas(candle_array):
    close_prices = candle_array.tolist()  
     
    # Check length     
    if len(close_prices) < EMA_FAST_PERIOD:
       return 0, 0  
        
    # Replace NaN    
    close_prices = np.nan_to_num(close_prices, nan=0.0)  
      
    # Calculate EMAs        
    ema_slow = talib.EMA(np.array(close_prices), timeperiod=EMA_SLOW_PERIOD)[-1]    
    ema_fast = talib.EMA(np.array(close_prices), timeperiod=EMA_FAST_PERIOD)[-1]
      
    return ema_slow, ema_fast


##################################################
##################################################

# Define ema moving averages crosses and getting percentage dist. from close to each of them:

def calculate_diff(close, ema, hist):  
    
    # Calculate difference         
    diff = abs((close - ema) / close) * 100
    
    if np.isnan(diff):
        # Calculate average of history 
        diff = np.nanmean(hist[ema])   
    
    if ema not in hist:
        hist[ema] = []
       
    # Update history           
    hist[ema].append(diff)  
        
    # Keep last 10 values 
    hist[ema] = hist[ema][-10:]
            
    return np.nanmean(hist[ema])


print()

##################################################
##################################################

NO_SIGNAL = 0

def calc_signal(candle_map):  

    signals = []
      
    for timeframe in candle_map:     
        candle_array = np.array([c["close"] for c in candle_map[timeframe]])   
        ema_slow, ema_fast = calculate_emas(candle_array)
                
        slow_diff_hist = {"slow": []}
        fast_diff_hist = {"fast": []}  
                
        close = candle_array[-1]
      
        slow_diff = calculate_diff(close, ema_slow,  
                                   slow_diff_hist)      
        fast_diff = calculate_diff(close, ema_fast,  
                                   fast_diff_hist)
          
        signals.append({
           'timeframe': timeframe, 
           'slow_diff': slow_diff,
           'fast_diff': fast_diff   
        })
            
    signal, _ = NO_SIGNAL, None
            
    for sig in signals:  
        print(f"{sig['timeframe']} - {sig['slow_diff']:.2f}% from slow EMA, {sig['fast_diff']:.2f}% from fast EMA")     
            
    return signal, _

signal, _ = calc_signal(candle_map)  

if signal == NO_SIGNAL:     
   print("No clear dominance at the moment")

signal, _ = calc_signal(candle_map)

print()

##################################################
##################################################

def get_signal():
    for timeframe, candles in candle_map.items():
        candle_array = np.array([candle["close"] for candle in candles])  
        ema_slow, ema_fast = get_emas(candle_array)

        if len(candle_array) == 0:
            print(f"No candles found for {timeframe}")
            continue
        
        close = candle_array[-1]        
        
        if candle_array[-1] < ema_slow:
            print(f"{timeframe} - Close below slow EMA, potential reversal UP point.")
        
        if candle_array[-1] < ema_fast:
            print(f"{timeframe} - Close below fast EMA, potential support.")
        
        if candle_array[-1] < ema_slow and candle_array[-1] < ema_fast:
            print(f"{timeframe} - Close below both EMAs, strong reversal UP signal.")
            
        if candle_array[-1] > ema_slow:
            print(f"{timeframe} - Close above slow EMA, potential reversal DOWN point.")
            
        if candle_array[-1] > ema_fast:
            print(f"{timeframe} - Close above fast EMA, potential resistance.")   
            
        if candle_array[-1] > ema_slow and candle_array[-1] > ema_fast:
            print(f"{timeframe} - Close above both EMAs, strong reversal DOWN signal.")
            
    return NO_SIGNAL, None

# Call function
signal, timeframe = get_signal()
            
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

def get_multi_timeframe_rsi():
    """Calculate RSI from multiple timeframes and average"""
    rsis = []
    
    for timeframe in ['1m', '5m', '15m']:
        
       # Get candle data               
       candles = candle_map[timeframe][-100:]  
        
       # Calculate RSI
       rsi = talib.RSI(np.array([c["close"] for c in candles]))
       rsis.append(rsi[-1])
       
    # Average RSIs        
    avg_rsi = sum(rsis) / len(rsis)
        
    return avg_rsi

mtf_rsi = get_multi_timeframe_rsi()
print("MTF rsi value is now at: ", mtf_rsi)

print()

##################################################
##################################################

def get_mtf_rsi_market_mood():
    rsi = get_multi_timeframe_rsi()
    closes = get_closes_last_n_minutes("1m", 50)

    # Define the indicators   
    indicator1 = rsi  
    indicator2 = 50

    # Define the thresholds for dip and top reversals
    dip_threshold = 30
    top_threshold = 70

    # Check if the close price is in the dip reversal area
    if closes[-1] <= dip_threshold and closes[-2] > dip_threshold:
        mood = "dip up reversal"
    # Check if the close price is in the top reversal area
    elif closes[-1] >= top_threshold and closes[-2] < top_threshold:
        mood = "top down reversal"
    # Check if the close price is in the accumulation area
    elif closes[-1] > dip_threshold and closes[-1] < indicator2:
        mood = "accumulation"
    # Check if the close price is in the distribution area
    elif closes[-1] < top_threshold and closes[-1] > indicator2:
        mood = "distribution"
    # Check if the market is in a downtrend
    elif indicator1 < indicator2 and indicator1 < 50:
        mood = "downtrend"
    # Check if the market is in an uptrend
    elif indicator1 > indicator2 and indicator1 > 50:
        mood = "uptrend"
    else:
        mood = "neutral"

    return mood

mood = get_mtf_rsi_market_mood()
print("MTF rsi mood: ", mood)

print()

##################################################
##################################################

def get_multi_timeframe_momentum():
    """Calculate momentum from multiple timeframes and average"""
    momentums = []
    
    for timeframe in timeframes:
        
        # Get candle data               
        candles = candle_map[timeframe][-100:]  
        
        # Calculate momentum using talib MOM
        momentum = talib.MOM(np.array([c["close"] for c in candles]), timeperiod=14)
        momentums.append(momentum[-1])
       
    # Average momentums        
    avg_momentum = sum(momentums) / len(momentums)
        
    return avg_momentum

print()

##################################################
##################################################

def get_mtf_market_mood():
    rsi_mood = get_mtf_rsi_market_mood()
    momentum = get_multi_timeframe_momentum()

    # Define the thresholds for momentum signals
    small_range_threshold = 5
    medium_range_threshold = 10

    # Define the indicators   
    indicator1 = momentum  
    indicator2 = 0

    # Check if the momentum signal is in the small range
    if abs(indicator1) < small_range_threshold:
        mood = "small range"
    # Check if the momentum signal is in the medium range
    elif abs(indicator1) < medium_range_threshold:
        mood = "medium range"
    # Check if the market is in a downtrend
    elif indicator1 < indicator2:
        mood = "MTF trend downtrend"
    # Check if the market is in an uptrend
    elif indicator1 > indicator2:
        mood = "MTF trend uptrend"
    else:
        mood = "MTF trend neutral"

    # Combine the RSI mood and momentum mood
    if rsi_mood == "dip up reversal" or rsi_mood == "uptrend":
        mood += " momentum bullish"
    elif rsi_mood == "top down reversal" or rsi_mood == "downtrend":
        mood += " momentum bearish"
    else:
        mood += " momentum neutral"

    return mood

print()

mtf_market_mood = get_mtf_market_mood()
print("MTF RSI and MOM market mood: ", mtf_market_mood)

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
for timeframe in timeframes:
    momentum = get_momentum(timeframe)
    print(f"Momentum for {timeframe}: {momentum}")

print()

##################################################
##################################################

# Scale current close price to sine wave       
def scale_to_sine(timeframe):  
  
    close_prices = np.array(get_closes(timeframe))
  
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

    # Take average % distances    
    avg_dist_min = sum(dist_min) / len(dist_min)
    avg_dist_max = sum(dist_max) / len(dist_max) 

    print(f"{timeframe} Close is now at "       
          f"dist. to min: {dist_from_close_to_min:.2f}% "
          f"and at "
          f"dist. to max: {dist_from_close_to_max:.2f}%")

    return dist_from_close_to_min, dist_from_close_to_max

# Call function           
#for timeframe in timeframes:        
    #scale_to_sine(timeframe)

print()

##################################################
##################################################
# timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

overall_ranges = [
    ['1m', '5m'],
    ['3m', '15m'],
    ['5m', '30m'],
    ['15m', '1h'],
    ['30m', '2h'],
    ['1h', '4h'],
    ['2h', '6h'],
    ['6h', '12h'],
    ['8h', '1d'],
]

def collect_results():
    results = []
    
    for timeframes_for_range in overall_ranges:
        dist_min_sum = 0
        dist_max_sum = 0
        for timeframe in timeframes_for_range:
            # Call existing function 
            dist_to_min, dist_to_max = scale_to_sine(timeframe)  
        
            # Add distances to running sum
            dist_min_sum += dist_to_min
            dist_max_sum += dist_to_max
        
        # Calculate average distances for this range
        num_timeframes = len(timeframes_for_range)
        dist_min_avg = dist_min_sum / num_timeframes
        dist_max_avg = dist_max_sum / num_timeframes
        
        # Append result tuple
        results.append((dist_min_avg, dist_max_avg))
        
    # Calculate overall percentages      
    overall_dist_min = sum([r[0] for r in results]) / len(results)    
    overall_dist_max = sum([r[1] for r in results]) / len(results)
    
    return overall_dist_min, overall_dist_max, results

# Call function      
overall_dist_min, overall_dist_max, results = collect_results()
print()

print("Overall distances:")
print(f"  To minimum: {overall_dist_min:.2f}%")  
print(f"  To maximum: {overall_dist_max:.2f}%")

print()

for i in range(len(overall_ranges)):
    timeframes_for_range = overall_ranges[i]
    dist_min_avg, dist_max_avg = results[i]
    print(f"Overall range {i+1} ({', '.join(timeframes_for_range)}):")
    print(f"  To minimum: {dist_min_avg:.2f}%")  
    print(f"  To maximum: {dist_max_avg:.2f}%")
    print()

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
    if next_reversal:
        last_reversal = next_reversal
        last_reversal_value_on_sine = next_reversal_value_on_sine

        next_reversal = None
        next_reversal_value_on_sine = None


    # Print last and next reversal info
    if last_reversal:
        print(f"Last reversal was at {last_reversal} on the sine wave at {last_reversal_value_on_sine:.2f} ")

    # Return the momentum sorter, market mood, close prices between min and max sine, and reversal info
    return momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max, last_reversal, last_reversal_value_on_sine, last_reversal_value_on_price, next_reversal, next_reversal_value_on_sine, next_reversal_value_on_price

momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max, last_reversal, last_reversal_value_on_sine, last_reversal_value_on_price, next_reversal, next_reversal_value_on_sine, next_reversal_value_on_price = generate_momentum_sinewave(timeframes)

print()

#print("Close price values between last reversals on sine: ")
#print(close_prices_between_min_and_max)

print()

print("Current close on sine value now at: ", current_sine)
print("last reversal is: ", last_reversal)
print("last reversal value on sine :", last_reversal_value_on_sine)
print("distances as percentages from close to min: ", dist_from_close_to_min, "%")
print("distances as percentages from close to max: ", dist_from_close_to_max, "%")
print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
print("Mood on 1min timeframe is now at: ", market_mood[-12])

print()

##################################################
##################################################

def generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Calculate the sine wave using HT_SINE
    sine_wave, _ = talib.HT_SINE(close_prices)
    
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

    # Determine the trend direction based onthe EM phase differences
    if em_phase_q1 - em_phase_q2 >= math.pi/2 and em_phase_q2 - em_phase_q3 >= math.pi/2 and em_phase_q3 - em_phase_q4 >= math.pi/2:
        trend_direction = "Up"
    elif em_phase_q1 - em_phase_q2 <= -math.pi/2 and em_phase_q2 - em_phase_q3 <= -math.pi/2 and em_phase_q3 - em_phase_q4 <= -math.pi/2:
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

#sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)
#print(sine_wave)

print()

##################################################
##################################################

def generate_market_mood_forecast(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Call generate_new_momentum_sinewave to get the sine wave and other features
    sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=percent_to_max_val, percent_to_min_val=percent_to_min_val)

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

generate_market_mood_forecast(close_prices, candles, percent_to_max_val=50, percent_to_min_val=50)

market_mood_forecast = generate_market_mood_forecast(close_prices, candles, percent_to_max_val=50, percent_to_min_val=50)

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

def reversals_unit_circle(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Call generate_new_momentum_sinewave to get the sine wave and other features
    sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=percent_to_max_val, percent_to_min_val=percent_to_min_val)

    # Get the current quadrant and EM phase of the sine wave
    current_quadrant = sine_wave["current_quadrant"]
    em_phase = sine_wave["em_phase"]

    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177

    # Define PI constant with 15 decimals
    PI = 3.1415926535897932384626433832795028842

    # Define the frequency bands and their corresponding emotional values
    frequency_bands = {"Delta": -0.5, "Theta": -0.25, "Alpha": 0, "Beta": 0.25, "Gamma": 0.5}

    # Calculate the emotional value of each frequency band using phi
    emotional_values = {band: frequency_bands[band] * PHI for band in frequency_bands}

    # Divide the unit circle into 4 quadrants based on Metatron's Cube geometry
    quadrants = {"Apex": 0, "Left": math.pi/2, "Base": math.pi, "Right": 3*math.pi/2}

    # Map the emotional values to the corresponding quadrants and calculate the emotional amplitude and phase values
    quadrant_emotional_values = {}
    for quadrant in quadrants:
        quadrant_phase = quadrants[quadrant]
        emotional_value = emotional_values["Gamma"] * math.sin(em_phase + quadrant_phase)
        quadrant_amplitude = (emotional_value + PHI) / (2 * PHI)
        quadrant_emotional_values[quadrant] = {"amplitude": quadrant_amplitude, "phase": quadrant_phase}

    # Calculate the forecast mood for each quadrant based on the phi value of each frequency band and mapping that to an emotional state
    forecast_moods = {}
    for quadrant in quadrant_emotional_values:
        quadrant_amplitude = quadrant_emotional_values[quadrant]["amplitude"]
        quadrant_phase = quadrant_emotional_values[quadrant]["phase"]
        quadrant_em_value = emotional_values["Gamma"] * math.sin(em_phase + quadrant_phase)
        quadrant_forecast_moods = {}
        for band in emotional_values:
            emotional_value = emotional_values[band]
            phi_value = PHI ** (emotional_value / math.sqrt(PHI * math.sqrt(5)))
            quadrant_forecast_moods[band] = quadrant_amplitude * math.cos(phi_value + quadrant_em_value)
        forecast_moods[quadrant] = quadrant_forecast_moods

    # Calculate the average moods for the highest and lowest 3 frequencies in each quadrant to determine the overall trend
    avg_moods = {}
    for quadrant in forecast_moods:
        quadrant_forecast_moods = forecast_moods[quadrant]
        sorted_frequencies = sorted(quadrant_forecast_moods, key=lambda x: quadrant_forecast_moods[x])
        high_moods = [quadrant_forecast_moods[band] for band in sorted_frequencies[-3:]]
        low_moods = [quadrant_forecast_moods[band] for band in sorted_frequencies[:3]]
        avg_high_mood = sum(high_moods) / len(high_moods)
        avg_low_mood = sum(low_moods) / len(low_moods)
        avg_moods[quadrant] = {"avg_high_mood": avg_high_mood, "avg_low_mood": avg_low_mood}

    # Identify the minimum and maximum frequency nodes to determine reversal points
    min_node = None
    max_node = None
    min_mood = 1
    max_mood = -1
    for quadrant in forecast_moods:
        quadrant_forecast_moods = forecast_moods[quadrant]
        for band in quadrant_forecast_moods:
            mood = quadrant_forecast_moods[band]
            if mood < min_mood:
                min_mood = mood
                min_node = quadrant + " " + band
            if mood > max_mood:
                max_mood = mood
                max_node = quadrant + " " + band

    #Print the minimum and maximum frequency nodes and their corresponding moods
    print("Minimum frequency node: {}, Mood: {}".format(min_node, min_mood))
    print("Maximum frequency node: {}, Mood: {}".format(max_node, max_mood))

    # Calculate the weighted average moods for the highest and lowest 3 frequencies in each quadrant to get a more nuanced forecast
    weighted_moods = {}
    for quadrant in forecast_moods:
        quadrant_forecast_moods = forecast_moods[quadrant]
        sorted_frequencies = sorted(quadrant_forecast_moods, key=lambda x: quadrant_forecast_moods[x])
        high_weights = [quadrant_forecast_moods[band] / sum(quadrant_forecast_moods.values()) for band in sorted_frequencies[-3:]]
        low_weights = [quadrant_forecast_moods[band] / sum(quadrant_forecast_moods.values()) for band in sorted_frequencies[:3]]
        weighted_high_mood = sum([high_weights[i] * high_moods[i] for i in range(len(high_moods))])
        weighted_low_mood = sum([low_weights[i] * low_moods[i] for i in range(len(low_moods))])
        weighted_moods[quadrant] = {"weighted_high_mood": weighted_high_mood, "weighted_low_mood": weighted_low_mood}

    # Determine the forecast direction based on the quadrant with the highest average mood
    sorted_avg_moods = sorted(avg_moods, key=lambda x: avg_moods[x]["avg_high_mood"]+avg_moods[x]["avg_low_mood"], reverse=True)
    forecast_direction = sorted_avg_moods[0]

    # Determine the mood reversal forecast based on the minimum and maximum frequency nodes
    mood_reversal_forecast = None
    if min_mood < 0 and max_mood > 0:
        if sorted_frequencies.index(min_node.split()[-1]) < sorted_frequencies.index(max_node.split()[-1]):
            mood_reversal_forecast = "Positive mood reversal expected in the near term"
        else:
            mood_reversal_forecast = "Negative mood reversal expected in the near term"

    # Determine the overall market mood based on the highest and lowest 3 frequencies in each quadrant
    sorted_weighted_moods = sorted(weighted_moods, key=lambda x: weighted_moods[x]["weighted_high_mood"]+weighted_moods[x]["weighted_low_mood"], reverse=True)
    market_mood = None
    if weighted_moods[sorted_weighted_moods[0]]["weighted_high_mood"] > weighted_moods[sorted_weighted_moods[-1]]["weighted_low_mood"]:
        market_mood = "Positive"
    elif weighted_moods[sorted_weighted_moods[-1]]["weighted_low_mood"] > weighted_moods[sorted_weighted_moods[0]]["weighted_high_mood"]:
        market_mood = "Negative"
    else:
        market_mood = "Neutral"

    # Print the forecast results
    print("Forecast Direction: {}".format(forecast_direction))
    for quadrant in quadrant_emotional_values:
        print("\n{} Quadrant".format(quadrant))
        print("Emotional Amplitude: {}".format(quadrant_emotional_values[quadrant]["amplitude"]))
        print("Emotional Phase: {}".format(quadrant_emotional_values[quadrant]["phase"]))
        for band in frequency_bands:
            print("{}: {}".format(band, forecast_moods[quadrant][band]))
        print("Average High Mood: {}".format(avg_moods[quadrant]["avg_high_mood"]))
        print("Average Low Mood: {}".format(avg_moods[quadrant]["avg_low_mood"]))
        print("Weighted High Mood: {}".format(weighted_moods[quadrant]["weighted_high_mood"]))
        print("Weighted Low Mood: {}".format(weighted_moods[quadrant]["weighted_low_mood"]))
    print("\nMood Reversal Forecast: {}".format(mood_reversal_forecast))
    print("Market Mood: {}".format(market_mood))

reversals = reversals_unit_circle(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)
print(reversals)

print()

##################################################
##################################################

def generate_market_mood_forecast_gr(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    """
    Generates a market mood forecast based on the given inputs.

    Args:
    close_prices (list): A list of close prices.
    candles (list): A list of candlestick data.
    percent_to_max_val (float): The percent to maximum value for the sine wave.
    percent_to_min_val (float): The percent to minimum value for the sine wave.

    Returns:
    A dictionary containing the market mood forecast and various mood and market forecasts.
    """

    # Define constants and ratios
    pi = 3.14159
    max_val = max(close_prices)
    min_val = min(close_prices)
    total_range = max_val - min_val
    time_dilation = 1.0 / len(close_prices)
    quadrants = ["I", "II", "III", "IV"]

    # Determine frequency bands
    frequencies = {}
    for i in range(1, 5):
        frequencies[i] = 1 / (total_range / (i * pi))
    sorted_frequencies = sorted(frequencies, key=frequencies.get)
    min_node = sorted_frequencies[0]
    max_node = sorted_frequencies[-1]

    # Calculate emotional values for frequency bands
    forecast_moods = {}
    for band in frequencies:
        phase = 2 * pi * (band / 4)
        val = percent_to_max_val * abs(math.sin(phase)) + percent_to_min_val
        forecast_moods[band] = val

    # Apply time dilation to frequency bands
    dilated_forecast_moods = {}
    for band in forecast_moods:
        dilated_forecast_moods[band] = forecast_moods[band] * time_dilation

    # Calculate average moods for dilated frequencies
    num_frequencies = 2
    dilated_high_moods = []
    dilated_low_moods = []

    if dilated_forecast_moods:
        dilated_high_moods = [dilated_forecast_moods[band] for band in sorted_frequencies[-num_frequencies:]]
        dilated_low_moods = [dilated_forecast_moods[band] for band in sorted_frequencies[:num_frequencies]]

    dilated_avg_high_mood = sum(dilated_high_moods) / len(dilated_high_moods) if dilated_high_moods else 0
    dilated_avg_low_mood = sum(dilated_low_moods) / len(dilated_low_moods) if dilated_low_moods else 0

    # Calculate weighted averages for dilated frequencies
    weights = [num_frequencies - i for i in range(num_frequencies)]
    dilated_weighted_high_mood = 0
    dilated_weighted_low_mood = 0

    if dilated_forecast_moods:
        dilated_high_freqs = list(dilated_forecast_moods.keys())[-num_frequencies:]
        dilated_high_moods = [dilated_forecast_moods[freq] for freq in dilated_high_freqs]

        for i in range(num_frequencies):
            dilated_weighted_high_mood += weights[i] * dilated_high_moods[i]

        dilated_weighted_high_mood /= sum(weights)

        dilated_low_freqs = list(dilated_forecast_moods.keys())[:num_frequencies]
        dilated_low_moods = [dilated_forecast_moods[freq] for freq in dilated_low_freqs]

        for i in range(num_frequencies):
            dilated_weighted_low_mood += weights[i] * dilated_low_moods[i]

        dilated_weighted_low_mood /= sum(weights)

    # Determine reversal forecast based on dilated frequencies
    dilated_mood_reversal_forecast = ""

    if dilated_forecast_moods and dilated_forecast_moods[min_node] < dilated_avg_low_mood and dilated_forecast_moods[max_node] > dilated_avg_high_mood:
        dilated_mood_reversal_forecast = "Mood reversal possible"

    # Determine market mood based on dilated average moods
    dilated_market_mood = ""

    if dilated_avg_high_mood > 0 and dilated_avg_low_mood > 0:
        dilated_market_mood = "Positive" if dilated_avg_high_mood > dilated_avg_low_mood else "Negative"

    # Create dictionary of mood and market forecasts
    forecast_dict = {
        "dilated_mood_reversal_forecast": dilated_mood_reversal_forecast,
        "dilated_market_mood": dilated_market_mood,
        "dilated_avg_high_mood": dilated_avg_high_mood,
        "dilated_avg_low_mood": dilated_avg_low_mood,
        "dilated_weighted_high_mood": dilated_weighted_high_mood,
        "dilated_weighted_low_mood": dilated_weighted_low_mood
    }

    # Determine quadrant based on current price position
    current_price = close_prices[-1]
    quadrant = ""
    if current_price > max_val - (total_range / 2):
        if current_price > max_val - (total_range / 4):
            quadrant = quadrants[0]
        else:
            quadrant = quadrants[1]
    else:
        if current_price < min_val + (total_range / 4):
            quadrant = quadrants[2]
        else:
            quadrant = quadrants[3]

    # Add quadrant to forecast dictionary
    forecast_dict["quadrant"] = quadrant

    return forecast_dict

forecast = generate_market_mood_forecast_gr(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)

print(forecast)

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



    return em_amp, em_phase

sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                                           percent_to_max_val=5, 
                                           percent_to_min_val=5)      
  
sine_wave_max = sine_wave["max"]   
sine_wave_min = sine_wave["min"]

octa_metatron_cube(close_prices, candles)  
print(octa_metatron_cube(close_prices, candles))

print()

##################################################
##################################################

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
#print("Average MTF:", avg_mtf)

#print("Range of prices within distance from current close price:")
#print(range_price[-1])

print()

##################################################
##################################################

def get_next_minute_target(closes, n_components):
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

    return target_price

# Example usage
closes = get_closes("5m")
n_components = 5
targets = []

for i in range(len(closes) - 1):
    # Decompose the signal up to the current minute and predict the target for the next minute
    target = get_next_minute_target(closes[:i+1], n_components)
    targets.append(target)

# Print the predicted targets for the next minute
print("Target for next minutes:", targets[-1])

print()

##################################################
##################################################
from sklearn.linear_model import LinearRegression

def harmonic_analysis(closes, frequency):
    time = np.arange(len(closes))
    cosines = np.cos(2*np.pi*frequency*time)
    sines = np.sin(2*np.pi*frequency*time)
    X = np.column_stack((cosines, sines))
    model = LinearRegression().fit(X, closes)
    trend = model.predict(X)
    residuals = closes - trend
    amplitude = np.sqrt(np.square(model.coef_[0]) + np.square(model.coef_[1]))
    phase = np.arctan2(model.coef_[1], model.coef_[0])
    return amplitude, phase, residuals

def poly_slope(closes, degree):
    time = np.arange(len(closes))
    coeffs = np.polyfit(time, closes, degree)
    slope = coeffs[-2]
    return slope

def get_consecutive_targets(closes, n_components, num_targets, current_time, current_close):
    print(f"current_time: {current_time}")
    print(f"current_close: {current_close}")
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

    # Calculate the targets for the next num_targets minutes
    targets = []
    target_times = []
    est_time_diffs = []
    price_diffs = []

    # Calculate the first target as the next value after the last closing price in the filtered signal
    target_price = filtered_signal[-1]

    # Calculate the remaining targets with increasing increments from the previous target price
    for i in range(num_targets):
        harmonic_freq = top_frequencies[-1-i%3] if i%3 < 3 else top_frequencies[i%3-3] # Use the last 3 highest and lowest frequencies for harmonic analysis
        amplitude, phase, residuals = harmonic_analysis(closes, harmonic_freq)
        slope = poly_slope(residuals, 1) # Use linear regression to estimate the slope of the residuals
        if harmonic_freq == 0:
            time_to_target = 1e6
        else:
            time_to_target = np.abs(np.arctan2(target_price-current_close, amplitude*np.cos(harmonic_freq*(len(closes)+i+1)+phase)+slope*(len(closes)+i+1))-np.arctan2(current_close-current_close, amplitude*np.cos(harmonic_freq*(len(closes))+phase)+slope*(len(closes))))/harmonic_freq # Use trigonometry to calculate the time to the target
        if time_to_target < 1e5:
            target_time = (current_time + datetime.timedelta(minutes=time_to_target)).strftime('%H:%M:%S')
            est_time_diff = f"{time_to_target:.2f} minutes from now"
            target_price += target_price * 0.005 # Increase the target price by 0.5% for each additional minute
            price_diff = target_price - current_close
            targets.append(target_price)
            target_times.append(target_time)
            est_time_diffs.append(est_time_diff)
            price_diffs.append(price_diff)
        else:
            print(f"Failed to calculate target {i+1}.")

    print()

    if len(targets) == len(target_times) == len(est_time_diffs) == len(price_diffs) == num_targets:
        print("Targets:")
        for target_time, target_price, price_diff in zip(target_times, targets, price_diffs):
            print(f"{target_time}: {target_price:.2f} ({price_diff:+.4f})")

    print()

        print("Estimated time differences:")
        for target_time, est_time_diff in zip(target_times, est_time_diffs):
            print(f"{target_time}: {est_time_diff}")
    else:
        print("Failed to calculate all targets.")

    return targets, target_times, est_time_diffs, price_diffs

# Set the number of components and targets
n_components = 5
num_targets = 3


# Define the current time and close price
current_time = datetime.datetime.now()
current_close = closes[-1]

print()

# # Call the get_consecutive_targets function
targets, target_times, est_time_diffs, price_diffs = get_consecutive_targets(closes, n_components, num_targets, current_time, current_close)

print()

# Print the results
#print("Targets:")
#for i in range(num_targets):
    #print(f"{target_times[i]}: {targets[i]:.2f} ({price_diffs[i]:+.4f})")

print()

##################################################
##################################################

print()

##################################################
##################################################
