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
        if account['asset'] == 'BUSD':
            bUSD_balance = float(account['balance'])
            return bUSD_balance

# Get the USDT balance of the futures account
bUSD_balance = float(get_account_balance())

# Print account balance
print("BUSD Futures balance:", bUSD_balance)
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
        limit = 1000  # default limit
        tf_value = int(timeframe[:-1])  # extract numeric value of timeframe
        if tf_value >= 4:  # check if timeframe is 4h or above
            limit = 2000  # increase limit for 4h timeframe and above
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

# Get SMA for all timeframes and for all intervals lengths

def get_sma(timeframe, length):
   closes = get_closes(timeframe)  
    
   sma = talib.SMA(np.array(closes), timeperiod=length)

   # Filter out NaN  
   sma = sma[np.isnan(sma) == False]
      
   return sma

# Call the function
sma_lengths = [5, 12, 21, 27, 56, 72, 100, 150, 200, 250, 369]

#for timeframe in timeframes:
#    for length in sma_lengths:
#       sma = get_sma(timeframe, length)
#       print(f"SMA {length} for {timeframe}: {sma}")

print()

##################################################
##################################################

def get_sma_diff(timeframe, length):
    close = get_close(timeframe)
    sma = get_sma(timeframe, length)[-1]
    
    diff = (close - sma) / sma * 100 if sma != 0 else 0
    
    position = "CLOSE ABOVE" if close > sma else "CLOSE BELOW"
    
    return diff, position

# Call the function   
for timeframe in timeframes:
    for length in sma_lengths:
       
       # Get close price
       close = get_close(timeframe)
       
       # Get SMA 
       sma = get_sma(timeframe, length)[-1]
       
       # Calculate % diff from close  
       close_diff = (close - sma) / sma * 100  
       
       # Initialize diff arrays 
       sma_diffs = []
       sma_values = []
           
       # Get SMA value for all lengths 
       for l in sma_lengths:
           sma_value = get_sma(timeframe, l)[-1]
           sma_values.append(sma_value)
           
       # Calculate diff between all SMAs       
       for i in range(len(sma_values)-1):
           sma_diff = abs(sma_values[i] - sma_values[i+1])
           sma_diffs.append(sma_diff)
            
       # Calculate average SMA diff     
       avg_sma_diff = mean(sma_diffs)       
         
##################################################
##################################################

def get_sma_ratio(timeframe):    
    above_ratios = []    
    below_ratios = []
    
    for length in sma_lengths:
         diff, position = get_sma_diff(timeframe, length)
          
         if position == "CLOSE ABOVE":
             above_ratios.append(diff)
         else:  
             below_ratios.append(diff)
            
    if above_ratios:        
        above_avg = statistics.mean(above_ratios)
    else:
        above_avg = 0
        
    if below_ratios:                
        below_avg = statistics.mean(below_ratios)          
    else:
        below_avg = 0
            
    return above_avg, below_avg

# Call the function
for timeframe in timeframes:
    above_avg, below_avg =  get_sma_ratio(timeframe)
    
    if below_avg > above_avg:
        print(f"{timeframe} Close is below SMAs at local DIP")        
    elif above_avg > below_avg:
        print(f"{timeframe} Close is above SMAs at local TOP")

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

EMA_THRESHOLD = 3

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

# New sinewave to build here

print()

##################################################
##################################################

def calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2):
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
    min_threshold = min_close - (max_close - min_close) * min_percentage_custom  
    max_threshold = max_close + (max_close - min_close) * max_percentage_custom
        
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


# Call function with minimum percentage of 2% and maximum percentage of 2%
min_threshold, max_threshold, avg_mtf, momentum_signal = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)

print("Momentum signal:", momentum_signal)
print()

print("Minimum threshold:", min_threshold)
print("Maximum threshold:", max_threshold)
print("Average MTF:", avg_mtf)

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
closes = get_closes("1m")
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


print()

##################################################
##################################################


print()

##################################################
##################################################

print()

##################################################
##################################################
