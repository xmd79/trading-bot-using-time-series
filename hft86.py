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
sma_lengths = [5, 7, 9, 12, 15, 18, 21, 27, 56, 72, 100, 127, 150, 200]

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
    return {"cycle_direction": cycle_direction, "quadrant_emotional_values": quadrant_emotional_values, "forecast_moods": forecast_moods, "sorted_frequencies": sorted_frequencies, "avg_high_mood": avg_high_mood, "avg_low_mood": avg_low_mood, "weighted_high_mood": weighted_high_mood, "weighted_low_mood": weighted_low_mood, "mapped_quadrants": mapped_quadrants, "min_node": min_node, "max_node": max_node, "mood_reversal_forecast": mood_reversal_forecast, "market_mood": market_mood}

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

print()

##################################################
##################################################

print()

##################################################
##################################################

print()

##################################################
##################################################