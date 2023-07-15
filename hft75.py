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
        klines = client.get_klines(
            symbol=symbol, 
            interval=timeframe,  
            limit=1000  
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
print("Minimum threshold:", min_threshold)
print("Maximum threshold:", max_threshold)
print("Average MTF:", avg_mtf)
print("Momentum signal:", momentum_signal)

print()

##################################################
##################################################


print()

##################################################
##################################################

