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
#print(type(closes))

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
       diff, position = get_sma_diff(timeframe, length)
        
       print(f"SMA {length} diff for {timeframe}: {diff}% - {position}") 


print()

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
        print(f"{timeframe} Close is at a local dip! {below_avg}% below SMAs")        
    elif above_avg > below_avg:
        print(f"{timeframe} Close is at a local top! {above_avg}% above SMAs")

print()

##################################################
##################################################

def forecast_market(timeframe):
    
    moods = {
        'uptrend': 0,   
        'downtrend': 0,    
        'accumulation': 0,   
        'distribution': 0    
    }
    
    signals = []
    
    above_avg, below_avg = get_sma_ratio(timeframe)
    
    if below_avg > above_avg:
        if below_avg < 50:
            signals.append('accumulation')    
            moods['accumulation'] += 1  
        else: 
            signals.append('downtrend')     
            moods['downtrend'] += 1           
            
    elif above_avg > below_avg:   
        if above_avg < 50:
           signals.append('distribution') 
           moods['distribution'] += 1          
        else:
           signals.append('uptrend')      
           moods['uptrend'] += 1
                    
    print(f"{timeframe} mood: {max(moods, key=moods.get)} - {'%.2f' % ((moods[max(moods, key=moods.get)]/len(candle_map[timeframe]))*100)}%")

# Forecast for small, medium and large timeframes    
forecast_market('1m')     # Short-term  
forecast_market('5m')     # Medium-term
forecast_market('1h')     # Long-term

print()

##################################################
##################################################

