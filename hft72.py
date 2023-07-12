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

# Define binance client reading api key and secret from local file:

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

candles = get_candles(TRADE_SYMBOL, timeframes)

candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]  
    if timeframe not in candle_map:
        candle_map[timeframe] = []        
    candle_map[timeframe].append(candle)

##################################################
##################################################

def get_close(candles, timeframes):
    # Get close price of last candle
    return candles[-1]['close']

##################################################
##################################################

EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26

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

def calc_signal(candle):  
      
    slow_diff_hist: Histories = {"slow": []} 
    fast_diff_hist: Histories = {"fast": []}
      
    entry_long_signal = 0    
    entry_short_signal = 0
    
    for timeframe in candle_map:
        close = 0
        for candle in candle_map[timeframe]:
            candle_array = np.array([candle["close"] for candle in candle_map[timeframe]]) 
                
            ema_slow = talib.EMA(candle_array, EMA_SLOW_PERIOD)[-1]     
            ema_fast = talib.EMA(candle_array, EMA_FAST_PERIOD)[-1]   
                
            ema_slow = np.nan_to_num(ema_slow, nan=0.0)
            ema_fast = np.nan_to_num(ema_fast, nan=0.0)
          
            if np.isnan(ema_slow).any() or np.isnan(ema_fast).any():
                continue  
                
            close = candle_array[-1]
       
            slow_diff = calculate_diff(close, ema_slow, slow_diff_hist)                       
            fast_diff = calculate_diff(close, ema_fast, fast_diff_hist)
      
            if close < ema_slow * 0.95 and close < ema_fast * 0.8 and ema_slow < ema_fast:      
                entry_long_signal += 1  
         
            if close > ema_slow * 1.2 and close > ema_fast * 1.05 and ema_slow > ema_fast:
                entry_short_signal += 1
            
            if entry_long_signal > EMA_THRESHOLD:
                signal = 1        
            elif entry_short_signal > EMA_THRESHOLD:
                signal = -1        
            else:     
                signal = NO_SIGNAL

            close = candle_array[-1]

    return signal, timeframe

signal, _ = calc_signal(candle_map)

if signal == NO_SIGNAL:    
   print("No actual signal generated")

print()

##################################################
##################################################

# Function to check where close is relative to EMAs     
def get_signal():
    for timeframe, candles in candle_map.items():
        candle_array = np.array([candle["close"] for candle in candles])
        ema_slow, ema_fast = get_emas(candle_array)

        if len(candle_array) == 0:
            print(f"No candles found for {timeframe}")
            continue

        if candle_array[-1] < ema_slow:
            print(f"{timeframe} - Close below slow EMA, potential reversal point.")

        if candle_array[-1] < ema_fast:
            print(f"{timeframe} - Close below fast EMA, potential support.")

        if candle_array[-1] < ema_slow and candle_array[-1] < ema_fast:
            print(f"{timeframe} - Close below both EMAs, strong reversal signal.")

        if candle_array[-1] > ema_slow:
            print(f"{timeframe} - Close above slow EMA, potential resistance.")

        if candle_array[-1] > ema_fast:
            print(f"{timeframe} - Close above fast EMA, potential resistance.")

        if candle_array[-1] > ema_slow and candle_array[-1] > ema_fast:
            print(f"{timeframe} - Close above both EMAs, strong bullish signal.")

    return NO_SIGNAL, None

# Call function
signal, timeframe = get_signal()
            
print()

##################################################
##################################################

def forecast_low_high():
    
    lows = []
    highs = []
    
    for timeframe, candles in candle_map.items():
        
        # Get last 100 candles
        recent_candles = candles[-100:]
        
        # Find lowest low and highest high
        lowest_low = min([c["low"] for c in recent_candles])
        highest_high = max([c["high"] for c in recent_candles])
        
        lows.append(lowest_low)
        highs.append(highest_high)
        
    # Take average of lows and highs  
    forecast_low = np.mean(lows)
    forecast_high = np.mean(highs)
        
    return forecast_low, forecast_high

forecast_low, forecast_high = forecast_low_high()

print(f"Forecast low: {forecast_low}")
print(f"Forecast high: {forecast_high}")

print()

##################################################
##################################################


