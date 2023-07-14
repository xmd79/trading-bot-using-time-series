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

from scipy.stats import linregress
import traceback2 as traceback

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

# Organize candles by timeframe        
candle_map = {}  
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

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
print(mtf_rsi)

##################################################
##################################################

def get_mtf_rsi_market_mood():

    rsi = get_multi_timeframe_rsi()

    # Define the indicators   
    indicator1 = rsi  
    indicator2 = 50

    # Logic to determine market mood 
    # Based on indicators
    if indicator1 > indicator2:
        return "bullish"
    elif indicator1 < indicator2:
        return "bearish"
    else:
        return "neutral"

mood = get_mtf_rsi_market_mood()
print("MTF momentum rsi mood: ", mood)

print()

##################################################
##################################################

period = 20

def bollinger_bands(timeframe, period=20, std=2):    
   
    candle_data = candle_map[timeframe]
    closes = [candle['close'] for candle in candle_data[-period:]]   
    #print(f"Number of closes: {len(closes)}")  
   
    # Set default low and high   
    bb_low = min(closes)     
    bb_high = max(closes)
   
    if len(closes) < period:       
        return bb_low, bb_high 
        
    closes_array = np.array(closes) 
   
    sma = talib.SMA(closes_array, timeperiod=period)       
    stdev = talib.STDDEV(closes_array, timeperiod=period)  
            
    bb_upper = sma + (std * stdev)    
    bb_lower = sma - (std * stdev)

    # Replace NaN with 0    
    bb_lower = np.nan_to_num(bb_lower)  
    bb_upper = np.nan_to_num(bb_upper)
        
    # Get valid lower and upper bands     
    bb_lower = bb_lower[np.isfinite(bb_lower) & (bb_lower > 0)]
    bb_upper = bb_upper[np.isfinite(bb_upper) & (bb_upper > 0)]
         
    # Get first non-zero value  
    bb_low = bb_lower[0]
    bb_high = bb_upper[0]  
         
    return bb_low, bb_high

for timeframe in timeframes:
    bb_low, bb_high = bollinger_bands(timeframe, period=20, std=2)

    print(f"Timeframe: {timeframe}")   
    print("BB low at : ", bb_low)      
    print("BB high at : ", bb_high) 

print()

##################################################
##################################################


print()



##################################################
##################################################



