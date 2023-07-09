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

# Calculate the trade size based on the USDT balance with 20x leverage
TRADE_SIZE = bUSD_balance * 20

# Global variables
TRADE_SYMBOL = 'BTCBUSD'

TRADE_TYPE = ''
TRADE_LVRG = 20

STOP_LOSS_THRESHOLD = 0.0144 # define 1.44% for stoploss
TAKE_PROFIT_THRESHOLD = 0.0144 # define 1.44% for takeprofit

EMA_SLOW_PERIOD = 200
EMA_FAST_PERIOD = 50

BUY_THRESHOLD = 3
SELL_THRESHOLD = 3

EMA_THRESHOLD_LONG = 5 # Close must be within 5% of slow EMA for long entry
EMA_THRESHOLD_SHORT = 20 # Close must be within 20% of fast EMA for short entry

closed_positions = []

OPPOSITE_SIDE = {'long': 'SELL', 'short': 'BUY'}

##################################################
##################################################

# Initialize variables for tracking trade state
trade_open = False
trade_side = None

trade_entry_pnl = 0
trade_exit_pnl = 0

trade_entry_time = 0
trade_exit_time = 0

trade_percentage = 0

##################################################
##################################################

# Define timeframes
timeframes = ['1min', '3min', '5min', '15min', '30min', '1h', '2h', '4h',  '6h', '8h', '12h', '1D',  '3D']

##################################################
##################################################

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

def get_candles(symbol, timeframe):
    """Get all candles for a symbol and timeframe"""
    klines = client.get_klines(
        symbol=symbol, 
        interval=timeframe,  
        limit=1000 # Get max 1000 candles
    )
    
    candles = []

    # Convert klines to candle dict
    for k in klines:
        candle = {
            "time": k[0] / 1000, # Unix timestamp in seconds
            "open": float(k[1]), 
            "high": float(k[2]),  
            "low": float(k[3]),   
            "close": float(k[4]),    
            "volume": float(k[5])   
        }
        candles.append(candle)
        
    return candles

##################################################
##################################################

def get_1m_candles(symbol):
    return get_candles(symbol, client.KLINE_INTERVAL_1MINUTE)

def get_3m_candles(symbol):
    return get_candles(symbol, client.KLINE_INTERVAL_3MINUTE)

def get_5m_candles(symbol):
    return get_candles(symbol, client.KLINE_INTERVAL_5MINUTE)

def get_15m_candles(symbol): 
    return get_candles(symbol, client.KLINE_INTERVAL_15MINUTE)

def get_30m_candles(symbol):
    return get_candles(symbol, client.KLINE_INTERVAL_30MINUTE)  

def get_1h_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_1HOUR)

def get_2h_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_2HOUR)

def get_4h_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_4HOUR)

def get_6h_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_6HOUR)

def get_8h_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_8HOUR)

def get_12h_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_12HOUR)

def get_1d_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_1DAY)

def get_3d_candles(symbol):   
    return get_candles(symbol, client.KLINE_INTERVAL_3DAY)

##################################################
##################################################

# getting KLINE data from binance servers for asset symbol:

# Get 1 minute candles
candles_1m = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_1MINUTE) 

# Get 3 minute candles
candles_3m = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_3MINUTE) 

# Get 5 minute candles
candles_5m = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_5MINUTE)

# Get 15 minute candles
candles_15m = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_15MINUTE)

# Get 30 minute candles
candles_30m = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_30MINUTE)

# Get 1hour candles
candles_1h = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_1HOUR)

# Get 2hour candles
candles_2h = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_2HOUR)

# Get 4hour candles
candles_4h = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_4HOUR)

# Get 6hour candles
candles_6h = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_6HOUR)

# Get 8hour candles
candles_8h = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_8HOUR)

# Get 6hour candles
candles_12h = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_12HOUR)

# Get 1day candles
candles_1D = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_1DAY)

# Get 3day candles
candles_3D = get_candles(TRADE_SYMBOL, client.KLINE_INTERVAL_3DAY)

##################################################
##################################################

# moving KLINE data received from binance server within candle_map{} dict. :

candle_map = {
    '1min': candles_1m,  
    '3min' : candles_3m,  
    '5min' : candles_5m,
    '15min': candles_15m,
    '30min': candles_30m,  
    '1h'  : candles_1h,   
    '2h'  : candles_2h,
    '4h'  : candles_4h,
    '6h'  : candles_6h,
    '8h'  : candles_8h,
    '12h'  : candles_12h,
    '1D'  : candles_1D,
    '3D'  : candles_3D
}

##################################################
##################################################

# Define ema moving averages crosses and getting percentage dist. from close to them:

def get_emacross_mtf_signal(): 

    entry_long_signal = 0    
    entry_short_signal = 0

    slow_diff_history = []
    fast_diff_history = []

    for timeframe, candles in candle_map.items():
        
        candle_array = np.array([candle["close"] for candle in candles])
        
        ema_slow = talib.EMA(candle_array, timeperiod=EMA_SLOW_PERIOD)[-1]     
        ema_fast = talib.EMA(candle_array, timeperiod=EMA_FAST_PERIOD)[-1] 
    
        # Replace NaN with 0
        ema_slow = np.nan_to_num(ema_slow, nan=0.0)  
        ema_fast = np.nan_to_num(ema_fast, nan=0.0)

        # Filter out values <= 0      
        ema_slow = ema_slow[ema_slow > 0]  
        ema_fast = ema_fast[ema_fast > 0]  
 
        close = candle_array[-1]
                
        slow_diff = abs((close - ema_slow)/close) * 100

        if np.isnan(slow_diff):
            # If NaN, calculate avg of previous values    
            slow_diff = np.mean(slow_diff_history)  
    
            # Store current value in history    
            slow_diff_history.append(slow_diff)
        else:
            # If not NaN, store current value in history
           slow_diff_history.append(slow_diff)

        # Limit history to last 10 values     
        slow_diff_history = slow_diff_history[-10:]

        slow_diff = np.mean(slow_diff)

        fast_diff = abs((close - ema_fast)/close) * 100

        if np.isnan(fast_diff):
            # If NaN, calculate avg of previous values    
            fast_diff = np.mean(fast_diff_history)  
    
            # Store current value in history    
            fast_diff_history.append(fast_diff)
        else:
            # If not NaN, store current value in history
           fast_diff_history.append(fast_diff)

        # Limit history to last 10 values     
        fast_diff_history = fast_diff_history[-10:]

        fast_diff = np.mean(fast_diff) 

        print(f"{timeframe} - Close: {close}")
        print(f"{timeframe} - Fast EMA: {ema_fast}")   
        print(f"{timeframe} - Slow EMA: {ema_slow}") 

        print(f"{timeframe} - Close price value to Slow EMA value diff: {slow_diff:.2f}%")
        print(f"{timeframe} - Close price value to Fast EMA value diff: {fast_diff:.2f}%") 

        print()

        if ema_slow.size > 0 and ema_fast.size > 0 and close < ema_slow * 0.95 and close < ema_fast * 0.8 and ema_slow < ema_fast:
                print(f"{timeframe} - Potential long entry")       
                entry_long_signal += 1  
        
        if ema_slow.size > 0 and ema_fast.size > 0 and close > ema_slow * 1.2 and close > ema_fast * 1.05 and ema_slow >  ema_fast:      
                print(f"{timeframe} - Potential short entry")        
                entry_short_signal += 1 
                
    if entry_long_signal > BUY_THRESHOLD: 
       signal = 1      
    elif entry_short_signal > BUY_THRESHOLD:
       signal = -1
    else:
       signal = 0
        
    return signal, timeframe

print()

##################################################
##################################################

signal, timeframe = get_emacross_mtf_signal()

if timeframe == "Long":
    print("Generated long signal on:", timeframe)

elif timeframe == "Short":      
    print("Generated short signal on:", timeframe)  
    
else:
    print("No signal generated")

print()

##################################################
##################################################

