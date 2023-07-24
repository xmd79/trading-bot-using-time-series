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

def get_client():
    # Read credentials from file    
    with open("credentials.txt", "r") as f:   
         lines = f.readlines()
         api_key = lines[0].strip()  
         api_secret = lines[1].strip()  
          
    # Instantiate client        
    client = BinanceClient(api_key, api_secret)
          
    return client

# Call the function to get the client  
client = get_client()

##################################################
##################################################

# Initialize variables for tracking trade state:

TRADE_SYMBOL = "BTCUSDT"

##################################################
##################################################

# Define timeframes

timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

##################################################
##################################################

# Get current price as <class 'float'>

def get_price(symbol):
    try:
        client = get_client()
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

# Define function to retrieve candles from Binance API
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

# Organize candles by timeframe        
candle_map = {}  
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

# Get entire list of close prices as <class 'list'> type
def get_closes(timeframe):
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

closes = get_closes('1m')
print(closes)
