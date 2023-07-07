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

# binance module imports
from binance.client import Client as BinanceClient  
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *

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

candles = get_5m_candles("BTCUSDT")
print(candles)

# Get the last price of a given symbol pair from Binance    
def get_current_price(symbol):  
    """Get the current price of a futures symbol"""
    ticker = client.futures_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

price = get_current_price("BTCUSDT")
#print(price)

period = 200

def calculate_ema(candles, period):
    prices = [float(candle['close']) for candle in candles]
    ema = []
    sma = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)
    ema.append(sma)
    for price in prices[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

calculate_ema(candles, period)

