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

price_level = 100
sine_indicator = 0.5

VOLATILITY = 0.05 # 5% volatility

# Define PHI constant with 15 decimals
PHI = 1.6180339887498948482045868343656381177

def golden_func(n):
    if isinstance(n,np.ndarray):
        result= PHI**n[:,np.newaxis] 
    else: result= PHI**n     
    result=result.round(15)        
    return result

def calculate_sine(data): 
    data = np.array([[c['open'], c['high'], c['low'],  
                      c['close'], c['volume']] for c in candles])   
    norm_sine=(data[:, 3]-np.min(data[:, 3]))/(np.max(data[:, 3])-np.min(data[:, 3]))       
    return norm_sine      

def calculate_reversal(price_level,norm_sine):      
    try:     
        if norm_sine<0 or norm_sine>1: raise ValueError  
    except ValueError:         
        return 0         
    result= PHI**norm_sine         
    result=result.round(15)       
    reversal_point=result*price_level       
    return reversal_point

def get_reversal_points(candles):       
    if len(candles)==0:
        return 0         
    close=candles[-1]['close']       
    data=np.array([[c['open'],c['high'],c['low'],c['close'],c['volume']]for c in candles])        
    norm_sine=calculate_sine(data)        
    high=np.max(data[:,1])         
    low=np.min(data[:,2])             
    quad_1_high=high*1.618   
    quad_2_low=high   
    quad_3_high=low/1.618
    quad_4_low=low 
    reversal= 0                  
    if close>quad_1_high:current_quadrant=2        
    elif close<quad_3_high:current_quadrant=4            
    else : current_quadrant=1 if close>high/2 else 3     
    if current_quadrant==2:                               
        reversal=quad_2_low*calculate_reversal(quad_2_low,norm_sine)         
    return reversal
    print(current_quadrant)
    print(norm_sine)

reversal = get_reversal_points(candles)
print(reversal)

# Get the next 3 reversal points and prices
def get_next_reversals(candles):
    reversals = []
    for i in range(3):
        candles.append({
            "open": candles[-1]['close'],
            "high": candles[-1]['close'],  
            "low": candles[-1]['close'],
            "close": candles[-1]['close'],   
            "volume": 1   
        })
        # Calculate reversal point                
        reversal = get_reversal_points(candles)  
        
        # Append result              
        reversals.append({
            "reversal": reversal,  
            "price": candles[-1]['close']       
        })
        
        # Remove the appended candle
        candles = candles[:-1]
        
    return reversals

reversals = get_next_reversals(candles)
print(reversals)
