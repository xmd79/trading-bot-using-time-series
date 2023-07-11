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

from typing import Tuple

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

# Define vars and constants for data calculations:

# Calculate the trade size based on the USDT balance with 20x leverage
TRADE_SIZE = bUSD_balance * 20

# Global variables
TRADE_SYMBOL = 'BTCBUSD'

TRADE_TYPE = ''
TRADE_LVRG = 20

# define % for stoploss
STOP_LOSS_THRESHOLD = 0.0234 

# define % for takeprofit
TAKE_PROFIT_THRESHOLD = 0.0234 

# Define EMA-s:
EMA_SLOW_PERIOD = 12
EMA_FAST_PERIOD = 5

BUY_THRESHOLD = 5
SELL_THRESHOLD = 3

EMA_THRESHOLD_LONG = 12 # Close must be within 5% of slow EMA for long entry
EMA_THRESHOLD_SHORT = 5 # Close must be within 5% of fast EMA for short entry

LONG_SIGNAL = 1
SHORT_SIGNAL = -1
NO_SIGNAL = 0 

closed_positions = []

OPPOSITE_SIDE = {'long': 'SELL', 'short': 'BUY'}

##################################################
##################################################

# Initialize variables for tracking trade state:

trade_open = False
trade_side = None

trade_entry_pnl = 0
trade_exit_pnl = 0

trade_entry_time = 0
trade_exit_time = 0

trade_percentage = 0

##################################################
##################################################

# Define timeframes:

timeframes = ['1min', '3min', '5min', '15min', '30min', '1h', '2h', '4h',  '6h', '8h', '12h', '1D',  '3D']

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

# Extrct kline data for candels via api from binance eserver:

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

# move KLINE data received from binance server within candle_map{} dict. :

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

# Get close price as array
def get_close(candles):
  "Get last close price from candles"
  return candles[-1]["close"]

##################################################
##################################################

# Function to get EMAs with np arrays
def get_emas(close_prices: np.ndarray) -> Tuple[float,float]:
    ema_slow = talib.EMA(close_prices, timeperiod=EMA_SLOW_PERIOD)[-1]   
    ema_fast = talib.EMA(close_prices, timeperiod=EMA_FAST_PERIOD)[-1]   
    return ema_slow, ema_fast

##################################################
##################################################

#Checking EMA crosees:

def check_cross(ema_slow: float, ema_fast: float) -> bool:  
  "Check if EMAs have crossed"
  return ema_slow < ema_fast  

# Define ema moving averages crosses and getting percentage dist. from close to each of them:

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

# Function to check where close is relative to EMAs   
def get_signal() -> Tuple[int, str]:
      
    for timeframe, candles in candle_map.items():
        
        candle_array = np.array([candle["close"] for candle in candles])   
        ema_slow, ema_fast = get_emas(candle_array)  
          
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

def get_market_mood():
    candles = candles_1h # Use 1 hour candles
    
    high = max([c["high"] for c in candles[-20:]]) # Get highest high in last 20 candles
    low = min([c["low"] for c in candles[-20:]]) # Get lowest low in last 20 candles  
    
    small_band = (high - low) * 0.1
    medium_band = (high - low) * 0.25  
    large_band = (high - low) * 0.5
    
    close = get_close(candles)
    
    if close > high - small_band:
        mood = "Bullish"
    elif close < low + small_band:
        mood = "Bearish"        
    elif close > high - medium_band:
        if high - close < close - low:
            mood = "Tending bullish"
        else:
            mood = "Somewhat bullish"
    elif close < low + medium_band: 
        if close - low < high - close:      
            mood = "Tending bearish" 
        else:      
            mood = "Somewhat bearish"       
    elif close > high - large_band:
        mood = "Mildly bullish"        
    elif close < low + large_band:
        mood = "Mildly bearish"       
    else:
        mood = "Neutral"      

    return mood
    print()

candles = candles_1h # Use 1 hour candles

high = max([c["high"] for c in candles[-20:]])  
low = min([c["low"] for c in candles[-20:]])

print(f"Highest high in last 20 candles: {high}")
print(f"Lowest low in last 20 candles: {low}")

small_band = (high - low) * 0.1
medium_band = (high - low) * 0.25  
large_band = (high - low) * 0.5

print(f"Small band: {small_band}") 
print(f"Medium band: {medium_band}")
print(f"Large band: {large_band}")   

close = get_close(candles)

print(f"Current close: {close}")

distance_to_high = high - close        
distance_to_low = close - low

print(f"Distance to high: {distance_to_high}")      
print(f"Distance to low: {distance_to_low}")  

mood = get_market_mood()

print(f"Market mood: {mood}")  

print()


##################################################
##################################################

def ratio_on_sine(): 
    # Distance from close to min/max in degrees
    distance_min = 30   
    distance_max = 150

    # Calculate sine values    
    sin_close = math.sin(math.radians(90)) 
    sin_min = math.sin(math.radians(90 - distance_min))    
    sin_max = math.sin(math.radians(90 + distance_max))
        
    # Calculate ratios        
    sin_close_min_ratio = sin_close / sin_min
    sin_close_max_ratio = sin_close / sin_max   
        
    # Convert to percentages        
    min_diff = (1 - sin_close_min_ratio) * 100  
    max_diff = (1 - sin_close_max_ratio) * 100
        
    return min_diff, max_diff

# Call with different distances
ratio_on_sine() 

# Call function   
min_diff, max_diff = ratio_on_sine()
 
distance_min = 20     
distance_max = 120

print("Relative reversal potential:")    
print(f"Potential min reversal: {min_diff:.2f}%")    
print(f"Potential max reversal: {max_diff:.2f}%")
print()



##################################################
##################################################

# Function to calculate momentum forecast 
def momentum_forecast():
    
    # Get candles from timeframes
    candles_1m  = get_1m_candles(TRADE_SYMBOL)
    candles_3m  = get_3m_candles(TRADE_SYMBOL)
    candles_5m  = get_5m_candles(TRADE_SYMBOL) 
    candles_15m = get_15m_candles(TRADE_SYMBOL)
    candles_1h  = get_1h_candles(TRADE_SYMBOL)
    
    # Initialize momentum variables    
    small_momentum = 0
    medium_momentum = 0
    
    # Calculate momentum on small timeframes 
    for candles in [candles_1m, candles_3m, candles_5m]:
        closes = [c["close"] for c in candles[-10:]]   
        momentum = (closes[-1] - closes[0]) / closes[0] * 100       
        small_momentum += momentum  
    
    # Calculate average momentum        
    small_momentum = small_momentum / 3
    
    # Calculate momentum on medium timeframes      
    for candles in [candles_15m, candles_30m, candles_1h]:
        closes = [c["close"] for c in candles[-10:]]   
        momentum = (closes[-1] - closes[0]) / closes[0] * 100       
        medium_momentum += momentum
        
    medium_momentum = medium_momentum / 3
    

    # Average small and medium momentum     
    avg_momentum = (small_momentum + medium_momentum) / 2

    return small_momentum, medium_momentum, avg_momentum
      
# Call function
momentum_forecast()
min_diff, max_diff = ratio_on_sine()

small_momentum, medium_momentum, avg_momentum = momentum_forecast()

print("Relative reversal potential:")    
print(f"Potential min reversal: {min_diff:.2f}%")    
print(f"Potential max reversal: {max_diff:.2f}%")

print("Momentum:")  
print(f"Average 1-5m momentum: {small_momentum:.2f}%")  
print(f"Average 15m-1h momentum: {medium_momentum:.2f}%") 
print(f"Overall momentum: {avg_momentum:.2f}%")

print()

##################################################
##################################################

def forecast_levels():
   
    # Get forecast low and high from existing function
    forecast_low, forecast_high = forecast_low_high()
    
    # Get market mood   
    market_mood = get_market_mood()
    
    # Calculate sine ratios
    ratio_on_sine()
    
    # Get momentum forecast
    momentum_forecast()

    if market_mood == "Bullish":
        
        support1 = forecast_low - (forecast_high - forecast_low) * 0.1
        support2 = forecast_low - (forecast_high - forecast_low) * 0.2
        support3 = forecast_low          
        
    elif market_mood == "Bearish":  
      
        resistance1 = forecast_high + (forecast_high - forecast_low) * 0.1
        resistance2 = forecast_high + (forecast_high - forecast_low) * 0.2
        resistance3 = forecast_high
        
    else:
        
        support1 = resistance1 = forecast_low   
        support2 = resistance2 = (forecast_high + forecast_low) / 2
        support3 = resistance3 = forecast_high
        
    print(f"Support 1: {support1}")    
    print(f"Support 2: {support2}")    
    print(f"Support 3: {support3}")
            
    print()
    
    print(f"Resistance 1: {resistance1}")        
    print(f"Resistance 2: {resistance2}")   
    print(f"Resistance 3: {resistance3}")


# Call forecast_levels() function
forecast_levels()

print()

# Call functions and assign returned values  
small_momentum, medium_momentum, avg_momentum = momentum_forecast()

min_diff, max_diff = ratio_on_sine() 

# Use returned values  
print(f"Average 1-5m momentum: {small_momentum:.2f}%")
print(f"Potential min reversal: {min_diff:.2f}%")

# Define variables globally  
support1, support2, support3 = 0,0,0 
resistance1, resistance2, resistance3 = 0,0,0

print()

# Call get_market_mood() function 
market_mood = get_market_mood()

print("Market mood:")
print(market_mood)

print()  
 

##################################################
##################################################


def get_angle(ratio):
    if ratio <= 0.1:
        angle = 90 - ratio*180  
        return angle
            
    if ratio <= 0.2:  
        angle = 90   
        return angle  
             
    if ratio <= 0.3: 
        angle = 60
        return angle   
            
    if ratio <= 0.4:   
        angle = 120   
        return angle
        
    if ratio <= 0.5:   
        angle = 180
        return angle 
        
    if ratio <= 0.6:    
        angle = 240   
        return angle 
            
    if ratio <= 0.7:   
        angle = 300
        return angle
      
    if ratio <= 0.8: 
        angle = 0 
        return angle
            
    if ratio <= 0.9: 
        angle = 30
        return angle
        
    if ratio <= 1.0:
        angle = 330
        return angle

def get_sincos(angle):    
     angle_rad = math.radians(angle)
     return math.sin(angle_rad),  math.cos(angle_rad)
     
def get_quadrant(angle):
     
    if angle >= 0 and angle < 90:
        return 1
         
    if angle >= 90 and angle < 180:  
        return 2
         
    if angle >= 180 and angle < 270:
        return 3
         
    if angle >= 270 and angle < 360:
        return 4 

def unit_circle_ratios(candles):

    # Get close price
    close = get_close(candles)

    # Calculate close ratio
    close_ratio = close / high  
    
    # Calculate close angle    
    close_angle = get_angle(close_ratio) 

    # Now call get_sincos() after defining close_angle    
    sin_close, cos_close = get_sincos(close_angle)
    
    print(f"Close angle: {close_angle}")  

    # Calculate sin and cos for potential support 
    support_angle = get_angle(support_ratio)        
    sin_support, cos_support = get_sincos(support_angle) 
    
    print(f"Support angle: {support_angle}")
    
    # Calculate ratio for potential support    
    ratio_support = sin_close / sin_support  
    
    print(f"Ratio for potential support: {ratio_support}")
    
    # Calculate sin and cos for potential resistance     
    resistance_angle = get_angle(resistance_ratio)    
    sin_resistance, cos_resistance = get_sincos(resistance_angle)  

    print(f"Resistance angle: {resistance_angle}")
    
    # Calculate ratio for potential resistance    
    ratio_resistance = sin_close / sin_resistance
    
    print(f"Ratio for potential resistance: {ratio_resistance}")
        
    # Get quadrants for close, support and resistance
    close_quadrant =  get_quadrant(close_angle)    
    support_quadrant = get_quadrant(support_angle)    
    resistance_quadrant = get_quadrant(resistance_angle)
        
    print(f"Close quadrant: {close_quadrant}")
    print(f"Support quadrant: {support_quadrant}")
    print(f"Resistance quadrant: {resistance_quadrant}")
        
    return (
        ratio_support, 
        support_quadrant,
        ratio_resistance,
        resistance_quadrant,
        close_quadrant
    )

support_ratio = resistance_ratio = 0.5

# Get 5m candles 
candles = get_5m_candles(TRADE_SYMBOL)

# Call function with candle data
ratio_support, support_quadrant, ratio_resistance, resistance_quadrant, close_quadrant = unit_circle_ratios(candles)  

print(ratio_support)
