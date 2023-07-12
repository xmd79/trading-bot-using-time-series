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

timeframes = ['1min', '3min', '5min', '15min', '30min', '1h', '2h', '4h',  '6h', '8h', '12h', '1D']

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

candles = get_candles(TRADE_SYMBOL, timeframes)

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

def get_market_mood(candles, tf):
    high = max([c["high"] for c in candles[-20:]]) # Get highest high in last 20 candles
    low = min([c["low"] for c in candles[-20:]]) # Get lowest low in last 20 candles  
    
    small_band = (high - low) * 0.1
    medium_band = (high - low) * 0.25  
    large_band = (high - low) * 0.5
    
    close = get_close(candles, timeframes)
    
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

# Define the candle data for each timeframe
candle_1m_1 = {"open": 100, "high": 110, "low": 90, "close": 105}
candle_1m_2 = {"open": 105, "high": 120, "low": 100, "close": 115}
candle_1m_3 = {"open": 115, "high": 125, "low": 105, "close": 120}
candle_1m_4 = {"open": 120, "high": 130, "low": 110, "close": 125}

candle_5m_1 = {"open": 120, "high": 130, "low": 110, "close": 125}
candle_5m_2 = {"open": 125, "high": 135, "low": 115, "close": 130}
candle_5m_3 = {"open": 130, "high": 140, "low": 120, "close": 135}
candle_5m_4 = {"open": 135, "high": 145, "low": 125, "close": 140}

candle_15m_1 = {"open": 135, "high": 145, "low": 125, "close": 140}
candle_15m_2 = {"open": 140, "high": 150, "low": 130, "close": 145}
candle_15m_3 = {"open": 145, "high": 155, "low": 135, "close": 150}
candle_15m_4 = {"open": 150, "high": 160, "low": 140, "close": 155}

candle_1h_1 = {"open": 155, "high": 165, "low": 145, "close": 160}
candle_1h_2 = {"open": 160, "high": 170, "low": 150, "close": 165}
candle_1h_3 = {"open": 165, "high": 175, "low": 155, "close": 170}
candle_1h_4 = {"open": 170, "high": 180, "low": 160, "close": 175}

candle_4h_1 = {"open": 175, "high": 185, "low": 165, "close": 180}
candle_4h_2 = {"open": 180, "high": 190, "low": 170, "close": 185}
candle_4h_3 = {"open": 185, "high": 195, "low": 175, "close": 190}
candle_4h_4 = {"open": 190, "high": 200, "low": 180, "close": 195}

candle_1D_1 = {"open": 195, "high": 205, "low": 185, "close": 200}
candle_1D_2 = {"open": 200, "high": 210, "low": 190, "close": 205}
candle_1D_3 = {"open": 205, "high": 215, "low": 195, "close": 210}
candle_1D_4 = {"open": 210, "high": 220, "low": 200, "close": 215}

# Define the candle_map dictionary
candle_map = {
    "1m": [candle_1m_1, candle_1m_2, candle_1m_3, candle_1m_4],
    "5m": [candle_5m_1, candle_5m_2, candle_5m_3, candle_5m_4],
    "15m": [candle_15m_1, candle_15m_2, candle_15m_3, candle_15m_4],      
    "1h": [candle_1h_1, candle_1h_2, candle_1h_3, candle_1h_4],     
    "4h": [candle_4h_1, candle_4h_2, candle_4h_3, candle_4h_4],      
    "1D": [candle_1D_1, candle_1D_2, candle_1D_3, candle_1D_4],  
}

# Define the candle data for each timeframe
candle_1m_1 = {"open": 100, "high": 110, "low": 90, "close": 105}
candle_1m_2 = {"open": 105, "high": 120, "low": 100, "close": 115}
candle_1m_3 = {"open": 115, "high": 125, "low": 105, "close": 120}
candle_1m_4 = {"open": 120, "high": 130, "low": 110, "close": 125}

candle_5m_1 = {"open": 120, "high": 130, "low": 110, "close": 125}
candle_5m_2 = {"open": 125, "high": 135, "low": 115, "close": 130}
candle_5m_3 = {"open": 130, "high": 140, "low": 120, "close": 135}
candle_5m_4 = {"open": 135, "high": 145, "low": 125, "close": 140}

candle_15m_1 = {"open": 135, "high": 145, "low": 125, "close": 140}
candle_15m_2 = {"open": 140, "high": 150, "low": 130, "close": 145}
candle_15m_3 = {"open": 145, "high": 155, "low": 135, "close": 150}
candle_15m_4 = {"open": 150, "high": 160, "low": 140, "close": 155}

candle_1h_1 = {"open": 155, "high": 165, "low": 145, "close": 160}
candle_1h_2 = {"open": 160, "high": 170, "low": 150, "close": 165}
candle_1h_3 = {"open": 165, "high": 175, "low": 155, "close": 170}
candle_1h_4 = {"open": 170, "high": 180, "low": 160, "close": 175}

candle_4h_1 = {"open": 175, "high": 185, "low": 165, "close": 180}
candle_4h_2 = {"open": 180, "high": 190, "low": 170, "close": 185}
candle_4h_3 = {"open": 185, "high": 195, "low": 175, "close": 190}
candle_4h_4 = {"open": 190, "high": 200, "low": 180, "close": 195}

candle_1D_1 = {"open": 195, "high": 205, "low": 185, "close": 200}
candle_1D_2 = {"open": 200, "high": 210, "low": 190, "close": 205}
candle_1D_3 = {"open": 205, "high": 215, "low": 195, "close": 210}
candle_1D_4 = {"open": 210, "high": 220, "low": 200, "close": 215}

# Define the candle_map dictionary
candle_map = {
    "1m": [candle_1m_1, candle_1m_2, candle_1m_3, candle_1m_4],
    "5m": [candle_5m_1, candle_5m_2, candle_5m_3, candle_5m_4],
    "15m": [candle_15m_1, candle_15m_2, candle_15m_3, candle_15m_4],
    "1h": [candle_1h_1, candle_1h_2, candle_1h_3, candle_1h_4],
    "4h": [candle_4h_1, candle_4h_2, candle_4h_3, candle_4h_4],
    "1D": [candle_1D_1, candle_1D_2, candle_1D_3, candle_1D_4],
}

# Create an empty dictionary to store the results
market_moods = {}

# Iterate through the candle_map dictionary
for tf, candles in candle_map.items():
    # Call the get_market_mood function for each timeframe and store the result
    market_moods[tf] = get_market_mood(candles, tf)

# Print the results
for tf, mood in market_moods.items():
    print(f"{tf} market mood: {mood}")

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

close = get_close(candles, timeframes)

print(f"Current close: {close}")

distance_to_high = high - close        
distance_to_low = close - low

print(f"Distance to high: {distance_to_high}")      
print(f"Distance to low: {distance_to_low}")  

mood = get_market_mood(candles, tf)

print(f"Market mood: {mood}")  

print()

##################################################
##################################################

def get_high(candles, timeframe):
    return candles[-1]['high']

def get_close(candles, timeframe):     
    return candles[-1]['close']

##################################################
##################################################

def ratio_on_sine(candles):
    """
    Calculate potential reversal from ratio of current candle 
    to moving average and unit circle function. 
    """
    
    closes = [c["close"] for c in candle_map[tf][-5:]]
    ma5 = sum(closes) / 5 # Simple 5 period moving average
    
    # Calculate ratio of current close to MA 
    current_close = candle_map[tf][-1]["close"] 
    ratio = current_close / ma5  
        
    # Define angles based on range of ratio 
    if ratio < 0.95:
        angle_low = 0
        angle_high = 45
    elif ratio < 1.05:
        angle_low = 45 
        angle_high = 90            
    else:
        angle_low = 90
        angle_high = 135
        
    # Calculate potential reversal based on ratio and angles   
    min_diff = angle_low * ratio # Potential min reversal  
    max_diff = angle_high * ratio # Potential max reversal
    
    return min_diff, max_diff

# Call function and unpack return values    
min_support, max_resistance = ratio_on_sine(candles)

print(f"Potential support level: {min_support:.2f}%")
print(f"Potential resistance level: {max_resistance:.2f}%")

##################################################
##################################################

# Function to calculate momentum forecast 
def momentum_forecast(candles, start, end):
    small_momentum = 0  
    medium_momentum = 0
      
    for i in range(start, end):   
        candle = candles[i]
        
        candle["timeframe"] = candle.get("timeframe", "Unknown")
        
        close = candle["close"]
          
        if "timeframe" in candle and candle["timeframe"] in timeframes:      
            small_momentum += (close - candles[0]["close"]) / candles[0]["close"] * 100
        else:                      
            medium_momentum += (close - candles[0]["close"]) / candles[0]["close"] * 100
              
    avg_momentum = (small_momentum + medium_momentum) / 2
    return small_momentum, medium_momentum, avg_momentum


small_momentum = 0
medium_momentum = 0

for tf in candle_map:    
    tf_candles = candle_map[tf]
    
    start = 0  
    end = len(tf_candles)
    
    sm, mm, am = momentum_forecast(tf_candles, start, end)
    
    small_momentum += sm   
    medium_momentum += mm

small_momentum = small_momentum / len(candle_map)      
medium_momentum = medium_momentum / len(candle_map) 

avg_momentum = (small_momentum + medium_momentum) / 2

print(f"Average small momentum: {small_momentum:.2f}%") 
print(f"Average medium momentum: {medium_momentum:.2f}%")
print(f"Overall momentum: {avg_momentum:.2f}%")

print()

##################################################
##################################################

def forecast_levels():
   
    # Get forecast low and high from existing function
    forecast_low, forecast_high = forecast_low_high()
    
    # Get market mood   
    market_mood = get_market_mood(candles, tf)
    
    # Calculate sine ratios
    ratio_on_sine(candles)
    
    # Get momentum forecast
    momentum_forecast(candles, start, end)

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
small_momentum, medium_momentum, avg_momentum = momentum_forecast(candles, start, end)

min_diff, max_diff = ratio_on_sine(candles) 

# Use returned values  
print(f"Average 1-5m momentum: {small_momentum:.2f}%")
print(f"Potential min reversal: {min_diff:.2f}%")

# Define variables globally  
support1, support2, support3 = 0,0,0 
resistance1, resistance2, resistance3 = 0,0,0

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

    # Extract high and close from candles
    high = get_high(candles, timeframe)

    # Get close price
    close = get_close(candles, timeframe)

    # Calculate close ratio
    close_ratio = close / high  
    
    # Calculate close angle    
    close_angle = get_angle(close_ratio) 

    close_quadrant =  get_quadrant(close_angle)

    if close_quadrant == 1: # First quadrant
        
        # Build support ratio  
        support_ratio = close_ratio - 0.1 
        
        # Build resistance ratio
        resistance_ratio = close_ratio + 0.1
        
    elif close_quadrant == 2: # Second quadrant  
        
        # Build support ratio 
        support_ratio = (2 - close_ratio) - 0.1  
      
        # Build resistance ratio   
        resistance_ratio = (2 - close_ratio) + 0.1
        
    elif close_quadrant == 3: # Third quadrant
        
        # Build support ratio     
        support_ratio = (close_ratio - 2) + 0.1
               
        # Build resistance ratio     
        resistance_ratio = (close_ratio - 2) - 0.1  
        
    elif close_quadrant == 4: # Fourth quadrant 
        
        # Build support ratio     
        support_ratio = close_ratio + 0.1  
             
        # Build resistance ratio     
        resistance_ratio = close_ratio - 0.1

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

angle_degrees = 45

# Convert to radians        
angle_radians = math.radians(angle_degrees)
sin_value = math.sin(angle_radians)        
cos_value = math.cos(angle_radians)

for timeframe in timeframes:  
    # Get candles for this timeframe    
    candles = get_candles(TRADE_SYMBOL, timeframes)
    
    # Call function to calculate ratios    
    ratio_support, support_quadrant, ratio_resistance, resistance_quadrant, close_quadrant = unit_circle_ratios(candles, timeframe)
        
    # Print results for this timeframe
    print(f"Results for {timeframe} timeframe:")        
    print(f"Ratio support: {ratio_support}")
    print(f"Ratio resistance: {ratio_resistance}") 
        
