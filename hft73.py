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

def calculate_emas(candle_array):
    close_prices = candle_array.tolist()  
     
    # Check length     
    if len(close_prices) < EMA_FAST_PERIOD:
       return 0, 0  
        
    # Replace NaN    
    close_prices = np.nan_to_num(close_prices, nan=0.0)  
      
    # Calculate EMAs        
    ema_slow = talib.EMA(np.array(close_prices), timeperiod=EMA_SLOW_PERIOD)[-1]    
    ema_fast = talib.EMA(np.array(close_prices), timeperiod=EMA_FAST_PERIOD)[-1]
      
    return ema_slow, ema_fast


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

def calc_signal(candle_map):  

    signals = []
      
    for timeframe in candle_map:     
        candle_array = np.array([c["close"] for c in candle_map[timeframe]])   
        ema_slow, ema_fast = calculate_emas(candle_array)
                
        slow_diff_hist = {"slow": []}
        fast_diff_hist = {"fast": []}  
                
        close = candle_array[-1]
      
        slow_diff = calculate_diff(close, ema_slow,  
                                   slow_diff_hist)      
        fast_diff = calculate_diff(close, ema_fast,  
                                   fast_diff_hist)
          
        signals.append({
           'timeframe': timeframe, 
           'slow_diff': slow_diff,
           'fast_diff': fast_diff   
        })
            
    signal, _ = NO_SIGNAL, None
            
    for sig in signals:  
        print(f"{sig['timeframe']} - {sig['slow_diff']:.2f}% from slow EMA, {sig['fast_diff']:.2f}% from fast EMA")     
            
    return signal, _

signal, _ = calc_signal(candle_map)  

if signal == NO_SIGNAL:     
   print("No clear dominance pattern overall")

signal, _ = calc_signal(candle_map)

if signal == NO_SIGNAL:    
   print("No clear dominance pattern overall")

print()

##################################################
##################################################

def get_signal():
    for timeframe, candles in candle_map.items():
        candle_array = np.array([candle["close"] for candle in candles])  
        ema_slow, ema_fast = get_emas(candle_array)

        if len(candle_array) == 0:
            print(f"No candles found for {timeframe}")
            continue
        
        close = candle_array[-1]        
        
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

def detect_phi_pattern(candles):
    
    middle_close = candles[0]['close']  
      
    patterns = 0   
    for candle in candles:  
        if candle['close'] < candle['open']:         
           patterns +=1
           
           if patterns == 3:
               middle_close = candle['close']
                
           if candle['close'] >= candle['open'] and patterns >= 3:      
               patterns = 0      
               
           if patterns == 0 and candle['close'] > middle_close:
               patterns +=1
               
           if patterns == 2:               
               patterns = 0
                
    return patterns

##################################################
##################################################

def detect_square_of_9(candles):
   
   patterns  = 0  
    
   while len(candles) >= 9:
       highest = [candles[0]['high'] for _ in range(9)]  
       lowest = [candles[0]['low'] for _ in range(9)]  
       
       for i in range(1,9):
           highest[i] = max(highest[i-1], candles[i]['high'])  
           lowest[i] = min(lowest[i-1], candles[i]['low'])
           
       if max(highest) == highest[4] or min(lowest) == lowest[4]:
          patterns += 1  
            
       candles = candles[1:]   
     
   # Check remaining candles   
   if len(candles) >= 1:      
      for candle in candles:  
         highest.append(candle['high']) 
         lowest.append(candle['low'])  
            
      if max(highest) == highest[4] or min(lowest) == lowest[4]:
          patterns += 1
             
   return patterns

##################################################
##################################################

def detect_forty_five_angle(candles):
    patterns = 0
    
    for i in range(len(candles)-1):  
        start_open = candles[i]['open']
        start_close = candles[i]['close'] 
        end_open = candles[i+1]['open']   
        end_close = candles[i+1]['close']
        
        angle = math.acos((start_close-start_open) / math.sqrt(start_close**2+start_open**2))  
        angle += math.acos((end_close-end_open) / math.sqrt(end_close**2+end_open**2)) 
        
        if 40 < angle < 50:           
            patterns += 1
            
        if len(candles) >= 1:       
           return patterns + 1  
            
    return patterns

##################################################
##################################################

def check_patterns():
    results = {}
    
    for timeframe, candles in candle_map.items():
        
        results[timeframe] = {}
        results[timeframe]['phi'] = {
            'count': detect_phi_pattern(candles), 
            'signal': 'dip' if candles[-1]['close'] < candles[-1]['open'] else 'top'
        }
        results[timeframe]['square'] = {
            'count': detect_square_of_9(candles), 
            'signal': 'dip' if candles[-1]['close'] < candles[-1]['open']  else 'top'
        }
        results[timeframe]['forty_five'] = {
            'count': detect_forty_five_angle(candles), 
            'signal': 'dip' if candles[-1]['close'] < candles[-1]['open'] else 'top' 
        }
    
    return results

patterns = check_patterns()

for timeframe, data in patterns.items():
    print(f"{timeframe} timeframe:")
    print(f"  Phi patterns: {data['phi']['count']} ({data['phi']['signal']})")
    print(f"Square patterns: {data['square']['count']} ({data['square']['signal']})")
    print(f"45 deg patterns: {data['forty_five']['count']} ({data['forty_five']['signal']})")

print()

##################################################
##################################################

def forecast_signal():
    signals = []
    moods = {
        'bearish': 0,
        'bullish': 0,
        'accumulation': 0, 
        'distribution': 0
    }
        
    for timeframe, candles in candle_map.items():

        candle_array = np.array([candle["close"] for candle in candles])
        ema_slow, ema_fast = np.array(get_emas(candle_array))
        
        if len(candle_array) == 0:
            continue
            
        close = candle_array[-1]       
        slow_diff = (close - ema_slow) / ema_slow * 100
        fast_diff = (close - ema_fast) / ema_fast * 100
               
        if close < ema_slow:      
           signals.append({
               'timeframe': timeframe,
               'signal': f'{slow_diff:.2f}% below slow EMA',
               'mood': 'bearish' 
           })
           
        if close < ema_fast:       
           signals.append({
               'timeframe': timeframe,  
               'signal': f'{fast_diff:.2f}% below fast EMA',
               'mood': 'accumulation' 
            })
 
        if close < ema_slow:
            signals.append({
               'timeframe': timeframe,  
               'signal': f'{fast_diff:.2f}% below fast EMA',
                'mood': 'bearish'
            })
            
        if close < ema_fast:       
           signals.append({
               'timeframe': timeframe,  
               'signal': f'{fast_diff:.2f}% below fast EMA',   
               'mood': 'accumulation'
            })
            
        if close > ema_slow:
           signals.append({
               'mood': 'bullish' 
           })
           
        if close > ema_fast:     
           signals.append({
               'mood': 'distribution'
            })
         
        # Do the same for close above EMAs                
            
    for signal in signals:
        moods[signal['mood']] += 1
            
    for mood, count in moods.items():
        print(f"{mood} mood: {count/len(signals)*100:.2f}%")   
            
    print("Overall market mood:", max(moods, key=moods.get))


forecast_signal()

##################################################
##################################################

moods = {
    'uptrend': 0,   
    'downtrend': 0,    
    'accumulation': 0,   
    'distribution': 0    
}

def calculate_market_mood():   

    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '8h']
    
    overall_moods = {
        'uptrend': 0,   
        'downtrend': 0,    
        'accumulation': 0,   
        'distribution': 0    
    }

    for timeframe in timeframes:  
        candle_array = np.array([candle["close"] for candle in candle_map[timeframe]])
        ema_slow, ema_fast = np.array(get_emas(candle_array))

        close = candle_array[-1]

        slow_diff = (close - ema_slow) / ema_slow * 100
        fast_diff = (close - ema_fast) / ema_fast * 100

        if close < ema_slow:    
            moods['downtrend'] += 1  
        elif slow_diff < -2:  
            moods['distribution'] += 1    
        elif slow_diff > 2:     
            moods['accumulation'] += 1    
        elif close > ema_slow:       
            moods['uptrend'] += 1
  
        if close < ema_slow:     
            moods['downtrend'] += 1
        elif close > ema_slow:
            moods['uptrend'] += 1
            
    reversals = []
    for timeframe in timeframes:
        if moods['uptrend'] > 5:      
            reversals.append(timeframe)
            
        if moods['downtrend'] > 5:
            reversals.append(timeframe)
            
    print(f"Market mood: {max(moods, key=moods.get)}")        
    print(f"Potential reversals at: {reversals}")
    print(f"{timeframe} mood: {max(moods, key=moods.get)} - {'%.2f' % ((moods[max(moods, key=moods.get)]/len(candle_array))*100)}%")

    moods['uptrend'] = 0
    moods['downtrend'] = 0 

calculate_market_mood()

print()

##################################################
##################################################

def forecast_market():
    
    moods = {
        'uptrend': 0,   
        'downtrend': 0,    
        'accumulation': 0,   
        'distribution': 0    
    }
    
    signals = []
    reversals = []
    
    for timeframe, candles in candle_map.items():

        candle_array = np.array([candle["close"] for candle in candle_map[timeframe]])
        ema_slow, ema_fast = np.array(get_emas(candle_array))

        close = candle_array[-1]

        slow_diff = (close - ema_slow) / ema_slow * 100
        fast_diff = (close - ema_fast) / ema_fast * 100
        
        # Check if close is below fast/slow EMA      
        if close < ema_slow:       
            signals.append({'mood': 'downtrend'})
            moods['downtrend'] += 1

            if moods['downtrend'] > 5:  
                reversals.append(timeframe)

        elif slow_diff < -2:     
            signals.append({'mood': 'distribution'})
            moods['distribution'] += 1
            
        elif slow_diff > 2:       
            signals.append({'mood': 'accumulation'})    
            moods['accumulation'] += 1
              
        elif close > ema_slow:    
            signals.append({'mood': 'uptrend'})      
            moods['uptrend'] += 1
            
            if moods['uptrend'] > 5:  
                reversals.append(timeframe)
                
    print(f"Market mood: {max(moods, key=moods.get)}")        
    print(f"Potential reversals at: {reversals}")
    print(f"{timeframe} mood: {max(moods, key=moods.get)} - {'%.2f' % ((moods[max(moods, key=moods.get)]/len(candle_array))*100)}%")

forecast_market()

print()

##################################################
##################################################

def analyze_timeframes():

    total_results = {}

    for timeframe, candles in candle_map.items(): 

        results = {}
        
        candle_array = np.array([candle["close"] for candle in candles])  
        close = candle_array[-1]   
        
        # Calculate EMAs
        def get_ema(period):       
            ema = talib.EMA(candle_array, period)[-1]       
            return np.nan_to_num(ema, nan=0.0)

        ema_slow = get_ema(26)    
        ema_fast = get_ema(12)
      
        # Get % differences        
        def get_percentage(value):
            return abs((close - value) / close) * 100
      
        # Get % differences       
        slow_diff = get_percentage(ema_slow)

        # Define fast diff     
        fast_diff = get_percentage(ema_fast)

        results["ema_diffs"] = {
            "slow": slow_diff,    
            "fast": fast_diff
        }  

        # Determine mood
        if slow_diff > 2:  
            results["mood"] = "accumulation" 
        elif slow_diff < -2:
            results["mood"] = "distribution"
        elif close > ema_fast:   
            results["mood"] = "uptrend"   
        elif close < ema_slow:     
            results["mood"] = "downtrend"
        else:  
            results["mood"] = "neutral"
         
        total_results[timeframe] = results    
                  
    # Print results       
    for timeframe, results in total_results.items():    
        print(f"Market mood: {results['mood']}")
        total_results[timeframe] = results

    # Print results
    for timeframe, results in total_results.items():

        print(f"{timeframe}:")  
        print(f"  Close {results['ema_diffs']['slow']:.2f}% from slow EMA")
        print(f"   {results['ema_diffs']['fast']:.2f}% from fast EMA")
        print(f"Market mood: {results['mood']}")    

        print()

analyze_timeframes()

print()

##################################################
##################################################



print()

##################################################
##################################################

print()

##################################################
##################################################

print()

##################################################
##################################################

print()

##################################################
##################################################

print()

##################################################
##################################################
