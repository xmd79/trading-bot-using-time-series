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
#print(candles)

# Organize candles by timeframe        
candle_map = {}  
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

#print(candle_map)

##################################################
##################################################

def get_latest_candle(symbol, interval, start_time=None):
    """Retrieve the latest candle for a given symbol and interval"""
    if start_time is None:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=1)
    else:
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, limit=1)
    candle = {
        "time": klines[0][0],
        "open": float(klines[0][1]),
        "high": float(klines[0][2]),
        "low": float(klines[0][3]),
        "close": float(klines[0][4]),
        "volume": float(klines[0][5]),
        "timeframe": interval
    }
    return candle

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

#print(closes)

##################################################
##################################################

# Scale current close price to sine wave       
def scale_to_sine(timeframe):  
  
    close_prices = np.array(get_closes(timeframe))
  
    # Get last close price 
    current_close = close_prices[-1]      
        
    # Calculate sine wave        
    sine_wave, leadsine = talib.HT_SINE(close_prices)
            
    # Replace NaN values with 0        
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave
        
    # Get the sine value for last close      
    current_sine = sine_wave[-1]
            
    # Calculate the min and max sine           
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # Calculate % distances            
    dist_min, dist_max = [], []
 
    for close in close_prices:    
        # Calculate distances as percentages        
        dist_from_close_to_min = ((current_sine - sine_wave_min) /  
                           (sine_wave_max - sine_wave_min)) * 100            
        dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                           (sine_wave_max - sine_wave_min)) * 100
                
        dist_min.append(dist_from_close_to_min)       
        dist_max.append(dist_from_close_to_max)

    # Take average % distances    
    avg_dist_min = sum(dist_min) / len(dist_min)
    avg_dist_max = sum(dist_max) / len(dist_max) 

    #print(f"{timeframe} Close is now at "       
          #f"dist. to min: {dist_from_close_to_min:.2f}% "
          #f"and at "
          #f"dist. to max: {dist_from_close_to_max:.2f}%")

    return dist_from_close_to_min, dist_from_close_to_max, current_sine

# Call function           
#for timeframe in timeframes:        
    #scale_to_sine(timeframe)

print()


##################################################
##################################################

def get_closes_last_n_minutes(interval, n):
    """Generate mock closing prices for the last n minutes"""
    closes = []
    for i in range(n):
        closes.append(random.uniform(0, 100))
    return closes

print()

##################################################
##################################################

def get_multi_timeframe_rsi():
    """Calculate RSI from multiple timeframes and average"""
    rsis = []
    
    for timeframe in ['1m', '3m', '5m', '15m', '1h']:
        
       # Get candle data               
       candles = candle_map[timeframe][-100:]  
        
       # Calculate RSI
       rsi = talib.RSI(np.array([c["close"] for c in candles]), timeperiod=21)
       rsis.append(rsi[-1])
       
    # Average RSIs        
    avg_rsi = sum(rsis) / len(rsis)
        
    return avg_rsi

def get_tsi():
    """Calculate TSI"""
    closes = np.array([c["close"] for c in candle_map['1d'][-200:]])
    macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=25, slowperiod=50, signalperiod=10)
    ema1 = talib.EMA(macd, timeperiod=13)
    ema2 = talib.EMA(ema1, timeperiod=21)
    tsi = ema2[-1]
    return tsi

def get_bollinger_bands():
    """Calculate Bollinger Bands"""
    closes = np.array([c["close"] for c in candle_map['1h'][-100:]])
    upperband, middleband, lowerband = talib.BBANDS(closes, timeperiod=20)
    return upperband[-1], middleband[-1], lowerband[-1]

def get_mtf_rsi_market_mood():
    rsi = get_multi_timeframe_rsi()
    tsi = get_tsi()
    ub, mb, lb = get_bollinger_bands()
    closes = get_closes_last_n_minutes("15m", 50)

    # Define the thresholds for dip and top reversals
    dip_threshold = lb
    top_threshold = ub

    # Check if the close price is in the dip reversal area
    if closes[-1] <= dip_threshold and closes[-2] > dip_threshold:
        mood = "dip up reversal"
    # Check if the close price is in the top reversal area
    elif closes[-1] >= top_threshold and closes[-2] < top_threshold:
        mood = "top down reversal"
    # Check if the close price is in the accumulation area
    elif closes[-1] > dip_threshold and closes[-1] < mb:
        mood = "accumulation"
    # Check if the close price is in the distribution area
    elif closes[-1] < top_threshold and closes[-1] > mb:
        mood = "distribution"
    # Check if the market is in a downtrend
    elif rsi < 50 and tsi < 0:
        mood = "downtrend"
    # Check if the market is in an uptrend
    elif rsi > 50 and tsi > 0:
        mood = "uptrend"
    # Check if the market is neutral with a bullish tendency
    elif rsi >= 50 and tsi >= 0:
        mood = "neutral with bullish tendency"
    # Check if the market is neutral with a bearish tendency
    elif rsi < 50 and tsi >= 0:
        mood = "neutral with bearish tendency"
    else:
        mood = "neutral"

    return mood

mood = get_mtf_rsi_market_mood()
print("MTF rsi mood: ", mood)

print()

##################################################
##################################################

def get_multi_timeframe_momentum():
    """Calculate momentum from multiple timeframes and average"""
    momentums = []
    
    for timeframe in timeframes:
        
        # Get candle data               
        candles = candle_map[timeframe][-100:]  
        
        # Calculate momentum using talib MOM
        momentum = talib.MOM(np.array([c["close"] for c in candles]), timeperiod=14)
        momentums.append(momentum[-1])
       
    # Average momentums        
    avg_momentum = sum(momentums) / len(momentums)
        
    return avg_momentum

print()

##################################################
##################################################

def get_mtf_market_mood():
    rsi_mood = get_mtf_rsi_market_mood()
    momentum = get_multi_timeframe_momentum()

    # Define the thresholds for momentum signals
    small_range_threshold = 5
    medium_range_threshold = 10

    # Define the indicators   
    indicator1 = momentum  
    indicator2 = 0

    # Check if the momentum signal is in the small range
    if abs(indicator1) < small_range_threshold:
        mood = "small range"
    # Check if the momentum signal is in the medium range
    elif abs(indicator1) < medium_range_threshold:
        mood = "medium range"
    # Check if the market is in a downtrend
    elif indicator1 < indicator2:
        mood = "MTF trend downtrend"
    # Check if the market is in an uptrend
    elif indicator1 > indicator2:
        mood = "MTF trend uptrend"
    else:
        mood = "MTF trend neutral"

    # Combine the RSI mood and momentum mood
    if rsi_mood == "dip up reversal" or rsi_mood == "uptrend":
        mood += " momentum bullish"
    elif rsi_mood == "top down reversal" or rsi_mood == "downtrend":
        mood += " momentum bearish"
    else:
        mood += " momentum neutral"

    return mood

print()

mtf_market_mood = get_mtf_market_mood()
print("MTF RSI and MOM market mood: ", mtf_market_mood)

print()

##################################################
##################################################

def get_momentum(timeframe):
    """Calculate momentum for a single timeframe"""
    # Get candle data               
    candles = candle_map[timeframe][-100:]  
    # Calculate momentum using talib MOM
    momentum = talib.MOM(np.array([c["close"] for c in candles]), timeperiod=14)
    return momentum[-1]

# Calculate momentum for each timeframe
for timeframe in timeframes:
    momentum = get_momentum(timeframe)
    print(f"Momentum for {timeframe}: {momentum}")

print()

##################################################
##################################################

# timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

overall_ranges = [
    ['1m', '5m'],
    ['3m', '15m'],
    ['5m', '30m'],
    ['15m', '1h'],
    ['30m', '2h'],
    ['1h', '4h'],
    ['2h', '6h'],
    ['6h', '12h'],
    ['8h', '1d'],
]

def collect_results():
    results = []
    
    for timeframes_for_range in overall_ranges:
        dist_min_sum = 0
        dist_max_sum = 0
        for timeframe in timeframes_for_range:
            # Call existing function 
            dist_to_min, dist_to_max, current_sine = scale_to_sine(timeframe)  
        
            # Add distances to running sum
            dist_min_sum += dist_to_min
            dist_max_sum += dist_to_max
        
        # Calculate average distances for this range
        num_timeframes = len(timeframes_for_range)
        dist_min_avg = dist_min_sum / num_timeframes
        dist_max_avg = dist_max_sum / num_timeframes
        
        # Append result tuple
        results.append((dist_min_avg, dist_max_avg))
        
    # Calculate overall percentages      
    overall_dist_min = sum([r[0] for r in results]) / len(results)    
    overall_dist_max = sum([r[1] for r in results]) / len(results)
    
    return overall_dist_min, overall_dist_max, results

# Call function      
overall_dist_min, overall_dist_max, results = collect_results()
print()

print("Overall distances:")
print(f"  To minimum: {overall_dist_min:.2f}%")  
print(f"  To maximum: {overall_dist_max:.2f}%")

print()

for i in range(len(overall_ranges)):
    timeframes_for_range = overall_ranges[i]
    dist_min_avg, dist_max_avg = results[i]
    print(f"Overall range {i+1} ({', '.join(timeframes_for_range)}):")
    print(f"  To minimum: {dist_min_avg:.2f}%")  
    print(f"  To maximum: {dist_max_avg:.2f}%")
    print()

print()

##################################################
##################################################

def generate_momentum_sinewave(timeframes):
    # Initialize variables
    momentum_sorter = []
    market_mood = []
    last_reversal = None
    last_reversal_value_on_sine = None
    last_reversal_value_on_price = None
    next_reversal = None
    next_reversal_value_on_sine = None
    next_reversal_value_on_price = None

    # Loop over timeframes
    for timeframe in timeframes:
        # Get close prices for current timeframe
        close_prices = np.array(get_closes(timeframe))

        # Get last close price
        current_close = close_prices[-1]

        # Calculate sine wave for current timeframe
        sine_wave, leadsine = talib.HT_SINE(close_prices)

        # Replace NaN values with 0
        sine_wave = np.nan_to_num(sine_wave)
        sine_wave = -sine_wave

        # Get the sine value for last close
        current_sine = sine_wave[-1]

        # Calculate the min and max sine
        sine_wave_min = np.nanmin(sine_wave) # Use nanmin to ignore NaN values
        sine_wave_max = np.nanmax(sine_wave)

        # Calculate price values at min and max sine
        sine_wave_min_price = close_prices[sine_wave == sine_wave_min][0]
        sine_wave_max_price = close_prices[sine_wave == sine_wave_max][0]
     

        # Calculate the difference between the max and min sine
        sine_wave_diff = sine_wave_max - sine_wave_min

        # If last close was the lowest, set as last reversal                                  
        if current_sine == sine_wave_min:
            last_reversal = 'dip'
            last_reversal_value_on_sine = sine_wave_min 
            last_reversal_value_on_price = sine_wave_min_price
        
        # If last close was the highest, set as last reversal                                 
        if current_sine == sine_wave_max:
            last_reversal = 'top'
            last_reversal_value_on_sine = sine_wave_max
            last_reversal_value_on_price = sine_wave_max_price

        # Calculate % distances
        newsine_dist_min, newsine_dist_max = [], []
        for close in close_prices:
            # Calculate distances as percentages
            dist_from_close_to_min = ((current_sine - sine_wave_min) /  
                                      sine_wave_diff) * 100            
            dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                                      sine_wave_diff) * 100
                
            newsine_dist_min.append(dist_from_close_to_min)       
            newsine_dist_max.append(dist_from_close_to_max)

        # Take average % distances
        avg_dist_min = sum(newsine_dist_min) / len(newsine_dist_min)
        avg_dist_max = sum(newsine_dist_max) / len(newsine_dist_max)

        # Determine market mood based on % distances
        if avg_dist_min <= 15:
            mood = "At DIP Reversal and Up to Bullish"
            if last_reversal != 'dip':
                next_reversal = 'dip'
                next_reversal_value_on_sine = sine_wave_min
                next_reversal_value_on_price = close_prices[sine_wave == sine_wave_min][0]
        elif avg_dist_max <= 15:
            mood = "At TOP Reversal and Down to Bearish"
            if last_reversal != 'top':
                next_reversal = 'top'
                next_reversal_value_on_sine = sine_wave_max
                next_reversal_value_on_price = close_prices[sine_wave == sine_wave_max][0]
        elif avg_dist_min < avg_dist_max:
            mood = "Bullish"
        else:
            mood = "Bearish"

        # Append momentum score and market mood to lists
        momentum_score = avg_dist_max - avg_dist_min
        momentum_sorter.append(momentum_score)
        market_mood.append(mood)

        # Print distances and market mood
        print(f"{timeframe} Close is now at "       
              f"dist. to min: {avg_dist_min:.2f}% "
              f"and at "
              f"dist. to max: {avg_dist_max:.2f}%. "
              f"Market mood: {mood}")

        # Update last and next reversal info
        if next_reversal:
            last_reversal = next_reversal
            last_reversal_value_on_sine = next_reversal_value_on_sine
            last_reversal_value_on_price = next_reversal_value_on_price
            next_reversal = None
            next_reversal_value_on_sine = None
            next_reversal_value_on_price = None

    # Get close prices for the 1-minute timeframe and last 3 closes
    close_prices = np.array(get_closes('1m'))

    # Calculate sine wave
    sine_wave, leadsine = talib.HT_SINE(close_prices)

    # Replace NaN values with 0
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    # Get the sine value for last close
    current_sine = sine_wave[-1]

    # Get current date and time
    now = datetime.datetime.now()

    # Calculate the min and max sine
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate the difference between the maxand min sine
    sine_wave_diff = sine_wave_max - sine_wave_min

    # Calculate % distances
    dist_from_close_to_min = ((current_sine - sine_wave_min) / 
                              sine_wave_diff) * 100
    dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                              sine_wave_diff) * 100

    # Determine market mood based on % distances
    if dist_from_close_to_min <= 15:
        mood = "At DIP Reversal and Up to Bullish"
        if last_reversal != 'dip':
            next_reversal = 'dip'
            next_reversal_value_on_sine = sine_wave_min

    elif dist_from_close_to_max <= 15:
        mood = "At TOP Reversal and Down to Bearish"
        if last_reversal != 'top':
            next_reversal = 'top'
            next_reversal_value_on_sine = sine_wave_max

    elif dist_from_close_to_min < dist_from_close_to_max:
        mood = "Bullish"
    else:
        mood = "Bearish"

    # Get the close prices that correspond to the min and max sine values
    close_prices_between_min_and_max = close_prices[(sine_wave >= sine_wave_min) & (sine_wave <= sine_wave_max)]

    print()

    # Print distances and market mood for 1-minute timeframe
    print(f"On 1min timeframe,Close is now at "       
          f"dist. to min: {dist_from_close_to_min:.2f}% "
          f"and at "
          f"dist. to max:{dist_from_close_to_max:.2f}%. "
          f"Market mood: {mood}")

    min_val = min(close_prices_between_min_and_max)
    max_val = max(close_prices_between_min_and_max)

    print("The lowest value in the array is:", min_val)
    print("The highest value in the array is:", max_val)

    print()

    # Update last and next reversal info
    #if next_reversal:
        #last_reversal = next_reversal
        #last_reversal_value_on_sine = next_reversal_value_on_sine

        #next_reversal = None
        #next_reversal_value_on_sine = None


    # Print last and next reversal info
    #if last_reversal:
        #print(f"Last reversal was at {last_reversal} on the sine wave at {last_reversal_value_on_sine:.2f} ")

    # Return the momentum sorter, market mood, close prices between min and max sine, and reversal info
    return momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max      

momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max = generate_momentum_sinewave(timeframes)

print()

#print("Close price values between last reversals on sine: ")
#print(close_prices_between_min_and_max)

print()

print("Current close on sine value now at: ", current_sine)
print("distances as percentages from close to min: ", dist_from_close_to_min, "%")
print("distances as percentages from close to max: ", dist_from_close_to_max, "%")
print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
print("Mood on 1min timeframe is now at: ", market_mood[-12])

print()

##################################################
##################################################

def generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Calculate the sine wave using HT_SINE
    sine_wave, _ = talib.HT_SINE(close_prices)
    
    # Replace NaN values with 0 using nan_to_num
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    print("Current close on Sine wave:", sine_wave[-1])

    # Calculate the minimum and maximum values of the sine wave
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate the distance from close to min and max as percentages on a scale from 0 to 100%
    dist_from_close_to_min = ((sine_wave[-1] - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_from_close_to_max = ((sine_wave_max - sine_wave[-1]) / (sine_wave_max - sine_wave_min)) * 100

    print("Distance from close to min:", dist_from_close_to_min)
    print("Distance from close to max:", dist_from_close_to_max)

    # Calculate the range of values for each quadrant
    range_q1 = (sine_wave_max - sine_wave_min) / 4
    range_q2 = (sine_wave_max - sine_wave_min) / 4
    range_q3 = (sine_wave_max - sine_wave_min) / 4
    range_q4 = (sine_wave_max - sine_wave_min) / 4

    # Set the EM amplitude for each quadrant based on the range of values
    em_amp_q1 = range_q1 / percent_to_max_val
    em_amp_q2 = range_q2 / percent_to_max_val
    em_amp_q3 = range_q3 / percent_to_max_val
    em_amp_q4 = range_q4 / percent_to_max_val

    # Calculate the EM phase for each quadrant
    em_phase_q1 = 0
    em_phase_q2 = math.pi/2
    em_phase_q3 = math.pi
    em_phase_q4 = 3*math.pi/2

    # Calculate the current position of the price on the sine wave
    current_position = (sine_wave[-1] - sine_wave_min) / (sine_wave_max - sine_wave_min)
    current_quadrant = 0

    # Determine which quadrant the current position is in
    if current_position < 0.25:
        # In quadrant 1
        em_amp = em_amp_q1
        em_phase = em_phase_q1
        current_quadrant = 1
        print("Current position is in quadrant 1. Distance from 0% to 25% of range:", (current_position - 0.0) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    elif current_position < 0.5:
        # In quadrant 2
        em_amp = em_amp_q2
        em_phase = em_phase_q2
        current_quadrant = 2
        print("Current position is in quadrant 2. Distance from 25% to 50% of range:", (current_position - 0.25) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    elif current_position < 0.75:
        # In quadrant 3
        em_amp = em_amp_q3
        em_phase = em_phase_q3
        current_quadrant = 3
        print("Current position is in quadrant 3. Distance from 50% to 75% of range:", (current_position - 0.5) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    else:
        # In quadrant 4
        em_amp = em_amp_q4
        em_phase = em_phase_q4
        current_quadrant = 4
        print("Current position is in quadrant 4. Distance from 75% to 100% of range:", (current_position - 0.75) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)

    print("EM amplitude:", em_amp)
    print("EM phase:", em_phase)

    # Calculate the EM value
    em_value = em_amp * math.sin(em_phase)

    print("EM value:", em_value)

    # Determine the trend direction based on the EM phase differences
    if em_phase_q4 - em_phase_q3 > 0 and em_phase_q3 - em_phase_q2 > 0 and em_phase_q2 - em_phase_q1 > 0:
        trend_direction = "Down"
    elif em_phase_q4 - em_phase_q3 < 0 and em_phase_q3 - em_phase_q2 < 0 and em_phase_q2 - em_phase_q1 < 0:
        trend_direction = "Up"
    else:
        trend_direction = "Sideways"

    print("Trend direction:", trend_direction)

    # Calculate the percentage of the price range
    price_range = candles[-1]["high"] - candles[-1]["low"]
    price_range_percent = (close_prices[-1] - candles[-1]["low"]) / price_range * 100

    print("Price range percent:", price_range_percent)

    # Calculate the momentum value
    momentum = em_value * price_range_percent / 100

    print("Momentum value:", momentum)

    print()

    # Return a dictionary of all the features
    return {
        "current_close": sine_wave[-1],
        "dist_from_close_to_min": dist_from_close_to_min,
        "dist_from_close_to_max": dist_from_close_to_max,
        "current_quadrant": current_quadrant,
        "em_amplitude": em_amp,
        "em_phase": em_phase,
        "trend_direction": trend_direction,
        "price_range_percent": price_range_percent,
        "momentum": momentum,
        "max": sine_wave_max,
        "min": sine_wave_min
    }
   
#sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)
#print(sine_wave)

print()

##################################################
##################################################

def generate_market_mood_forecast(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Call generate_new_momentum_sinewave to get the sine wave and other features
    sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=percent_to_max_val, percent_to_min_val=percent_to_min_val)

    current_quadrant = sine_wave["current_quadrant"]
    em_phase_q1 = sine_wave["em_phase"]
    em_phase_q2 = em_phase_q1 + math.pi/2
    em_phase_q3 = em_phase_q1 + math.pi
    em_phase_q4 = em_phase_q1 + 3*math.pi/2

    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177

    # Calculate the Brun constant from the phi ratio and sqrt(5)
    brun_constant = math.sqrt(PHI * math.sqrt(5))

    # Define PI constant with 15 decimals
    PI = 3.1415926535897932384626433832795028842

    # Calculate sacred frequency
    sacred_freq = (432 * PHI ** 2) / 360

    # Calculate Alpha and Omega ratios
    alpha_ratio = PHI / PI
    omega_ratio = PI / PHI

    # Calculate Alpha and Omega spiral angle rates
    alpha_spiral = (2 * math.pi * sacred_freq) / alpha_ratio
    omega_spiral = (2 * math.pi * sacred_freq) / omega_ratio

    # Calculate quadrature phase shift based on current quadrant
    if current_quadrant == 1:
        # Up cycle from Q1 to Q4
        quadrature_phase = em_phase_q1
        em_phase = alpha_spiral
    elif current_quadrant == 2:
        quadrature_phase = em_phase_q2
        em_phase = omega_spiral
    elif current_quadrant == 3:
        quadrature_phase = em_phase_q3
        em_phase = omega_spiral
    else:
        quadrature_phase = em_phase_q4
        em_phase = alpha_spiral

    cycle_direction = "UP"
    next_quadrant = 1

    if current_quadrant == 1:
        next_quadrant = 2
        cycle_direction = "UP"

    elif current_quadrant == 2:
        if cycle_direction == "UP":
            next_quadrant = 3
        elif cycle_direction == "DOWN":
            next_quadrant = 1

    elif current_quadrant == 3:
        if cycle_direction == "UP":
            next_quadrant = 4
        elif cycle_direction == "DOWN":
            next_quadrant = 2

    elif current_quadrant == 4:
        if cycle_direction == "UP":
            next_quadrant = 3
            cycle_direction = "DOWN"

    current_point = ""
    if current_quadrant == 1:
        current_point = "Apex"  
    if current_quadrant == 2:  
        current_point = "Left"       
    if current_quadrant == 3:
        current_point = "Base"        
    if current_quadrant == 4:
        current_point = "Right"  
            
    next_point = ""        
    if next_quadrant == 1:
        next_point = "Apex"   
    if next_quadrant == 2:
         next_point = "Left"          
    if next_quadrant == 3:
        next_point = "Base"       
    if next_quadrant == 4:
        next_point = "Right"

    # Calculate quadrature phase
    if next_quadrant == 1:
        next_quadrature_phase = em_phase_q1
    elif next_quadrant == 2:
        next_quadrature_phase = em_phase_q2
    elif next_quadrant == 3:
        next_quadrature_phase = em_phase_q3
    else:
        next_quadrature_phase = em_phase_q4

    # Calculate EM value
    em_value = sine_wave["em_amplitude"] * math.sin(em_phase)

    # Calculate quadrature phase shift from current to next quadrant
    quadrature = next_quadrature_phase - quadrature_phase

    if quadrature > 0:
        cycle_direction = "UP"
    else:
        cycle_direction = "DOWN"

    # Define the frequency bands and their corresponding emotional values
    frequency_bands = {"Delta": -0.5, "Theta": -0.25, "Alpha": 0, "Beta": 0.25, "Gamma": 0.5}

    # Calculate the emotional value of each frequency band using phi
    emotional_values = {band: frequency_bands[band] * PHI for band in frequency_bands}

    # Divide the frequency spectrum into 4 quadrants based on Metatron's Cube geometry
    quadrant_amplitudes = {"Apex": 1, "Left": 0.5, "Base": 0, "Right": 0.5}
    quadrant_phases = {"Apex": 0, "Left": math.pi/2, "Base": math.pi, "Right": 3*math.pi/2}

    # Calculate emotional amplitude and phase values for each quadrant
    quadrant_emotional_values = {}
    for quadrant in quadrant_amplitudes:
        amplitude = quadrant_amplitudes[quadrant]
        phase = quadrant_phases[quadrant]
        quadrant_emotional_values[quadrant] = {"amplitude": amplitude, "phase": phase}

    # Calculate the forecast mood for the next 1 hour based on the phi value of each frequency band and mapping that to an emotional state
    forecast_moods = {}
    for band in emotional_values:
        emotional_value = emotional_values[band]
        phi_value = PHI ** (emotional_value / brun_constant)
        forecast_moods[band] = math.cos(phi_value + em_value)

    # Sort the frequencies from most negative to most positive based on their emotional value
    sorted_frequencies = sorted(forecast_moods, key=lambda x: forecast_moods[x])

    # Calculate average moods for the highest 3 and lowest 3 frequencies to determine the overall trend
    high_moods = [forecast_moods[band] for band in sorted_frequencies[-3:]]
    low_moods = [forecast_moods[band] for band in sorted_frequencies[:3]]
    avg_high_mood = sum(high_moods) / len(high_moods)
    avg_low_mood = sum(low_moods) / len(low_moods)

    # Calculate weighted averages of the top n and bottom n frequencies to get a more nuanced forecast
    n = 2
    weighted_high_mood = sum([forecast_moods[band] * (n - i) for i, band in enumerate(sorted_frequencies[-n:])]) / sum(range(1, n+1))
    weighted_low_mood = sum([forecast_moods[band] * (n - i) for i, band in enumerate(sorted_frequencies[:n])]) / sum(range(1, n+1))

    # Map the 4 quadrants to the 4 points of Metatron's Cube (Apex, Left, Base, Right)
    mapped_quadrants = {}
    for quadrant in quadrant_emotional_values:
        amplitude = quadrant_emotional_values[quadrant]["amplitude"]
        phase = quadrant_emotional_values[quadrant]["phase"]
        mapped_quadrants[quadrant] = amplitude * math.sin(em_value + phase)

    # Identify the minimum and maximum frequency nodes to determine reversal points
    min_node = sorted_frequencies[0]
    max_node = sorted_frequencies[-1]

    # Based on the minimum and maximum nodes, calculate forecasts for when mood reversals may occur
    if forecast_moods[min_node] < avg_low_mood and forecast_moods[max_node] > avg_high_mood:
        mood_reversal_forecast = "Mood reversal expected in the short term"
    elif forecast_moods[min_node] < weighted_low_mood and forecast_moods[max_node] > weighted_high_mood:
        mood_reversal_forecast = "Mood may reverse soon"
    else:
        mood_reversal_forecast = "No mood reversal expected in the near term"

    # Determine an overall market mood based on the highest and lowest 3 frequencies - positive, negative or neutral
    if avg_high_mood > 0 and avg_low_mood > 0:
        market_mood = "Positive"
    elif avg_high_mood < 0 and avg_low_mood < 0:
        market_mood = "Negative"
    else:
        market_mood = "Neutral"

    # Return the market mood forecast
    return {"cycle_direction": cycle_direction, "quadrant_emotional_values": quadrant_emotional_values, "forecast_moods": forecast_moods, "sorted_frequencies": sorted_frequencies, "avg_high_mood": avg_high_mood, "avg_low_mood": avg_low_mood, "weighted_high_mood": weighted_high_mood, "weighted_low_mood": weighted_low_mood, "mapped_quadrants": mapped_quadrants, "min_node": min_node, "max_node": max_node, "mood_reversal_forecast": mood_reversal_forecast, "market_mood": market_mood, "current_point": current_point, "next_point": next_point}
  
generate_market_mood_forecast(close_prices, candles, percent_to_max_val=50, percent_to_min_val=50)

market_mood_forecast = generate_market_mood_forecast(close_prices, candles, percent_to_max_val=50, percent_to_min_val=50)

cycle_direction = market_mood_forecast["cycle_direction"]
quadrant_emotional_values = market_mood_forecast["quadrant_emotional_values"]
forecast_moods = market_mood_forecast["forecast_moods"]
sorted_frequencies = market_mood_forecast["sorted_frequencies"]
avg_high_mood = market_mood_forecast["avg_high_mood"]
avg_low_mood = market_mood_forecast["avg_low_mood"]
weighted_high_mood = market_mood_forecast["weighted_high_mood"]
weighted_low_mood = market_mood_forecast["weighted_low_mood"]
mapped_quadrants = market_mood_forecast["mapped_quadrants"]
min_node = market_mood_forecast["min_node"]
max_node = market_mood_forecast["max_node"]
mood_reversal_forecast = market_mood_forecast["mood_reversal_forecast"]
market_mood = market_mood_forecast["market_mood"]

current_point = market_mood_forecast["current_point"] 
next_point = market_mood_forecast["next_point"]

print(f"Current point: {current_point}")
print(f"Next point: {next_point}")

print("Cycle direction:", cycle_direction)
print("Quadrant emotional values:", quadrant_emotional_values)
print("Forecast moods:", forecast_moods)
print("Sorted frequencies:", sorted_frequencies)
print("Average high mood:", avg_high_mood)
print("Average low mood:", avg_low_mood)
print("Weighted high mood:", weighted_high_mood)
print("Weighted low mood:", weighted_low_mood)
print("Mapped quadrants:", mapped_quadrants)
print("Minimum node:", min_node)
print("Maximum node:", max_node)
print("Mood reversal forecast:", mood_reversal_forecast)
print("Market mood:", market_mood)

print()

##################################################
##################################################

def reversals_unit_circle(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Call generate_new_momentum_sinewave to get the sine wave and other features
    sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=percent_to_max_val, percent_to_min_val=percent_to_min_val)

    # Get the current quadrant and EM phase of the sine wave
    current_quadrant = sine_wave["current_quadrant"]
    em_phase = sine_wave["em_phase"]

    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177

    # Define PI constant with 15 decimals
    PI = 3.1415926535897932384626433832795028842

    # Define the frequency bands and their corresponding emotional values
    frequency_bands = {"Delta": -0.5, "Theta": -0.25, "Alpha": 0, "Beta": 0.25, "Gamma": 0.5}

    # Calculate the emotional value of each frequency band using phi
    emotional_values = {band: frequency_bands[band] * PHI for band in frequency_bands}

    # Divide the unit circle into 4 quadrants based on Metatron's Cube geometry
    quadrants = {"Apex": 0, "Left": math.pi/2, "Base": math.pi, "Right": 3*math.pi/2}

    # Map the emotional values to the corresponding quadrants and calculate the emotional amplitude and phase values
    quadrant_emotional_values = {}
    for quadrant in quadrants:
        quadrant_phase = quadrants[quadrant]
        emotional_value = emotional_values["Gamma"] * math.sin(em_phase + quadrant_phase)
        quadrant_amplitude = (emotional_value + PHI) / (2 * PHI)
        quadrant_emotional_values[quadrant] = {"amplitude": quadrant_amplitude, "phase": quadrant_phase}

    # Calculate the forecast mood for each quadrant based on the phi value of each frequency band and mapping that to an emotional state
    forecast_moods = {}
    for quadrant in quadrant_emotional_values:
        quadrant_amplitude = quadrant_emotional_values[quadrant]["amplitude"]
        quadrant_phase = quadrant_emotional_values[quadrant]["phase"]
        quadrant_em_value = emotional_values["Gamma"] * math.sin(em_phase + quadrant_phase)
        quadrant_forecast_moods = {}
        for band in emotional_values:
            emotional_value = emotional_values[band]
            phi_value = PHI ** (emotional_value / math.sqrt(PHI * math.sqrt(5)))
            quadrant_forecast_moods[band] = quadrant_amplitude * math.cos(phi_value + quadrant_em_value)
        forecast_moods[quadrant] = quadrant_forecast_moods

    # Calculate the average moods for the highest and lowest 3 frequencies in each quadrant to determine the overall trend
    avg_moods = {}
    for quadrant in forecast_moods:
        quadrant_forecast_moods = forecast_moods[quadrant]
        sorted_frequencies = sorted(quadrant_forecast_moods, key=lambda x: quadrant_forecast_moods[x])
        high_moods = [quadrant_forecast_moods[band] for band in sorted_frequencies[-3:]]
        low_moods = [quadrant_forecast_moods[band] for band in sorted_frequencies[:3]]
        avg_high_mood = sum(high_moods) / len(high_moods)
        avg_low_mood = sum(low_moods) / len(low_moods)
        avg_moods[quadrant] = {"avg_high_mood": avg_high_mood, "avg_low_mood": avg_low_mood}

    # Identify the minimum and maximum frequency nodes to determine reversal points
    min_node = None
    max_node = None
    min_mood = 1
    max_mood = -1
    for quadrant in forecast_moods:
        quadrant_forecast_moods = forecast_moods[quadrant]
        for band in quadrant_forecast_moods:
            mood = quadrant_forecast_moods[band]
            if mood < min_mood:
                min_mood = mood
                min_node = quadrant + " " + band
            if mood > max_mood:
                max_mood = mood
                max_node = quadrant + " " + band

    #Print the minimum and maximum frequency nodes and their corresponding moods
    print("Minimum frequency node: {}, Mood: {}".format(min_node, min_mood))
    print("Maximum frequency node: {}, Mood: {}".format(max_node, max_mood))

    # Calculate the weighted average moods for the highest and lowest 3 frequencies in each quadrant to get a more nuanced forecast
    weighted_moods = {}
    for quadrant in forecast_moods:
        quadrant_forecast_moods = forecast_moods[quadrant]
        sorted_frequencies = sorted(quadrant_forecast_moods, key=lambda x: quadrant_forecast_moods[x])
        high_weights = [quadrant_forecast_moods[band] / sum(quadrant_forecast_moods.values()) for band in sorted_frequencies[-3:]]
        low_weights = [quadrant_forecast_moods[band] / sum(quadrant_forecast_moods.values()) for band in sorted_frequencies[:3]]
        weighted_high_mood = sum([high_weights[i] * high_moods[i] for i in range(len(high_moods))])
        weighted_low_mood = sum([low_weights[i] * low_moods[i] for i in range(len(low_moods))])
        weighted_moods[quadrant] = {"weighted_high_mood": weighted_high_mood, "weighted_low_mood": weighted_low_mood}

    # Determine the forecast direction based on the quadrant with the highest average mood
    sorted_avg_moods = sorted(avg_moods, key=lambda x: avg_moods[x]["avg_high_mood"]+avg_moods[x]["avg_low_mood"], reverse=True)
    forecast_direction = sorted_avg_moods[0]

    # Determine the mood reversal forecast based on the minimum and maximum frequency nodes
    mood_reversal_forecast = None
    if min_mood < 0 and max_mood > 0:
        if sorted_frequencies.index(min_node.split()[-1]) < sorted_frequencies.index(max_node.split()[-1]):
            mood_reversal_forecast = "Positive mood reversal expected in the near term"
        else:
            mood_reversal_forecast = "Negative mood reversal expected in the near term"

    # Determine the overall market mood based on the highest and lowest 3 frequencies in each quadrant
    sorted_weighted_moods = sorted(weighted_moods, key=lambda x: weighted_moods[x]["weighted_high_mood"]+weighted_moods[x]["weighted_low_mood"], reverse=True)
    market_mood = None
    if weighted_moods[sorted_weighted_moods[0]]["weighted_high_mood"] > weighted_moods[sorted_weighted_moods[-1]]["weighted_low_mood"]:
        market_mood = "Positive"
    elif weighted_moods[sorted_weighted_moods[-1]]["weighted_low_mood"] > weighted_moods[sorted_weighted_moods[0]]["weighted_high_mood"]:
        market_mood = "Negative"
    else:
        market_mood = "Neutral"

    # Print the forecast results
    print("Forecast Direction: {}".format(forecast_direction))
    for quadrant in quadrant_emotional_values:
        print("\n{} Quadrant".format(quadrant))
        print("Emotional Amplitude: {}".format(quadrant_emotional_values[quadrant]["amplitude"]))
        print("Emotional Phase: {}".format(quadrant_emotional_values[quadrant]["phase"]))
        for band in frequency_bands:
            print("{}: {}".format(band, forecast_moods[quadrant][band]))
        print("Average High Mood: {}".format(avg_moods[quadrant]["avg_high_mood"]))
        print("Average Low Mood: {}".format(avg_moods[quadrant]["avg_low_mood"]))
        print("Weighted High Mood: {}".format(weighted_moods[quadrant]["weighted_high_mood"]))
        print("Weighted Low Mood: {}".format(weighted_moods[quadrant]["weighted_low_mood"]))
    print("\nMood Reversal Forecast: {}".format(mood_reversal_forecast))
    print("Market Mood: {}".format(market_mood))

    return {
        "quadrant_emotional_values": quadrant_emotional_values,
        "forecast_moods": forecast_moods,
        "min_node": min_node,
        "max_node": max_node,
        "avg_moods": avg_moods,
        "weighted_moods": weighted_moods,
        "forecast_direction": forecast_direction,
        "mood_reversal_forecast": mood_reversal_forecast, 
        "market_mood": market_mood  
    }

results = reversals_unit_circle(close_prices, candles)

quadrant_emotional_values, forecast_moods, min_node, max_node, avg_moods, weighted_moods, forecast_direction, mood_reversal_forecast, market_mood = results.values()

# Now you have all the results as separate variables  
print(quadrant_emotional_values, forecast_moods, min_node, max_node, avg_moods, weighted_moods, forecast_direction, mood_reversal_forecast, market_mood)

##################################################
##################################################

def generate_market_mood_forecast_gr(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    """
    Generates a market mood forecast based on the given inputs.

    Args:
    close_prices (list): A list of close prices.
    candles (list): A list of candlestick data.
    percent_to_max_val (float): The percent to maximum value for the sine wave.
    percent_to_min_val (float): The percent to minimum value for the sine wave.

    Returns:
    A dictionary containing the market mood forecast and various mood and market forecasts.
    """

    # Define constants and ratios
    pi = 3.14159
    max_val = max(close_prices)
    min_val = min(close_prices)
    total_range = max_val - min_val
    time_dilation = 1.0 / len(close_prices)
    quadrants = ["I", "II", "III", "IV"]

    # Determine frequency bands
    frequencies = {}
    for i in range(1, 5):
        frequencies[i] = 1 / (total_range / (i * pi))
    sorted_frequencies = sorted(frequencies, key=frequencies.get)
    min_node = sorted_frequencies[0]
    max_node = sorted_frequencies[-1]

    # Calculate emotional values for frequency bands
    forecast_moods = {}
    for band in frequencies:
        phase = 2 * pi * (band / 4)
        val = percent_to_max_val * abs(math.sin(phase)) + percent_to_min_val
        forecast_moods[band] = val

    # Apply time dilation to frequency bands
    dilated_forecast_moods = {}
    for band in forecast_moods:
        dilated_forecast_moods[band] = forecast_moods[band] * time_dilation

    # Calculate average moods for dilated frequencies
    num_frequencies = 2
    dilated_high_moods = []
    dilated_low_moods = []

    if dilated_forecast_moods:
        dilated_high_moods = [dilated_forecast_moods[band] for band in sorted_frequencies[-num_frequencies:]]
        dilated_low_moods = [dilated_forecast_moods[band] for band in sorted_frequencies[:num_frequencies]]

    dilated_avg_high_mood = sum(dilated_high_moods) / len(dilated_high_moods) if dilated_high_moods else 0
    dilated_avg_low_mood = sum(dilated_low_moods) / len(dilated_low_moods) if dilated_low_moods else 0

    # Calculate weighted averages for dilated frequencies
    weights = [num_frequencies - i for i in range(num_frequencies)]
    dilated_weighted_high_mood = 0
    dilated_weighted_low_mood = 0

    if dilated_forecast_moods:
        dilated_high_freqs = list(dilated_forecast_moods.keys())[-num_frequencies:]
        dilated_high_moods = [dilated_forecast_moods[freq] for freq in dilated_high_freqs]

        for i in range(num_frequencies):
            dilated_weighted_high_mood += weights[i] * dilated_high_moods[i]

        dilated_weighted_high_mood /= sum(weights)

        dilated_low_freqs = list(dilated_forecast_moods.keys())[:num_frequencies]
        dilated_low_moods = [dilated_forecast_moods[freq] for freq in dilated_low_freqs]

        for i in range(num_frequencies):
            dilated_weighted_low_mood += weights[i] * dilated_low_moods[i]

        dilated_weighted_low_mood /= sum(weights)

    # Determine reversal forecast based on dilated frequencies
    dilated_mood_reversal_forecast = ""

    if dilated_forecast_moods and dilated_forecast_moods[min_node] < dilated_avg_low_mood and dilated_forecast_moods[max_node] > dilated_avg_high_mood:
        dilated_mood_reversal_forecast = "Mood reversal possible"

    # Determine market mood based on dilated average moods
    dilated_market_mood = ""

    if dilated_avg_high_mood > 0 and dilated_avg_low_mood > 0:
        dilated_market_mood = "Positive" if dilated_avg_high_mood > dilated_avg_low_mood else "Negative"

    # Create dictionary of mood and market forecasts
    forecast_dict = {
        "dilated_mood_reversal_forecast": dilated_mood_reversal_forecast,
        "dilated_market_mood": dilated_market_mood,
        "dilated_avg_high_mood": dilated_avg_high_mood,
        "dilated_avg_low_mood": dilated_avg_low_mood,
        "dilated_weighted_high_mood": dilated_weighted_high_mood,
        "dilated_weighted_low_mood": dilated_weighted_low_mood
    }

    # Determine quadrant based on current price position
    current_price = close_prices[-1]
    quadrant = ""
    if current_price > max_val - (total_range / 2):
        if current_price > max_val - (total_range / 4):
            quadrant = quadrants[0]
        else:
            quadrant = quadrants[1]
    else:
        if current_price < min_val + (total_range / 4):
            quadrant = quadrants[2]
        else:
            quadrant = quadrants[3]

    # Add quadrant to forecast dictionary
    forecast_dict["quadrant"] = quadrant

    return forecast_dict

forecast = generate_market_mood_forecast_gr(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)

print(forecast)

print()

##################################################
##################################################

def octa_metatron_cube(close_prices, candles,  
                       percent_to_max_val=5,  
                       percent_to_min_val=5):

    sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                               percent_to_max_val, 
                               percent_to_min_val)  
  
    current_quadrant = sine_wave["current_quadrant"]
    current_value = sine_wave["current_close"]
    current_quadrant = 0

    sine_wave_max = sine_wave["max"]    
    sine_wave_min = sine_wave["min"] 

    if sine_wave["current_quadrant"] == 1:
         print("In quadrant 1!")
    elif sine_wave["current_quadrant"] == 2:  
         print("In quadrant 2!")            
    elif sine_wave["current_quadrant"] == 3:               
         print("In quadrant 3!")           
    elif sine_wave["current_quadrant"] == 4:                 
         print("In quadrant 4!")
    
    print()

    print("Close price map on sine now at: ", sine_wave)

    print()

    print("Min at:", sine_wave_min)
    print("Max at: ", sine_wave_max)

    print() 
 
    # Get the current quadrant and EM phase of the sine wave
    current_quadrant = sine_wave["current_quadrant"]
    em_phase = sine_wave["em_phase"]
    em_amp = sine_wave["em_amplitude"]

    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177  
   
    # Calculate the Brun constant from the phi ratio and sqrt(5)
    brun_constant = math.sqrt(PHI * math.sqrt(5))

    # Define PI constant with 15 decimals    
    PI = 3.1415926535897932384626433832795028842
   
    # Define e constant with 15 decimals   
    e =  2.718281828459045235360287471352662498  

    # Calculate sacred frequency
    sacred_freq = (432 * PHI ** 2) / 360
    
    # Calculate Alpha and Omega ratios   
    alpha_ratio = PHI / PI       
    omega_ratio = PI / PHI
          
    # Calculate Alpha and Omega spiral angle rates     
    alpha_spiral = (2 * math.pi * sacred_freq) / alpha_ratio
    omega_spiral = (2 * math.pi * sacred_freq) / omega_ratio

    start_value = 0.0 

    frequencies = []
    frequencies_next = []
    forecast = []

    freq_range = 25
    
    print(em_phase)
    print(em_amp) 

    current_quadrant_amplitude = em_amp
    current_quadrant_phase= em_amp

    print()
  
    #Calculate X:Y ratio from golden ratio     
    ratio = 2 * (PHI - 1)  

    # Calculate EM phases based on dividing whole range by X:Y ratio  
    em_phase_q1 = 0  
    em_phase_q2= PI * ratio / 2 
    em_phase_q3 = PI * ratio           
    em_phase_q4 = PI * ratio * 1.5  

    # Determine the trend direction based on the EM phase differences
    em_phase_diff_q1_q2 = em_phase_q2 - em_phase_q1
    em_phase_diff_q2_q3 = em_phase_q3 - em_phase_q2
    em_phase_diff_q3_q4 = em_phase_q4 - em_phase_q3
    em_phase_diff_q4_q1 = 2*math.pi - (em_phase_q4 - em_phase_q1)

    # Assign EM amplitudes based on X:Y ratio
    em_amp_q1 = (sine_wave_max - sine_wave_min) * ratio / 4
    em_amp_q2 = (sine_wave_max - sine_wave_min) * ratio / 4
    em_amp_q3 = (sine_wave_max - sine_wave_min) * ratio / 4
    em_amp_q4 = (sine_wave_max - sine_wave_min) * ratio / 4

    for i in range(freq_range):
        frequency = i * sacred_freq      
        #em_value = current_quadrant_amplitude * math.sin(current_quadrant_phase * frequency)

       # Calculate EM value based on frequency   
        if current_quadrant == 1: 
            # Most negative frequencies in Q1
            em_value = em_amp_q1 * math.sin(em_phase_q1 * frequency) 
            print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 2: 
            # Most negative frequencies in Q1
            em_value = em_amp_q2 * math.sin(em_phase_q2 * frequency) 
            print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 3: 
            # Most negative frequencies in Q1
            em_value = em_amp_q3 * math.sin(em_phase_q3 * frequency) 
            print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 4: 
            # Most negative frequencies in Q1
            em_value = em_amp_q4 * math.sin(em_phase_q4 * frequency) 
            print(em_value)

        frequencies.append({
            'number': i,
            'frequency': frequency,  
            'em_amp': 0,
            'em_phase': 0,         
            'em_value': 0,
            'phi': PHI,    
            'pi': PI,
            'e': e, 
            'mood': 'neutral'      
        }) 

    for freq in frequencies:
        forecast.append(freq)  
                
    for freq in frequencies:

        # Get PHI raised to the frequency number         
        phi_power = PHI ** freq['number'] 

        if phi_power < 1.05:
            freq['mood'] = 'extremely positive'
        elif phi_power < 1.2:       
            freq['mood'] = 'strongly positive'
        elif phi_power < 1.35:       
            freq['mood'] = 'positive'    
        elif phi_power < 1.5:       
            freq['mood'] = 'slightly positive'
        elif phi_power < 2:          
            freq['mood'] = 'neutral'              
        elif phi_power < 2.5:     
            freq['mood'] = 'slightly negative'      
        elif phi_power < 3.5:     
            freq['mood'] = 'negative'
        elif phi_power < 4.5:     
            freq['mood'] = 'strongly negative'   
        else:                     
            freq['mood'] = 'extremely negative'

        forecast.append(freq)    # Define current_quadrant variable

    #Calculate midpoints of each quadrant
    mid_q1 = sine_wave_min + em_amp_q1 / 2
    mid_q2 = mid_q1 + em_amp_q1
    mid_q3 = mid_q2 + em_amp_q2
    mid_q4 = float(sine_wave['max'])

    #Compare current sine wave value to determine quadrant
    if current_value <= mid_q1:
        current_quadrant = 1
        current_em_amp = em_amp_q1
        current_em_phase = em_phase_q1

    elif current_value <= mid_q2:
        current_quadrant = 2
        current_em_amp = em_amp_q2
        current_em_phase = em_phase_q2

    elif current_value <= mid_q3:
        current_quadrant = 3
        current_em_amp = em_amp_q3
        current_em_phase = em_phase_q3

    elif current_value <= mid_q4:
        current_quadrant = 4
        current_em_amp = em_amp_q4
        current_em_phase = em_phase_q4

    else:
        # Assign a default value
        current_em_amp = 0 
        current_em_phase = 0

    #Assign current EM amplitude and phase
    em_amp = current_em_amp
    em_phase = current_em_phase

    if percent_to_min_val < 20:
        print("Bullish momentum in trend")

        if current_quadrant == 1:
            # In quadrant 1, distance from min to 25% of range
            print("Bullish momentum in Q1")
        elif current_quadrant == 2:
            # In quadrant 2, distance from 25% to 50% of range
            print("Bullish momentum in Q2")
        elif current_quadrant == 3:
            # In quadrant 3, distance from 50% to 75% of range
            print("Bullish momentum in Q3")
        elif current_quadrant == 4:
            # In quadrant 4, distance from 75% to max of range
            print("Bullish momentum in Q4")

    elif percent_to_max_val < 20:
        print("Bearish momentum in trend")

        if current_quadrant == 1:
            # In quadrant 1, distance from min to 25% of range
            print("Bearish momentum in Q1")
        elif current_quadrant == 2:
            # In quadrant 2, distance from 25% to 50% of range
            print("Bearish momentum in Q2")
        elif current_quadrant == 3:
            # In quadrant 3, distance from 50% to 75% of range
            print("Bearish momentum in Q3")
        elif current_quadrant == 4:
            # In quadrant 4, distance from 75% to max of range
            print("Bearish momentum in Q4")

    # Calculate quadrature phase shift based on current quadrant  
    if current_quadrant == 1:  
        # Up cycle from Q1 to Q4   
        quadrature_phase = em_phase_q1
        em_phase = alpha_spiral
    elif current_quadrant == 2:
        quadrature_phase = em_phase_q2
        em_phase = omega_spiral          
    elif current_quadrant  == 3:      
        quadrature_phase = em_phase_q3
        em_phase = omega_spiral      
    else:          
        quadrature_phase = em_phase_q4
        em_phase = alpha_spiral 

    cycle_direction = "UP"
    next_quadrant = 1

    if current_quadrant == 1:
        next_quadrant = 2
        cycle_direction = "UP"

    elif current_quadrant == 2:
        if cycle_direction == "UP":
            next_quadrant = 3
        elif cycle_direction == "DOWN":
            next_quadrant = 1

    elif current_quadrant == 3:
        if cycle_direction == "UP":
            next_quadrant = 4
        elif cycle_direction == "DOWN":
            next_quadrant = 2

    elif current_quadrant == 4:
        if cycle_direction == "UP":
            next_quadrant = 3
            cycle_direction = "DOWN"

    # Calculate quadrature phase                       
    if next_quadrant == 1:     
        next_quadrature_phase = em_phase_q1            
    elif next_quadrant == 2:        
        next_quadrature_phase = em_phase_q2          
    elif next_quadrant == 3:                 
        next_quadrature_phase = em_phase_q3             
    else:              
        next_quadrature_phase = em_phase_q4

    # Calculate EM value
    em_value = em_amp * math.sin(em_phase)  

    # Calculate quadrature phase shift from current to next quadrant      
    quadrature = next_quadrature_phase - quadrature_phase

    if quadrature > 0:
        # Up cycle from Q1 to Q4  
        print("Up cycle now")  
    else:  
        # Down cycle from Q4 to Q1 
        print("Down cycle now")

    if current_quadrant == 1:
        # Quadrant 1
                
        if freq['number'] <= 10:
            # Most negative frequencies
            freq['em_amp'] = em_amp_q1
            freq['em_phase'] = em_phase_q1                 
            freq['mood'] = 'extremely negative'  

        elif freq['number'] >= 20:              
            freq['em_amp'] = em_amp_q1
            freq['em_phase'] = em_phase_q1  
            freq['mood'] = 'extremely positive'

    elif current_quadrant == 2:
        # Quadrant 2
                
        if freq['number'] > 10 and freq['number'] <= 15:                 
            freq['em_amp'] = em_amp_q2
            freq['em_phase'] = em_phase_q2
            freq['mood'] = 'strongly negative'
                        
        elif freq['number'] > 15 and freq['number'] <= 20:                 
            freq['em_amp'] = em_amp_q2
            freq['em_phase'] = em_phase_q2
            freq['mood'] = 'strongly positive'

    elif current_quadrant == 3: 
        # Quadrant 3
            
        if freq['number'] > 15 and freq['number'] < 20:            
            freq['em_amp'] = em_amp_q3                  
            freq['em_phase'] = em_phase_q3
            freq['mood'] = 'negative'              
           

        elif freq['number'] > 10 and freq['number'] < 15:            
            freq['em_amp'] = em_amp_q3                  
            freq['em_phase'] = em_phase_q3
            freq['mood'] = 'positive'
 
    else:      
        # Quadrant 4 
            
        if freq['number'] >= 20:                    
            freq['em_amp'] = em_amp_q4
            freq['em_phase'] = em_phase_q4  
            freq['mood'] = 'partial negative'       

        elif freq['number'] <= 10:                    
            freq['em_amp'] = em_amp_q4
            freq['em_phase'] = em_phase_q4  
            freq['mood'] = 'partial positive'

    freq['em_value'] = freq['em_amp'] * math.sin(freq['em_phase'])
        
    # Sort frequencies from most negative to most positive       
    frequencies.sort(key=lambda x: x['em_value'])   
        
    print("Quadrant is in: " + cycle_direction + " cycle")  
 
        
    for freq in frequencies:               
        print(freq['number'], freq['em_value'], freq['mood'])    
        
    # Calculate frequency spectrum index range based on most negative and positive frequencies
    mood_map = {
        'extremely negative': -4,  
        'strongly negative': -3,  
        'negative': -2,        
        'partial negative': -1,           
        'neutral': 0,
        'partial positive': 1, 
        'positive': 2,       
        'strongly positive': 3,    
        'extremely positive': 4   
        }

    if frequencies[0]['mood'] != 'neutral' and frequencies[-1]['mood'] != 'neutral':   
        total_mood = frequencies[0]['mood'] + " and " +  frequencies[-1]['mood']
    else:
        total_mood = 'neutral'

    print()

    # Update the frequencies for the next quadrant     
    if next_quadrant == 1:       
        # Update frequencies for next quadrant (Q1)               
        for freq in frequencies_next:       
            freq['em_amp'] = em_amp_q1       
            freq['em_phase'] = em_phase_q1
             
    elif next_quadrant == 2:
        # Update frequencies for Q2        
        for freq in frequencies_next:                 
            freq['em_amp'] = em_amp_q2   
            freq['em_phase'] = em_phase_q2 

    elif next_quadrant == 3:
        # Update frequencies for Q3        
        for freq in frequencies_next:                 
            freq['em_amp'] = em_amp_q3   
            freq['em_phase'] = em_phase_q3

    elif next_quadrant == 4:
        # Update frequencies for Q4        
        for freq in frequencies_next:                 
            freq['em_amp'] = em_amp_q4   
            freq['em_phase'] = em_phase_q4

    quadrant_1_amplitude = random.uniform(0.4, 0.6)  
    quadrant_1_phase = random.uniform(0.1, 0.3)

    quadrant_2_amplitude = random.uniform(0.8, 1.2)   
    quadrant_2_phase = random.uniform(0.4, 0.6)

    quadrant_3_amplitude = random.uniform(0.6, 1.0)        
    quadrant_3_phase = random.uniform(0.6, 0.8)

    quadrant_4_amplitude = random.uniform(1.0, 1.4)       
    quadrant_4_phase = random.uniform(0.8, 1.0)

    lowest_frequency = float('inf')
    highest_frequency = 0

    min_quadrant = None  
    max_quadrant = None

    if current_quadrant == 1:
        frequency_amplitude = quadrant_1_amplitude  
        frequency_phase = quadrant_1_phase  
        current_frequency = frequency_amplitude * frequency_phase

    elif current_quadrant == 2:        
        frequency_amplitude = quadrant_2_amplitude   
        frequency_phase = quadrant_2_phase 
        current_frequency = frequency_amplitude * frequency_phase

    elif current_quadrant == 3:        
        frequency_amplitude = quadrant_3_amplitude
        frequency_phase = quadrant_3_phase
        current_frequency = frequency_amplitude * frequency_phase
  
    elif current_quadrant == 4:        
        frequency_amplitude = quadrant_4_amplitude       
        frequency_phase = quadrant_4_phase
        current_frequency = frequency_amplitude * frequency_phase

    if current_frequency == lowest_frequency:
        min_node = {'frequency': current_frequency, 'quadrant': current_quadrant}

    if current_frequency == highest_frequency:      
        max_node = {'frequency': current_frequency, 'quadrant': current_quadrant}

    # Get next quadrant phi 
    next_phi = PHI ** freq['number'] 

    # Map moods based on inverse phi power         
    if next_phi < 1.2:
        freq['mood'] = 'extremely positive' 
    elif next_phi < 1.4:
        freq['mood'] = 'positive'

    highest_3 = frequencies[:3]
    lowest_3 = frequencies[-3:]

    mood_map = {
        'extremely negative': -4,  
        'strongly negative': -3,  
        'negative': -2,        
        'partial negative': -1,           
        'neutral': 0,
        'partial positive': 1, 
        'positive': 2,       
        'strongly positive': 3,    
        'extremely positive': 4   
         }

    highest_3_mood_values = []
    for freq in highest_3:   
        if freq['mood'] == 'neutral':
            highest_3_mood_values.append(0)   
        else:
            highest_3_mood_values.append(mood_map[freq['mood']])

    lowest_3_mood_values = []        
    for freq in lowest_3:   
        if freq['mood'] == 'neutral':
            lowest_3_mood_values.append(0)        
        else:
            lowest_3_mood_values.append(mood_map[freq['mood']])      

    highest_3_mood_values = [mood_map[freq['mood']] for freq in highest_3]
    highest_3_mood = statistics.mean(highest_3_mood_values)

    lowest_3_mood_values = [mood_map[freq['mood']] for freq in lowest_3]
    lowest_3_mood = statistics.mean(lowest_3_mood_values)

    print(f"Current quadrant: {current_quadrant}")
    print(f"Next quadrant: {next_quadrant}")
    print(f"Highest 3 frequencies: {highest_3_mood}")        
    print(f"Lowest 3 frequencies: {lowest_3_mood}")

    if highest_3_mood > 0:
        print(f"Cycle mood is negative")
    elif highest_3_mood < 0:      
        print(f"Cycle mood is positive") 
    else:
        print("Cycle mood is neutral")

    if frequencies[0]['mood'] != 'neutral' and frequencies[-1]['mood'] != 'neutral':        
        if mood_map[frequencies[0]['mood']] < 0:
            total_mood = f"{frequencies[0]['mood']} and  {frequencies[-1]['mood']}"
            print(f"Frequency spectrum index range: {total_mood} ")
            print(f"Freq. range is negative")
        else:    
            total_mood = f"{frequencies[0]['mood']} and {frequencies[-1]['mood']}"
            print(f"Frequency spectrum index range: {total_mood}") 
            print(f"Freq. range is positive")   
    else:
        print(f"Frequency spectrum index range: neutral")
        print(f"Freq. range is neutral") 

    print()

    # Sort forecast from most negative to most positive       
    forecast.sort(key=lambda f: mood_map[f['mood']])  
    
    # Get average mood of highest/lowest 3 frequencies 
    highest_3 = forecast[:3]
    highest_3_mood = 0
   
    for freq in highest_3:
        mood_val  = mood_map[freq['mood']] 
        highest_3_mood += mood_val
        highest_3_mood = highest_3_mood / len(highest_3)

    lowest_3 = forecast[-3:]
    lowest_3_mood = 0

    for freq in lowest_3:
        mood_val  = mood_map[freq['mood']]
        lowest_3_mood += mood_val  
        lowest_3_mood = lowest_3_mood / len(lowest_3)

    for freq in forecast:
        freq['magnitude'] = np.abs(freq['em_value'])

    n = 10

    def calculate_weighted_avg(weights, values):
        total = 0 
        total_weight = 0
    
        for w, v in zip(weights, values):
            magnitude = w 
            mood = v
            total += magnitude * mood
            total_weight += magnitude
        
            if total_weight == 0:
                return 0

            return total / total_weight

    # Calculate weighted averages  
    top_n_weights = [freq['magnitude'] for freq in forecast[:n]]
    top_n_moods = [mood_map[freq['mood']] for freq in forecast[:n]]
    top_n_weighted_avg = calculate_weighted_avg(top_n_weights, top_n_moods)

    bottom_n_weights = [freq['magnitude'] for freq in forecast[-n:]]  
    bottom_n_moods = [mood_map[freq['mood']] for freq in forecast[-n:]]
    bottom_n_weighted_avg = calculate_weighted_avg(bottom_n_weights, bottom_n_moods)

    overall_mood = top_n_weighted_avg - bottom_n_weighted_avg

    if overall_mood > 2:
        print("Strongly bullish mood")
    elif overall_mood > 1:       
        print("Bullish mood")
    elif overall_mood > 0:
        print("Mildly bullish mood")     
    elif overall_mood == 0:
        print("Neutral mood")          
    elif overall_mood > -1:        
        print("Mildly Bearish mood")      
    elif overall_mood > -2:       
        print("Bearish mood")  
    else:
        print("Strongly bearish mood")

    if overall_mood < -3:
        print("Extremely bearish")  
    elif overall_mood < -2:           
        print("Strongly bearish")
    elif overall_mood < -1:       
        print("Bearish")           
    elif overall_mood == 0:
        print("Neutral")
    elif overall_mood > 1:        
        print("Bullish")      
    elif overall_mood > 2:        
        print("Strongly Bullish")            
    else:
        print("Extremely bullish")

    # Define stationary circuit variables
    stationary_circuit = []

    # Map quadrants to triangle points of Metatron's Cube
    quadrant_map = {
        1: 'Apex',
        2: 'Left',  
        3: 'Base',
        4: 'Right' 
        }

    # Add triangle points to circuit
    stationary_circuit.append('Apex')
    stationary_circuit.append('Left')     
    stationary_circuit.append('Base')
    stationary_circuit.append('Right')

    # Loop through each quadrant cycle        
    for quadrant in [1,2,3,4]:
    
        #print(f"Quadrant {quadrant}")
            
        # Get triangle point from quadrant map               
        point = quadrant_map[quadrant]    
    
        #print(f"Current point: {point}")
    
        # Get next point based on circuit        
        if point == 'Apex':
            next_point = 'Left'
        elif point == 'Left':
            next_point = 'Base'         
        elif point == 'Base':
            next_point = 'Right'
        elif point == 'Right':
            next_point = 'Apex'
        
        #print(f"Next point: {next_point}")       
    
        # Get frequency and mood forecast
        frequency = frequencies[quadrant]['frequency']
        mood = frequencies[quadrant]['mood']
    
        #print(f"Frequency: {frequency} Hz - Mood: {mood}")
    
    if current_frequency > lowest_frequency:
        lowest_frequency = current_frequency    
    if current_frequency < highest_frequency:     
        highest_frequency = current_frequency

    # Define min and max nodal points based on frequencies
    min_node = { 
        'frequency': lowest_frequency,  
        'quadrant': min_quadrant   
        }

    max_node = {
        'frequency': highest_frequency,
        'quadrant': max_quadrant         
        }

    if current_frequency < lowest_frequency:
        lowest_frequency = current_frequency    
        quadrant = current_quadrant   

    min_node = {'frequency': lowest_frequency, 'quadrant': quadrant}  

    if current_frequency > highest_frequency:      
        highest_frequency = current_frequency   
        max_quadrant = current_quadrant   

    max_node = {'frequency': highest_frequency, 'quadrant': max_quadrant}

    # Loop through each quadrant cycle        
    for quadrant in [1,2,3,4]:
    
        # Check if current quadrant is a min or max node
        if quadrant == min_node['quadrant']:
            print(f"Node reached at frequency {min_node['frequency']} Hz")
        
        elif quadrant == max_node['quadrant']:
            print(f"Node reached at frequency {max_node['frequency']} Hz")

    print()

    # Calculate forecast mood based on frequencies and nodal points
    forecast = {
        'mood' : None,
        'min_reversal' : {
            'time': None,
            'quadrant': None
            },  
        'max_reversal' : {
            'time': None,     
            'quadrant': None      
            }       
        }

    if lowest_3_mood < 0:
        forecast['mood'] = 'positive'
    elif highest_3_mood > 0:      
        forecast['mood'] = 'negative'
    else:
        forecast['mood'] = 'neutral'

    # Calculate time to min nodal point reversal
    freq = min_node['frequency']  
    period = 1/freq   
    min_time = period/4    
    forecast['min_reversal']['time'] = min_time
    forecast['min_reversal']['quadrant'] = min_node['quadrant']

    # Calculate time to max nodal point reversal    
    freq = max_node['frequency']         
    period = 1/freq   
    max_time = period/4  
                        
    forecast['max_reversal']['time'] = max_time
    forecast['max_reversal']['quadrant'] = max_node['quadrant']

    # Print forecast   
    print(forecast)

    # Print overall mood  
    print(f"Overall market mood: {forecast['mood']}")

    print()

    # Define octahedron points mapped to Metatron's Cube
    octahedron = [
        {'point': 'Apex',       'frequency': None, 'mood': None},
        {'point': 'Left',       'frequency': None, 'mood': None},
        {'point': 'Base',       'frequency': None, 'mood': None},
        {'point': 'Right',      'frequency': None, 'mood': None},
        {'point': 'Phi',        'frequency': None, 'mood': None},  
        {'point': 'Pi',         'frequency': None, 'mood': None},
        {'point': 'e',          'frequency': None, 'mood': None},
        {'point': 'Origin',     'frequency': None, 'mood': None}    
        ]
 

    # Update octahedron points with frequencies and moods
    for point in octahedron:
        if point['point'] == quadrant_map[current_quadrant]:
            point['frequency'] = current_frequency
            point['mood'] = frequencies[current_quadrant]['mood']

    valid_points = [p for p in octahedron if p['frequency'] is not None] 

    # Find minimum and maximum frequency points   
    min_point = min(valid_points, key=lambda p: p['frequency'])   
    max_point = max(valid_points, key=lambda p: p['frequency'])

    # Calculate reversal predictions from min and max points
    forecast = {
        'min_reversal': {
            'time': None,
            'point': None
            },
        'max_reversal': {
            'time': None,
            'point': None   
            }
        }

    freq = min_point['frequency']
    period = 1/freq
    forecast['min_reversal']['time'] = period/4               
    forecast['min_reversal']['point'] = min_point['point']

    freq = max_point['frequency']
    period = 1/freq                             
    forecast['max_reversal']['time'] = period/4               
    forecast['max_reversal']['point'] = max_point['point']

    # Prints                    
    print(f"Apex: {octahedron[0]['frequency']}")    
    print(f"Left: {octahedron[1]['frequency']}")      
    print(f"Base: {octahedron[2]['frequency']}")
    print(f"Right: {octahedron[3]['frequency']}")
    print(f"Phi: {octahedron[4]['frequency']}")
    print(f"Pi: {octahedron[5]['frequency']}")
    print(f"e: {octahedron[6]['frequency']}")
    print(f"Origin: {octahedron[7]['frequency']}")

    print("Current point is at: ", forecast[f'min_reversal']['point'] if forecast[f'min_reversal']['point'] == point else forecast[f'max_reversal']['point']) 

    # Extract current quadrant and EM phase
    current_quadrant = sine_wave["current_quadrant"]   
    em_phase = sine_wave["em_phase"]

    mood_to_corr = {
        'extremely negative': -0.9,   
        'strongly negative': -0.7,
        'negative': -0.5,       
        'partial negative': -0.3,           
        'neutral': 0,
        'partial positive': 0.3,  
        'positive': 0.5,       
        'strongly positive': 0.7,    
        'extremely positive': 0.9   
        }

    momentum_map = {      
        'extremely negative': -5,   
        'strongly negative': -4,     
        'negative': -3,       
        'partial negative': -2,           
        'neutral': 0,
        'partial positive': 2,   
        'positive': 3,       
        'strongly positive': 4,    
        'extremely positive': 5  
        }

    current_momentum = 0

    for freq in frequencies:

        corr = mood_to_corr[freq['mood']]
        z = np.arctanh(corr) 
        freq['z'] = z
        freq['pos'] = 0 if corr == 0 else z

        # Get momentum score       
        momentum = momentum_map[freq['mood']]
      
        # Calculate weighted momentum score      
        weighted_momentum = momentum * freq['magnitude']         
  
        current_momentum += weighted_momentum
                        
    frequencies.sort(key=lambda f: f['z'])

    # Calculate average momentum      
    current_momentum /= len(frequencies)

    # Forecast mood 
    if current_momentum < -2:
        forecast_mood = 'bearish'
    elif current_momentum < 0:
        forecast_mood = 'slightly bearish'    
    elif current_momentum == 0:        
        forecast_mood = 'neutral'
    elif current_momentum > 0:        
        forecast_mood = 'slightly bullish'
    elif current_momentum > 2:
        forecast_mood = 'bullish'

    print(f"Current momentum: {current_momentum}")     
    print(f"Trend forecast: {forecast_mood}")    

    print()

    em_amp = []  
    em_phase = []

    em_amp.append(current_em_amp)
    em_phase.append(current_em_phase)



    return em_amp, em_phase, current_momentum, forecast_mood, point, next_point

sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                                           percent_to_max_val=5, 
                                           percent_to_min_val=5)      

sine_wave_max = sine_wave["max"]   
sine_wave_min = sine_wave["min"]

octa_metatron_cube(close_prices, candles)  
print(octa_metatron_cube(close_prices, candles))

print()

##################################################
##################################################

def get_octant_coordinates(emotional_values):
    if isinstance(emotional_values, dict):
        emotional_values = [emotional_values.get(key, {}).get('phase', 0) for key in ['Apex', 'Left', 'Base', 'Right']]
    octant_coordinates = []
    for emotional_value in emotional_values:
        if isinstance(emotional_value, (int, float)):
            x = math.cos(emotional_value * math.pi / 180)
            y = math.sin(emotional_value * math.pi / 180)
            octant_coordinates.append((x, y))
    return octant_coordinates

def metatron_reversals_unit_circle(quadrant_emotional_values, close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    close_prices = np.array(close_prices) # Convert close_prices to a numpy.ndarray
    results = reversals_unit_circle(close_prices, candles, percent_to_max_val, percent_to_min_val)
    forecast_moods = results['forecast_moods']
    min_node = results['min_node']
    max_node = results['max_node']
    avg_moods = results['avg_moods']
    weighted_moods = results['weighted_moods']
    forecast_direction = results['forecast_direction']
    mood_reversal_forecast = results['mood_reversal_forecast']
    market_mood = results['market_mood']

    octant_coordinates = get_octant_coordinates(quadrant_emotional_values)
    output = {
        "octant_coordinates": octant_coordinates,
        "forecast_moods": forecast_moods,
        "min_node": min_node,
        "max_node": max_node,
        "avg_moods": avg_moods,
        "weighted_moods": weighted_moods,
        "forecast_direction": forecast_direction,
        "mood_reversal_forecast": mood_reversal_forecast,
        "market_mood": market_mood
    }
    return output

print()

quadrant_emotional_values = {
    'Apex': {'amplitude': 0.75, 'phase': 0},
    'Left': {'amplitude': 0.5, 'phase': 1.5707963267948966},
    'Base': {'amplitude': 0.25, 'phase': 3.141592653589793},
    'Right': {'amplitude': 0.49999999999999994, 'phase': 4.71238898038469}
}

percent_to_max_val = 5
percent_to_min_val = 5

print("quadrant_emotional_values: ", quadrant_emotional_values)
  
print()

##################################################
##################################################

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate thresholds and averages based on min and max percentages. 
    """
  
    # Get min/max close    
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    # Convert close_prices to numpy array
    close_prices = np.array(close_prices)
    
    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=period)
    
    # Get min/max momentum    
    min_momentum = np.nanmin(momentum)   
    max_momentum = np.nanmax(momentum)
    
    # Calculate custom percentages 
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    # Calculate thresholds       
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    # Calculate range of prices within a certain distance from the current close price
    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)

    # Filter close prices
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
        
    # Calculate avg    
    avg_mtf = np.nanmean(filtered_close)

    # Get current momentum       
    current_momentum = momentum[-1]

    # Calculate % to min/max momentum    
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) /   
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               

        percent_to_max_momentum = ((current_momentum - min_momentum) / 
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
 
    # Calculate combined percentages              
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    # Combined momentum signal     
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price


# Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

print("Momentum signal:", momentum_signal)
print()

print("Minimum threshold:", min_threshold)
print("Maximum threshold:", max_threshold)
print("Average MTF:", avg_mtf)

#print("Range of prices within distance from current close price:")
#print(range_price[-1])

print()

##################################################
##################################################

# Define the current time and close price
current_time = datetime.datetime.now()
current_close = closes[-1]

print("Current local Time is now at: ", current_time)
print("Current close price is at : ", current_close)

print()

##################################################
##################################################

def get_next_minute_target(closes, n_components):
    # Calculate FFT of closing prices
    fft = fftpack.fft(closes)
    frequencies = fftpack.fftfreq(len(closes))

    # Sort frequencies by magnitude and keep only the top n_components
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]

    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = np.real(fftpack.ifft(filtered_fft))

    # Calculate the target price as the next value after the last closing price
    target_price = filtered_signal[-1]

    return target_price

# Example usage
closes = get_closes("1m")
n_components = 5
targets = []

for i in range(len(closes) - 1):
    # Decompose the signal up to the current minute and predict the target for the next minute
    target = get_next_minute_target(closes[:i+1], n_components)
    targets.append(target)

# Print the predicted targets for the next minute
print("Target for 1min tf:", targets[-1])

##################################################
##################################################

# Example usage
closes = get_closes("3m")
n_components = 5
targets = []

for i in range(len(closes) - 1):
    # Decompose the signal up to the current minute and predict the target for the next minute
    target = get_next_minute_target(closes[:i+1], n_components)
    targets.append(target)

##################################################
##################################################

# Print the predicted targets for the next minute
print("Target for 3min tf:", targets[-1])

# Example usage
closes = get_closes("5m")
n_components = 5
targets = []

for i in range(len(closes) - 1):
    # Decompose the signal up to the current minute and predict the target for the next minute
    target = get_next_minute_target(closes[:i+1], n_components)
    targets.append(target)

# Print the predicted targets for the next minute
print("Target for 5min tf:", targets[-1])

print()

##################################################
##################################################

def get_next_minute_targets(closes, n_components):
    # Calculate FFT of closing prices
    fft = fftpack.fft(closes)
    frequencies = fftpack.fftfreq(len(closes))

    # Sort frequencies by magnitude and keep only the top n_components
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]

    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = np.real(fftpack.ifft(filtered_fft))

    # Calculate the target price as the next value after the last closing price
    target_price = filtered_signal[-1]
    
    # Calculate the stop loss and target levels
    entry_price = closes[-1]
    stop_loss = entry_price - np.std(closes)
    target1 = target_price + np.std(closes)
    target2 = target_price + 2*np.std(closes)
    target3 = target_price + 3*np.std(closes)

    return entry_price, stop_loss, target1, target2, target3

# Example usage
closes = get_closes("1m")
n_components = 5
targets = []

for i in range(len(closes) - 1):
    # Decompose the signal up to the current minute and predict the target for the next minute
    entry_price, stop_loss, target1, target2, target3 = get_next_minute_targets(closes[:i+1], n_components)
    targets.append((entry_price, stop_loss, target1, target2, target3))

# Print the predicted levels for the next minute
print("Entry price:", targets[-1][0])
print("Stop loss:", targets[-1][1])
print("Target 1:", targets[-1][2])
print("Target 2:", targets[-1][3])
print("Target 3:", targets[-1][4])

print()

##################################################
##################################################

def analyze_reversals(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Call reversals_unit_circle to get the forecast results
    results = reversals_unit_circle(close_prices, candles, percent_to_max_val=percent_to_max_val, percent_to_min_val=percent_to_min_val)

    # Get the quadrant emotional values, forecast moods, and average moods
    quadrant_emotional_values = results["quadrant_emotional_values"]
    forecast_moods = results["forecast_moods"]
    avg_moods = results["avg_moods"]

    # Print the emotional values and forecast moods for each quadrant
    for quadrant in quadrant_emotional_values:
        print("\n{} Quadrant".format(quadrant))
        print("Emotional Amplitude: {}".format(quadrant_emotional_values[quadrant]["amplitude"]))
        print("Emotional Phase: {}".format(quadrant_emotional_values[quadrant]["phase"]))
        for band in forecast_moods[quadrant]:
            print("{}: {}".format(band, forecast_moods[quadrant][band]))

    # Print the minimum and maximum frequency nodes and their corresponding moods
    print("\nMinimum frequency node: {}, Mood: {}".format(results["min_node"], forecast_moods[results["min_node"].split()[0]][results["min_node"].split()[-1]]))
    print("Maximum frequency node: {}, Mood: {}".format(results["max_node"], forecast_moods[results["max_node"].split()[0]][results["max_node"].split()[-1]]))

    # Print the average moods for each quadrant
    for quadrant in avg_moods:
        print("\n{} Quadrant".format(quadrant))
        print("Average High Mood: {}".format(avg_moods[quadrant]["avg_high_mood"]))
        print("Average Low Mood: {}".format(avg_moods[quadrant]["avg_low_mood"]))

    # Determine the forecast direction and print it
    forecast_direction = results["forecast_direction"]
    print("\nForecast Direction: {}".format(forecast_direction))

    # Determine the mood reversal forecast and print it
    mood_reversal_forecast = results["mood_reversal_forecast"]
    if mood_reversal_forecast:
        print("Mood Reversal Forecast: {}".format(mood_reversal_forecast))

    # Determine the market mood and print it
    market_mood = results["market_mood"]
    print("Market Mood: {}".format(market_mood))

    # Perform further calculations based on the forecast results
    if forecast_direction == "Apex":
        print("\nMarket is in a consolidation phase")
    elif forecast_direction == "Base":
        print("\nMarket is in a downtrend")
    else:
        print("\nMarket is in an uptrend")

    if mood_reversal_forecast:
        if forecast_direction == "Apex":
            print("Expect sideways movement in the near term")
        elif forecast_direction == "Base":
            print("Expect a reversal to an uptrend in the near term")
        else:
            print("Expect a reversal to a downtrend in the near term")

    if market_mood == "Positive":
        print("Market sentiment is positive")
    elif market_mood == "Negative":
        print("Market sentiment is negative")
    else:
        print("Market sentiment is neutral")

    print()

    # Return the forecast direction, mood reversal forecast, and market mood
    return forecast_direction, mood_reversal_forecast, market_mood

print()

##################################################
##################################################

def get_rf_band(frequency):
    if frequency >= 300:
        return "THz or Tremendously high frequency"      
    elif frequency >= 100 and frequency < 300:
        return "W band (75 - 110 GHz)"           
    elif frequency >= 60 and frequency < 100:
        return "E band (60 - 90 GHz)"     
    elif frequency >= 40 and frequency < 60:     
        return "V band (40 - 75 GHz)"    
    elif frequency >= 30 and frequency <40:    
        return "Ka band (27 - 40 GHz)"       
    elif frequency >= 18 and frequency < 26.5:      
        return "Ku band (12.4 - 18 GHz)"           
    elif frequency >= 12.4 and frequency < 18:   
        return "K band (18 - 26.5 GHz)"      
    elif frequency >= 8 and frequency < 12.4:     
        return "X band (8.2 - 12.4 GHz)"    
    elif frequency >= 4 and frequency < 8:     
        return "C band (5.85 - 8.2 GHz)"    
    elif frequency >= 2 and frequency < 4:   
        return "S band (2.6 - 3.95 GHz)"      
    elif frequency >= 0.3 and frequency < 2:       
        return "L band (1 - 2 GHz)"    
    elif frequency >= 0.03 and frequency < 0.3:    
        return "VHF band (30MHz - 300MHz)"       
    elif frequency >= 0.003 and frequency < 0.03:  
        return "HF band (3MHz - 30MHz)"        
    else:
        return "Frequency out of range"


def subatomic_map_function(close_prices, candles, particle_freqs):
    # Check input data format
    for freq in particle_freqs.values():
        if not isinstance(freq, float):
            raise TypeError("All frequency values must be of type 'float'.")

    # Call the original functions 
    results1 = reversals_unit_circle(close_prices, candles, percent_to_max_val=5.0, percent_to_min_val=5.0)  
    results2 = metatron_reversals_unit_circle(results1["quadrant_emotional_values"], close_prices, candles, percent_to_max_val=5.0, percent_to_min_val=5.0)
  
    # Build subatomic particle map
    particles = {"Electron": {}, 
                 "Proton":{},
                 "Neutron":{},
                 "Photon": {},
                 "Quark": {},
                 "Muon": {},
                 "Tau": {},
                 "Neutrino": {},
                 "Antineutrino": {},
                 "W Boson": {},
                 "Z Boson": {},
                 "Gluon": {},
                 "Higgs Boson": {}}
                   
    # Map momentum for each particle               
    particles["Electron"]["Momentum"] = results1["min_node"]  
    particles["Proton"]["Momentum"]  = results1["max_node"]  
  
    # Map other relevant properties               
    particles["Electron"]["Properties"] = "Negatively charged, low mass, spin 1/2"
    particles["Proton"]["Properties"] = "Positively charged, similar mass to neutron, spin 1/2"
    particles["Neutron"]["Properties"] = "No net charge, similar mass to proton, spin 1/2"
    particles["Photon"]["Properties"] = "Zero mass, spin 1"
    particles["Quark"]["Properties"] = "Fractional charges and spins, confined in hadrons"
    particles["Muon"]["Properties"] = "Negatively charged, heavier than electron, spin 1/2"
    particles["Tau"]["Properties"] = "Negatively charged, heavier than muon, spin 1/2"
    particles["Neutrino"]["Properties"] = "Very low mass, spin 1/2, weakly interacting"
    particles["Antineutrino"]["Properties"] = "Same properties as neutrino, opposite charge"
    particles["W Boson"]["Properties"] = "Massive, charged, mediates weak force"
    particles["Z Boson"]["Properties"] = "Massive, neutral, mediates weak force"
    particles["Gluon"]["Properties"] = "Massless, spin 1, mediates strong force"
    particles["Higgs Boson"]["Properties"] = "Scalar particle, mediates Higgs field"

    # Map transitions and frequencies for each particle
    particles["Electron"]["Transitions"] = "Electron transitions between energy levels"
    particles["Proton"]["Transitions"]  = "Proton/neutron spin flip transitions"
    particles["Neutron"]["Transitions"] = "Neutron spin flip transitions and electron capture"
    particles["Photon"]["Transitions"] = "Emitted from electron, proton and neutron transitions"
    particles["Quark"]["Transitions"] = "Quark interactions within hadrons"
    particles["Muon"]["Transitions"] = "Muon decay"
    particles["Tau"]["Transitions"] = "Tau decay"
    particles["Neutrino"]["Transitions"] = "Weak interactions"
    particles["Antineutrino"]["Transitions"] = "Weak interactions"
    particles["W Boson"]["Transitions"] = "Mediates weak force interactions"
    particles["Z Boson"]["Transitions"] = "Mediates weak force interactions"
    particles["Gluon"]["Transitions"] = "Quark-antiquark interactions"
    particles["Higgs Boson"]["Transitions"] = "Higgs field interactions"
  
    particles["Electron"]["Frequencies"] = "Correspond to energy differences between levels"
    particles["Proton"]["Frequencies"] = "Radiofrequency range"
    particles["Neutron"]["Frequencies"] = "Radiofrequencies and gamma rays"
    particles["Muon"]["Frequencies"] = "Radiofrequencies"
    particles["Tau"]["Frequencies"] = "Radiofrequencies"
    particles["Neutrino"]["Frequencies"] = "Very high frequencies"
    particles["Antineutrino"]["Frequencies"] = "Very high frequencies"
    particles["W Boson"]["Frequencies"] = "Very high frequencies"
    particles["Z Boson"]["Frequencies"] = "Very high frequencies"
    particles["Gluon"]["Frequencies"] = "RF frequencies"
    particles["Higgs Boson"]["Frequencies"] = "RF frequencies"
        
    # Map RF frequencies for each particle (only for electrons, protons, and neutrons)
    particles["Electron"]["RF band"] = get_rf_band(particle_freqs["Electron"])  
    particles["Proton"]["RF band"] = get_rf_band(particle_freqs["Proton"])  
    particles["Neutron"]["RF band"] = get_rf_band(particle_freqs["Neutron"])
    
    return particles

# Example usage
particle_freqs = {"Electron": 100.0, 
                  "Proton": 5.0, 
                  "Neutron": 20.0, 
                  "Photon": 0.0, 
                  "Quark": 0.0, 
                  "Muon": 0.0, 
                  "Tau": 0.0, 
                  "Neutrino": 0.0, 
                  "Antineutrino": 0.0, 
                  "W Boson": 0.0, 
                  "Z Boson": 0.0, 
                  "Gluon": 0.0, 
                  "Higgs Boson": 0.0}

particles = subatomic_map_function(close_prices, candles, particle_freqs)

print()

print(particles)         

print()

# Access particle details   
electron_momentum = particles["Electron"]["Momentum"]
print(electron_momentum)
# "min_node: Delta Quadrant Electron"

proton_properties = particles["Proton"]["Properties"]       
print(proton_properties)                         
# "Positively charged, similar mass to neutron, spin 1/2"

neutron_transitions = particles["Neutron"]["Transitions"]                 
print(neutron_transitions)
# "Neutron spin flip transitions and electron capture" 

print()

##################################################
##################################################

print()

import math

def subatomic_function(emotional_amplitudes, emotional_phases, frequency_values):
    # Define variables for the 147 258 369 pattern algorithm
    triangle_length = 1 / math.sqrt(5)
    circle_radius = triangle_length / math.sqrt(3)
    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]

    # Create a dictionary of subatomic particles with their associated properties
    particles = {'Electron': {'Momentum': 'Left Gamma', 'Properties': 'Negatively charged, low mass, spin 1/2', 'Transitions': 'Electron transitions between energy levels', 'Frequencies': 'Correspond to energy differences between levels', 'RF band': 'W band (75 - 110 GHz)'}, 'Proton': {'Momentum': 'Apex Delta', 'Properties': 'Positively charged, similar mass to neutron, spin 1/2', 'Transitions': 'Proton/neutron spin flip transitions', 'Frequencies': 'Radiofrequency range', 'RF band': 'C band (5.85 - 8.2 GHz)'}, 'Neutron': {'Properties': 'No net charge, similar mass to proton, spin 1/2', 'Transitions': 'Neutron spin flip transitions and electron capture', 'Frequencies': 'Radiofrequencies and gamma rays', 'RF band': 'Ku band (12.4 - 18 GHz)'}, 'Photon': {'Properties': 'Zero mass, spin 1', 'Transitions': 'Emitted from electron, proton and neutron transitions'} , 'Quark': {'Properties': 'Fractional charges and spins, confined in hadrons', 'Transitions': 'Quark interactions within hadrons'}, 'Muon': {'Properties': 'Negatively charged, heavier than electron, spin 1/2', 'Transitions': 'Muon decay', 'Frequencies': 'Radiofrequencies'}, 'Tau': {'Properties': 'Negatively charged, heavier than muon, spin 1/2', 'Transitions': 'Tau decay', 'Frequencies': 'Radiofrequencies'}, 'Neutrino': {'Properties': 'Very low mass, spin 1/2, weakly interacting', 'Transitions': 'Weak interactions', 'Frequencies': 'Very high frequencies'}, 'Antineutrino': {'Properties': 'Same properties as neutrino, opposite charge', 'Transitions': 'Weak interactions', 'Frequencies': 'Very high frequencies'}, 'W Boson': {'Properties': 'Massive, charged, mediates weak force', 'Transitions': 'Mediates weak force interactions', 'Frequencies': 'Very high frequencies'}, 'Z Boson': {'Properties': 'Massive, neutral, mediates weak force', 'Transitions': 'Mediates weak force interactions', 'Frequencies': 'Very high frequencies'}, 'Gluon': {'Properties': 'Massless, spin 1, mediates strong force', 'Transitions': 'Quark-antiquark interactions', 'Frequencies': 'RF frequencies'}, 'Higgs Boson': {'Properties': 'Scalar particle, mediates Higgs field', 'Transitions': 'Higgs field interactions', 'Frequencies': 'RF frequencies'}}

    # Loop through each particle in the dictionary
    for particle in particles:
        # Calculate the particle's corresponding frequency, momentum energy, intensity, and distance between reversals
        frequency = frequency_values[particle]
        momentum = particles[particle]['Momentum'] if 'Momentum' in particles[particle] else 'N/A'
        energy = emotional_amplitudes[particle] * frequency_values[particle]
        intensity = emotional_amplitudes[particle] * emotional_phases[particle]
        distance = circle_radius * math.sin((angles[emotional_phases[particle] % 3] + angles[(emotional_phases[particle] + 1) % 3]) / 2)

        # Print out all the detailed calculations and outputs for each particle
        print("Particle:", particle)
        print("Momentum:", momentum)
        print("Properties:", particles[particle]['Properties'])
        print("Transitions:", particles[particle]['Transitions'])
        if 'Frequencies' in particles[particle]:
            print("Frequencies:", particles[particle]['Frequencies'])
        else:
            print("Frequencies: N/A")
        print("RF band:", particles[particle]['RF band'] if 'RF band' in particles[particle] else 'N/A')
        print("Frequency:", frequency)
        print("Momentum Energy:", energy)
        print("Intensity:", intensity)
        print("Distance from current particle between reversals:", distance)
        print()

# Sample inputs
emotional_amplitudes = {'Electron': 0.5, 'Proton': 0.8, 'Neutron': 0.6, 'Photon': 1.0, 'Quark': 0.7, 'Muon': 0.9, 'Tau': 0.4, 'Neutrino': 0.3, 'Antineutrino': 0.2, 'W Boson': 0.6, 'Z Boson': 0.8, 'Gluon': 0.5, 'Higgs Boson': 0.3}
emotional_phases = {'Electron': 0, 'Proton': 2, 'Neutron': 1, 'Photon': 0, 'Quark': 2, 'Muon': 1, 'Tau': 0, 'Neutrino': 1, 'Antineutrino': 2, 'W Boson': 0, 'Z Boson': 1, 'Gluon': 2, 'Higgs Boson': 0}
frequency_values = {'Electron': 2.41799 * 10 ** 14, 'Proton': 1.303 * 10 ** 10, 'Neutron': 9.661 * 10 ** 6, 'Photon': 0, 'Quark': 0, 'Muon': 6.8 * 10 ** 6, 'Tau': 1.777 * 10 ** 8, 'Neutrino': 1.7 * 10 ** 15, 'Antineutrino': 1.7 * 10 ** 15, 'W Boson': 2.085 * 10 ** 23, 'Z Boson': 9.118 * 10 ** 23, 'Gluon': 3 * 10 ** 26, 'Higgs Boson': 1.2 * 10 ** 15}

# Execute the function
subatomic_function(emotional_amplitudes, emotional_phases, frequency_values)



