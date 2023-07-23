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
        "price_range_percent": price_range_percent,
        "momentum": momentum,
        "min": sine_wave_min,
        "max": sine_wave_max
    }
   


#sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)
#print(sine_wave)

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
            #print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 2: 
            # Most negative frequencies in Q1
            em_value = em_amp_q2 * math.sin(em_phase_q2 * frequency) 
            #print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 3: 
            # Most negative frequencies in Q1
            em_value = em_amp_q3 * math.sin(em_phase_q3 * frequency) 
            #print(em_value)

       # Calculate EM value based on frequency   
        elif current_quadrant == 4: 
            # Most negative frequencies in Q1
            em_value = em_amp_q4 * math.sin(em_phase_q4 * frequency) 
            #print(em_value)

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
    
    print()

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

    print()

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

    print()

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

    print()

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
 
        
    #for freq in frequencies:               
        #print(freq['number'], freq['em_value'], freq['mood'])    

        
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

    for freq in frequencies: 
        freq['freq'] = current_frequency  # Add this

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

    print()

    if highest_3_mood > 0:
        print(f"Cycle mood is negative")
    elif highest_3_mood < 0:      
        print(f"Cycle mood is positive") 
    else:
        print("Cycle mood is neutral")

    print()

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

    freq_index_range_min = frequencies[0]['mood']
    freq_index_range_max = frequencies[-1]['mood'] 
    freq_range =  total_mood

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


    print("added code from here")

    # Add this  
    for freq in frequencies:
        freq.setdefault('freq', '')
        freq.setdefault('mood', '')
        freq.setdefault('em_value', '')
        
    for freq in frequencies:
        try:  
            print(f"Frequency: {freq['freq']}  Hz")  
            print(f"Mood: {freq['mood']}")   
            print(f"EM Value: {freq['em_value']}")  
        except KeyError:
            print("No frequency data available.")

    print("to here")


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

    up_circuit = True

    # Loop through each quadrant cycle        
    for quadrant in [1,2,3,4]:
    
        #print(f"Quadrant {quadrant}")
            
        # Get triangle point from quadrant map               
        point = quadrant_map[quadrant]    
    
        #print(f"Current point: {point}")

        last_point = 'Apex' 
        next_point = 'Left'

        # Get next point based on circuit
        
        if up_circuit:
            if point == 'Apex':
                next_point = 'Left' 
                last_point = 'Left'  
            
            elif point == 'Left':
                if last_point == 'Apex':      
                    next_point = 'Base'    
                else:      
                    next_point = 'Apex'
            
            elif point == 'Base':
                if last_point == 'Left':      
                    next_point = 'Right'    
                else:      
                    next_point = 'Left' 
            
        else: # Down circuit   
            if point =='Right':    
                next_point = 'Base'       
                last_point = 'Base'
            
            elif point == 'Base':       
                next_point = 'Left'  
            
            elif point == 'Left':
                next_point = 'Apex' 
                last_point = 'Apex'
            
        # Toggle circuit           
        up_circuit = not up_circuit

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

    current_point = forecast[f'min_reversal']['point'] if forecast[f'min_reversal']['point'] == point else forecast[f'max_reversal']['point']

    print("Current point is at: ", current_point) 

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

    return (
        current_em_amp,  
        current_em_phase,
        cycle_direction,
        current_momentum,    
        forecast_mood, 
        freq_index_range_min,
        freq_index_range_max,
        freq_range,    
        current_point,
        next_point,       
        forecast,       
        frequencies 
    )

print()

current_em_amp, current_em_phase, cycle_direction, current_momentum, forecast_mood, freq_index_range_min, freq_index_range_max, freq_range, current_point, next_point, forecast, frequencies = octa_metatron_cube(close_prices, candles)

print()

print(f"Current EM Amplitude: {current_em_amp}")
print(f"Current EM Phase: {current_em_phase}")
print(f"Current Momentum: {current_momentum}")  
print(f"Forecast Mood: {forecast_mood}")
print(f"Current Point: {current_point}")
print(f"Next Point: {next_point}")
print(f"Forecast: {forecast}")
print(f"Cycle_direction: {cycle_direction}")


#print("Frequencies:")
#for freq in frequencies:  
    #try:
       #print(f"Frequency: {freq['freq']}  Hz") 
        #print(f"Mood: {freq['mood']}")
        #print(f"EM Value: {freq['em_value']}")
    #except KeyError:
        #print("No frequency data available.")

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

def get_target(closes, n_components, target_distance=0.001):
    # Calculate FFT of closing prices
    fft = fftpack.fft(closes) 
    frequencies = fftpack.fftfreq(len(closes), d=0.25)
    
    # Sort frequencies by magnitude and keep only the top n_components 
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]
    
    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = np.real(fftpack.ifft(filtered_fft))
    
    # Calculate the target price as the next value after the last closing price, plus a small constant
    current_close = closes[-1]
    target_price = filtered_signal[-1] + target_distance
    
    # Get the current time           
    current_time = datetime.datetime.now()
    
    # Calculate the market mood based on the predicted target price and the current close price
    diff = target_price - current_close
    if diff > 0:           
        market_mood = "Bullish"
        faster_target = current_close - target_distance
    elif diff < 0:                 
        market_mood = "Bearish"
        faster_target = current_close + target_distance
    else:           
        market_mood = "Neutral"
        faster_target = current_close + target_distance
    
    # Calculate the stop loss and target levels
    entry_price = closes[-1]    
    stop_loss =  entry_price - 2*np.std(closes)   
    target1 = target_price + 0.5*np.std(closes)  
    target2 = target_price + np.std(closes)  
    target3 = target_price + 1.5*np.std(closes)            
    
    return current_time, entry_price, stop_loss, target1, target2, target3, filtered_signal, target_price, faster_target, market_mood

closes = get_closes("1m")     
n_components = 5

current_time, entry_price, stop_loss, target1, target2, target3, filtered_signal, target_price, fastest_target, market_mood = get_target(closes, n_components, target_distance=56)

print("Current local Time is now at: ", current_time)
print("Market mood is: ", market_mood)

print()


print("Fastest target is: ", fastest_target)
print("Fast target is: ", target_price)   

print()

print("Entry price is at : ", entry_price)
print("Stop loss is: ", stop_loss)

print() 

print("Target 1 is: ", target1)           
print("Target 2 is: ", target2)
print("Target 3 is: ", target3)

print()

##################################################
##################################################

print("Init main() loop: ")

print()

##################################################
##################################################
import numpy
import math

beta = 16
n = 1024
delta = 0.3
alpha = 0.25
L = 10
sigma = 0.002
a = 0
b = 0
B = 16


def phi(x):
    return numpy.arctan(x)

def compute_P(sigma, a, b, n):
    P = [0] * n
    for i in range(n):
        P[i] = numpy.cos(2 * numpy.pi * sigma * ((i + a) % n) / n + b)
    return P

def GB(alpha, delta, x):
    n = len(x)
    G = [0] * n
    for k in range(n):
        G[k] = alpha ** k * delta ** (n - k) * x[k]
    return G

def HASHTOBINS(x, zb, sigma, a, b, B, delta, alpha):
    n = len(x)
    y = GB(alpha, delta, compute_P(sigma, a, b, n))
    P = [compute_P(sigma, 0, zb[i], n) for i in range(beta)]
    y_prime = [y[k] - P[k % B][k] for k in range(n)]
    ub = [y_prime[k % B] / B for k in range(n)]
    return ub

def NOISELESSSPARSEFFT(x, k, sigma, a, b, B):
    zb = [0.0] * beta
    k_prime = beta * math.ceil(k / beta)
    for t in range(int(numpy.log2(k))):
        zb_prime = list(zb)
        zb_prime[:len(zb)] = [zb_prime[i] + NOISELESSSPARSEFFTINNER(x, k_prime, zb, sigma, a, b, B)[i] for i in range(len(zb_prime))]
        zb = [zb_prime[i] if i < len(zb_prime) else 0.0 for i in range(len(zb))]
        for i in range(beta):
            if abs(zb[i]) >= L:
                zb[i] = 0.0
    return zb

def NOISELESSSPARSEFFTINNER(x, k, zb, sigma, a, b, B):
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        if i < k:
            y[i] = x[i]
        else:
            y[i] = y[i - k]
    y_hat = numpy.fft.fft(y)
    y_prime_hat = [y_hat[i] if i < (n // 2 + 1) else numpy.conj(y_hat[n - i]) for i in range(n)]
    y_prime = numpy.fft.ifft(y_prime_hat)
    ub = HASHTOBINS(x, zb, sigma, a, b, B, delta, alpha)
    z = [y_prime[i] - ub[i] for i in range(n)]
    return z[:k]

# Preprocess close prices to compute log returns

log_returns = [numpy.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]

# Compute NSFT of log returns
k = 64
x = log_returns[:k]
zb = [0.0] * beta
zb_prime = list(zb)
zb_prime[:len(zb)] = NOISELESSSPARSEFFTINNER(x, k, zb, sigma, a, b, B)
zb = [zb_prime[i] if i < len(zb_prime) else 0.0 for i in range(len(zb))]
ub = HASHTOBINS(x, zb, sigma, a, b, B, delta, alpha)

print(zb)
