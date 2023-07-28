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

##################################################
##################################################

# Get current price as <class 'float'>

def get_price(symbol):
    try:
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

# Get entire list of close prices as <class 'list'> type
def get_close(timeframe):
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
  
    close_prices = np.array(get_close(timeframe))
  
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
for timeframe in timeframes:        
    scale_to_sine(timeframe)

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

def get_closes_last_n_minutes(interval, n):
    """Generate mock closing prices for the last n minutes"""
    closes = []
    for i in range(n):
        closes.append(random.uniform(0, 100))
    return closes

print()

##################################################
##################################################

import numpy as np
import scipy.fftpack as fftpack
import datetime

def get_target(closes, n_components, target_distance=0.01):
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
    
    # Calculate the target price as the next value after the last closing price, plus a small constant
    current_close = closes[-1]
    target_price = filtered_signal[-1] + target_distance
    
    # Get the current time           
    current_time = datetime.datetime.now()
    
    # Calculate the market mood based on the predicted target price and the current close price
    diff = target_price - current_close
    if diff > 0:           
        market_mood = "Bullish"
        fastest_target = current_close + target_distance/2
        fast_target1 = current_close + target_distance/4
        fast_target2 = current_close + target_distance/8
        fast_target3 = current_close + target_distance/16
        fast_target4 = current_close + target_distance/32
        target1 = target_price + np.std(closes)/16
        target2 = target_price + np.std(closes)/8
        target3 = target_price + np.std(closes)/4
        target4 = target_price + np.std(closes)/2
        target5 = target_price + np.std(closes)
    elif diff < 0:                 
        market_mood = "Bearish"
        fastest_target = current_close - target_distance/2
        fast_target1 = current_close - target_distance/4
        fast_target2 = current_close - target_distance/8
        fast_target3 = current_close - target_distance/16
        fast_target4 = current_close - target_distance/32
        target1 = target_price - np.std(closes)/16
        target2 = target_price - np.std(closes)/8
        target3 = target_price - np.std(closes)/4
        target4 = target_price - np.std(closes)/2
        target5 = target_price - np.std(closes)
    else:           
        market_mood = "Neutral"
        fastest_target = current_close + target_distance/2
        fast_target1 = current_close - target_distance/4
        fast_target2 = current_close + target_distance/8
        fast_target3 = current_close - target_distance/16
        fast_target4 = current_close + target_distance/32
        target1 = target_price + np.std(closes)/16
        target2 = target_price - np.std(closes)/8
        target3 = target_price + np.std(closes)/4
        target4 = target_price - np.std(closes)/2
        target5 = target_price + np.std(closes)
    
    # Calculate the stop loss and target levels
    entry_price = closes[-1]    
    stop_loss =  entry_price - 3*np.std(closes)   
    target6 = target_price + np.std(closes)
    target7 = target_price + 2*np.std(closes)
    target8 = target_price + 3*np.std(closes)
    target9 = target_price + 4*np.std(closes)
    target10 = target_price + 5*np.std(closes)
    
    return current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood

closes = get_closes("1m")     
n_components = 5

current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood = get_target(closes, n_components, target_distance=56)

print("Current local Time is now at: ", current_time)
print("Market mood is: ", market_mood)

print()

print("Current close price is at : ", current_close)

print()

print("Fast target 1 is: ", fast_target4)
print("Fast target 2 is: ", fast_target3)
print("Fast target 3 is: ", fast_target2)
print("Fast target 4 is: ", fast_target1)

print()

print("Fastest target is: ", fastest_target)

print()

print("Target 1 is: ", target1)
print("Target 2 is: ", target2)
print("Target 3 is: ", target3)
print("Target 4 is: ", target4)
print("Target 5 is: ", target5)

print()


##################################################
##################################################

def get_current_price():
    url = "https://fapi.binance.com/fapi/v1/ticker/price"
    params = {
        "symbol": "BTCUSDT" 
    }
    response = requests.get(url, params=params)
    data = response.json()
    price = float(data["price"])
    return price

# Get the current price
price = get_current_price()

print()

##################################################
##################################################

def get_support_resistance_levels(close):
    # Convert close list to numpy array
    close_prices = np.array(close)

    # Calculate EMA50 and EMA200
    ema50 = talib.EMA(close_prices, timeperiod=50)
    ema200 = talib.EMA(close_prices, timeperiod=200)

    # Check if ema50 and ema200 have at least one element
    if len(ema50) == 0 or len(ema200) == 0:
        return []

    # Get the last element of ema50 and ema200
    ema50 = ema50[-1]
    ema200 = ema200[-1]

    # Calculate Phi Ratio levels
    range_ = ema200 - ema50
    phi_levels = [ema50, ema50 + range_/1.618, ema50 + range_]

    # Calculate Gann Square levels
    current_price = close_prices[-1]
    high_points = [current_price, ema200, max(phi_levels)]
    low_points = [min(phi_levels), ema50, current_price]

    gann_levels = []
    for i in range(1, min(4, len(high_points))):
        for j in range(1, min(4, len(low_points))):
            gann_level = ((high_points[i-1] - low_points[j-1]) * 0.25 * (i + j)) + low_points[j-1]
            gann_levels.append(gann_level)

    # Combine levels and sort
    levels = phi_levels + gann_levels
    levels.sort()

    return levels

print()

# Get the support and resistance levels
levels = get_support_resistance_levels(close_prices)

support_levels, resistance_levels = [], []

for level in levels:
    if level < close_prices[-1]:
        support_levels.append(level)
    else:
        resistance_levels.append(level)

# Determine the market mood
if len(levels) > 0:
    support_levels = []
    resistance_levels = []
    for level in levels:
        if level < close_prices[-1]:
            support_levels.append(level)
        else:
            resistance_levels.append(level)

    if len(support_levels) > 0 and len(resistance_levels) > 0:
        market_mood_sr = "Neutral"
    elif len(support_levels) > 0:
        market_mood_sr = "Bullish"
    elif len(resistance_levels) > 0:
        market_mood_sr = "Bearish"
    else:
        market_mood_sr = "Undefined"

    # Calculate support and resistance ranges
    if len(support_levels) > 0:
        support_range = max(support_levels) - min(support_levels)
        print("Support range: {:.2f}".format(support_range))
    else:
        print("Support range: None")

    if len(resistance_levels) > 0:
        resistance_range = max(resistance_levels) - min(resistance_levels)
        print("Resistance range: {:.2f}".format(resistance_range))
    else:
        print("Resistance range: None")

    # Print the levels and market mood
    print("Potential support levels:")
    if len(support_levels) > 0:
        for level in support_levels:
            print("  - {:.2f}".format(level))
    else:
        print("  None found.")

    print("Potential resistance levels:")
    if len(resistance_levels) > 0:
        for level in resistance_levels:
            print("  - {:.2f}".format(level))
    else:
        print("  None found.")

    incoming_bullish_reversal = None
    incoming_bearish_reversal = None

    if market_mood_sr == "Neutral":
        print("Market mood: {}".format(market_mood_sr))
        if len(support_levels) > 0:
            support = max(support_levels)
            support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            support = None
            support_percentage = None

        if len(resistance_levels) > 0:
            top = min(resistance_levels)
            top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            top = None
            top_percentage = None

        print("Best dip: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))

        if support_percentage >= 3.0:
            incoming_bullish_reversal = True

    elif market_mood_sr == "Bullish":
        print("Market mood: {}".format(market_mood_sr))
        if len(resistance_levels) > 0:
            top = min(resistance_levels)
            top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            top = None
            top_percentage = None

        if top is not None:
            print("Best breakout: {:.2f}% (Resistance level: {:.2f})".format(top_percentage, top))
        else:
            print("Best breakout: None")

        if top_percentage is not None and top_percentage >= 3.0:
            incoming_bullish_reversal = True

    elif market_mood_sr == "Bearish":
        print("Market mood: {}".format(market_mood_sr))
        if len(support_levels) > 0:
            support = max(support_levels)
            support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            support = None
            support_percentage = None

        if support is not None:
            print("Best bounce: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))
        else:
            print("Best bounce: None")

        if support_percentage is not None and support_percentage >= 3.0:
            incoming_bearish_reversal = True

    else:
        print("Market mood: {}".format(market_mood_sr))

    # Print incoming reversal signals
    if incoming_bullish_reversal:
        print("Incoming bullish reversal signal!")

    if incoming_bearish_reversal:
        print("Incoming bearish reversal signal!")


print()

##################################################
##################################################

from scipy.signal import argrelextrema

import talib
import numpy as np
from scipy.signal import argrelextrema

def calculate_reversals(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate new reversal lows and highs and incoming direction based on the reciprocal Fibonacci constant for different time cycles.
    """
  
    # Get min/max close    
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    # Convert close_prices to numpy array
    close_prices = np.array(close_prices)
    
    # Calculate HT_SINE
    sine, leadsine = talib.HT_SINE(close_prices)
    sine = -sine

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
    
    # Calculate incoming direction
    incoming_direction = "Unknown"
    if momentum_signal > 0:
        incoming_direction = "Up"
    elif momentum_signal < 0:
        incoming_direction = "Down"
        
    # Calculate new reversal lows and highs
    highs = argrelextrema(leadsine, np.greater, order=period)[0]
    lows = argrelextrema(leadsine, np.less, order=period)[0]
    
    # Determine market mood based on reciprocal Fibonacci constant for different cycles
    very_fast_cycle = 3
    medium_cycle = 5
    slow_cycle = 8
    very_slow_cycle = 13
    
    #rfib_const = 0.03833  # Reciprocal Fibonacci constant
    rfib_const = 3.3598856662

    very_fast_cycle_mood = "Unknown"
    medium_cycle_mood = "Unknown"
    slow_cycle_mood = "Unknown"
    very_slow_cycle_mood = "Unknown"
    
    if len(lows) > 0 and len(highs) > 0:
        last_low = lows[-1]
        last_high = highs[-1]
        
        # Calculate cycles based on last reversal low and high
        very_fast_cycle_length = last_high - last_low
        medium_cycle_length = very_fast_cycle_length * rfib_const ** medium_cycle
        slow_cycle_length = very_fast_cycle_length * rfib_const ** slow_cycle
        very_slow_cycle_length = very_fast_cycle_length * rfib_const ** very_slow_cycle
        
        # Check if current price is below or above the last reversal low or high
        if np.all(close_prices < last_low):
            very_fast_cycle_mood = "Bearish"
            medium_cycle_mood = "Bearish"
            slow_cycle_mood = "Bearish"
            very_slow_cycle_mood = "Bearish"
            print("Current price is below the last reversal low.")
        elif np.all(close_prices > last_high):
            very_fast_cycle_mood = "Bullish"
            medium_cycle_mood = "Bullish"
            slow_cycle_mood = "Bullish"
            very_slow_cycle_mood = "Bullish"
            print("Current price is above the last reversal high.")
        else:
            # Calculate cycles based on current price
            very_fast_cycle_length = np.abs(avg_mtf - close_prices[-1])
            medium_cycle_length = very_fast_cycle_length * rfib_const ** medium_cycle
            slow_cycle_length = very_fast_cycle_length * rfib_const ** slow_cycle
            very_slow_cycle_length = very_fast_cycle_length * rfib_const ** very_slow_cycle
            
            # Calculate mood for different cycles
            if close_prices[-1] < avg_mtf:
                very_fast_cycle_mood = "Bearish"
                medium_cycle_mood = "Bullish"
                slow_cycle_mood = "Bearish"
                very_slow_cycle_mood = "Bullish"
                print("Current price is below the average filtered close price.")
            else:
                very_fast_cycle_mood = "Bullish"
                medium_cycle_mood = "Bearish"
                slow_cycle_mood = "Bullish"
                very_slow_cycle_mood = "Bearish"
                print("Current price is above the average filtered close price.")
    
    # Create dictionary to store all the calculated values
    results = {
        "min_close": min_close,
        "max_close": max_close,
        "sine": sine[-1],
        "leadsine": leadsine[-1],
        "momentum": momentum[-1],
        "min_momentum": min_momentum,
        "max_momentum": max_momentum,
        "min_percentage_custom": min_percentage_custom,
        "max_percentage_custom": max_percentage_custom,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "filtered_close": filtered_close[-1],
        "avg_mtf": avg_mtf,
        "current_momentum": current_momentum,
        "percent_to_min_momentum": percent_to_min_momentum,
        "percent_to_max_momentum": percent_to_max_momentum,
        "percent_to_min_combined": percent_to_min_combined,
        "percent_to_max_combined": percent_to_max_combined,
        "momentum_signal": momentum_signal,
        "incoming_direction": incoming_direction,
        "highs": highs[-1],
        "lows": lows[-1],
        "very_fast_cycle_mood": very_fast_cycle_mood,
        "medium_cycle_mood": medium_cycle_mood,
        "slow_cycle_mood": slow_cycle_mood,
        "very_slow_cycle_mood": very_slow_cycle_mood,
        "very_fast_cycle_length": very_fast_cycle_length,
        "medium_cycle_length": medium_cycle_length,
        "slow_cycle_length": slow_cycle_length,
        "very_slow_cycle_length": very_slow_cycle_length,
    }
    
    # Print all the calculated values
    for key, value in results.items():
        print(f"{key}: {value}")
        
    return results
    
# convert close_prices to numpy array of type float64
close_prices = np.array(close_prices, dtype=np.float64)

# Call the function with default parameters
results = calculate_reversals(close_prices)

# Print the last element of each array, if it exists
if isinstance(results["min_close"], np.ndarray):
    print("min_close:", results["min_close"][-1] if results["min_close"].size > 0 else None)
if isinstance(results["max_close"], np.ndarray):
    print("max_close:", results["max_close"][-1] if results["max_close"].size > 0 else None)
if isinstance(results["sine"], np.ndarray):
    print("sine:", results["sine"][-1] if results["sine"].size > 0 else None)
if isinstance(results["leadsine"], np.ndarray):
    print("leadsine:", results["leadsine"][-1] if results["leadsine"].size > 0 else None)
if isinstance(results["momentum"], np.ndarray):
    print("momentum:", results["momentum"][-1] if results["momentum"].size > 0 else None)
if isinstance(results["min_momentum"], np.ndarray):
    print("min_momentum:", results["min_momentum"][-1] if results["min_momentum"].size > 0 else None)
if isinstance(results["max_momentum"], np.ndarray):
    print("max_momentum:", results["max_momentum"][-1] if results["max_momentum"].size > 0 else None)
if isinstance(results["min_percentage_custom"], np.ndarray):
    print("min_percentage_custom:", results["min_percentage_custom"][-1] if results["min_percentage_custom"].size > 0 else None)
if isinstance(results["max_percentage_custom"], np.ndarray):
    print("max_percentage_custom:", results["max_percentage_custom"][-1] if results["max_percentage_custom"].size > 0 else None)
if isinstance(results["min_threshold"], np.ndarray):
    print("min_threshold:", results["min_threshold"][-1] if results["min_threshold"].size > 0 else None)
if isinstance(results["max_threshold"], np.ndarray):
    print("max_threshold:", results["max_threshold"][-1] if results["max_threshold"].size > 0 else None)
if isinstance(results["filtered_close"], np.ndarray):
    print("filtered_close:", results["filtered_close"][-1] if results["filtered_close"].size > 0 else None)
if isinstance(results["avg_mtf"], np.ndarray):
    print("avg_mtf:", results["avg_mtf"][-1] if results["avg_mtf"].size > 0 else None)
if isinstance(results["current_momentum"], np.ndarray):
    print("current_momentum:", results["current_momentum"][-1] if results["current_momentum"].size > 0 else None)
if isinstance(results["percent_to_min_momentum"], np.ndarray):
    print("percent_to_min_momentum:", results["percent_to_min_momentum"][-1] if results["percent_to_min_momentum"].size > 0 else None)
if isinstance(results["percent_to_max_momentum"], np.ndarray):
    print("percent_to_max_momentum:", results["percent_to_max_momentum"][-1] if results["percent_to_max_momentum"].size > 0 else None)
if isinstance(results["percent_to_min_combined"], np.ndarray):
    print("percent_to_min_combined:", results["percent_to_min_combined"][-1] if results["percent_to_min_combined"].size > 0 else None)
if isinstance(results["percent_to_max_combined"], np.ndarray):
    print("percent_to_max_combined:", results["percent_to_max_combined"][-1] if results["percent_to_max_combined"].size > 0 else None)
if isinstance(results["momentum_signal"], np.ndarray):
    print("momentum_signal:", results["momentum_signal"][-1] if results["momentum_signal"].size > 0 else None)
if isinstance(results["incoming_direction"], np.ndarray):
    print("incoming_direction:", results["incoming_direction"][-1] if results["incoming_direction"].size > 0 else None)
if isinstance(results["highs"], np.ndarray):
    print("highs:", results["highs"][-1] if results["highs"].size > 0 else None)
if isinstance(results["lows"], np.ndarray):
    print("lows:", results["lows"][-1] if results["lows"].size > 0 else None)
if isinstance(results["very_fast_cycle_mood"], np.ndarray):
    print("very_fast_cycle_mood:", results["very_fast_cycle_mood"][-1] if results["very_fast_cycle_mood"].size > 0 else None)
if isinstance(results["medium_cycle_mood"], np.ndarray):
    print("medium_cycle_mood:", results["medium_cycle_mood"][-1] if results["medium_cycle_mood"].size > 0 else None)
if isinstance(results["slow_cycle_mood"], np.ndarray):
    print("slow_cycle_mood:", results["slow_cycle_mood"][-1] if results["slow_cycle_mood"].size > 0 else None)
if isinstance(results["very_slow_cycle_mood"], np.ndarray):
    print("very_slow_cycle_mood:", results["very_slow_cycle_mood"][-1] if results["very_slow_cycle_mood"].size > 0 else None)
if isinstance(results["very_fast_cycle_length"], np.ndarray):
    print("very_fast_cycle_length:", results["very_fast_cycle_length"][-1] if results["very_fast_cycle_length"].size > 0 else None)
if isinstance(results["medium_cycle_length"], np.ndarray):
    print("medium_cycle_length:", results["medium_cycle_length"][-1] if results["medium_cycle_length"].size > 0 else None)
if isinstance(results["slow_cycle_length"], np.ndarray):
    print("slow_cycle_length:", results["slow_cycle_length"][-1] if results["slow_cycle_length"].size > 0 else None)
if isinstance(results["very_slow_cycle_length"], np.ndarray):
    print("very_slow_cycle_length:", results["very_slow_cycle_length"][-1] if results["very_slow_cycle_length"].size > 0 else None)

# Calculate very fast cycle length 
very_fast_cycle_length = np.abs(avg_mtf - close_prices[-1])   

# Calculate next 3 prices based on cycle length and current price        
price1 = close_prices[-1] + very_fast_cycle_length    
price2 = close_prices[-1] + 2 * very_fast_cycle_length   
price3 = close_prices[-1] + 3 * very_fast_cycle_length

# Print the next 3 forecast prices
print(f"Next 3 forecast prices: {price1}, {price2}, {price3}")

# Get current momentum
current_momentum = results["momentum"]

# Calculate very fast cycle momentum
very_fast_cycle_momentum = 0.75 * current_momentum

# Calculate medium cycle momentum  
medium_cycle_momentum = 0.5 * current_momentum

# Calculate slow cycle momentum
slow_cycle_momentum = 0.25 * current_momentum

# Print next 3 momentum forecast cycles    
print(f"Next 3 momentum forecasts: {very_fast_cycle_momentum}, {medium_cycle_momentum}, {slow_cycle_momentum}")


# Calculate very fast cycle momentum        
very_fast_cycle_momentum = 0.75 * current_momentum

# Determine very fast cycle mood based on momentum        
if very_fast_cycle_momentum < 0:
    very_fast_cycle_mood = "Bearish"
else:
    very_fast_cycle_mood = "Bullish"  

# Calculate medium cycle momentum       
medium_cycle_momentum = 0.5 * current_momentum

# Determine medium cycle mood        
if medium_cycle_momentum < 0:
    medium_cycle_mood = "Bearish"    
else:
    medium_cycle_mood = "Bullish" 

# Calculate slow cycle momentum    
slow_cycle_momentum = 0.25 * current_momentum

# Determine slow cycle mood
if slow_cycle_momentum < 0:
    slow_cycle_mood = "Bearish"      
else:      
    slow_cycle_mood = "Bullish"  

# Print market mood forecasts
print(f"Momentum market mood forecasts: {very_fast_cycle_mood}, {medium_cycle_mood}, {slow_cycle_mood}")

# Compare current momentum to different cycle lengths
if current_momentum > results['very_fast_cycle_length']:
    print("Current momentum is greater than very fast cycle length.")
elif current_momentum < results['very_fast_cycle_length']:
    print("Current momentum is less than very fast cycle length.")
else:
    print("Current momentum is equal to very fast cycle length.")

if current_momentum > results['medium_cycle_length']:
    print("Current momentum is greater than medium cycle length.")
elif current_momentum < results['medium_cycle_length']:
    print("Current momentum is less than medium cycle length.")
else:
    print("Current momentum is equal to medium cycle length.")

if current_momentum > results['slow_cycle_length']:
    print("Current momentum is greater than slow cycle length.")
elif current_momentum < results['slow_cycle_length']:
    print("Current momentum is less than slow cycle length.")
else:
    print("Current momentum is equal to slow cycle length.")

if current_momentum > results['very_slow_cycle_length']:
    print("Current momentum is greater than very slow cycle length.")
elif current_momentum < results['very_slow_cycle_length']:
    print("Current momentum is less than very slow cycle length.")
else:
    print("Current momentum is equal to very slow cycle length.")

# Compare current price to different cycle lengths
if np.all(close_prices < results['very_fast_cycle_length']):
    print("Current price is below very fast cycle length.")
elif np.all(close_prices > results['very_fast_cycle_length']):
    print("Current price is above very fast cycle length.")
else:
    print("Current price is at very fast cycle length.")

if np.all(close_prices < results['medium_cycle_length']):
    print("Current price is below medium cycle length.")
elif np.all(close_prices > results['medium_cycle_length']):
    print("Current price is above medium cycle length.")
else:
    print("Current price is at medium cycle length.")

if np.all(close_prices < results['slow_cycle_length']):
    print("Current price is below slow cycle length.")
elif np.all(close_prices > results['slow_cycle_length']):
    print("Current price is above slow cycle length.")
else:
    print("Current price is at slow cycle length.")

if np.all(close_prices < results['very_slow_cycle_length']):
    print("Current price is below very slow cycle length.")
elif np.all(close_prices > results['very_slow_cycle_length']):
    print("Current price is above very slow cycle length.")
else:
    print("Current price is at very slow cycle length.")

print()

##################################################
##################################################

print()

##################################################
##################################################

print("Init main() loop: ")

print()

##################################################
##################################################

def main():

    ##################################################
    ##################################################

    # Define timeframes
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    TRADE_SYMBOL = "BTCUSDT"

    ##################################################
    ##################################################

    while True:

        ##################################################
        ##################################################
        
        # Get fresh closes for the current timeframe
        closes = get_closes('1m')
       
        # Get close price as <class 'float'> type
        close = get_close('1m')
        
        # Get fresh candles  
        candles = get_candles(TRADE_SYMBOL, timeframes)
                
        # Update candle_map with fresh data
        for candle in candles:
            timeframe = candle["timeframe"]  
            candle_map.setdefault(timeframe, []).append(candle)
                
        ##################################################
        ##################################################

        try:     
            ##################################################
            ##################################################

            # Calculate fresh sine wave  
            close_prices = np.array(closes)
            sine, leadsine = talib.HT_SINE(close_prices)
                
            # Call scale_to_sine() function   
            #dist_from_close_to_min, dist_from_close_to_max = scale_to_sine('1m')

            #for timeframe in timeframes:
                #dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)

                # Print results        
                #print(f"On {timeframe} Close price value on sine is now at: {current_sine})")
                #print(f"On {timeframe} Distance from close to min perc. is now at: {dist_from_close_to_min})")
                #print(f"On {timeframe} Distance from close to max perc. is now at: {dist_from_close_to_max})")

            ##################################################
            ##################################################

            print()

            momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max = generate_momentum_sinewave(timeframes)
        
            print()

            print("Current close on sine value now at: ", current_sine)
            print("Distance as percentages from close to min: ", dist_from_close_to_min, "%")
            print("Distance as percentages from close to max: ", dist_from_close_to_max, "%")
            #print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
            print("Mood on 1min timeframe is now at: ", market_mood[-12])

            print()


            ##################################################
            ##################################################

            sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                                               percent_to_max_val=5, 
                                               percent_to_min_val=5)      

            sine_wave_max = sine_wave["max"]   
            sine_wave_min = sine_wave["min"]

            # Call the function
            results = generate_new_momentum_sinewave(
                close_prices, 
                candles,  
                percent_to_max_val=5,  
                percent_to_min_val=5
                )
  
            # Unpack the returned values    
            current_close = results["current_close"]  
            dist_from_close_to_min = results["dist_from_close_to_min"]  
            dist_from_close_to_max = results["dist_from_close_to_max"]
            current_quadrant = results["current_quadrant"]
            em_amp = results["em_amplitude"]
            em_phase = results["em_phase"]  
            price_range_percent = results["price_range_percent"] 
            momentum = results["momentum"]
            sine_wave_min = results["min"]
            sine_wave_max = results["max"]

            print()

            ##################################################
            ##################################################

            url = "https://fapi.binance.com/fapi/v1/ticker/price"

            params = {
                "symbol": "BTCUSDT" 
                }

            response = requests.get(url, params=params)
            data = response.json()

            price = data["price"]
            #print(f"Current BTCUSDT price: {price}")

            # Define the current time and close price
            current_time = datetime.datetime.now()
            current_close = price

            #print("Current local Time is now at: ", current_time)
            #print("Current close price is at : ", current_close)

            print()

            ##################################################
            ##################################################

            # Print the variables from generate_new_momentum_sinewave()
            print(f"Distance from close to min: {dist_from_close_to_min}") 
            print(f"Distance from close to max: {dist_from_close_to_max}")
            print(f"Current_quadrant now at: {current_quadrant}")

            print()

            ##################################################
            ##################################################

            timeframe = '1m'
            momentum = get_momentum(timeframe)
            print("Momentum on 1min tf is at: ", momentum)

            ##################################################
            ##################################################

            # Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
            min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

            print("Momentum sinewave signal:", momentum_signal)
            print()

            print("Minimum threshold:", min_threshold)
            print("Maximum threshold:", max_threshold)
            print("Average MTF:", avg_mtf)

            print()

            ##################################################
            ##################################################

            closes = get_closes("1m")     
            n_components = 5

            current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood = get_target(closes, n_components, target_distance=56)

            print("Current local Time is now at: ", current_time)
            print("Market mood is: ", market_mood)

            print()

            print("Current close price is at : ", current_close)

            print()

            print("Fast target 1 is: ", fast_target4)
            print("Fast target 2 is: ", fast_target3)
            print("Fast target 3 is: ", fast_target2)
            print("Fast target 4 is: ", fast_target1)

            print()

            print("Fastest target is: ", fastest_target)

            print()

            print("Target 1 is: ", target1)
            print("Target 2 is: ", target2)
            print("Target 3 is: ", target3)
            print("Target 4 is: ", target4)
            print("Target 5 is: ", target5)

            # Get the current price
            price = get_current_price()

            print()

            price = float(price)

            print()

            ##################################################
            ##################################################

            # Initialize variables
            trigger_long = False 
            trigger_short = False

            current_time = datetime.datetime.utcnow() + timedelta(hours=3)

            range_threshold = max_threshold - min_threshold
            dist_from_min = price - min_threshold
            dist_from_max = max_threshold - price

            pct_diff_to_min = (dist_from_min / range_threshold) * 100
            pct_diff_to_max = (dist_from_max / range_threshold) * 100

            print("Percentage difference to min threshold:", pct_diff_to_min)
            print("Percentage difference to max threshold:", pct_diff_to_max)

            print()

            # Get the support and resistance levels
            levels = get_support_resistance_levels(close_prices)

            support_levels, resistance_levels = [], []

            for level in levels:
                if level < close_prices[-1]:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)

            # Determine the market mood
            if len(levels) > 0:
                support_levels = []
                resistance_levels = []
                for level in levels:
                    if level < close_prices[-1]:
                        support_levels.append(level)
                    else:
                        resistance_levels.append(level)

            if len(support_levels) > 0 and len(resistance_levels) > 0:
                market_mood_sr = "Neutral"
            elif len(support_levels) > 0:
                market_mood_sr = "Bullish"
            elif len(resistance_levels) > 0:
                market_mood_sr = "Bearish"
            else:
                market_mood_sr = "Undefined"

            # Calculate support and resistance ranges
            if len(support_levels) > 0:
                support_range = max(support_levels) - min(support_levels)
                print("Support range: {:.2f}".format(support_range))
            else:
                print("Support range: None")

            if len(resistance_levels) > 0:
                resistance_range = max(resistance_levels) - min(resistance_levels)
                print("Resistance range: {:.2f}".format(resistance_range))
            else:
                print("Resistance range: None")

            # Print the levels and market mood
            print("Potential support levels:")
            if len(support_levels) > 0:
                for level in support_levels:
                    print("  - {:.2f}".format(level))
            else:
                print("  None found.")

            print("Potential resistance levels:")
            if len(resistance_levels) > 0:
                for level in resistance_levels:
                    print("  - {:.2f}".format(level))
            else:
                print("  None found.")

            incoming_bullish_reversal = None
            incoming_bearish_reversal = None

            if market_mood_sr == "Neutral":
                print("Market mood: {}".format(market_mood_sr))
                if len(support_levels) > 0:
                    support = max(support_levels)
                    support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    support = None
                    support_percentage = None

                if len(resistance_levels) > 0:
                    top = min(resistance_levels)
                    top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    top = None
                    top_percentage = None

                print("Best dip: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))

                if support_percentage >= 3.0:
                    incoming_bullish_reversal = True

            elif market_mood_sr == "Bullish":
                print("Market mood: {}".format(market_mood_sr))
                if len(resistance_levels) > 0:
                    top = min(resistance_levels)
                    top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    top = None
                    top_percentage = None

                if top is not None:
                    print("Best breakout: {:.2f}% (Resistance level: {:.2f})".format(top_percentage, top))
                else:
                    print("Best breakout: None")

                if top_percentage is not None and top_percentage >= 3.0:
                    incoming_bullish_reversal = True

            elif market_mood_sr == "Bearish":
                print("Market mood: {}".format(market_mood_sr))
                if len(support_levels) > 0:
                    support = max(support_levels)
                    support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    support = None
                    support_percentage = None

                if support is not None:
                    print("Best bounce: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))
                else:
                    print("Best bounce: None")

                if support_percentage is not None and support_percentage >= 3.0:
                    incoming_bearish_reversal = True

            else:
                print("Market mood: {}".format(market_mood_sr))

            # Print incoming reversal signals
            if incoming_bullish_reversal:
                print("Incoming bullish reversal signal!")

            if incoming_bearish_reversal:
                print("Incoming bearish reversal signal!")

            print()

            # convert close_prices to numpy array of type float64
            close_prices = np.array(close_prices, dtype=np.float64)

            # Call the function with default parameters
            results = calculate_reversals(close_prices)

            # Print the last element of each array, if it exists
            if isinstance(results["min_close"], np.ndarray):
                print("min_close:", results["min_close"][-1] if results["min_close"].size > 0 else None)
            if isinstance(results["max_close"], np.ndarray):
                print("max_close:", results["max_close"][-1] if results["max_close"].size > 0 else None)
            if isinstance(results["sine"], np.ndarray):
                print("sine:", results["sine"][-1] if results["sine"].size > 0 else None)
            if isinstance(results["leadsine"], np.ndarray):
                print("leadsine:", results["leadsine"][-1] if results["leadsine"].size > 0 else None)
            if isinstance(results["momentum"], np.ndarray):
                print("momentum:", results["momentum"][-1] if results["momentum"].size > 0 else None)
            if isinstance(results["min_momentum"], np.ndarray):
                print("min_momentum:", results["min_momentum"][-1] if results["min_momentum"].size > 0 else None)
            if isinstance(results["max_momentum"], np.ndarray):
                print("max_momentum:", results["max_momentum"][-1] if results["max_momentum"].size > 0 else None)
            if isinstance(results["min_percentage_custom"], np.ndarray):
                print("min_percentage_custom:", results["min_percentage_custom"][-1] if results["min_percentage_custom"].size > 0 else None)
            if isinstance(results["max_percentage_custom"], np.ndarray):
                print("max_percentage_custom:", results["max_percentage_custom"][-1] if results["max_percentage_custom"].size > 0 else None)
            if isinstance(results["min_threshold"], np.ndarray):
                print("min_threshold:", results["min_threshold"][-1] if results["min_threshold"].size > 0 else None)
            if isinstance(results["max_threshold"], np.ndarray):
                print("max_threshold:", results["max_threshold"][-1] if results["max_threshold"].size > 0 else None)
            if isinstance(results["filtered_close"], np.ndarray):
                print("filtered_close:", results["filtered_close"][-1] if results["filtered_close"].size > 0 else None)
            if isinstance(results["avg_mtf"], np.ndarray):
                print("avg_mtf:", results["avg_mtf"][-1] if results["avg_mtf"].size > 0 else None)
            if isinstance(results["current_momentum"], np.ndarray):
                print("current_momentum:", results["current_momentum"][-1] if results["current_momentum"].size > 0 else None)
            if isinstance(results["percent_to_min_momentum"], np.ndarray):
                print("percent_to_min_momentum:", results["percent_to_min_momentum"][-1] if results["percent_to_min_momentum"].size > 0 else None)
            if isinstance(results["percent_to_max_momentum"], np.ndarray):
                print("percent_to_max_momentum:", results["percent_to_max_momentum"][-1] if results["percent_to_max_momentum"].size > 0 else None)
            if isinstance(results["percent_to_min_combined"], np.ndarray):
                print("percent_to_min_combined:", results["percent_to_min_combined"][-1] if results["percent_to_min_combined"].size > 0 else None)
            if isinstance(results["percent_to_max_combined"], np.ndarray):
                print("percent_to_max_combined:", results["percent_to_max_combined"][-1] if results["percent_to_max_combined"].size > 0 else None)
            if isinstance(results["momentum_signal"], np.ndarray):
                print("momentum_signal:", results["momentum_signal"][-1] if results["momentum_signal"].size > 0 else None)
            if isinstance(results["incoming_direction"], np.ndarray):
                print("incoming_direction:", results["incoming_direction"][-1] if results["incoming_direction"].size > 0 else None)
            if isinstance(results["highs"], np.ndarray):
                print("highs:", results["highs"][-1] if results["highs"].size > 0 else None)
            if isinstance(results["lows"], np.ndarray):
                print("lows:", results["lows"][-1] if results["lows"].size > 0 else None)
            if isinstance(results["very_fast_cycle_mood"], np.ndarray):
                print("very_fast_cycle_mood:", results["very_fast_cycle_mood"][-1] if results["very_fast_cycle_mood"].size > 0 else None)
            if isinstance(results["medium_cycle_mood"], np.ndarray):
                print("medium_cycle_mood:", results["medium_cycle_mood"][-1] if results["medium_cycle_mood"].size > 0 else None)
            if isinstance(results["slow_cycle_mood"], np.ndarray):
                print("slow_cycle_mood:", results["slow_cycle_mood"][-1] if results["slow_cycle_mood"].size > 0 else None)
            if isinstance(results["very_slow_cycle_mood"], np.ndarray):
                print("very_slow_cycle_mood:", results["very_slow_cycle_mood"][-1] if results["very_slow_cycle_mood"].size > 0 else None)
            if isinstance(results["very_fast_cycle_length"], np.ndarray):
                print("very_fast_cycle_length:", results["very_fast_cycle_length"][-1] if results["very_fast_cycle_length"].size > 0 else None)
            if isinstance(results["medium_cycle_length"], np.ndarray):
                print("medium_cycle_length:", results["medium_cycle_length"][-1] if results["medium_cycle_length"].size > 0 else None)
            if isinstance(results["slow_cycle_length"], np.ndarray):
                print("slow_cycle_length:", results["slow_cycle_length"][-1] if results["slow_cycle_length"].size > 0 else None)
            if isinstance(results["very_slow_cycle_length"], np.ndarray):
                print("very_slow_cycle_length:", results["very_slow_cycle_length"][-1] if results["very_slow_cycle_length"].size > 0 else None)

            # Calculate very fast cycle length 
            very_fast_cycle_length = np.abs(avg_mtf - close_prices[-1])   

            # Calculate next 3 prices based on cycle length and current price        
            price1 = close_prices[-1] + very_fast_cycle_length    
            price2 = close_prices[-1] + 2 * very_fast_cycle_length   
            price3 = close_prices[-1] + 3 * very_fast_cycle_length

            # Print the next 3 forecast prices
            print(f"Next 3 forecast prices: {price1}, {price2}, {price3}")

            # Get current momentum
            current_momentum = results["momentum"]

            # Calculate very fast cycle momentum
            very_fast_cycle_momentum = 0.75 * current_momentum

            # Calculate medium cycle momentum  
            medium_cycle_momentum = 0.5 * current_momentum

            # Calculate slow cycle momentum
            slow_cycle_momentum = 0.25 * current_momentum

            # Print next 3 momentum forecast cycles    
            print(f"Next 3 momentum forecasts: {very_fast_cycle_momentum}, {medium_cycle_momentum}, {slow_cycle_momentum}")

            # Calculate very fast cycle momentum        
            very_fast_cycle_momentum = 0.75 * current_momentum

            # Determine very fast cycle mood based on momentum        
            if very_fast_cycle_momentum < 0:
                very_fast_cycle_mood = "Bearish"
            else:
                very_fast_cycle_mood = "Bullish"  

            # Calculate medium cycle momentum       
            medium_cycle_momentum = 0.5 * current_momentum

            # Determine medium cycle mood        
            if medium_cycle_momentum < 0:
                medium_cycle_mood = "Bearish"    
            else:
                medium_cycle_mood = "Bullish" 

            # Calculate slow cycle momentum    
            slow_cycle_momentum = 0.25 * current_momentum

            # Determine slow cycle mood
            if slow_cycle_momentum < 0:
                slow_cycle_mood = "Bearish"      
            else:      
                slow_cycle_mood = "Bullish"  

            # Print market mood forecasts
            print(f"Momentum market mood forecasts: {very_fast_cycle_mood}, {medium_cycle_mood}, {slow_cycle_mood}")

            # Compare current momentum to different cycle lengths
            if current_momentum > results['very_fast_cycle_length']:
                print("Current momentum is greater than very fast cycle length.")
            elif current_momentum < results['very_fast_cycle_length']:
                print("Current momentum is less than very fast cycle length.")
            else:
                print("Current momentum is equal to very fast cycle length.")

            if current_momentum > results['medium_cycle_length']:
                print("Current momentum is greater than medium cycle length.")
            elif current_momentum < results['medium_cycle_length']:
                print("Current momentum is less than medium cycle length.")
            else:
                print("Current momentum is equal to medium cycle length.")

            if current_momentum > results['slow_cycle_length']:
                print("Current momentum is greater than slow cycle length.")
            elif current_momentum < results['slow_cycle_length']:
                print("Current momentum is less than slow cycle length.")
            else:
                print("Current momentum is equal to slow cycle length.")

            if current_momentum > results['very_slow_cycle_length']:
                print("Current momentum is greater than very slow cycle length.")
            elif current_momentum < results['very_slow_cycle_length']:
                print("Current momentum is less than very slow cycle length.")
            else:
                print("Current momentum is equal to very slow cycle length.")

            # Compare current price to different cycle lengths
            if np.all(close_prices < results['very_fast_cycle_length']):
                print("Current price is below very fast cycle length.")
            elif np.all(close_prices > results['very_fast_cycle_length']):
                print("Current price is above very fast cycle length.")
            else:
                print("Current price is at very fast cycle length.")

            if np.all(close_prices < results['medium_cycle_length']):
                print("Current price is below medium cycle length.")
            elif np.all(close_prices > results['medium_cycle_length']):
                print("Current price is above medium cycle length.")
            else:
                print("Current price is at medium cycle length.")

            if np.all(close_prices < results['slow_cycle_length']):
                print("Current price is below slow cycle length.")
            elif np.all(close_prices > results['slow_cycle_length']):
                print("Current price is above slow cycle length.")
            else:
                print("Current price is at slow cycle length.")

            if np.all(close_prices < results['very_slow_cycle_length']):
                print("Current price is below very slow cycle length.")
            elif np.all(close_prices > results['very_slow_cycle_length']):
                print("Current price is above very slow cycle length.")
            else:
                print("Current price is at very slow cycle length.")

            print()

            # Convert price1 price2 and price3 to Python float
            price1 = float(price1)
            price2 = float(price2)
            price3 = float(price3)

            # Convert avg_mtf, fastest_target and target1 to Python float
            avg_mtf = float(avg_mtf)
            fastest_target = float(fastest_target)
            target1 = float(target1)

            with open("signals.txt", "a") as f:   
                # Get data and calculate indicators here...
                timestamp = current_time.strftime("%d %H %M %S")

                if price <= min_threshold and price < price1 and price < price2 and price < price3 and very_fast_cycle_mood == "Bullish":
                    if momentum > 0:
                        trigger_long = True

                elif price >= max_threshold and price > price1 and price > price2 and price > price3 and very_fast_cycle_mood == "Bearish":
                    if momentum < 0:
                        trigger_short = True

                if current_quadrant == 1: 

                    # Add percentage difference condition from close to min           
                    if price < price1 and price < price2 and price < price3:
                        if dist_from_close_to_min <= 15:
                            if momentum > 0:
                                if price < avg_mtf and price < fastest_target and price < target1 and market_mood_sr == "Bullish" and very_fast_cycle_mood == "Bullish":
                                    trigger_long = True


                elif current_quadrant == 4: 

                    # Add percentage difference condition from close to max        
                    if price > price1 and price > price2 and price > price3:
                        if dist_from_close_to_max <= 15:
                            if momentum < 0:
                                if price > avg_mtf and price > fastest_target and price > target1 and market_mood_sr == "Bearish" and very_fast_cycle_mood == "Bearish":
                                    trigger_short = True  

                if market_mood_sr == "Bullish" and very_fast_cycle_mood == "Bullish":
                    trigger_long = True

                elif market_mood_sr == "Bearish" and very_fast_cycle_mood == "Bearish":
                    trigger_short = True  
              
                if very_fast_cycle_mood == "Bullish":
                    trigger_long = True

                elif very_fast_cycle_mood == "Bearish":
                    trigger_short = True 

                if trigger_long:          
                    print("LONG signal!")  
                    f.write(f"{timestamp} LONG {price}\n") 
                    trigger_long = False
         
                if trigger_short:
                    print("SHORT signal!")
                    f.write(f"{timestamp} SHORT {price}\n")
                    trigger_short = False

                ##################################################
                ##################################################

            print()

            ##################################################
            ##################################################

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

        time.sleep(5)
        print()

        ##################################################
        ##################################################

print()

##################################################
##################################################

# Run the main function
if __name__ == '__main__':
    main()
