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
#print(closes)

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

print("Current close on sine value now at: ", current_sine)
print("distances as percentages from close to min: ", dist_from_close_to_min, "%")
print("distances as percentages from close to max: ", dist_from_close_to_max, "%")
print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
print("Mood on 1min timeframe is now at: ", market_mood[-12])

print()

##################################################
##################################################