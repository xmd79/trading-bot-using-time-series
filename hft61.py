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

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Define a function to get the account balance in BUSD
def get_account_balance():
    accounts = client.futures_account_balance()
    for account in accounts:
        if account['asset'] == 'BUSD':
            bUSD_balance = float(account['balance'])
            return bUSD_balance

# Get the USDT balance of the futures account
bUSD_balance = float(get_account_balance())

# Calculate the trade size based on the USDT balance with 20x leverage
TRADE_SIZE = bUSD_balance * 20

# Global variables
TRADE_SYMBOL = 'BTCBUSD'
TRADE_TYPE = ''
TRADE_LVRG = 20
STOP_LOSS_THRESHOLD = 0.0144 # define 1.44% for stoploss
TAKE_PROFIT_THRESHOLD = 0.0144 # define 1.44% for takeprofit
BUY_THRESHOLD = -10
SELL_THRESHOLD = 10
EMA_SLOW_PERIOD = 56
EMA_FAST_PERIOD = 12
closed_positions = []
OPPOSITE_SIDE = {'long': 'SELL', 'short': 'BUY'}

# Initialize variables for tracking trade state
trade_open = False
trade_side = None
trade_entry_pnl = 0
trade_exit_pnl = 0
trade_entry_time = 0
trade_percentage = 0

print()

# Print account balance
print("BUSD Futures balance:", bUSD_balance)

# Define timeframes
timeframes = ['1m', '3m', '5m']
print(timeframes)

print()

# Define start and end time for historical data
start_time = int(time.time()) - (86400 * 30)  # 30 days ago
end_time = int(time.time())

# Fetch historical data for BTCBUSD pair
candles = {}
for interval in timeframes:
    tf_candles = client.futures_klines(symbol=TRADE_SYMBOL, interval=interval, startTime=start_time * 1000, endTime=end_time * 1000)
    candles[interval] = []
    for candle in tf_candles:
        candles[interval].append({
            'timestamp': candle[0],
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        })

# Print the historical data for BTCUSDT pair
#for interval in timeframes:
#    print(f"Data for {interval} interval:")
#    print(candles[interval])

print()


# Create close prices array for each time frame
close_prices = {}
for interval in timeframes:
    close_prices[interval] = np.array([c['close'] for c in candles[interval]], dtype=np.double)
    print(f"Close prices for {interval} time frame:")
    print(close_prices[interval])
    print()

print()

# Global variables
closed_positions = []

print()

def get_mtf_signal(candles, timeframes, percent_to_min=5, percent_to_max=5):
    signals = {}

    # Get the OHLCV data for the 1-minute timeframe
    data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles['1m']], dtype=np.double)

    # Get the HT sine wave indicator for the 1-minute timeframe
    sine, leadsine = talib.HT_SINE(data[:, 3])

    # Normalize the HT sine wave indicator to the minimum and maximum prices in the market data
    min_price = np.nanmin(data[:, 3])
    max_price = np.nanmax(data[:, 3])
    norm_sine = (sine - min_price) / (max_price - min_price)

    # Get the minimum and maximum values of the normalized HT Sine Wave indicator
    min_sine = np.nanmin(norm_sine)
    max_sine = np.nanmax(norm_sine)

    # Calculate the percentage distance from the current close on sine to the minimum and maximum values of the normalized HT Sine Wave indicator
    close = data[-1][-2]
    percent_to_min_val = (max_sine - norm_sine[-1]) / (max_sine - min_sine) * 100
    percent_to_max_val = (norm_sine[-1] - min_sine) / (max_sine - min_sine) * 100

    # Print percentages
    print(f"Current close on sine is {percent_to_min_val:.2f}% away from the minimum value")
    print(f"Current close on sine is {percent_to_max_val:.2f}% away from the maximum value")
    print()

    # Calculate the distance from the current momentum to the closest reversal keypoint
    if norm_sine[-1] >= max_sine - (max_sine - min_sine) * percent_to_max / 100:
        mtf_signal = "bearish"
        reversal_keypoint = max_sine
        momentum_distance_min = 100 * ((close - max_sine) / (max_price - min_price))
        momentum_distance_max = 100 * ((close - min_sine) / (max_price - min_price))
    elif norm_sine[-1] <= min_sine + (max_sine - min_sine) * percent_to_min / 100:
        mtf_signal = "bullish"
        reversal_keypoint = min_sine
        momentum_distance_min = 100 * ((min_sine - close) / (max_price - min_price))
        momentum_distance_max = 100 * ((max_sine - close) / (max_price - min_price))
    else:
        # Calculate the average percentage across all timeframes
        if signals and len(signals) > 0:
            avg_percent = sum([signals[tf] for tf in signals]) / len(signals)
        else:
            avg_percent = 0.0

        # Calculate the distance between the average percentage and the minimum and maximum percentages
        dist_to_min = abs(avg_percent - percent_to_min_val)
        dist_to_max = abs(avg_percent - percent_to_max_val)

        if dist_to_min < dist_to_max:
            mtf_signal = "bullish"
        else:
            mtf_signal = "bearish"

        reversal_keypoint = None
        momentum_distance_min = None
        momentum_distance_max = None

    # Store the percentage distance for each timeframe in the signals dictionary
    for tf in timeframes:
        # Get the OHLCV data for the specified timeframe
        tf_data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles[tf]], dtype=np.double)

        # Get the HT sine wave indicator for the specified timeframe
        tf_sine, tf_leadsine = talib.HT_SINE(tf_data[:, 3])

        # Normalize the HT sine wave indicator to the minimum and maximum prices in the market data
        tf_min_price = np.nanmin(tf_data[:, 3])
        tf_max_price = np.nanmax(tf_data[:, 3])
        tf_norm_sine = (tf_sine - tf_min_price) / (tf_max_price - tf_min_price)

        # Get the minimum and maximum values of the normalized HT Sine Wave indicator
        tf_min_sine = np.nanmin(tf_norm_sine)
        tf_max_sine = np.nanmax(tf_norm_sine)

        # Calculate the percentage distance from the current close on the sine wave to the minimum and maximum values of the normalized HT Sine Wave indicator
        tf_close = tf_data[-1][-2]
        tf_percent_to_min = (tf_max_sine - tf_norm_sine[-1]) / (tf_max_sine - tf_min_sine) * 100
        tf_percent_to_max = (tf_norm_sine[-1] - tf_min_sine) / (tf_max_sine - tf_min_sine) * 100

        # Store the percentage distance in the signals dictionary
        signals[tf] = tf_percent_to_min if mtf_signal == "bullish" else tf_percent_to_max

    return signals, mtf_signal

# Get the MTF signals
signals, mtf_signal = get_mtf_signal(candles, timeframes, percent_to_min=5, percent_to_max=5)

# Print the signals for all timeframes
print("MTF signals:")

print()
for tf, signal in signals.items():
    print(f"{tf} - {signal}")

print()
# Print the buy/sell signal based on the MTF signals
print("MTF buy/sell signal:", mtf_signal)

print()

def check_mtf_signal(candles, timeframes, mtf_signal):
    signal = "No Signal"
    # Get the OHLCV data for the 1-minute timeframe
    data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles['1m']], dtype=np.double)

    # Get the HT sine wave indicator for the 1-minute timeframe
    sine, leadsine = talib.HT_SINE(data[:, 3])

    # Normalize the HT sine wave indicator to the minimum and maximum prices in the market data
    min_price = np.nanmin(data[:, 3])
    max_price = np.nanmax(data[:, 3])
    norm_sine = (sine - min_price) / (max_price - min_price)
    norm_leadsine = (leadsine - min_price) / (max_price - min_price)

    # Get the minimum and maximum values of the normalized HT Sine Wave indicator
    min_sine = np.nanmin(norm_sine)
    max_sine = np.nanmax(norm_sine)

    # Calculate the time difference between the minimum and maximum values
    if np.isnan(min_sine) or np.isnan(max_sine):
        cycle_time_str = "N/A"
    else:
        cycle_time = int(abs(np.nanargmax(norm_sine) - np.nanargmin(norm_sine)) * 0.25)
        cycle_time_str = str(timedelta(minutes=cycle_time, seconds=0)).split(".")[0]

        # Calculate the time remaining until the cycle completes
        remaining_time = cycle_time % 30
        if remaining_time == 0:
            remaining_time = 30

    # Check if the sine wave fits the market cycle
    close = data[-1][-2]
    if norm_sine[-1] == min_sine and close <= np.nanmin(data[-timeframes['1m']*30:, 3]):
        print("Close is near the last low on price. Sine wave fits the market cycle.")
    elif norm_sine[-1] == max_sine and close >= np.nanmax(data[-timeframes['1m']*30:, 3]):
        print("Close is near the last high on price. Sine wave fits the market cycle.")
    else:
        print("Sine wave momentum 1min tf does not fit the market cycle reversals but in range between key points...seeking reversal")

    print()

    # Calculate the percentage distance from the current close on sine to the minimum and maximum values of the normalized HT Sine Wave indicator
    percent_to_min = 100 * ((max_sine - norm_sine[-1]) / (max_sine - min_sine))
    percent_to_max = 100 * ((norm_sine[-1] - min_sine) / (max_sine - min_sine))

    # Print percentages
    print(f"Current close on sine is {percent_to_min:.2f}% away from the minimum value")
    print(f"Current close on sine is {percent_to_max:.2f}% away from the maximum value")
    print()

    # Calculate the distance from the current momentum to the closest reversal keypoint
    if mtf_signal == "bearish":
        reversal_keypoint = max_sine
        momentum_distance_min = 100 * ((close - max_sine) / (max_price - min_price))
        momentum_distance_max = 100 * ((close - min_sine) / (max_price - min_price))
    else:
        reversal_keypoint = min_sine
        momentum_distance_min = 100 * ((min_sine - close) / (max_price - min_price))
        momentum_distance_max = 100 * ((max_sine - close) / (max_price - min_price))

    # Calculate the range between 0 to 100% from close to first reversal incoming closest to current value of close on sine
    if mtf_signal == "bearish":
        momentum_range = np.arange(norm_sine[-1], max_sine + 0.0001, (max_sine - norm_sine[-1]) / 100)
    else:
        momentum_range = np.arange(min_sine - 0.0001, norm_sine[-1], (norm_sine[-1] - min_sine) / 100)

    # Determine the trade signal based on momentum and trend signals
    if mtf_signal == "bearish" and norm_sine[-1] >= reversal_keypoint:
        signal = "bearish"
    elif mtf_signal == "bullish" and norm_sine[-1] <= reversal_keypoint:
        signal = "bullish"
    else:
        if percent_to_min > 80:
            signal = "Momentum Bearish"
        elif percent_to_max > 80:
            signal = "Momentum Bullish"
    
    print()
    return signal, momentum_distance_min, momentum_distance_max, momentum_range, cycle_time_str, remaining_time, percent_to_min, percent_to_max

mtf = check_mtf_signal(candles, timeframes, mtf_signal)
print(mtf[0])
print()

def get_mtf_signal_v2(candles, timeframes, percent_to_min=5, percent_to_max=5):
    signals = {}
    
    # Get the OHLCV data for the 1-minute timeframe
    data_1m = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles['1m']], dtype=np.double)
    
    # Get the HT sine wave indicator for the 1-minute timeframe
    sine, leadsine = talib.HT_SINE(data_1m[:, 3])
    
    # Normalize the HT sine wave indicator to the minimum and maximum prices in the market data
    min_price = np.nanmin(data_1m[:, 3])
    max_price = np.nanmax(data_1m[:, 3])
    norm_sine = (sine - min_price) / (max_price - min_price)
    
    # Get the minimum and maximum values of the normalized HT Sine Wave indicator
    min_sine = np.nanmin(norm_sine)
    max_sine = np.nanmax(norm_sine)
    
    # Calculate the percentage distance from the current close on sine to the minimum and maximum values of the normalized HT Sine Wave indicator
    close = data_1m[-1][-2]
    percent_to_min_val = (max_sine - norm_sine[-1]) / (max_sine - min_sine) * 100
    percent_to_max_val = (norm_sine[-1] - min_sine) / (max_sine - min_sine) * 100
    
    for timeframe in timeframes:
        # Get the OHLCV data for the given timeframe
        ohlc_data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles[timeframe]], dtype=np.double)
        
        # Calculate the momentum signal for the given timeframe
        close_prices = ohlc_data[:, 3]
        momentum = talib.MOM(close_prices, timeperiod=14)
        
        # Calculate the minimum and maximum values for the momentum signal
        min_momentum = np.nanmin(momentum)
        max_momentum = np.nanmax(momentum)
        
        # Calculate the percentage distance from the current momentum to the minimum and maximum values of the momentum signal
        current_momentum = momentum[-1]
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * 100
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * 100
        
        # Calculate the new momentum signal based on percentages from the MTF signal and the initial momentum signal
        percent_to_min_combined = (percent_to_min_val + percent_to_min_momentum) / 2
        percent_to_max_combined = (percent_to_max_val + percent_to_max_momentum) / 2
        momentum_signal = percent_to_max_combined - percent_to_min_combined
        
        # Calculate the new average for the MTF signal based on the percentage distance from the current close to the minimum and maximu values of the normalized HT Sine Wave indicator, and the given percentage thresholds
        min_mtf = np.nanmin(ohlc_data[:, 3])
        max_mtf = np.nanmax(ohlc_data[:, 3])
        percent_to_min_custom = percent_to_min / 100
        percent_to_max_custom = percent_to_max / 100
        min_threshold = min_mtf + (max_mtf - min_mtf) * percent_to_min_custom
        max_threshold = max_mtf - (max_mtf - min_mtf) * percent_to_max_custom
        filtered_close = np.where(ohlc_data[:, 3] < min_threshold, min_threshold, ohlc_data[:, 3])
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
        avg_mtf = np.nanmean(filtered_close)
        
        # Store the signals for the given timeframe
        signals[timeframe] = {'momentum': momentum_signal, 'ht_sine_percent_to_min': percent_to_min_val, 'ht_sine_percent_to_max': percent_to_max_val, 'mtf_average': avg_mtf, 'min_threshold': min_threshold, 'max_threshold': max_threshold}
    
    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=3)

    # Print the results
    print("Current time:", current_time.strftime('%Y-%m-%d %H:%M:%S'))
    print(f"HT Sine Wave Percent to Min: {percent_to_min_val:.2f}%")
    print(f"HT Sine Wave Percent to Max: {percent_to_max_val:.2f}%")
    print(f"Momentum Percent to Min: {percent_to_min_momentum:.2f}%")
    print(f"Momentum Percent to Max: {percent_to_max_momentum:.2f}%")
    print(f"Combined Percent to Min: {percent_to_min_combined:.2f}%")
    print(f"Combined Percent to Max: {percent_to_max_combined:.2f}%")
    print(f"New Momentum Signal: {momentum_signal:.2f}")
    print(f"New MTF Average:")
    for timeframe in timeframes:
        print(f"{timeframe}: {signals[timeframe]['mtf_average']:.2f} (min threshold: {signals[timeframe]['min_threshold']:.2f}, max threshold: {signals[timeframe]['max_threshold']:.2f})")
    print()

    return signals

get_mtf_signal_v2(candles, timeframes, percent_to_min=5, percent_to_max=5)

def get_historical_candles(symbol, start_time, end_time, timeframe):
    candles = client.futures_klines(symbol=symbol, interval=timeframe, startTime=start_time * 1000, endTime=end_time * 1000)
    candles_by_timeframe = {}
    for tf in ['1m', '3m', '5m']:
        if tf == timeframe:
            candles_by_timeframe[tf] = [ {'open': float(candle[1]), 'high': float(candle[2]), 'low': float(candle[3]), 'close': float(candle[4]), 'volume': float(candle[5])} for candle in candles ]
        else:
            resampled_candles = []
            for i in range(0, len(candles), int(tf[:-1])):
                candles_chunk = candles[i:i+int(tf[:-1])]
                if len(candles_chunk) == int(tf[:-1]):
                    open_price = float(candles_chunk[0][1])
                    high_price = max([float(candle[2]) for candle in candles_chunk])
                    low_price = min([float(candle[3]) for candle in candles_chunk])
                    close_price = float(candles_chunk[-1][4])
                    total_volume = sum([float(candle[5]) for candle in candles_chunk])
                    resampled_candles.append({'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price, 'volume': total_volume})
            candles_by_timeframe[tf] = resampled_candles

    return candles_by_timeframe

def get_current_price(symbol):
    ticker = client.futures_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def get_min_order_quantity(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        symbol_info = next(filter(lambda x: x['symbol'] == symbol, exchange_info['symbols']))
        min_qty = float(symbol_info['filters'][2]['minQty'])
        return min_qty
    except Exception as e:
        print(f"Error getting minimum order quantity for {symbol}: {e}")
        return None

def entry_long(symbol):
    try:     
        # Get balance and leverage     
        account_balance = get_account_balance()   
        trade_leverage = 20
    
        # Get symbol price    
        symbol_price = client.futures_symbol_ticker(symbol=symbol)['price']
        
        # Get step size from exchange info
        info = client.futures_exchange_info()
        filters = [f for f in info['symbols'] if f['symbol'] == symbol][0]['filters']
        step_size = [f['stepSize'] for f in filters if f['filterType']=='LOT_SIZE'][0]
                    
        # Calculate max quantity based on balance, leverage, and price
        max_qty = int(account_balance * trade_leverage / float(symbol_price) / float(step_size)) * float(step_size)  
                    
        # Create buy market order    
        order = client.futures_create_order(
            symbol=symbol,        
            side='BUY',           
            type='MARKET',         
            quantity=max_qty)          
                    
        if 'orderId' in order:
            return True
          
        else: 
            print("Error creating long order.")  
            return False
            
    except BinanceAPIException as e:
        print(f"Error creating long order: {e}")
        return False

def entry_short(symbol):
    try:     
        # Get balance and leverage     
        account_balance = get_account_balance()   
        trade_leverage = 20
    
        # Get symbol price    
        symbol_price = client.futures_symbol_ticker(symbol=symbol)['price']
        
        # Get step size from exchange info
        info = client.futures_exchange_info()
        filters = [f for f in info['symbols'] if f['symbol'] == symbol][0]['filters']
        step_size = [f['stepSize'] for f in filters if f['filterType']=='LOT_SIZE'][0]
                    
        # Calculate max quantity based on balance, leverage, and price
        max_qty = int(account_balance * trade_leverage / float(symbol_price) / float(step_size)) * float(step_size)  
                    
        # Create sell market order    
        order = client.futures_create_order(
            symbol=symbol,        
            side='SELL',           
            type='MARKET',         
            quantity=max_qty)          
                    
        if 'orderId' in order:
            return True
          
        else: 
            print("Error creating short order.")  
            return False
            
    except BinanceAPIException as e:
        print(f"Error creating short order: {e}")
        return False

def exit_trade():
    # Get all open positions
    positions = client.futures_position_information()
    
    # Loop through each position
    for position in positions:
        symbol = position['symbol']
        position_amount = float(position['positionAmt'])
        
        # Determine order side
        if position['positionSide'] == 'LONG':
            order_side = 'SELL'
        else: 
            order_side = 'BUY'  
            
        # Place order to exit position      
        if position_amount != 0:
            order = client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='MARKET',
                quantity=abs(position_amount))
                
            print(f"{order_side} order created to exit {abs(position_amount)} {symbol}.")
                
    print("All positions exited!")

def calculate_ema(candles, period):
    prices = [float(candle['close']) for candle in candles]
    ema = []
    sma = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)
    ema.append(sma)
    for price in prices[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

print()
print("Init main() loop: ")
print()

def main():
    # Load credentials from file
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()

    # Instantiate Binance client
    client = BinanceClient(api_key, api_secret)

    # Define EM amplitude variables for each quadrant
    em_amp_q1 = 0
    em_amp_q2 = 0
    em_amp_q3 = 0
    em_amp_q4 = 0

    # Define EM phase variables for each quadrant
    em_phase_q1 = 0
    em_phase_q2 = math.pi/2
    em_phase_q3 = math.pi
    em_phase_q4 = 3*math.pi/2

    # Define minimum and maximum values for the sine wave
    sine_wave_min = -1
    sine_wave_max = 1

    # Define min and max values for percentages from close to min and max of talib HT_SINE
    percent_to_min_val = 10
    percent_to_max_val = 10

    trade_open = False
    current_quadrant = None
    trade_entry_pnl = 0
    trade_exit_pnl = 0
    trade_side = None

    # Define constants
    trade_symbol = "BTCBUSD"
     
    fast_ema = 12       
    slow_ema = 26         
    
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

    # Calculate frequencies spectrum index
    frequencies = []    
    frequencies_next = [] 

    range = 26

    for i in __builtins__.range(1,26):
        frequency = i * sacred_freq
        
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
      
        frequencies_next.append({
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


    # Define trade variables    
    position = None   
    trade_open = False       
    trade_side = None
    entry_price = 0
    entry_time = 0

    trade_exit_triggered = False

    url = "https://api.binance.com/api/v3/time"
    while True:       
        try:   
            # Get the server time
            server_time = requests.get('https://api.binance.com/api/v3/time').json()['serverTime']
            
            # Calculate the timestamp of your request       
            timestamp = int(server_time - 500) # Subtract5 seconds
            
            # Make your request with the adjusted timestamp
            response = requests.get(url, params={'timestamp': timestamp})
            response = requests.get('https://api.binance.com/api/v3/time', auth=(api_key, api_secret))
            print(response.json())

            # Define current_quadrant variable
            current_quadrant = 0

            # Define start and end time for historical data
            start_time = int(time.time()) - (1800 * 4)  # 60-minute interval (4 candles)
            end_time = int(time.time())

            # Define the candles and timeframes to use for the signals
            candles = get_historical_candles(TRADE_SYMBOL, start_time, end_time, '1m')
            timeframes = ['1m', '3m', '5m']

            # Check if candles is empty
            if not candles:
                print("Error: No historical candles found.")
                continue

            bUSD_balance = float(get_account_balance())
            print("My BUSD balance from futures wallet is at: ", bUSD_balance)

            # Get the MTF signal
            signals = get_mtf_signal_v2(candles, timeframes, percent_to_min=1, percent_to_max=1)
            print(signals)
            
            print()

            initial_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])

            stop_loss = -0.0144 * initial_pnl
            take_profit = 0.0144 * initial_pnl

            # Check if the '1m' key exists in the signals dictionary
            if '1m' in signals:
                # Check if the percent to min/max signal keys exist in the '1m' dictionary
                if 'ht_sine_percent_to_min' in signals['1m'] and 'ht_sine_percent_to_max' in signals['1m']:
                    percent_to_min_val = signals['1m']['ht_sine_percent_to_min']
                    percent_to_max_val = signals['1m']['ht_sine_percent_to_max']

                    close_prices = np.array([candle['close'] for candle in candles['1m']])
                    print("Close price:", close_prices[-1])

                    # Calculate the sine wave using HT_SINE
                    sine_wave, _ = talib.HT_SINE(close_prices)

                    # Replace NaN values with 0 using nan_to_num
                    sine_wave = np.nan_to_num(sine_wave)
                    sine_wave = -sine_wave

                    print("Current close on Sine wave:", sine_wave[-1])

                    # Calculate the minimum and maximum values of the sine wave
                    sine_wave_min = np.min(sine_wave)
                    sine_wave_max = np.max(sine_wave)

                    # Calculate the distance from close on sine to min and max as percentages on a scale from 0 to 100%
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
                    em_phase_diff_q1_q2 = em_phase_q2 - em_phase_q1
                    em_phase_diff_q2_q3 = em_phase_q3 - em_phase_q2
                    em_phase_diff_q3_q4 = em_phase_q4 - em_phase_q3
                    em_phase_diff_q4_q1 = 2*math.pi - (em_phase_q4 - em_phase_q1)

                    # Check if EMA periods have been defined
                    if EMA_SLOW_PERIOD and EMA_FAST_PERIOD:
                        # Calculate the EMAs
                        ema_slow = talib.EMA(close_prices, timeperiod=EMA_SLOW_PERIOD)[-1]
                        ema_fast = talib.EMA(close_prices, timeperiod=EMA_FAST_PERIOD)[-1]

                        print("EMA slow:", ema_slow)
                        print("EMA fast:", ema_fast)

                        close_prices = np.array([candle['close'] for candle in candles['1m']])

                        #Calculate X:Y ratio from golden ratio     
                        ratio = 2 * (PHI - 1)  

                        # Assign EM amplitudes based on X:Y ratio
                        em_amp_q1 = (sine_wave_max - sine_wave_min) * ratio / 4
                        em_amp_q2 = (sine_wave_max - sine_wave_min) * ratio / 4 
                        em_amp_q3 = (sine_wave_max - sine_wave_min) * ratio / 4 
                        em_amp_q4 = (sine_wave_max - sine_wave_min) * ratio / 4  

                        # Calculate EM phases based on dividing whole range by X:Y ratio  
                        em_phase_q1 = 0  
                        em_phase_q2 = PI * ratio / 2 
                        em_phase_q3 = PI * ratio           
                        em_phase_q4 = PI * ratio * 1.5  

                        #Calculate midpoints of each quadrant
                        mid_q1 = sine_wave_min + em_amp_q1 / 2
                        mid_q2 = mid_q1 + em_amp_q1
                        mid_q3 = mid_q2 + em_amp_q2
                        mid_q4 = max(sine_wave)

                        #Compare current sine wave value to determine quadrant
                        if sine_wave.any() <= mid_q1:
                            current_quadrant = 1
                            current_em_amp = em_amp_q1
                            current_em_phase = em_phase_q1
                        elif sine_wave.any() <= mid_q2:
                            current_quadrant = 2
                            current_em_amp = em_amp_q2
                            current_em_phase = em_phase_q2
                        elif sine_wave.any() <= mid_q3:
                            current_quadrant = 3
                            current_em_amp = em_amp_q3
                            current_em_phase = em_phase_q3
                        elif sine_wave.any() <= mid_q4:
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


                        # Check if the current price is above the EMAs and the percent to min signals are below 20%
                        if close_prices[-1] < ema_slow and close_prices[-1] < ema_fast and percent_to_min_val < 20:
                            print("Buy signal!")

                        # Check if the current price is below the EMAs and the percent to max signals are below 20%
                        elif close_prices[-1] > ema_slow and close_prices[-1] > ema_fast and percent_to_max_val < 20:
                            print("Sell signal!")

                        elif percent_to_min_val < 20:
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

                        print()

                        next_1h_forecast = []

                        for freq in frequencies:

                            # Get PHI raised to the frequency number         
                            phi_power = PHI ** freq['number'] 

                            if phi_power < 1.05:
                                freq['mood_next_1h'] = 'extremely positive'
                            elif phi_power < 1.2:       
                                freq['mood_next_1h'] = 'strongly positive'
                            elif phi_power < 1.35:       
                                freq['mood_next_1h'] = 'positive'    
                            elif phi_power < 1.5:       
                                freq['mood_next_1h'] = 'slightly positive'
                            elif phi_power < 2:          
                                freq['mood_next_1h'] = 'neutral'              
                            elif phi_power < 2.5:     
                                freq['mood_next_1h'] = 'slightly negative'      
                            elif phi_power < 3.5:     
                                freq['mood_next_1h'] = 'negative'
                            elif phi_power < 4.5:     
                                freq['mood_next_1h'] = 'strongly negative'   
                            else:                     
                                freq['mood_next_1h'] = 'extremely negative'

                            next_1h_forecast.append(freq)

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

                        quadrant_1_amplitude = 0.5
                        quadrant_1_phase = 0.2

                        quadrant_2_amplitude = 1.0  
                        quadrant_2_phase = 0.5

                        quadrant_3_amplitude = 0.8        
                        quadrant_3_phase = 0.7

                        quadrant_4_amplitude = 1.2
                        quadrant_4_phase = 0.9

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
                        next_1h_forecast.sort(key=lambda f: mood_map[f['mood_next_1h']])  
    
                        # Get average mood of highest/lowest 3 frequencies 
                        highest_3 = next_1h_forecast[:3]
                        highest_3_mood = 0
   
                        for freq in highest_3:
                            mood_val  = mood_map[freq['mood_next_1h']] 
                            highest_3_mood += mood_val
                            highest_3_mood = highest_3_mood / len(highest_3)

                        lowest_3 = next_1h_forecast[-3:]
                        lowest_3_mood = 0

                        for freq in lowest_3:
                            mood_val  = mood_map[freq['mood_next_1h']]
                            lowest_3_mood += mood_val  
                            lowest_3_mood = lowest_3_mood / len(lowest_3)

                        for freq in next_1h_forecast:
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
                        top_n_weights = [freq['magnitude'] for freq in next_1h_forecast[:n]]
                        top_n_moods = [mood_map[freq['mood_next_1h']] for freq in next_1h_forecast[:n]]
                        top_n_weighted_avg = calculate_weighted_avg(top_n_weights, top_n_moods)

                        bottom_n_weights = [freq['magnitude'] for freq in next_1h_forecast[-n:]]  
                        bottom_n_moods = [mood_map[freq['mood_next_1h']] for freq in next_1h_forecast[-n:]]
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
                        print()

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

                        brun_low  = close_prices[-1] * brun_constant  
                        brun_high = close_prices[-1] / brun_constant 
                        #print(brun_low)
                        #print(brun_high)

                        sma50 = talib.SMA(np.array(close_prices), timeperiod=50)
                        sma50 = sma50[-1]  
                        print("SMA50 is now at: ", sma50)

                        sma100 = talib.SMA(np.array(close_prices), timeperiod=100)
                        sma100 = sma100[-1]
                        print("SMA100 is now at: ", sma100)

                        phi_ratio = sma100 / sma50

                        deviation = close_prices[-1] - sma50  
                        threshold = 0.03

                        if deviation > threshold: # Reversal signal  
      
                            if close_prices[-1] > brun_high: # Above Brun level 
                                if phi_ratio < 1.2:  
        
                                    # Near term       
                                    forecast = f"Continuation likely for {15*phi_ratio} periods"      
                                    print(forecast)
                                elif 1.2 <= phi_ratio < 1.5:       
        
                                    # Medium term        
                                    range = (brun_high - brun_low) * 0.5
                                    forecast = f"Continuation likely {range*phi_ratio} points in the medium term"  
                                    print(forecast)
                                else:  
        
                                    # Long term     
                                    range = brun_high - brun_low
                                    forecast = f"Strong continuation likely {range*phi_ratio} points or more"
                                    print(forecast)
                            elif close_prices[-1] < brun_low: # Below Brun level  # Downtrend reversal signal    
        
                                if phi_ratio < 1.2:  
          
                                    forecast = f"Reversal likely in {15*phi_ratio} periods"
                                    print(forecast)
                                elif 1.2 <= phi_ratio < 1.5:
          
                                    range = sma50 - sma100
                                    forecast = f"Reversal of up to {range*phi_ratio} points possible in medium term"
                                    print(forecast)
                                else:
          
                                    forecast = f"Reversal of {sma100*phi_ratio} points or more possible in long term" 
                                    print(forecast)
                        elif deviation < threshold:  # No reversal signal
                         
                            if close_prices[-1] > sma50: # Uptrend
                   
                                if phi_ratio < 1.2:   
                                    forecast = "Uptrend likely to continue in near term"   
                                    print(forecast)
                                elif 1.2 <= phi_ratio < 1.5:               
                                    range = sma50 - sma100
                                    forecast = f"Uptrend likely to continue {range * phi_ratio} points in medium term"
                                    print(forecast)
                                else:               
                                    forecast = f"Uptrend likely to continue {sma100 * phi_ratio} points or more in long term"
                                    print(forecast)
                            elif close_prices[-1] > sma50: # Downtrend       
              
                                if phi_ratio < 1.2:  
                                    forecast = "Downtrend likely to continue in near term" 
                                    print(forecast)
                                elif 1.2 <= phi_ratio < 1.5:              
                                    range = sma100 - sma50       
                                    forecast ="Downtrend likely to continue {range * phi_ratio} points in medium term"
                                    print(forecast)
                                else:                 
                                    forecast = f"Downtrend likely to continue {sma100 * phi_ratio} points or more in long term"
                                    print(forecast)

                        print()

                        # Get all open positions
                        positions = client.futures_position_information()

                        # Loop through each position
                        for position in positions:
                            symbol = position['symbol']
                            position_amount = float(position['positionAmt'])

                        # Print position if there is nor not     
                        if position_amount != 0:
                            print("Position open: ", position_amount)
                       
                        elif position_amount == 0:
                            print("Position not open: ", position_amount)

                        print(f"Current PNL: {float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])}, Entry PNL: {trade_entry_pnl}, Exit PNL: {trade_exit_pnl}")

                        print()
                else:
                    print("Error: 'ht_sine_percent_to_min' or 'ht_sine_percent_to_max' keys not found in signals dictionary.")
            else:
                print("Error: '1m' key not found in signals dictionary.")

            time.sleep(5) # Sleep for 5 seconds      
                
        except BinanceAPIException as e:  
            print(e)

        except Exception as e:      
            print(f"An error occurred: {e}")    
            time.sleep(5) 

# Run the main function
if __name__ == '__main__':
    main()
