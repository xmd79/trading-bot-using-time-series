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
STOP_LOSS_THRESHOLD = 0.0112 # define 1.12% for stoploss
TAKE_PROFIT_THRESHOLD = 0.0336 # define 3.36% for stoploss
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
        # Get the available account balance and set the leverage to 20x
        bUSD_balance = float(get_account_balance())
        TRADE_LVRG = 20

        # Calculate the maximum order quantity based on the current price
        symbol_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])

        # Calculate the order quantity as the entire available balance
        quantity = round((bUSD_balance / symbol_price), 6)

        # Check that the resulting quantity meets the minimum order quantity for the asset
        min_quantity = float(get_min_order_quantity(symbol))
        if quantity < min_quantity:
            print(f"Order quantity is less than the minimum quantity: {quantity} < {min_quantity}")
            return False

        # Create the long order at market price
        order = client.futures_create_order(
            symbol=symbol,
            side=client.SIDE_BUY,
            type=client.ORDER_TYPE_MARKET,
            quantity=quantity)

        print(f"Long order created for {quantity} {symbol} at market price.")
        return True

    except BinanceAPIException as e:
        print(f"Error creating long order: {e}")
        return False

def entry_short(symbol):
    try:
        # Get the available account balance and set the leverage to 20x
        bUSD_balance = float(get_account_balance())
        TRADE_LVRG = 20

        # Calculate the maximum order quantity based on the current price
        symbol_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])

        # Calculate the order quantity as the entire available balance
        quantity = round((bUSD_balance / symbol_price), 6)

        # Check that the resulting quantity meets the minimum order quantity for the asset
        min_quantity = float(get_min_order_quantity(symbol))
        if quantity < min_quantity:
            print(f"Order quantity is less than the minimum quantity: {quantity} < {min_quantity}")
            return False

        # Create the short order at market price
        order = client.futures_create_order(
            symbol=symbol,
            side=client.SIDE_SELL,
            type=client.ORDER_TYPE_MARKET,
            quantity=quantity)

        print(f"Short order created for {quantity} {symbol} at market price.")
        return True

    except BinanceAPIException as e:
        print(f"Error creating short order: {e}")
        return False

def exit_trade(symbol):
    try:
        total_quantity = 0
        positions = client.futures_position_information(symbol=symbol)

        for position in positions:
            if float(position['positionAmt']) != 0:
                side = 'SELL' if position['positionSide'] == 'LONG' else 'BUY'
                quantity = abs(float(position['positionAmt']))
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=client.ORDER_TYPE_MARKET,
                    quantity=quantity)

                total_quantity += quantity
                print(f"Closed {quantity} {symbol} position with {side} order at market price.")

        print(f"Total of {total_quantity} {symbol} positions closed.")
        return True

    except BinanceAPIException as e:
        print(f"Error closing positions: {e}")
        return False

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
    # Define the sine wave parameters
    amplitude = 100
    period = 60
    phi = math.pi / 2
    cycle_length = period * 4

    # Define the price parameters
    starting_price = 100
    volatility = 0.1
    drift = 0.01

    # Define the time parameters
    current_time = time.time()
    forecast_time_5min = current_time + 5 * 60
    forecast_time_15min = current_time + 15 * 60
    forecast_time_30min = current_time + 30 * 60
    current_time_mmss = time.strftime('%M:%S', time.localtime(current_time))

    # Initialize the price and time variables
    current_price = starting_price
    time_step = 0

    # Initialize the trend direction variables
    trend_direction = "Sideways, reversal incoming"
    season_trend_direction = "Sideways, reversal incoming"
    micro_trend_direction = ""
    macro_trend_direction = ""

    # Initialize the reversal price variables
    low_reversal_price = current_price
    high_reversal_price = current_price

    # Initialize the price forecast variables
    price_forecast_5min = current_price
    price_forecast_15min = current_price
    price_forecast_30min = current_price

    # Initialize the Elliott Wave Oscillator (EMO) variables
    ema_short = 0
    ema_long = 0
    ema_diff = 0
    ema_diff_prev = 0

    # Start the time loop
    while True:
        # Calculate the sine wave value and the current quadrant
        cycle_position = time_step % cycle_length
        cycle_position_normalized = cycle_position / cycle_length
        sine_value = amplitude * math.sin(2 * math.pi * cycle_position_normalized + phi)
        current_quadrant = math.ceil(cycle_position_normalized * 4)

        # Calculate the distance from the current price to the minimum and maximum of the quadrant range
        q_min = amplitude * math.sin(2 * math.pi * (current_quadrant - 1) / 4 + phi)
        q_max = amplitude * math.sin(2 * math.pi * current_quadrant / 4 + phi)
        dist_from_close_to_min = abs(current_price - q_min)
        dist_from_close_to_max = abs(current_price - q_max)

        # Update the price with a random walk
        current_price += drift + volatility * (2 * (0.5 - random.random()))
        time_step += 1

        # Update the trend direction based on the cycle from the minimum of Q1 to the maximum of Q4 and the reverse cycle
        if current_quadrant == 1 and dist_from_close_to_min < 25:
            season_trend_direction = "Uptrend"
            if micro_trend_direction != "Uptrend":
                micro_trend_direction = "Uptrend"
                print("MICRO Trend Direction: Uptrend")
        elif current_quadrant == 2 and dist_from_close_to_max < 25:
            season_trend_direction = "Uptrend"
            if micro_trend_direction != "Uptrend":
                micro_trend_direction = "Uptrend"
                print("MICRO Trend Direction: Uptrend")
        elif current_quadrant == 3 and dist_from_close_to_min < 25:
            season_trend_direction = "Downtrend"
            if micro_trend_direction != "Downtrend":
                micro_trend_direction = "Downtrend"
                print("MICRO Trend Direction: Downtrend")
        elif current_quadrant == 4 and dist_from_close_to_max < 25:
            season_trend_direction = "Downtrend"
            if micro_trend_direction != "Downtrend":
                micro_trend_direction = "Downtrend"
                print("MICRO Trend Direction: Downtrend")
        else:
            season_trend_direction = "Sideways, reversal incoming"
            micro_trend_direction = ""
            if trend_direction != "Sideways, reversal incoming":
                print("MICRO Trend Direction: Sideways, reversal incoming")

        # Update the macro trend direction based on complete cycles from Q1 to Q4 and Q4 to Q1 repetitions
        if (cycle_position_normalized < 0.01 or cycle_position_normalized > 0.99) and season_trend_direction != "Sideways, reversal incoming":
            if macro_trend_direction != season_trend_direction:
                macro_trend_direction = season_trend_direction
                print("MACRO Trend Direction:", macro_trend_direction)

        # Update the low and high reversal prices
        if season_trend_direction == "Uptrend":
            if current_price < low_reversal_price:
                low_reversal_price = current_price
                print("New Low Reversal Price:", low_reversal_price)
            if current_price > high_reversal_price:
                high_reversal_price = current_price
        elif season_trend_direction == "Downtrend":
            if current_price > high_reversal_price:
                high_reversal_price = current_price
                print("New High Reversal Price:", high_reversal_price)
            if current_price < low_reversal_price:
                low_reversal_price = current_price

        # Update the price forecasts
        if time.time() >= forecast_time_5min:
            price_forecast_5min = current_price
            forecast_time_5min += 5 * 60
        if time.time() >= forecast_time_15min:
            price_forecast_15min = current_price
            forecast_time_15min += 15 * 60
        if time.time() >= forecast_time_30min:
            price_forecast_30min = current_price
            forecast_time_30min += 30 * 60

        # Update the Elliott Wave Oscillator (EMO)
        if time_step == 1:
            ema_short = current_price
            ema_long = current_price
        else:
            ema_short = 2 / (2 + 1) * (current_price - ema_short) + ema_short
            ema_long = 2 / (14 + 1) * (current_price - ema_long) + ema_long
            ema_diff_prev = ema_diff
            ema_diff = ema_short - ema_long

        # Print the current status
        print("Time: {} - Price: {:.2f} - Trend Direction: {} - Season Trend Direction: {} - Micro Trend Direction: {} - Low Reversal Price: {:.2f} - High Reversal Price: {:.2f} - Price Forecast (5min): {:.2f} - Price Forecast (15min): {:.2f} - Price Forecast (30min): {:.2f} - EMO: {:.2f}".format(current_time_mmss, current_price, trend_direction, season_trend_direction, micro_trend_direction, low_reversal_price, high_reversal_price, price_forecast_5min, price_forecast_15min, price_forecast_30min, ema_diff))

        # Sleep for 5 second
        time.sleep(5)

# Run the main function
if __name__ == '__main__':
    main()