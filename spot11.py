import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

def fetch_candles_in_parallel(timeframes, symbol='BTCUSDC', limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    
    return dict(zip(timeframes, results))

def get_candles(TRADE_SYMBOL, timeframe, limit=100):
    try:
        klines = client.get_klines(symbol=TRADE_SYMBOL, interval=timeframe, limit=limit)
        candles = []
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
    except BinanceAPIException as e:
        print(f"Error fetching candles: {e.message}")
        return []

def get_current_btc_price():
    try:
        ticker = client.get_symbol_ticker(symbol="BTCUSDC")
        return float(ticker['price'])
    except BinanceAPIException as e:
        print(f"Error fetching BTC price: {e.message}")
        return 0.0

def get_balance(asset='USDC'):
    try:
        balance_info = client.get_asset_balance(asset)
        return float(balance_info['free']) if balance_info else 0.0
    except BinanceAPIException as e:
        print(f"Error fetching balance for {asset}: {e.message}")
        return 0.0 

def buy_btc(amount):
    try:
        order = client.order_market_buy(
            symbol='BTCUSDC',
            quantity=amount
        )
        print(f"Market buy order executed: {order}")
        return order
    except BinanceAPIException as e:
        print(f"Error executing buy order: {e.message}")
        return None 

def check_exit_condition(initial_investment, btc_balance):
    current_value = btc_balance * get_current_btc_price()
    return current_value >= (initial_investment * 1.012)

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    close_prices = np.array(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = np.nanmin(momentum)
    max_momentum = np.nanmax(momentum)

    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)

    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)

    avg_mtf = np.nanmean(filtered_close)
    current_momentum = momentum[-1]

    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               
        percent_to_max_momentum = ((current_momentum - min_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan

    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2

    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum

def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            buy_volume[timeframe] = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] > candle["open"])
            sell_volume[timeframe] = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] < candle["open"])
    return buy_volume, sell_volume 

def check_volume_trend(candle_map):
    volume_trend_data = {}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            volume_data = [candle["volume"] for candle in candle_map[timeframe]]
            if len(volume_data) > 1:
                current_volume = volume_data[-1]
                previous_volume = volume_data[-2]
                trend = "Increasing" if current_volume > previous_volume else "Decreasing" if current_volume < previous_volume else "Stable"
                volume_trend_data[timeframe] = {
                    "current_volume": current_volume,
                    "previous_volume": previous_volume,
                    "trend": trend
                }
            else:
                volume_trend_data[timeframe] = {
                    "current_volume": 0,
                    "previous_volume": 0,
                    "trend": "N/A (Insufficient Data)"
                }
    return volume_trend_data 

def find_major_reversals(candles):
    lows = [candle['low'] for candle in candles]
    highs = [candle['high'] for candle in candles]
    last_bottom = min(lows)
    last_top = max(highs)
    return last_bottom, last_top 

def scale_to_sine(close_prices):
    sine_wave, _ = talib.HT_SINE(np.array(close_prices))
    current_sine = np.nan_to_num(sine_wave)[-1]
    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0

    return dist_from_close_to_min, dist_from_close_to_max, current_sine 

def log_trade(timestamp, btc_price, usdc_balance, operation, profit_percentage, usdc_diff):
    """ Log the trade details into a text file. """
    print(f"Trade: Timestamp: {timestamp}, BTC Price: {btc_price:.2f}, USDC Balance: {usdc_balance:.2f}, "
          f"Operation: {operation}, Profit Percentage: {profit_percentage:.2f}%, USDC Difference: {usdc_diff:.2f}")

def calculate_bullish_bearish_ratio(candle_map):
    volume_ratios = {}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            buy_volume = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] > candle["open"])
            sell_volume = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] < candle["open"])
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                volume_ratios[timeframe] = (buy_volume, sell_volume, (buy_volume / total_volume) * 100)
            else:
                volume_ratios[timeframe] = (0, 0, 0)
    return volume_ratios 

def calculate_45_degree_projection(last_bottom, last_top):
    """ Calculate the price projection based on a 45-degree angle. """
    angle_distance = last_top - last_bottom
    projected_price_45 = last_top + angle_distance
    return projected_price_45 

def calculate_golden_ratio_projection(last_close, angle_projection):
    """ Calculate a price projection based on the Golden Ratio rule. """
    phi = 1.618
    projected_price_golden = last_close + (angle_projection - last_close) * phi
    return projected_price_golden 

# Initialize variables for tracking trade state
TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m']

position_open = False
initial_investment = 0.0
btc_balance = 0.0 

while True:  
    current_local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Local Time: {current_local_time}")

    # Fetch candle data
    candle_map = fetch_candles_in_parallel(timeframes)
    usdc_balance = get_balance('USDC')
    current_btc_price = get_current_btc_price()

    # Calculate volume details
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)

    # Analyze each timeframe for its relevant data
    conditions = {
        "current_close_above_min_threshold": False,
        "dip_condition_met": False
    }

    # Initialize condition tracking
    conditions_status = {
        "current_close_above_min_threshold": False,
        "dip_condition_met": False,
        "volume_bullish_1m": False,
        "dist_to_min_less_than_max": False,
        "current_close_below_average_threshold": False
    }

    average_threshold_1m = None  # Store the average threshold for 1m candles

    for timeframe in timeframes:
        if timeframe in candle_map:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            # Store the average threshold for the 1-minute timeframe
            if timeframe == '1m':
                average_threshold_1m = avg_mtf

            # Find major reversals
            last_bottom, last_top = find_major_reversals(candle_map[timeframe])
            print(f"Last Bottom (Major Reversal): {last_bottom:.2f}")
            print(f"Last Top (Major Reversal): {last_top:.2f}")

            # Print relevant details
            print(f"Current Close: {closes[-1]:.2f}")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")

            # Get distances to min and max on SINE
            dist_to_min_sine, dist_to_max_sine, current_sine = scale_to_sine(closes)
            print(f"Distance to Min SINE: {dist_to_min_sine:.2f}%")
            print(f"Distance to Max SINE: {dist_to_max_sine:.2f}%")

            # 45 Degree Angle Price Calculation
            projected_price_45 = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price_45:.2f}")

            # Current price vs. 45-degree angle projection
            if current_btc_price < projected_price_45:
                print(f"Current Price: {current_btc_price:.2f} is BELOW the projected 45-degree angle price.")
            elif current_btc_price > projected_price_45:
                print(f"Current Price: {current_btc_price:.2f} is ABOVE the projected 45-degree angle price.")
            else:
                print(f"Current Price: {current_btc_price:.2f} is EQUAL to the projected 45-degree angle price.")

            # Calculate Golden Ratio Projection
            projected_price_golden = calculate_golden_ratio_projection(closes[-1], projected_price_45)
            print(f"Projected Price Using Golden Ratio: {projected_price_golden:.2f}")

            # Determine market mood based on last major reversals
            if abs(current_btc_price - last_bottom) < abs(current_btc_price - last_top):
                last_major_reversal = f"Last Major Reversal Found is at: DIP ({last_bottom:.2f})"
                market_mood = "Market Mood: Up Cycle" if current_btc_price < last_top else "Market Mood: Sideways"
            else:
                last_major_reversal = f"Last Major Reversal Found is at: TOP ({last_top:.2f})"
                market_mood = "Market Mood: Down Cycle" if current_btc_price > last_bottom else "Market Mood: Sideways"

            print(last_major_reversal)
            print(market_mood)

            # Record condition statuses
            if timeframe == '1m':
                # Check if the volume is bullish on 1m timeframe
                conditions_status["volume_bullish_1m"] = buy_volume['1m'] > sell_volume['1m']

                # Check distance to sine conditions
                dist_to_min_sine_1m, dist_to_max_sine_1m, _ = scale_to_sine(closes)
                if dist_to_min_sine_1m < dist_to_max_sine_1m:
                    conditions_status["dist_to_min_less_than_max"] = True

            if timeframe in ['1m', '5m']:
                # Check distance to sine conditions for both 1m and 5m
                dist_to_min_sine_tf, dist_to_max_sine_tf, _ = scale_to_sine(closes)
                if dist_to_min_sine_tf < dist_to_max_sine_tf:
                    conditions_status["dist_to_min_less_than_max"] = True

            # Check if current close is below the average threshold from 1-minute candles
            if average_threshold_1m is not None:
                conditions_status["current_close_below_average_threshold"] = closes[-1] < average_threshold_1m

            # Update conditions
            conditions_status["current_close_above_min_threshold"] = closes[-1] > min_threshold
            conditions_status["dip_condition_met"] = current_btc_price < projected_price_45  # Update dip condition

            print()  # Extra line for spacing between timeframes

    # Overall conditions summary
    true_conditions_count = sum(conditions_status.values())
    false_conditions_count = len(conditions_status) - true_conditions_count
    print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
    print("Conditions Details:")
    for cond, status in conditions_status.items():
        print(f"{cond}: {'True' if status else 'False'}")

    # Entry Logic
    if not position_open:
        if (conditions_status["current_close_above_min_threshold"] and
                conditions_status["dip_condition_met"] and
                conditions_status["volume_bullish_1m"] and
                conditions_status["dist_to_min_less_than_max"] and
                conditions_status["current_close_below_average_threshold"]):
            if usdc_balance > 0:  # Check if there is enough balance
                amount_to_invest = usdc_balance  # You can modify this logic to invest only a certain amount.
                btc_order = buy_btc(amount_to_invest)
                if btc_order:
                    initial_investment = amount_to_invest
                    btc_balance += amount_to_invest / current_btc_price
                    print(f"New position opened with {amount_to_invest} USDC at price {current_btc_price:.2f}.")
                    position_open = True 

    # Exit Logic
    if position_open and check_exit_condition(initial_investment, btc_balance):
        try:
            sell_order = client.order_market_sell(
                symbol='BTCUSDC',
                quantity=btc_balance
            )
            print(f"Market sell order executed: {sell_order}")
            print(f"Position closed. Sold BTC for USDC: {btc_balance * current_btc_price:.2f}")

            # Calculate profit details
            exit_usdc_balance = get_balance('USDC')
            usdc_diff = exit_usdc_balance - initial_investment
            profit_percentage = (usdc_diff / initial_investment) * 100 if initial_investment > 0 else 0.0

            # Log the trade
            log_trade(current_local_time, current_btc_price, exit_usdc_balance, "EXIT", profit_percentage, usdc_diff)

            position_open = False
            initial_investment = 0.0 
            btc_balance = 0.0 
        except BinanceAPIException as e:
            print(f"Error executing sell order: {e.message}")

    # Clean up references and collect garbage
    del candle_map
    gc.collect()  

    print()
    # Log current account information
    print(f"Current USDC balance: {usdc_balance:.2f} | Current BTC balance: {btc_balance:.4f} BTC | Current BTC price: {current_btc_price:.2f}")
    print()

    time.sleep(5)
