import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, filename='trading_bot.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.error(f"Error fetching candles: {e.message}")
        return []

def get_current_btc_price():
    try:
        ticker = client.get_symbol_ticker(symbol="BTCUSDC")
        return float(ticker['price'])
    except BinanceAPIException as e:
        logging.error(f"Error fetching BTC price: {e.message}")
        return 0.0

def get_balance(asset='USDC'):
    try:
        balance_info = client.get_asset_balance(asset)
        return float(balance_info['free']) if balance_info else 0.0
    except BinanceAPIException as e:
        logging.error(f"Error fetching balance for {asset}: {e.message}")
        return 0.0 

def buy_btc(amount):
    try:
        order = client.order_market_buy(
            symbol='BTCUSDC',
            quantity=amount
        )
        logging.info(f"Market buy order executed: {order}")
        return order
    except BinanceAPIException as e:
        logging.error(f"Error executing buy order: {e.message}")
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

def log_entry_signal(timestamp, btc_price, projected_price):
    """ Log the entry signal details into a text file, keeping only the last entry. """
    try:
        with open("entry_signals.txt", "w") as f:  # Opens in write mode to overwrite existing content
            f.write(f"Timestamp: {timestamp}, BTC Price: {btc_price:.2f}, Projected Price: {projected_price:.2f}\n")
        print(f"Logged entry signal: Timestamp: {timestamp}, BTC Price: {btc_price:.2f}, Projected Price: {projected_price:.2f}")
    except IOError as e:
        logging.error(f"Error writing to entry_signals.txt: {e}")
        print(f"Error writing to entry_signals.txt: {e}")

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
    logging.info(f"Current Local Time: {current_local_time}")

    # Fetch candle data
    candle_map = fetch_candles_in_parallel(timeframes)
    usdc_balance = get_balance('USDC')
    current_btc_price = get_current_btc_price()

    # Calculate volume details
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_trend_data = check_volume_trend(candle_map)

    # Calculate volume ratios
    volume_ratios = calculate_bullish_bearish_ratio(candle_map)

    # Analyze each timeframe for its relevant data
    entry_trigger_conditions = {
        "below_angle": False,
        "min_dist": False,
        "sine_dist": False,
        "buy_volume": False,
        "forecast_condition": False,
        "dist_to_min_sine_1m": False,
        "dist_to_max_sine_1m": False,
        "dist_to_min_sine_5m": False,
        "dist_to_max_sine_5m": False,
        "dip_condition_met": False,
        "current_close_above_min_threshold": False,
    }

    min_threshold_1m = None
    last_bottom_1m = None 

    for timeframe in timeframes:
        if timeframe in candle_map:
            # Print volume conditions for 1m, 3m, and 5m
            buy_vol = volume_ratios[timeframe][0]
            sell_vol = volume_ratios[timeframe][1]
            if buy_vol > sell_vol:
                print(f"{timeframe} Volume Condition: Bullish (Buy Volume: {buy_vol:.2f}, Sell Volume: {sell_vol:.2f})")
            else:
                print(f"{timeframe} Volume Condition: Bearish (Buy Volume: {buy_vol:.2f}, Sell Volume: {sell_vol:.2f})")

            # Calculate thresholds
            closes = [candle['close'] for candle in candle_map[timeframe]]
            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            # Store Minimum Threshold and major reversal from the 1m timeframe
            if timeframe == '1m':
                min_threshold_1m = min_threshold
                last_bottom_1m = find_major_reversals(candle_map[timeframe])[0]

            # Print relevant details
            print(f"--- {timeframe} ---")
            print(f"Current Close: {closes[-1]:.2f}")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")
            print(f"Volume Trend: {volume_trend_data[timeframe]['trend']}")

            # Get distances to min and max on SINE
            dist_to_min_sine, dist_to_max_sine, current_sine = scale_to_sine(closes)
            print(f"Distance to Min SINE: {dist_to_min_sine:.2f}%")
            print(f"Distance to Max SINE: {dist_to_max_sine:.2f}%")

            # Store the distances based on timeframe
            if timeframe == '1m':
                entry_trigger_conditions["dist_to_min_sine_1m"] = dist_to_min_sine
                entry_trigger_conditions["dist_to_max_sine_1m"] = dist_to_max_sine
            elif timeframe == '5m':
                entry_trigger_conditions["dist_to_min_sine_5m"] = dist_to_min_sine
                entry_trigger_conditions["dist_to_max_sine_5m"] = dist_to_max_sine

            # Find major reversals
            last_bottom, last_top = find_major_reversals(candle_map[timeframe])
            print(f"Last Bottom: {last_bottom:.2f}, Last Top: {last_top:.2f}")

            # 45 Degree Angle Price Calculation
            projected_price_45 = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price_45:.2f}")

            # Calculate Golden Ratio Projection
            projected_price_golden = calculate_golden_ratio_projection(closes[-1], projected_price_45)
            print(f"Projected Price Using Golden Ratio: {projected_price_golden:.2f}")

            # Check entry conditions
            if projected_price_45 > current_btc_price and projected_price_golden > current_btc_price:
                entry_trigger_conditions["forecast_condition"] = True
            
            if projected_price_45 > current_btc_price:
                entry_trigger_conditions["below_angle"] = True

    print()

    # Use bullish volume condition from the 5-minute timeframe
    if buy_volume['5m'] > sell_volume['5m']:
        entry_trigger_conditions["buy_volume"] = True

    # Check if current close is above the Minimum Threshold from 1m timeframe
    current_close = candle_map['1m'][-1]['close']
    entry_trigger_conditions["current_close_above_min_threshold"] = current_close > min_threshold_1m

    # Confirm if the last bottom (reversal) is below current close (i.e., it's a dip)
    if last_bottom_1m == min_threshold_1m:
        entry_trigger_conditions["dip_condition_met"] = True
        print("Confirmed dip condition found: Current Close is at the last major reversal (Minimum Threshold).")

    # Entry Logic
    if not position_open:
        if (entry_trigger_conditions["below_angle"] and
            entry_trigger_conditions["buy_volume"] and
            entry_trigger_conditions["forecast_condition"] and
            entry_trigger_conditions["dist_to_min_sine_1m"] < entry_trigger_conditions["dist_to_max_sine_1m"] and
            entry_trigger_conditions["dist_to_min_sine_5m"] < entry_trigger_conditions["dist_to_max_sine_5m"] and
            entry_trigger_conditions["current_close_above_min_threshold"] and
            entry_trigger_conditions["dip_condition_met"]):
            
            # Check if balance is sufficient for entry
            if usdc_balance > 0:  # Ensure we have some balance to invest
                amount_to_invest = usdc_balance
                btc_order = buy_btc(amount_to_invest)
                if btc_order:
                    initial_investment = amount_to_invest
                    btc_balance += amount_to_invest / current_btc_price
                    print(f"New position opened with {amount_to_invest} USDC at price {current_btc_price:.2f}.")
                    
                    # Log the entry signal with projected price
                    log_entry_signal(current_local_time, current_btc_price, projected_price_45)

                    position_open = True 
            else:
                print("Insufficient USDC balance to open a new position.")

    # Exit Logic
    if position_open and check_exit_condition(initial_investment, btc_balance):
        try:
            sell_order = client.order_market_sell(
                symbol='BTCUSDC',
                quantity=btc_balance
            )
            logging.info(f"Market sell order executed: {sell_order}")
            print(f"Market sell order executed: {sell_order}")
            print(f"Position closed. Sold BTC for USDC: {btc_balance * current_btc_price:.2f}")
            position_open = False
            initial_investment = 0.0 
            btc_balance = 0.0 
        except BinanceAPIException as e:
            logging.error(f"Error executing sell order: {e.message}")
            print(f"Error executing sell order: {e.message}")

    # Clean up references and collect garbage
    del candle_map, volume_trend_data, buy_volume, sell_volume
    gc.collect()  

    print()

    # Print overall entry trigger conditions status at the end of the iteration
    print("Overall Entry Trigger Conditions Status:")
    print()
    
    below_angle = entry_trigger_conditions['below_angle']
    buy_volume = entry_trigger_conditions['buy_volume']
    forecast_condition = entry_trigger_conditions['forecast_condition']
    dist_to_min_sine_1m = entry_trigger_conditions['dist_to_min_sine_1m']
    dist_to_max_sine_1m = entry_trigger_conditions['dist_to_max_sine_1m']
    dist_to_min_sine_5m = entry_trigger_conditions['dist_to_min_sine_5m']
    dist_to_max_sine_5m = entry_trigger_conditions['dist_to_max_sine_5m']
    current_close_above_min_threshold = entry_trigger_conditions['current_close_above_min_threshold']
    dip_condition_met = entry_trigger_conditions['dip_condition_met']

    print(f"Below Angle: {below_angle} (Expected: True)" if below_angle else "Below Angle: False")
    print(f"Buy Volume > Sell Volume: {buy_volume} (Expected: True)" if buy_volume else "Buy Volume > Sell Volume: False")
    print(f"Forecast Condition: {forecast_condition} (Expected: True)" if forecast_condition else "Forecast Condition: False")
    print(f"Distance to Min SINE (1m) < Distance to Max SINE (1m): {dist_to_min_sine_1m < dist_to_max_sine_1m} (Expected: True)")
    print(f"Distance to Min SINE (5m) < Distance to Max SINE (5m): {dist_to_min_sine_5m < dist_to_max_sine_5m} (Expected: True)")
    print(f"Current Close > Minimum Threshold (1m): {current_close_above_min_threshold} (Expected: True)")
    print(f"Dip Condition Met (Confirmed last reversal): {dip_condition_met} (Expected: True)")
    print()

    # Check if all conditions are true for entry
    if (below_angle and buy_volume and forecast_condition and
        dist_to_min_sine_1m < dist_to_max_sine_1m and
        dist_to_min_sine_5m < dist_to_max_sine_5m and
        current_close_above_min_threshold and
        dip_condition_met):
        print("Entry signal: all conditions trigger found TRUE")
    else:
        print("Seeking entry conditions...")

    # Log current account information
    print(f"Current USDC balance: {usdc_balance:.2f} | Current BTC balance: {btc_balance:.4f} BTC | Current BTC price: {current_btc_price:.2f}")
    logging.info(f"Current USDC balance: {usdc_balance:.2f} | Current BTC balance: {btc_balance:.4f} BTC | Current BTC price: {current_btc_price:.2f}")
    print()

    time.sleep(5)