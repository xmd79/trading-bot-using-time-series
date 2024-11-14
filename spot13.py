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
    return current_value >= (initial_investment * 1.01618)

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

def calculate_volume_ratio(buy_volume, sell_volume):
    """ Calculate the bullish and bearish volume ratios. """
    volume_ratio = {}
    for timeframe in buy_volume.keys():
        total_volume = buy_volume[timeframe] + sell_volume[timeframe]
        if total_volume > 0:
            ratio = (buy_volume[timeframe] / total_volume) * 100
            volume_ratio[timeframe] = {
                "buy_ratio": ratio,
                "sell_ratio": 100 - ratio,
                "status": "Bullish" if ratio > 50 else "Bearish" if ratio < 50 else "Neutral"
            }
        else:
            volume_ratio[timeframe] = {
                "buy_ratio": 0,
                "sell_ratio": 0,
                "status": "No Activity"
            }
    return volume_ratio

def find_major_reversals(candles, current_close, min_threshold, max_threshold):
    lows = [candle['low'] for candle in candles if candle['low'] >= min_threshold]
    highs = [candle['high'] for candle in candles if candle['high'] <= max_threshold]

    last_bottom = np.nanmin(lows) if lows else None
    last_top = np.nanmax(highs) if highs else None

    closest_reversal = None
    closest_type = None

    # Determine closest reversal and type (DIP or TOP)
    if last_bottom is not None and (closest_reversal is None or abs(last_bottom - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_bottom
        closest_type = 'DIP'
    
    if last_top is not None and (closest_reversal is None or abs(last_top - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_top
        closest_type = 'TOP'

    return last_bottom, last_top, closest_reversal, closest_type 

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

def evaluate_forecast(current_price, support, resistance):
    """ Simple placeholder for forecasting logic based on support and resistance. """
    if current_price < support:
        return support  # Price could bounce from support
    elif current_price > resistance:
        return resistance  # Price could reverse from resistance
    else:
        return current_price  # Assuming sideways movement within support/resistance

def calculate_stochastic_rsi(close_prices, high_prices, low_prices, length_rsi, length_stoch, smooth_k, smooth_d, lower_band, upper_band, min_threshold, max_threshold):
    rsi = talib.RSI(np.array(close_prices), timeperiod=length_rsi)

    # Calculate Stochastic values
    stoch_k, stoch_d = talib.STOCHF(np.array(high_prices), np.array(low_prices), np.array(close_prices), fastk_period=length_stoch, fastd_period=smooth_d)

    # Compute support and resistance based on thresholds instead of the current close prices
    support_candidates = [close_prices[i] for i in range(len(stoch_k)) if stoch_k[i] < lower_band and close_prices[i] >= min_threshold]
    resistance_candidates = [close_prices[i] for i in range(len(stoch_k)) if stoch_k[i] > upper_band and close_prices[i] <= max_threshold]

    # Determine support using thresholds
    if support_candidates:
        support = np.nanmin(support_candidates)
    else:
        support = min_threshold  # Default to minimum threshold if no candidates found

    # Determine resistance using thresholds
    if resistance_candidates:
        resistance = np.nanmax(resistance_candidates)
    else:
        resistance = max_threshold  # Default to maximum threshold if no candidates found

    return support, resistance

def determine_market_trend(last_bottom, last_top, support, resistance):
    if last_bottom < support and last_top > resistance:
        return "Uptrend"
    elif last_bottom > support and last_top < resistance:
        return "Downtrend"
    else:
        return "Sideways"

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
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

    conditions_status = {
        "current_close_above_min_threshold": False,
        "dip_condition_met": False,
        "volume_bullish_1m": False,
        "dist_to_min_less_than_max": False,
        "current_close_below_average_threshold": False,
        "current_close_above_last_major_reversal": False  # Added condition
    }

    average_threshold_1m = None  
    average_threshold_5m = None  

    for timeframe in timeframes:
        if timeframe in candle_map:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            highs = [candle['high'] for candle in candle_map[timeframe]]
            lows = [candle['low'] for candle in candle_map[timeframe]]

            length_rsi = 64  # RSI length for the stochastic RSI
            length_stoch = 64  # Stochastic length
            smooth_k = 3
            smooth_d = 3
            lower_band = 20
            upper_band = 80

            # Calculate thresholds and other indicators
            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            # Calculate Support/Resistance using Stochastic RSI and thresholds
            support, resistance = calculate_stochastic_rsi(closes, highs, lows, length_rsi, length_stoch, smooth_k, smooth_d, lower_band, upper_band, min_threshold, max_threshold)

            print(f"Calculated Support (Price): {support:.2f}, Calculated Resistance (Price): {resistance:.2f}")

            # Forecast price calculation based on dip confirmation
            if conditions_status["dip_condition_met"]:
                projected_forecast_price = max_threshold  # Use max threshold if dip is confirmed
            else:
                projected_forecast_price = min_threshold  # Use min threshold otherwise

            # Store the average thresholds for the respective timeframes
            if timeframe == '1m':
                average_threshold_1m = avg_mtf
            elif timeframe == '5m':
                average_threshold_5m = avg_mtf

            # Find major reversals close to the current close using thresholds
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_btc_price, min_threshold, max_threshold)
            print(f"Last Bottom (Major Reversal): {last_bottom:.2f} at Dip" if last_bottom else "No Last Bottom Found")
            print(f"Last Top (Major Reversal): {last_top:.2f} at Top" if last_top else "No Last Top Found")
            
            # Determine if dip_condition_met is TRUE or FALSE
            if closest_type == 'DIP':
                conditions_status["dip_condition_met"] = True
            elif closest_type == 'TOP':
                conditions_status["dip_condition_met"] = False
            
            if closest_reversal is not None:
                print(f"Closest Major Reversal: {closest_reversal:.2f} at {closest_type}")

            # Check if current close is above the last major reversal
            conditions_status["current_close_above_last_major_reversal"] = current_btc_price > closest_reversal if closest_reversal else False

            # Evaluate overall conditions
            current_close = closes[-1]
            conditions_status["current_close_above_min_threshold"] = current_close > min_threshold
            print(f"Current Close Above Min Threshold: {conditions_status['current_close_above_min_threshold']}")

            # Print relevant details
            print(f"Current Close: {closes[-1]:.2f}")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")
            print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.2f}% | Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.2f}% | Status: {volume_ratios[timeframe]['status']}")
            print(f"Forecast Price: {projected_forecast_price:.2f}")  # Print updated forecast price

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
            trend = determine_market_trend(last_bottom, last_top, support, resistance)
            print(f"Market Trend: {trend}")

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

            # Update conditions using 5m candles
            if average_threshold_5m is not None:
                conditions_status["current_close_below_average_threshold"] = closes[-1] < average_threshold_5m

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
                conditions_status["current_close_below_average_threshold"] and
                conditions_status["current_close_above_last_major_reversal"]):  # Added condition
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