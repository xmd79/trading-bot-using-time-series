import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import logging
import time
import concurrent.futures
import talib
import gc
from sklearn.linear_model import LinearRegression

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

# Calculate thresholds and averages based on min and max percentages
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
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

    # Calculate average
    avg_mtf = np.nanmean(filtered_close)

    # Get current momentum       
    current_momentum = momentum[-1]

    # Calculate % to min/max momentum    
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               
        percent_to_max_momentum = ((current_momentum - min_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan

    # Calculate combined percentages              
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    # Combined momentum signal     
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

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
            volume_data = [candle["volume"] for candle in candle_map[timeframe] if isinstance(candle, dict)]
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

    # Calculate % distances
    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0

    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_bullish_bearish_ratio(candle_map):
    volume_ratios = {}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            buy_volume = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] > candle["open"])
            sell_volume = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] < candle["open"])
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                volume_ratios[timeframe] = (buy_volume / total_volume) * 100
            else:
                volume_ratios[timeframe] = 0
    return volume_ratios

def linear_regression_forecast(candles):
    if len(candles) < 2:  # Regression requires at least two points
        return None
    close_prices = np.array([candle['close'] for candle in candles])
    time_points = np.array(range(len(candles))).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(time_points, close_prices)
    
    future_time = np.array([[len(candles)]])  # Predict next point
    forecasted_price = model.predict(future_time)[0]
    
    return forecasted_price

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

def log_entry_signal(timestamp, btc_price, forecast_price):
    """ Log the entry signal details into a text file. """
    with open("entry_signals.txt", "a") as f:
        f.write(f"Timestamp: {timestamp}, BTC Price: {btc_price}, Forecast Price: {forecast_price}\n")

# Initialize variables for tracking trade state
TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m']

position_open = False
initial_investment = 0.0
btc_balance = 0.0

while True:  
    current_local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Local Time: {current_local_time}")
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
        "bullish_volume_condition": False,
        "increasing_volume_condition": False
    }

    for timeframe in timeframes:
        if timeframe in candle_map:
            # Calculate thresholds
            closes = [candle['close'] for candle in candle_map[timeframe]]
            min_threshold, max_threshold, avg_mtf, momentum_signal, _ = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )
            
            # Print relevant details
            print(f"--- {timeframe} ---")
            print(f"Current Close: {closes[-1]:.2f}")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")
            print(f"Volume Trend: {volume_trend_data[timeframe]['trend']}")
            print(f"Buy Volume: {buy_volume[timeframe]:.2f}, Sell Volume: {sell_volume[timeframe]:.2f}")

            # Find major reversals
            last_bottom, last_top = find_major_reversals(candle_map[timeframe])
            print(f"Last Bottom: {last_bottom:.2f}, Last Top: {last_top:.2f}")

            # 45 Degree Angle Price Calculation
            projected_price_45 = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price_45:.2f}")

            # Current Price Distances
            distance_to_min = (closes[-1] - min_threshold) / (max_threshold - min_threshold) * 100
            distance_to_max = (max_threshold - closes[-1]) / (max_threshold - min_threshold) * 100
            print(f"Distance to Min Threshold: {distance_to_min:.2f}%")
            print(f"Distance to Max Threshold: {distance_to_max:.2f}%")

            # Calculate Golden Ratio Projection
            projected_price_golden = calculate_golden_ratio_projection(closes[-1], projected_price_45)
            print(f"Projected Price Using Golden Ratio: {projected_price_golden:.2f}")

            # Check entry conditions
            if current_btc_price < projected_price_45:
                entry_trigger_conditions["below_angle"] = True

            if distance_to_min < distance_to_max:
                entry_trigger_conditions["min_dist"] = True
            
            sine_dist_min, sine_dist_max, current_sine = scale_to_sine(closes)
            if sine_dist_min < sine_dist_max:
                entry_trigger_conditions["sine_dist"] = True
            
            # Assess Volume Condition
            if buy_volume[timeframe] > sell_volume[timeframe]:
                entry_trigger_conditions["buy_volume"] = True

            # Current Momentum
            momentum = talib.MOM(np.array(closes), timeperiod=14)
            if len(momentum) > 1:
                current_momentum = momentum[-1]
                previous_momentum = momentum[-2]
                momentum_trend = "Increasing" if current_momentum > previous_momentum else "Decreasing"
                print(f"Current Momentum: {current_momentum:.2f} ({momentum_trend})")

    print()

    # 1-Minute TF analysis for additional conditions
    if '1m' in candle_map:
        one_min_candles = candle_map['1m']
        forecast_price = linear_regression_forecast(one_min_candles)
        if forecast_price is not None:
            print(f"Forecasted Price for 1-min TF: {forecast_price:.2f}")

            if current_btc_price < forecast_price:
                entry_trigger_conditions["forecast_condition"] = True

        # Check bullish volume and increasing volume on 1-min TF
        bullish_volume = buy_volume['1m'] > sell_volume['1m']
        print(f"1m Bullish Volume Condition: {bullish_volume}")

        if bullish_volume:
            entry_trigger_conditions["bullish_volume_condition"] = True

        if volume_trend_data['1m']['trend'] == "Increasing":
            entry_trigger_conditions["increasing_volume_condition"] = True

    # Entry Logic
    if not position_open:
        if (entry_trigger_conditions["below_angle"] and
            entry_trigger_conditions["min_dist"] and
            entry_trigger_conditions["buy_volume"] and
            entry_trigger_conditions["forecast_condition"] and
            entry_trigger_conditions["bullish_volume_condition"] and
            entry_trigger_conditions["increasing_volume_condition"]):
            
            amount_to_invest = usdc_balance
            btc_order = buy_btc(amount_to_invest)
            if btc_order:
                initial_investment = amount_to_invest
                btc_balance += amount_to_invest / current_btc_price
                print(f"New position opened with {amount_to_invest} USDC at price {current_btc_price:.2f}.")
                
                # Log the entry signal
                log_entry_signal(current_local_time, current_btc_price, forecast_price)
                
                position_open = True  # Mark position as open

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
            initial_investment = 0.0  # Reset investment after exit
            btc_balance = 0.0  # Reset balance after exit
        except BinanceAPIException as e:
            logging.error(f"Error executing sell order: {e.message}")
            print(f"Error executing sell order: {e.message}")

    # Clean up references and collect garbage
    del candle_map, volume_trend_data, buy_volume, sell_volume
    gc.collect()  # Collect garbage to free memory
    
    print()  # Print a new line for better readability

    # Print overall entry trigger conditions status at the end of the iteration
    print("Overall Entry Trigger Conditions Status:")
    print()
    # Check if all conditions are true
    below_angle = entry_trigger_conditions['below_angle']
    min_dist = entry_trigger_conditions['min_dist']
    sine_dist = entry_trigger_conditions['sine_dist']
    buy_volume = entry_trigger_conditions['buy_volume']
    forecast_condition = entry_trigger_conditions['forecast_condition']
    bullish_volume_condition = entry_trigger_conditions['bullish_volume_condition']
    increasing_volume_condition = entry_trigger_conditions['increasing_volume_condition']

    print(f"Below Angle: {below_angle} (True)" if below_angle else "Below Angle: False")
    print(f"Min Distance < Max Distance: {min_dist} (True)" if min_dist else "Min Distance < Max Distance: False")
    print(f"Sine Distance: {sine_dist} (True)" if sine_dist else "Sine Distance: False")
    print(f"Buy Volume > Sell Volume: {buy_volume} (True)" if buy_volume else "Buy Volume > Sell Volume: False")
    print(f"Forecast Condition: {forecast_condition} (True)" if forecast_condition else "Forecast Condition: False")
    print(f"Bullish Volume Condition: {bullish_volume_condition} (True)" if bullish_volume_condition else "Bullish Volume Condition: False")
    print(f"Increasing Volume Condition: {increasing_volume_condition} (True)" if increasing_volume_condition else "Increasing Volume Condition: False")
    print()

    # Check if all conditions are true
    if (below_angle and min_dist and buy_volume and
            forecast_condition and bullish_volume_condition and
            increasing_volume_condition):
        print("Entry signal: all conditions trigger found TRUE")
    else:
        print("Seeking entry conditions...")

    # Log current account information
    print(f"Current USDC balance: {usdc_balance:.2f} | Current BTC balance: {btc_balance:.4f} BTC | Current BTC price: {current_btc_price:.2f}")
    logging.info(f"Current USDC balance: {usdc_balance:.2f} | Current BTC balance: {btc_balance:.4f} BTC | Current BTC price: {current_btc_price:.2f}")
    print()

    time.sleep(5)  # Sleep for a short period before the next iteration