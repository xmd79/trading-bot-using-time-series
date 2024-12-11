import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.fftpack as fftpack

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

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

def get_candles(symbol, timeframe, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
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

def get_average_entry_price_for_btc():
    """Fetch the average entry price for the current BTC balance based on order history."""
    try:
        trades = client.get_my_trades(symbol=TRADE_SYMBOL)
        if not trades:
            print("No trades found.")
            return 0.0
        
        total_cost = 0.0
        total_quantity = 0.0

        for trade in trades:
            print(trade)  # Debug: Print the trade to see its structure
            if 'isBuyer' in trade and trade['isBuyer']:
                price = float(trade['price'])
                qty = float(trade['qty'])
                total_cost += price * qty
                total_quantity += qty

        if total_quantity > 0:
            average_price = total_cost / total_quantity
            return average_price  # Return the average entry price
        return 0.0  # If no buy trades were found

    except BinanceAPIException as e:
        print(f"Error fetching trade history: {e.message}")
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
    return current_value >= (initial_investment * 1.0167)

def backtest_model(candles):
    closes = np.array([candle["close"] for candle in candles])
    X = np.arange(len(closes)).reshape(-1, 1)
    y = closes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, mae, predictions, y_test

def forecast_next_price(model, num_steps=1):
    last_index = model.n_features_in_
    future_steps = np.arange(last_index, last_index + num_steps).reshape(-1, 1)
    forecasted_prices = model.predict(future_steps)
    return forecasted_prices

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    close_prices = np.array([x for x in close_prices if not np.isnan(x) and x > 0])
    if len(close_prices) == 0:
        return None, None, None, None, None, None, None

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    # Use talib for momentum calculation 
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
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * 100 if max_momentum - min_momentum != 0 else np.nan
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * 100 if max_momentum - min_momentum != 0 else np.nan

    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum

def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in candle_map:
        buy_volume[timeframe] = []
        sell_volume[timeframe] = []
        total_buy = 0
        total_sell = 0
        for candle in candle_map[timeframe]:
            if candle["close"] > candle["open"]:
                total_buy += candle["volume"]
            elif candle["close"] < candle["open"]:
                total_sell += candle["volume"]
            buy_volume[timeframe].append(total_buy)
            sell_volume[timeframe].append(total_sell)

    return buy_volume, sell_volume 

def calculate_volume_ratio(buy_volume, sell_volume):
    volume_ratio = {}
    for timeframe in buy_volume.keys():
        total_volume = buy_volume[timeframe][-1] + sell_volume[timeframe][-1] 
        if total_volume > 0:
            ratio = (buy_volume[timeframe][-1] / total_volume) * 100
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

    if last_bottom is not None and (closest_reversal is None or abs(last_bottom - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_bottom
        closest_type = 'DIP'
    
    if last_top is not None and (closest_reversal is None or abs(last_top - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_top
        closest_type = 'TOP'

    if closest_type == 'TOP' and closest_reversal <= current_close:
        closest_type = None
        closest_reversal = None
    elif closest_type == 'DIP' and closest_reversal >= current_close:
        closest_type = None
        closest_reversal = None

    return last_bottom, last_top, closest_reversal, closest_type 

def scale_to_sine(close_prices):
    sine_wave, _ = talib.HT_SINE(np.array(close_prices))
    current_sine = np.nan_to_num(sine_wave)[-1]
    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0

    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_spectral_analysis(prices):
    fft_result = np.fft.fft(prices)
    power_spectrum = np.abs(fft_result) ** 2
    frequencies = np.fft.fftfreq(len(prices))

    negative_freqs = frequencies[frequencies < 0]
    negative_powers = power_spectrum[frequencies < 0]

    positive_freqs = frequencies[frequencies >= 0]
    positive_powers = power_spectrum[frequencies >= 0]

    return negative_freqs, negative_powers, positive_freqs, positive_powers

def determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, last_major_reversal_type, buy_volume, sell_volume):
    total_negative_power = np.sum(negative_powers)
    total_positive_power = np.sum(positive_powers)

    if last_major_reversal_type == 'DIP':
        if buy_volume > sell_volume:
            return "Bullish"
        else:
            return "Accumulation"
    elif last_major_reversal_type == 'TOP':
        if sell_volume > buy_volume:
            if total_positive_power > total_negative_power:
                return "Bearish"
            else:
                return "Distribution"
        else:
            return "Bullish"
    else:
        return "Neutral"

def get_min_trade_size(symbol):
    exchange_info = client.get_symbol_info(symbol)
    for filter in exchange_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            return float(filter['minQty'])
    return 0.0

def calculate_45_degree_projection(last_bottom, last_top):
    if last_bottom is not None and last_top is not None:
        distance = last_top - last_bottom
        projection = last_top + distance
        return projection
    return None

def calculate_wave_price(t, avg, min_price, max_price, omega, phi):
    return avg + ((max_price - min_price) / 2) * np.sin(omega * t + phi)

def calculate_independent_wave_price(current_price, avg, min_price, max_price, range_distance):
    noise = np.random.uniform(-1, 1) * range_distance
    return avg + (noise * (max_price - min_price) / 2)

def find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_close):
    support_levels = []
    resistance_levels = []

    for timeframe in candle_map:
        candles = candle_map[timeframe]
        for candle in candles:
            if candle["close"] < current_close and candle["close"] >= min_threshold:
                support_levels.append((candle["close"], candle["volume"], candle["time"]))
            if candle["close"] > current_close and candle["close"] <= max_threshold:
                resistance_levels.append((candle["close"], candle["volume"], candle["time"]))

    support_levels.sort(key=lambda x: x[1], reverse=True)  
    resistance_levels.sort(key=lambda x: x[1], reverse=True)  

    significant_support = support_levels[:3]  
    significant_resistance = resistance_levels[:3]  

    print("Most Significant Support Levels (Price, Volume):")
    for price, volume, timestamp in significant_support:
        print(f"Support Price: {price:.25f}, Volume: {volume:.25f}")

    print("Most Significant Resistance Levels (Price, Volume):")
    for price, volume, timestamp in significant_resistance:
        print(f"Resistance Price: {price:.25f}, Volume: {volume:.25f}")

    return [level[0] for level in significant_support], [level[0] for level in significant_resistance]

def calculate_stochastic_rsi(close_prices, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3):
    rsi = talib.RSI(np.array(close_prices), timeperiod=length_rsi)
    min_rsi = talib.MIN(rsi, timeperiod=length_stoch)
    max_rsi = talib.MAX(rsi, timeperiod=length_stoch)
    
    stoch_k = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    stoch_k_smooth = talib.EMA(stoch_k, timeperiod=smooth_k)
    stoch_d = talib.EMA(stoch_k_smooth, timeperiod=smooth_d)

    return stoch_k_smooth, stoch_d

# Volume analysis functions
def calculate_bullish_bearish_volume_ratios(candle_map):
    """Calculate Bullish and Bearish volume ratios for each timeframe."""
    volume_ratios = {}
    for timeframe, candles in candle_map.items():
        bullish_volume = 0
        bearish_volume = 0
        
        for candle in candles:
            if candle["close"] > candle["open"]:
                bullish_volume += candle["volume"]
            elif candle["close"] < candle["open"]:
                bearish_volume += candle["volume"]

        total_volume = bullish_volume + bearish_volume
        ratio = bullish_volume / bearish_volume if bearish_volume > 0 else float('inf')  # Handle division by zero

        volume_ratios[timeframe] = {
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "ratio": ratio,
            "status": "Bullish" if ratio > 1 else "Bearish" if ratio < 1 else "Neutral"
        }

        print(f"{timeframe} - Bullish Volume: {bullish_volume:.2f}, Bearish Volume: {bearish_volume:.2f}, Ratio: {ratio:.2f} ({volume_ratios[timeframe]['status']})")

    return volume_ratios

def analyze_volume_changes_over_time(candle_map):
    """Analyze volume changes over time cycles for each timeframe."""
    volume_trends = {}
    
    for timeframe, candles in candle_map.items():
        if len(candles) < 2:
            print(f"{timeframe} - Not enough data to analyze volume changes.")
            continue
            
        last_volume = candles[-1]["volume"]
        previous_volume = candles[-2]["volume"]
        
        volume_change = last_volume - previous_volume
        change_status = "Increasing" if volume_change > 0 else "Decreasing" if volume_change < 0 else "Stable"

        volume_trends[timeframe] = {
            "last_volume": last_volume,
            "previous_volume": previous_volume,
            "change": volume_change,
            "status": change_status
        }

        print(f"{timeframe} - Last Volume: {last_volume:.2f}, Previous Volume: {previous_volume:.2f}, Change: {volume_change:.2f} ({change_status})")

    return volume_trends

# Modified Fibonacci levels calculation logic
def calculate_fibonacci_levels_from_reversal(last_reversal, max_threshold, min_threshold, last_major_reversal_type):
    levels = {}

    if last_reversal is None:
        return levels  # Return empty levels if last reversal is not valid

    if last_major_reversal_type == 'DIP':  # Up cycle
        levels['0.0'] = last_reversal  # Starting point for cycle
        levels['1.0'] = max_threshold  # Ending point for cycle
        levels['0.5'] = last_reversal + (max_threshold - last_reversal) / 2  # Mid-point
    elif last_major_reversal_type == 'TOP':  # Down cycle
        levels['0.0'] = last_reversal  # Starting point for cycle
        levels['1.0'] = min_threshold  # Ending point for cycle
        levels['0.5'] = last_reversal - (last_reversal - min_threshold) / 2  # Mid-point

    return levels

# Forecasting Fibonacci target price for incoming reversal
def forecast_fibo_target_price(fib_levels):
    """Forecast Fibonacci target prices for incoming reversal."""
    if '1.0' not in fib_levels:
        return None  # Return None if the level doesn't exist
    return fib_levels['1.0']  # Return the '1.0' Fibonacci level directly

# Forecast Volume Function
def forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_price):
    forecasted_price = None

    if volume_ratios['1m']['buy_ratio'] > 50:
        forecasted_price = current_price + (current_price * 0.0267)
        print(f"Forecasting a bullish price increase to {forecasted_price:.25f}.")
    elif volume_ratios['1m']['sell_ratio'] > 50:
        forecasted_price = current_price - (current_price * 0.0267)
        print(f"Forecasting a bearish price decrease to {forecasted_price:.25f}.")
    else:
        print("No clear forecast direction based on volume ratios.")

    return forecasted_price

# Check market conditions and forecast
def check_market_conditions_and_forecast(support_levels, resistance_levels, current_price):
    forecast_decision = None

    if not support_levels and not resistance_levels:
        print("No support or resistance levels found; trade cautiously.")
        return "No trading signals available."

    first_support = support_levels[0] if support_levels else None
    first_resistance = resistance_levels[0] if resistance_levels else None

    if first_support is not None and current_price < first_support:
        forecast_decision = "Current price below key support level; consider selling."
        print(f"Current price {current_price:.25f} is below support {first_support:.25f}.")
    elif first_resistance is not None and current_price > first_resistance:
        forecast_decision = "Current price above key resistance level; consider buying."
        print(f"Current price {current_price:.25f} is above resistance {first_resistance:.25f}.")
    else:
        print(f"Current price {current_price:.25f} is within support {first_support} and resistance {first_resistance}.")
    
    return forecast_decision

# Get order book data
def get_order_book(symbol):
    """Fetch the order book for the given symbol."""
    try:
        return client.get_order_book(symbol=symbol)
    except BinanceAPIException as e:
        print(f"Error fetching order book: {e.message}")
        return None

def calculate_bai(order_book):
    """Calculate Bid-Ask Imbalance (BAI)."""
    total_bids = sum(float(level[1]) for level in order_book['bids'])
    total_asks = sum(float(level[1]) for level in order_book['asks'])
    return (total_bids - total_asks) / (total_bids + total_asks) if (total_bids + total_asks) > 0 else 0.0

# New target calculation using FFT
def get_target(closes, n_components, last_major_reversal_type, buy_volume, sell_volume):
    """Calculate price targets based on FFT analysis and market mood."""
    fft = fftpack.rfft(closes) 
    frequencies = fftpack.rfftfreq(len(closes))

    # Sort frequencies by magnitude and keep only the top n_components 
    idx = np.argsort(np.abs(fft))[::-1][:n_components]

    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)

    # Calculate the current close and adjust the target price based on the last signal
    current_close = closes[-1]
    target_price = np.nanmax(filtered_signal)  
    
    # Get the current time           
    current_time = datetime.datetime.now()

    # Calculate stop loss closer to the current price
    stop_loss = current_close - np.std(closes)

    # Determine market mood based on last major reversal type and volume dynamics
    market_mood = "Neutral"  
    if last_major_reversal_type == 'DIP':
        if buy_volume > sell_volume:
            market_mood = "Bullish"
    elif last_major_reversal_type == 'TOP':
        if sell_volume > buy_volume:
            if np.sum(sell_volume) < np.sum(buy_volume):  # heuristic
                market_mood = "Choppy"
            else:
                market_mood = "Bearish"

    return current_time, current_close, stop_loss, target_price, market_mood

# Instantiate the minimum trade size for the trading pair
min_trade_size = get_min_trade_size(TRADE_SYMBOL)

# Initialize variables for tracking trade state
position_open = False
initial_investment = 0.0  # Start out with no investment
btc_balance = 0.0
entry_price = 0.0  # To store entry price of the last buy order

# Initial balance check and average entry price retrieval
usdc_balance = get_balance('USDC')  # Get initial USDC balance at start
btc_balance = get_balance('BTC')    # Get initial BTC balance at start
print("Trading Bot Initialized!")

# Retrieve the average entry price for BTC
average_entry_price = get_average_entry_price_for_btc()
initial_investment = btc_balance * average_entry_price if average_entry_price else 0.0  # Calculate initial investment based on current BTC balance and average entry price
entry_price = average_entry_price  # Set entry price from the average entry retrieved

print(f"Initial USDC amount: {initial_investment:.25f}, Average Entry Price for current BTC balance: {entry_price:.25f}")

# Starting the main trading loop
while True:
    current_local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nCurrent Local Time: {current_local_time}")

    # Fetch candle data
    candle_map = fetch_candles_in_parallel(['1m', '3m', '5m']) 
    current_btc_price = get_current_btc_price()

    # Instant backtesting on the last 100 candles of the 1m timeframe
    model, backtest_mae, last_predictions, actuals = None, None, None, None
    if "1m" in candle_map:
        model, backtest_mae, last_predictions, actuals = backtest_model(candle_map["1m"])
        print(f"Backtest MAE: {backtest_mae:.25f}")

    # Forecast next price using the ML model
    adjusted_forecasted_price = None
    if model is not None:
        forecasted_prices = forecast_next_price(model, num_steps=1)
        if "1m" in candle_map:
            closes = [candle['close'] for candle in candle_map["1m"]]
            min_threshold, max_threshold, _, _, _, _, _ = calculate_thresholds(closes)
            adjusted_forecasted_price = np.clip(forecasted_prices[-1], min_threshold, max_threshold)
            print(f"Forecasted Price: {adjusted_forecasted_price:.25f}")

    # Calculate volume details
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

    # Find specific support and resistance levels
    support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_btc_price)

    # Perform volume analysis
    volume_ratios_details = calculate_bullish_bearish_volume_ratios(candle_map)
    volume_trends_details = analyze_volume_changes_over_time(candle_map)

    # Calculate Fibonacci levels and identify last major reversals
    fib_info = {}
    last_major_reverse_info = {timeframe: None for timeframe in ['1m', '3m', '5m']}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            closes_tf = [candle['close'] for candle in candle_map[timeframe]]
            high_tf = np.nanmax(closes_tf)
            low_tf = np.nanmin(closes_tf)

            last_bottom, last_top, last_reversal, last_reversal_type = find_major_reversals(candle_map[timeframe], current_btc_price, low_tf, high_tf)
            fib_levels = calculate_fibonacci_levels_from_reversal(last_reversal, high_tf, low_tf, last_reversal_type)
            
            if not fib_levels:
                print(f"No valid Fibonacci levels for {timeframe}.")
                fib_info[timeframe] = {}
                continue
            
            fib_info[timeframe] = fib_levels
            print(f"Fibonacci Levels for {timeframe}:")
            for level, price in fib_levels.items():
                print(f"Level {level}: {price:.25f}")

            # Calculate distances to min and max
            dist_to_min = ((current_btc_price - low_tf) / (high_tf - low_tf)) * 100 if (high_tf - low_tf) != 0 else 0
            dist_to_max = ((high_tf - current_btc_price) / (high_tf - low_tf)) * 100 if (high_tf - low_tf) != 0 else 0
            print(f"Distance from Current Close to Min Threshold ({low_tf:.25f}): {dist_to_min:.25f}%")
            print(f"Distance from Current Close to Max Threshold ({high_tf:.25f}): {dist_to_max:.25f}%")

            # Ensure symmetry between distance percentages
            symmetrical_min_distance = (high_tf - current_btc_price) / (high_tf - low_tf) * 100 if (high_tf - low_tf) != 0 else 0
            symmetrical_max_distance = (current_btc_price - low_tf) / (high_tf - low_tf) * 100 if (high_tf - low_tf) != 0 else 0
            
            print(f"Normalized Distance to Min Threshold (Symmetrical): {symmetrical_max_distance:.25f}%")
            print(f"Normalized Distance to Max Threshold (Symmetrical): {symmetrical_min_distance:.25f}%")

    # Forecast potential volumes based on conditions
    forecasted_price = forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_btc_price)

    # Check market conditions and determine forecast decisions
    forecast_decision = check_market_conditions_and_forecast(support_levels, resistance_levels, current_btc_price)

    # Initialize conditions status
    conditions_status = {
        "volume_bullish_1m": False,
        "ML_Forecasted_Price_over_Current_Close": False,
        "current_close_below_average_threshold_5m": False,
        "dip_confirmed_1m": False,
        "dip_confirmed_3m": False,
        "dip_confirmed_5m": False,
    }

    last_bottom = None
    last_top = None
    last_major_reversal_type = None

    # Check conditions for each timeframe
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = closes[-1]

            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_btc_price, min_threshold, max_threshold)
            dip_confirmed = closest_type == 'DIP'
            conditions_status[f'dip_confirmed_{timeframe}'] = dip_confirmed

            if timeframe == '1m':
                print(f"Dip Confirmed on 1min TF: {'True' if dip_confirmed else 'False'}")
                conditions_status["ML_Forecasted_Price_over_Current_Close"] = adjusted_forecasted_price is not None and adjusted_forecasted_price > current_close
            elif timeframe == '3m':
                print(f"Dip Confirmed on 3min TF: {'True' if dip_confirmed else 'False'}")
            elif timeframe == '5m':
                print(f"Dip Confirmed on 5min TF: {'True' if dip_confirmed else 'False'}")

            valid_closes = np.array([c for c in closes if not np.isnan(c) and c > 0])
            sma_lengths = [5, 7, 9, 12]
            smas = {length: talib.SMA(valid_closes, timeperiod=length)[-1] for length in sma_lengths}
            smas = {k: v for k, v in smas.items() if v is not np.nan and v > 0}

            print("Simple Moving Averages:")
            for length, sma in smas.items():
                print(f"SMA-{length}: {sma:.25f}, Current Close: {current_close:.25f} - {'Above' if current_close > sma else 'Below'}")

            if all(current_close > sma for sma in smas.values()):
                print("SELL Signal: Current close is above all SMAs.")
            elif all(current_close < sma for sma in smas.values()):
                print("BUY Signal: Current close is below all SMAs.")

            stoch_k, stoch_d = calculate_stochastic_rsi(closes, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3)
            current_stoch_k = stoch_k[-1]
            current_stoch_d = stoch_d[-1]

            print(f"Current Stochastic K: {current_stoch_k:.25f}")
            print(f"Current Stochastic D: {current_stoch_d:.25f}")

            negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
            market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, closest_type, buy_volume[timeframe][-1], sell_volume[timeframe][-1])
            print(f"Market Sentiment: {market_sentiment}")

            conditions_status["volume_bullish_1m"] = buy_volume['1m'][-1] > sell_volume['1m'][-1]
            conditions_status["current_close_below_average_threshold_5m"] = current_close < avg_mtf

            if closest_reversal is not None:
                print(f"Most Recent Major Reversal Type: {closest_type}")
                print(f"Last Major Reversal Found at Price: {closest_reversal:.25f}")
                last_major_reversal_type = closest_type  
            else:
                print("No Major Reversal Found")

            projected_price = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price:.25f}" if projected_price is not None else "No projection available")
            print(f"Current Close: {closes[-1]:.25f}")
            print(f"Minimum Threshold: {min_threshold:.25f}")
            print(f"Maximum Threshold: {max_threshold:.25f}")
            print(f"Average MTF: {avg_mtf:.25f}")
            print(f"Momentum Signal: {momentum_signal:.25f}")
            print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.25f}%")
            print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.25f}%")
            print(f"Status: {volume_ratios[timeframe]['status']}")

            avg = (min_threshold + max_threshold) / 2
            wave_price = calculate_wave_price(len(closes), avg, min_threshold, max_threshold, omega=0.1, phi=0)
            print(f"Calculated Wave Price: {wave_price:.25f}")

            independent_wave_price = calculate_independent_wave_price(current_btc_price, avg, min_threshold, max_threshold, range_distance=0.1)
            print(f"Calculated Independent Wave Price: {independent_wave_price:.25f}")

            current_time, entry_price_usdc, stop_loss, reversal_target, market_mood = get_target(
                closes, n_components=5, last_major_reversal_type=last_major_reversal_type, 
                buy_volume=buy_volume[timeframe][-1], sell_volume=sell_volume[timeframe][-1]
            )
            
            print(f"Current Time: {current_time}")
            print(f"Entry Price: {entry_price_usdc:.25f}")
            print(f"Stop Loss: {stop_loss:.25f}")
            print(f"Reversal Target: {reversal_target:.25f}")
            print(f"Market Mood (from FFT): {market_mood}")

            fib_reversal_price = forecast_fibo_target_price(fib_info[timeframe])
            print(f"{timeframe} Incoming Fibonacci Reversal Target (Forecast): {fib_reversal_price:.25f}" if fib_reversal_price is not None else f"{timeframe} Incoming Fibonacci Reversal Target: price not available.")

    # Print checks for available balances
    usdc_balance = get_balance('USDC')
    btc_balance = get_balance('BTC')

    # Ensure we get the initial USDC value for transaction calculation
    print(f"Initial USDC amount: {initial_investment:.25f}, Entry Price for last BTC purchased: {entry_price:.25f}")

    # Check USDC balance
    if usdc_balance > 0:
        print(f"Current USDC balance found: {usdc_balance:.25f}")
    else:
        print("No USDC balance available.")

    print(f"Current BTC balance: {btc_balance:.25f} BTC")

    if btc_balance > 0:
        # Calculate current value of BTC balance in USDC
        current_value_in_usdc = btc_balance * current_btc_price
        print(f"Value of current BTC balance: {current_value_in_usdc:.25f} USDC")

        # Fetch accurate average entry price based on the BTC balance
        average_entry_price = get_average_entry_price_for_btc()

        # Set target profit percentage
        target_profit_percentage = 1.67
        target_value = btc_balance * average_entry_price * (1 + target_profit_percentage / 100)  # target value calculation

        # Calculate percentage differences
        distance_to_initial_investment = ((current_value_in_usdc - initial_investment) / initial_investment) * 100 if initial_investment > 0 else 0.0
        distance_to_target_profit = ((current_value_in_usdc - target_value) / target_value) * 100 if target_value > 0 else 0.0
        
        # Display correct percentages
        print(f"Percentage distance to initial amount: {distance_to_initial_investment:.25f}%")
        print(f"Percentage distance to target profit (1.67%): {distance_to_target_profit:.25f}%")
        print(f"Target Value for Profit Calculation: {target_value:.25f} USDC")
        
        # Calculate and display target price
        if average_entry_price > 0:
            target_price = average_entry_price * (1 + target_profit_percentage / 100)
            print(f"Target Price for Profit Calculation: {target_price:.25f} USDC")

    else:
        print("No BTC balance available.")
        
    # Calculate percentage differences from the initial investment for exit evaluation
    if position_open:
        current_value_in_usdc = btc_balance * current_btc_price
        percentage_change_btc = ((current_value_in_usdc / initial_investment) - 1) * 100 if initial_investment > 0 else 0
        distance_from_initial_investment = (current_value_in_usdc - initial_investment) / initial_investment * 100 if initial_investment > 0 else 0
        
        print(f"Current BTC Balance Value in USDC: {current_value_in_usdc:.25f} (Change: {percentage_change_btc:.25f}%)")
        print(f"Distance from Initial USDC Investment: {distance_from_initial_investment:.25f}%")

        # Check if exit condition with the target profit is met
        if check_exit_condition(initial_investment, btc_balance):
            print("Target profit of 1.67% reached or exceeded. Initiating exit...")

            try:
                sell_order = client.order_market_sell(
                    symbol='BTCUSDC',
                    quantity=btc_balance
                )
                print(f"Market sell order executed: {sell_order}")
                print(f"Position closed. Sold BTC for USDC: {btc_balance * current_btc_price:.25f}")

                exit_usdc_balance = get_balance('USDC')
                usdc_diff = exit_usdc_balance - initial_investment
                profit_percentage = (usdc_diff / initial_investment) * 100 if initial_investment > 0 else 0.0

                print(f"Trade log: Time: {current_local_time}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}, Profit Percentage: {profit_percentage:.25f}%, Profit Amount: {usdc_diff:.25f}")

                position_open = False
                initial_investment = 0.0 
                btc_balance = 0.0 

            except BinanceAPIException as e:
                print(f"Error executing sell order: {e.message}")

    print()

    true_conditions_count = sum(int(status) for status in conditions_status.values())
    false_conditions_count = len(conditions_status) - true_conditions_count
    print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False\n")

    print("Individual Condition Status:")
    for condition, status in conditions_status.items():
        print(f"{condition}: {'True' if status else 'False'}")

    # Entry logic
    if not position_open:
        all_conditions_met = all(conditions_status.values())
        
        if all_conditions_met:
            print(f"All Conditions Met for Entry: Yes")
            usdc_balance = get_balance('USDC')  # Refresh USDC balance to reflect any changes
            print(f"Current USDC Balance Found: {usdc_balance:.25f}")
            
            if btc_balance > 0:
                current_value_in_usdc = btc_balance * current_btc_price
                largest_balance = max(usdc_balance, current_value_in_usdc)  # Determine the largest balance between USDC and BTC value
                
                # Evaluate if we should exit the BTC position
                if current_value_in_usdc >= (initial_investment * 1.0167):
                    print("Current BTC balance is sufficient; considering exit at profit.")
                    if check_exit_condition(initial_investment, btc_balance):
                        print("Exit conditions met for BTC; selling the position...")
                        try:
                            sell_order = client.order_market_sell(
                                symbol='BTCUSDC',
                                quantity=btc_balance
                            )
                            print(f"Market sell order executed for BTC position: {sell_order}")
                        except BinanceAPIException as e:
                            print(f"Error executing sell order: {e.message}")
            else:
                # Calculating the quantity to buy if no BTC is held
                quantity_to_buy = usdc_balance / current_btc_price

                # Check if the quantity to buy meets the minimum trade size requirement
                if quantity_to_buy >= min_trade_size and usdc_balance > 0:
                    btc_order = buy_btc(quantity_to_buy)
                    if btc_order:
                        initial_investment = usdc_balance  # Use the entire USDC balance for the order
                        entry_price_usdc = float(btc_order['fills'][0]['price'])  # Use the price from the order fill
                        print(f"BTC was bought at initial entry price of {entry_price_usdc:.25f} USDC for quantity: {quantity_to_buy:.25f} BTC.")
                        position_open = True  
                        print(f"New position opened with {usdc_balance:.25f} USDC at price {current_btc_price:.25f}.")
                    else:
                        print("Error placing buy order.")
                else:
                    if quantity_to_buy < min_trade_size:
                        print(f"Cannot place order: Quantity {quantity_to_buy:.25f} is less than minimum trade size {min_trade_size:.25f}.")
                    if usdc_balance <= 0:
                        print("No USDC balance to invest in BTC.")
        else:
            print("All Conditions Met for Entry: No.")

    # Clean up references and collect garbage
    del candle_map
    gc.collect()  

    # Print balances at the end of the loop
    print(f"\nCurrent USDC balance: {usdc_balance:.25f}")
    print(f"Current BTC balance: {btc_balance:.25f} BTC")
    print(f"Current BTC price: {current_btc_price:.25f}\n")

    time.sleep(5)  # Wait before the next iteration