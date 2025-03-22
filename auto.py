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
import math

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"  # Change to "MKRUSDC" if intended

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))

    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=100, retries=3, delay=2):
    for attempt in range(retries):
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
            print(f"Error fetching candles for {timeframe} (attempt {attempt + 1}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    return []

def get_current_price(retries=3, delay=2):
    for attempt in range(retries):
        try:
            ticker = client.get_symbol_ticker(symbol=TRADE_SYMBOL)
            price = float(ticker['price'])
            if price > 0:
                return price
            print(f"Invalid price {price} on attempt {attempt + 1}")
        except BinanceAPIException as e:
            print(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after retries.")
    return 0.0

def get_balance(asset='USDC'):
    try:
        balance_info = client.get_asset_balance(asset)
        return float(balance_info['free']) if balance_info else 0.0
    except BinanceAPIException as e:
        print(f"Error fetching balance for {asset}: {e.message}")
        return 0.0 

def get_last_buy_trade():
    try:
        trades = client.get_my_trades(symbol=TRADE_SYMBOL)
        if not trades:
            print("No trades found.")
            return None
        
        for trade in reversed(trades):
            if trade['isBuyer']:
                return {
                    "price": float(trade['price']),
                    "qty": float(trade['qty']),
                    "time": trade['time']
                }
    except BinanceAPIException as e:
        print(f"Error fetching trade history: {e.message}")
    return None

def get_average_entry_price():
    last_trade = get_last_buy_trade()
    if last_trade:
        entry_price = last_trade['price']
        print(f"Using last buy trade price as entry price: {entry_price:.25f}")
        return entry_price
    print(f"No valid last buy trade found for {TRADE_SYMBOL}; cannot calculate entry price.")
    return 0.0

def get_symbol_lot_size_info(symbol):
    try:
        exchange_info = client.get_symbol_info(symbol)
        for filter in exchange_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                return {
                    'minQty': float(filter['minQty']),
                    'stepSize': float(filter['stepSize'])
                }
        print(f"Could not find LOT_SIZE filter for {symbol}. Using defaults.")
        return {'minQty': 0.00001, 'stepSize': 0.00001}  # Default for BTCUSDC
    except BinanceAPIException as e:
        print(f"Error fetching symbol info for {symbol}: {e.message}")
        return {'minQty': 0.00001, 'stepSize': 0.00001}

def buy_asset():
    try:
        # Fetch current price and balance at the time of buy
        current_price = get_current_price()
        if current_price <= 0:
            print(f"Invalid current price {current_price} for buy order.")
            return None, None
        
        usdc_balance = get_balance('USDC')
        if usdc_balance <= 0:
            print("No USDC balance available to place buy order.")
            return None, None

        # Calculate raw quantity using the entire USDC balance
        raw_quantity = usdc_balance / current_price
        print(f"Raw quantity calculated: {raw_quantity:.10f} (USDC: {usdc_balance:.2f}, Price: {current_price:.2f})")
        
        # Get precision from step_size (number of decimal places)
        step_precision = int(-math.log10(step_size)) if step_size > 0 else 8  # Default to 8 if step_size is invalid
        
        # Adjust quantity to use maximum possible balance, floored to step_size
        adjusted_quantity = math.floor(raw_quantity / step_size) * step_size
        adjusted_quantity = round(adjusted_quantity, step_precision)
        cost = adjusted_quantity * current_price
        print(f"Adjusted quantity (max balance): {adjusted_quantity:.10f}, Cost: {cost:.2f} (Step size: {step_size}, Precision: {step_precision})")
        
        # Check minimum notional value (typically 10 USDC for Binance)
        min_notional = 10.0  # Binance minimum trade value
        if cost < min_notional:
            print(f"Cost {cost:.2f} USDC is below minimum notional value {min_notional}. Adjusting quantity.")
            min_quantity_for_notional = min_notional / current_price
            adjusted_quantity = math.ceil(min_quantity_for_notional / step_size) * step_size
            adjusted_quantity = round(adjusted_quantity, step_precision)
            cost = adjusted_quantity * current_price
            print(f"Re-adjusted quantity for notional: {adjusted_quantity:.10f}, New Cost: {cost:.2f}")

        # Ensure quantity meets minimum trade size
        if adjusted_quantity < min_trade_size:
            print(f"Adjusted quantity {adjusted_quantity:.10f} is below minimum trade size {min_trade_size}. Cannot execute trade.")
            return None, None
        
        # Verify cost doesn't exceed balance (shouldn't happen due to flooring, but safety check)
        if cost > usdc_balance:
            print(f"Cost {cost:.2f} exceeds available balance {usdc_balance:.2f}. This should not occur with proper flooring.")
            adjusted_quantity = math.floor((usdc_balance / current_price) / step_size) * step_size
            adjusted_quantity = round(adjusted_quantity, step_precision)
            cost = adjusted_quantity * current_price
            print(f"Re-adjusted quantity to fit balance: {adjusted_quantity:.10f}, Final Cost: {cost:.2f}")

        # Final validation
        if adjusted_quantity < min_trade_size:
            print(f"Final adjusted quantity {adjusted_quantity:.10f} still below minimum trade size {min_trade_size}. Cannot execute trade.")
            return None, None

        # Calculate remaining balance
        remaining_balance = usdc_balance - cost
        print(f"Using {cost:.2f} of {usdc_balance:.2f} USDC, Remaining Balance: {remaining_balance:.2f} USDC")

        # Execute the market buy order
        order = client.order_market_buy(
            symbol=TRADE_SYMBOL,
            quantity=adjusted_quantity
        )
        print(f"Market buy order executed: {order}")
        entry_price = float(order['fills'][0]['price'])
        return entry_price, adjusted_quantity
    except BinanceAPIException as e:
        print(f"Error executing buy order: {e.message}")
        return None, None

def check_exit_condition(initial_investment, asset_balance):
    current_value = asset_balance * get_current_price()
    return current_value >= (initial_investment * 1.03)  # 3% profit target

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
    return model.predict(future_steps)

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    close_prices = np.array([x for x in close_prices if not np.isnan(x) and x > 0])
    if len(close_prices) == 0:
        return None, None, None, None, None, None, None

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
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
    return frequencies[frequencies < 0], power_spectrum[frequencies < 0], frequencies[frequencies >= 0], power_spectrum[frequencies >= 0]

def determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, last_major_reversal_type, buy_volume, sell_volume):
    total_negative_power = np.sum(negative_powers)
    total_positive_power = np.sum(positive_powers)

    if last_major_reversal_type == 'DIP':
        return "Bullish" if buy_volume > sell_volume else "Accumulation"
    elif last_major_reversal_type == 'TOP':
        if sell_volume > buy_volume:
            return "Bearish" if total_positive_power > total_negative_power else "Distribution"
        return "Bullish"
    return "Neutral"

def calculate_45_degree_projection(last_bottom, last_top):
    if last_bottom is not None and last_top is not None:
        distance = last_top - last_bottom
        return last_top + distance
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
        for candle in candle_map[timeframe]:
            if candle["close"] < current_close and candle["close"] >= min_threshold:
                support_levels.append((candle["close"], candle["volume"], candle["time"]))
            if candle["close"] > current_close and candle["close"] <= max_threshold:
                resistance_levels.append((candle["close"], candle["volume"], candle["time"]))

    support_levels.sort(key=lambda x: x[1], reverse=True)  
    resistance_levels.sort(key=lambda x: x[1], reverse=True)  

    significant_support = support_levels[:3]  
    significant_resistance = resistance_levels[:3]  

    print("Most Significant Support Levels (Price, Volume):")
    for price, volume, _ in significant_support:
        print(f"Support Price: {price:.25f}, Volume: {volume:.25f}")

    print("Most Significant Resistance Levels (Price, Volume):")
    for price, volume, _ in significant_resistance:
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

def calculate_bullish_bearish_volume_ratios(candle_map):
    volume_ratios = {}
    for timeframe, candles in candle_map.items():
        bullish_volume = sum(candle["volume"] for candle in candles if candle["close"] > candle["open"])
        bearish_volume = sum(candle["volume"] for candle in candles if candle["close"] < candle["open"])
        
        total_volume = bullish_volume + bearish_volume
        ratio = bullish_volume / bearish_volume if bearish_volume > 0 else float('inf')

        volume_ratios[timeframe] = {
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "ratio": ratio,
            "status": "Bullish" if ratio > 1 else "Bearish" if ratio < 1 else "Neutral"
        }
        print(f"{timeframe} - Bullish Volume: {bullish_volume:.2f}, Bearish Volume: {bearish_volume:.2f}, Ratio: {ratio:.2f} ({volume_ratios[timeframe]['status']})")
    return volume_ratios

def analyze_volume_changes_over_time(candle_map):
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

def calculate_fibonacci_levels_from_reversal(last_reversal, max_threshold, min_threshold, last_major_reversal_type):
    levels = {}
    if last_reversal is None:
        return levels

    if last_major_reversal_type == 'DIP':
        levels['0.0'] = last_reversal
        levels['1.0'] = max_threshold
        levels['0.5'] = last_reversal + (max_threshold - last_reversal) / 2
    elif last_major_reversal_type == 'TOP':
        levels['0.0'] = last_reversal
        levels['1.0'] = min_threshold
        levels['0.5'] = last_reversal - (last_reversal - min_threshold) / 2
    return levels

def forecast_fibo_target_price(fib_levels):
    return fib_levels.get('1.0')

def forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_price):
    if '1m' in volume_ratios and volume_ratios['1m']['buy_ratio'] > 50:
        forecasted_price = current_price + (current_price * 0.0267)
        print(f"Forecasting a bullish price increase to {forecasted_price:.25f}.")
        return forecasted_price
    elif '1m' in volume_ratios and volume_ratios['1m']['sell_ratio'] > 50:
        forecasted_price = current_price - (current_price * 0.0267)
        print(f"Forecasting a bearish price decrease to {forecasted_price:.25f}.")
        return forecasted_price
    print("No clear forecast direction based on volume ratios.")
    return None

def check_market_conditions_and_forecast(support_levels, resistance_levels, current_price):
    if not support_levels and not resistance_levels:
        print("No support or resistance levels found; trade cautiously.")
        return "No trading signals available."

    first_support = support_levels[0] if support_levels else None
    first_resistance = resistance_levels[0] if resistance_levels else None

    if first_support and current_price < first_support:
        print(f"Current price {current_price:.25f} is below support {first_support:.25f}.")
        return "Current price below key support level; consider selling."
    elif first_resistance and current_price > first_resistance:
        print(f"Current price {current_price:.25f} is above resistance {first_resistance:.25f}.")
        return "Current price above key resistance level; consider buying."
    print(f"Current price {current_price:.25f} is within support {first_support} and resistance {first_resistance}.")
    return None

def get_target(closes, n_components, last_major_reversal_type, buy_volume, sell_volume):
    fft = fftpack.rfft(closes) 
    frequencies = fftpack.rfftfreq(len(closes))
    idx = np.argsort(np.abs(fft))[::-1][:n_components]

    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)

    current_close = closes[-1]
    target_price = np.nanmax(filtered_signal)  
    current_time = datetime.datetime.now()
    stop_loss = current_close - np.std(closes)

    market_mood = "Neutral"  
    if last_major_reversal_type == 'DIP' and buy_volume > sell_volume:
        market_mood = "Bullish"
    elif last_major_reversal_type == 'TOP' and sell_volume > buy_volume:
        market_mood = "Bearish" if np.sum(sell_volume) >= np.sum(buy_volume) else "Choppy"
    return current_time, current_close, stop_loss, target_price, market_mood

def forecast_price_per_time_pythagorean(timeframe, candles, min_threshold, max_threshold, current_price, time_window_minutes, last_reversal, last_reversal_type):
    threshold_range = max_threshold - min_threshold
    time_leg = time_window_minutes
    hypotenuse = np.sqrt(time_leg**2 + threshold_range**2)
    price_per_minute = threshold_range / time_leg if time_leg > 0 else 0.0
    forecast_price = max_threshold  # Default target for up cycle

    if last_reversal_type == 'DIP' and last_reversal is not None:
        dist_from_dip = current_price - last_reversal
        if dist_from_dip >= threshold_range:
            print(f"Pump incoming detected for {timeframe}: Distance from DIP ({dist_from_dip:.25f}) >= Threshold Range ({threshold_range:.25f})")
            forecast_price = last_reversal + (threshold_range * 1.5)
        else:
            forecast_price = max_threshold

    print(f"\n--- Fixed Pythagorean Forecast for {timeframe} ---")
    print(f"Min Threshold: {min_threshold:.25f}")
    print(f"Max Threshold: {max_threshold:.25f}")
    print(f"Threshold Range (Leg B): {threshold_range:.25f}")
    print(f"Time Window (Leg A): {time_leg} minutes")
    print(f"Hypotenuse (Price-Time Distance): {hypotenuse:.25f}")
    print(f"Base Price Change Rate: {price_per_minute:.25f} USDC per minute")
    if last_reversal_type == 'DIP' and last_reversal is not None:
        print(f"Last DIP Reversal: {last_reversal:.25f}")
        print(f"Distance from DIP to Current Price: {dist_from_dip:.25f}")
    print(f"Fixed Forecast Price: {forecast_price:.25f}")
    
    return forecast_price, price_per_minute

# Instantiate the minimum trade size and step size for the trading pair
lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
min_trade_size = lot_size_info['minQty']
step_size = lot_size_info['stepSize']
print(f"Initialized {TRADE_SYMBOL} - Min Trade Size: {min_trade_size}, Step Size: {step_size}")

# Initialize variables for tracking trade state
position_open = False
initial_investment = 0.0
asset_balance = 0.0
entry_price = 0.0

# Initial balance check
usdc_balance = get_balance('USDC')
asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])
print("Trading Bot Initialized!")

# Main trading loop
while True:
    current_local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nCurrent Local Time: {current_local_time}")

    candle_map = fetch_candles_in_parallel(['1m', '3m', '5m']) 
    if not candle_map.get('1m'):
        print("Error: '1m' candles not fetched. Check API connectivity or symbol.")
    current_price = get_current_price()
    if current_price == 0.0:
        print(f"Warning: Current {TRADE_SYMBOL} price is 0.0. API may be failing.")

    if "1m" in candle_map and candle_map['1m']:
        model, backtest_mae, last_predictions, actuals = backtest_model(candle_map["1m"])
        print(f"Backtest MAE: {backtest_mae:.25f}")

        forecasted_prices = forecast_next_price(model, num_steps=1)
        closes = [candle['close'] for candle in candle_map["1m"]]
        min_threshold, max_threshold, _, _, _, _, _ = calculate_thresholds(closes)
        adjusted_forecasted_price = np.clip(forecasted_prices[-1], min_threshold, max_threshold)
        print(f"Forecasted Price: {adjusted_forecasted_price:.25f}")
    else:
        min_threshold, max_threshold, adjusted_forecasted_price = None, None, None
        print("No 1m data available for forecasting.")

    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)
    support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold or 0, max_threshold or float('inf'), current_price)
    volume_ratios_details = calculate_bullish_bearish_volume_ratios(candle_map)
    volume_trends_details = analyze_volume_changes_over_time(candle_map)

    fib_info = {}
    pythagorean_forecasts = {'1m': {'price': None, 'rate': None}, '3m': {'price': None, 'rate': None}, '5m': {'price': None, 'rate': None}}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map and candle_map[timeframe]:
            closes_tf = [candle['close'] for candle in candle_map[timeframe]]
            high_tf = np.nanmax(closes_tf)
            low_tf = np.nanmin(closes_tf)

            last_bottom, last_top, last_reversal, last_reversal_type = find_major_reversals(candle_map[timeframe], current_price, low_tf, high_tf)
            fib_levels = calculate_fibonacci_levels_from_reversal(last_reversal, high_tf, low_tf, last_reversal_type)
            
            if not fib_levels:
                print(f"No valid Fibonacci levels for {timeframe}.")
                fib_info[timeframe] = {}
                continue
            
            fib_info[timeframe] = fib_levels
            print(f"Fibonacci Levels for {timeframe}:")
            for level, price in fib_levels.items():
                print(f"Level {level}: {price:.25f}")

            dist_to_min = ((current_price - low_tf) / (high_tf - low_tf)) * 100 if (high_tf - low_tf) != 0 else 0
            dist_to_max = ((high_tf - current_price) / (high_tf - low_tf)) * 100 if (high_tf - low_tf) != 0 else 0
            print(f"Distance from Current Close to Min Threshold ({low_tf:.25f}): {dist_to_min:.25f}%")
            print(f"Distance from Current Close to Max Threshold ({high_tf:.25f}): {dist_to_max:.25f}%")

            symmetrical_min_distance = (high_tf - current_price) / (high_tf - low_tf) * 100 if (high_tf - low_tf) != 0 else 0
            symmetrical_max_distance = (current_price - low_tf) / (high_tf - low_tf) * 100 if (high_tf - low_tf) != 0 else 0
            print(f"Normalized Distance to Min Threshold (Symmetrical): {symmetrical_max_distance:.25f}%")
            print(f"Normalized Distance to Max Threshold (Symmetrical): {symmetrical_min_distance:.25f}%")

            time_window = {'1m': 1, '3m': 3, '5m': 5}[timeframe]
            min_threshold_tf, max_threshold_tf, _, _, _, _, _ = calculate_thresholds(closes_tf)
            forecast_price, price_rate = forecast_price_per_time_pythagorean(
                timeframe, candle_map[timeframe], min_threshold_tf, max_threshold_tf, 
                current_price, time_window, last_reversal, last_reversal_type
            )
            pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
        else:
            print(f"Warning: No candle data available for {timeframe}. Skipping forecast.")

    forecasted_price = forecast_volume_based_on_conditions(volume_ratios, min_threshold or 0, current_price)
    forecast_decision = check_market_conditions_and_forecast(support_levels, resistance_levels, current_price)

    conditions_status = {
        "volume_bullish_1m": False,
        "ML_Forecasted_Price_over_Current_Close": False,
        "current_close_below_average_threshold_5m": False,
        "dip_confirmed_1m": False,
        "dip_confirmed_3m": False,
        "dip_confirmed_5m": False,
    }

    last_major_reversal_type = None
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map and candle_map[timeframe]:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = closes[-1]

            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_price, min_threshold, max_threshold)
            dip_confirmed = closest_type == 'DIP'
            conditions_status[f'dip_confirmed_{timeframe}'] = dip_confirmed

            if timeframe == '1m':
                print(f"Dip Confirmed on 1min TF: {'True' if dip_confirmed else 'False'}")
                conditions_status["ML_Forecasted_Price_over_Current_Close"] = adjusted_forecasted_price is not None and adjusted_forecasted_price > current_close
            elif timeframe == '3m':
                print(f"Dip Confirmed on 3min TF: {'True' if dip_confirmed else 'False'}")
            elif timeframe == '5m':
                print(f"Dip Confirmed on 5min TF: {'True' if dip_confirmed else 'False'}")
                conditions_status["current_close_below_average_threshold_5m"] = current_close < avg_mtf if avg_mtf is not None else False

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

            stoch_k, stoch_d = calculate_stochastic_rsi(closes)
            print(f"Current Stochastic K: {stoch_k[-1]:.25f}")
            print(f"Current Stochastic D: {stoch_d[-1]:.25f}")

            negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
            market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, closest_type, buy_volume[timeframe][-1], sell_volume[timeframe][-1])
            print(f"Market Sentiment: {market_sentiment}")

            conditions_status["volume_bullish_1m"] = buy_volume.get('1m', [0])[-1] > sell_volume.get('1m', [0])[-1]

            if closest_reversal is not None:
                print(f"Most Recent Major Reversal Type: {closest_type}")
                print(f"Last Major Reversal Found at Price: {closest_reversal:.25f}")
                last_major_reversal_type = closest_type  
            else:
                print("No Major Reversal Found")

            projected_price = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price:.25f}" if projected_price is not None else "No projection available")
            print(f"Current Close: {closes[-1]:.25f}")
            print(f"Minimum Threshold: {min_threshold:.25f}" if min_threshold is not None else "Minimum Threshold: Not available")
            print(f"Maximum Threshold: {max_threshold:.25f}" if max_threshold is not None else "Maximum Threshold: Not available")
            print(f"Average MTF: {avg_mtf:.25f}" if avg_mtf is not None else "Average MTF: Not available")
            print(f"Momentum Signal: {momentum_signal:.25f}" if momentum_signal is not None else "Momentum Signal: Not available")
            print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bullish Ratio: Not available")
            print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bearish Ratio: Not available")
            print(f"Status: {volume_ratios[timeframe]['status']}" if timeframe in volume_ratios else "Status: Not available")

            avg = (min_threshold + max_threshold) / 2 if min_threshold is not None and max_threshold is not None else current_price
            wave_price = calculate_wave_price(len(closes), avg, min_threshold or 0, max_threshold or float('inf'), omega=0.1, phi=0)
            print(f"Calculated Wave Price: {wave_price:.25f}")

            independent_wave_price = calculate_independent_wave_price(current_price, avg, min_threshold or 0, max_threshold or float('inf'), range_distance=0.1)
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

            fib_reversal_price = forecast_fibo_target_price(fib_info.get(timeframe, {}))
            print(f"{timeframe} Incoming Fibonacci Reversal Target (Forecast): {fib_reversal_price:.25f}" if fib_reversal_price is not None else f"{timeframe} Incoming Fibonacci Reversal Target: price not available.")
            
            if pythagorean_forecasts[timeframe]['price'] is not None:
                print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
            else:
                print(f"Pythagorean Forecast Price for {timeframe}: Not available")
            if pythagorean_forecasts[timeframe]['rate'] is not None:
                print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")
            else:
                print(f"Pythagorean Price Rate for {timeframe}: Not available")
        else:
            print(f"--- {timeframe} --- No data available.")

    usdc_balance = get_balance('USDC')
    asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])

    if position_open:
        current_value_in_usdc = asset_balance * current_price
        print(f"Current {TRADE_SYMBOL.split('USDC')[0]} Balance Value in USDC: {current_value_in_usdc:.25f}")
        print(f"Initial USDC amount: {initial_investment:.25f}, Entry Price for last {TRADE_SYMBOL.split('USDC')[0]} purchased: {entry_price:.25f}")
        
        percentage_change = ((current_value_in_usdc - initial_investment) / initial_investment) * 100 if initial_investment > 0 else 0
        print(f"Percentage Change from Initial Investment: {percentage_change:.25f}%")

        if check_exit_condition(initial_investment, asset_balance):
            print("Target profit of 3% reached or exceeded. Initiating exit...")
            try:
                step_precision = int(-math.log10(step_size)) if step_size > 0 else 8
                sell_quantity = math.floor(asset_balance / step_size) * step_size
                sell_quantity = round(sell_quantity, step_precision)
                
                if sell_quantity < min_trade_size:
                    print(f"Cannot sell: Adjusted quantity {sell_quantity:.10f} is below minimum trade size {min_trade_size}.")
                else:
                    sell_order = client.order_market_sell(
                        symbol=TRADE_SYMBOL,
                        quantity=sell_quantity
                    )
                    print(f"Market sell order executed: {sell_order}")
                    exit_usdc_balance = get_balance('USDC')
                    profit = exit_usdc_balance - initial_investment
                    profit_percentage = (profit / initial_investment) * 100 if initial_investment > 0 else 0.0

                    print(f"Position closed. Sold {TRADE_SYMBOL.split('USDC')[0]} for USDC: {current_value_in_usdc:.25f}")
                    print(f"Trade log: Time: {current_local_time}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}, Profit: {profit:.25f} USDC, Profit Percentage: {profit_percentage:.25f}%")

                    position_open = False
                    initial_investment = 0.0 
                    asset_balance = 0.0 
            except BinanceAPIException as e:
                print(f"Error executing sell order: {e.message}")
    else:
        if usdc_balance > 0:
            print(f"Current USDC balance found: {usdc_balance:.25f}")
        else:
            print("No USDC balance available.")
        print(f"Current {TRADE_SYMBOL.split('USDC')[0]} balance: {asset_balance:.25f} {TRADE_SYMBOL.split('USDC')[0]}")

    true_conditions_count = sum(int(status) for status in conditions_status.values())
    false_conditions_count = len(conditions_status) - true_conditions_count
    print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False\n")
    print("Individual Condition Status:")
    for condition, status in conditions_status.items():
        print(f"{condition}: {'True' if status else 'False'}")

    if not position_open:
        all_conditions_met = all(conditions_status.values())
        print(f"All Conditions Met for Entry: {'Yes' if all_conditions_met else 'No'}")
        if all_conditions_met:
            usdc_balance = get_balance('USDC')  # Refresh balance right before buy
            if asset_balance > 0:
                current_value_in_usdc = asset_balance * current_price
                if check_exit_condition(initial_investment, asset_balance):
                    print(f"Exit conditions met for existing {TRADE_SYMBOL.split('USDC')[0]}; selling the position...")
                    try:
                        step_precision = int(-math.log10(step_size)) if step_size > 0 else 8
                        sell_quantity = math.floor(asset_balance / step_size) * step_size
                        sell_quantity = round(sell_quantity, step_precision)
                        
                        if sell_quantity < min_trade_size:
                            print(f"Cannot sell: Adjusted quantity {sell_quantity:.10f} is below minimum trade size {min_trade_size}.")
                        else:
                            sell_order = client.order_market_sell(
                                symbol=TRADE_SYMBOL,
                                quantity=sell_quantity
                            )
                            print(f"Market sell order executed for {TRADE_SYMBOL.split('USDC')[0]} position: {sell_order}")
                            asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])
                            usdc_balance = get_balance('USDC')
                    except BinanceAPIException as e:
                        print(f"Error executing sell order: {e.message}")
            elif usdc_balance > 0:
                print(f"Trigger signal detected! Attempting to buy {TRADE_SYMBOL} with entire USDC balance: {usdc_balance:.25f} at price {current_price:.2f}")
                entry_price, quantity_bought = buy_asset()
                if entry_price is not None and quantity_bought is not None:
                    initial_investment = usdc_balance
                    print(f"{TRADE_SYMBOL.split('USDC')[0]} was bought at entry price of {entry_price:.25f} USDC for quantity: {quantity_bought:.10f} {TRADE_SYMBOL.split('USDC')[0]}.")
                    position_open = True
                    print(f"New position opened with {usdc_balance:.25f} USDC at price {current_price:.25f}.")
                    # Update balances after successful buy
                    usdc_balance = get_balance('USDC')
                    asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])
                else:
                    print("Error placing buy order.")
            else:
                print(f"No USDC balance to invest in {TRADE_SYMBOL.split('USDC')[0]}.")

    del candle_map
    gc.collect()

    print(f"\nCurrent USDC balance: {usdc_balance:.25f}")
    print(f"Current {TRADE_SYMBOL.split('USDC')[0]} balance: {asset_balance:.25f} {TRADE_SYMBOL.split('USDC')[0]}")
    print(f"Current {TRADE_SYMBOL} price: {current_price:.25f}\n")

    time.sleep(5)
