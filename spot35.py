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
    return current_value >= (initial_investment * 1.0267)

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
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * 100 if max_momentum - min_momentum != 0 else np.nan               
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * 100 if max_momentum - min_momentum != 0 else np.nan

    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum

def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in candle_map:
        if timeframe in candle_map:
            buy_volume[timeframe] = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] > candle["open"])
            sell_volume[timeframe] = sum(candle["volume"] for candle in candle_map[timeframe] if candle["close"] < candle["open"])
    return buy_volume, sell_volume 

def calculate_volume_ratio(buy_volume, sell_volume):
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

    if last_bottom is not None and (closest_reversal is None or abs(last_bottom - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_bottom
        closest_type = 'DIP'
    
    if last_top is not None and (closest_reversal is None or abs(last_top - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_top
        closest_type = 'TOP'

    return last_bottom, last_top, closest_reversal, closest_type 

def scale_to_sine(close_prices):
    """Scale close prices to sine wave and return distances to min and max."""
    sine_wave, _ = talib.HT_SINE(np.array(close_prices))
    current_sine = np.nan_to_num(sine_wave)[-1]
    
    # Remove NaN values from sine wave for proper calculations
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    if sine_wave_max <= sine_wave_min:
        return 0, 100, current_sine  # If the min and max are the same, avoid division errors.

    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    # Normalize distance to ensure it remains between 0 and 100
    dist_from_close_to_min = np.clip(dist_from_close_to_min, 0, 100)
    dist_from_close_to_max = np.clip(dist_from_close_to_max, 0, 100)

    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_stochastic_rsi(close_prices, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI."""
    rsi = talib.RSI(np.array(close_prices), timeperiod=length_rsi)
    min_rsi = talib.MIN(rsi, timeperiod=length_stoch)
    max_rsi = talib.MAX(rsi, timeperiod=length_stoch)

    stoch_k = np.where(max_rsi - min_rsi != 0, (rsi - min_rsi) / (max_rsi - min_rsi) * 100, 0)
    stoch_k_smooth = talib.EMA(stoch_k, timeperiod=smooth_k)
    stoch_d = talib.EMA(stoch_k_smooth, timeperiod=smooth_d)

    return stoch_k_smooth, stoch_d

def calculate_spectral_analysis(prices):
    """Calculate the FFT of the closing prices and analyze frequencies."""
    fft_result = np.fft.fft(prices)
    power_spectrum = np.abs(fft_result) ** 2
    frequencies = np.fft.fftfreq(len(prices))

    negative_freqs = frequencies[frequencies < 0]
    negative_powers = power_spectrum[frequencies < 0]

    positive_freqs = frequencies[frequencies >= 0]
    positive_powers = power_spectrum[frequencies >= 0]

    return negative_freqs, negative_powers, positive_freqs, positive_powers

def determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers):
    """Determine if the market is predominantly negative or positive based on frequencies."""
    total_negative_power = np.sum(negative_powers)
    total_positive_power = np.sum(positive_powers)

    if total_negative_power > total_positive_power:
        return "Predominantly Negative"
    elif total_positive_power > total_negative_power:
        return "Predominantly Positive"
    else:
        return "Neutral"

def get_min_trade_size(symbol):
    """Fetch the minimum trade size for a symbol from Binance."""
    exchange_info = client.get_symbol_info(symbol)
    for filter in exchange_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            return float(filter['minQty'])
    return 0.0

def calculate_45_degree_projection(last_bottom, last_top):
    """Calculate the projected price using a 45-degree angle."""
    if last_bottom is not None and last_top is not None:
        distance = last_top - last_bottom
        projection = last_top + distance
        return projection
    return None

def find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_close):
    """Identify specific support and resistance levels based on volume and thresholds."""
    support_levels = []
    resistance_levels = []

    for timeframe in candle_map:
        candles = candle_map[timeframe]
        for candle in candles:
            if candle["close"] < current_close and candle["close"] >= min_threshold:
                support_levels.append((candle["close"], candle["volume"], candle["time"]))
            if candle["close"] > current_close and candle["close"] <= max_threshold:
                resistance_levels.append((candle["close"], candle["volume"], candle["time"]))

    # Sort and select significant support and resistance levels
    support_levels.sort(key=lambda x: x[1], reverse=True)  # Sort by volume
    resistance_levels.sort(key=lambda x: x[1], reverse=True)  # Sort by volume

    significant_support = support_levels[:3]  # Get top 3 support levels
    significant_resistance = resistance_levels[:3]  # Get top 3 resistance levels

    print("Most Significant Support Levels (Price, Volume):")
    for price, volume, timestamp in significant_support:
        print(f"Support Price: {price:.25f}, Volume: {volume:.25f}")

    print("Most Significant Resistance Levels (Price, Volume):")
    for price, volume, timestamp in significant_resistance:
        print(f"Resistance Price: {price:.25f}, Volume: {volume:.25f}")

    return [level[0] for level in significant_support], [level[0] for level in significant_resistance]

def forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_price):
    """Forecast potential volumes based on conditions."""
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

# Enhanced Entry Probability Calculation
def calculate_entry_probability(candles):
    closes = np.array([candle['close'] for candle in candles])
    volumes = np.array([candle['volume'] for candle in candles])
    
    minimum_period = 14  # Define your period for calculations
    length = len(closes)  # Current length

    # Calculate moving averages
    volume_moving_avg = np.mean(volumes[-minimum_period:]) if length >= minimum_period else 0
    current_volume = volumes[-1]
    
    # Calculate Bollinger Bands
    upper_band, middle_band, lower_band = talib.BBANDS(closes, timeperiod=minimum_period, nbdevup=2, nbdevdn=2, matype=0)
    current_close = closes[-1]
    
    # Check the relation of current close to Bollinger Bands
    close_relation = {
        'below_lower': current_close < lower_band[-1],
        'above_upper': current_close > upper_band[-1],
        'middle': middle_band[-1]
    }

    # Check conditions for entry based on Bollinger Bands
    entry_condition = 0
    if close_relation['below_lower']:
        entry_condition = 1  # Potential buy signal
    elif close_relation['above_upper']:
        entry_condition = -1  # Potential sell signal

    # Print Bollinger Bands information
    print(f"Bollinger Bands: Lower: {lower_band[-1]:.2f}, Middle: {middle_band[-1]:.2f}, Upper: {upper_band[-1]:.2f}")
    print(f"Current Close: {current_close:.2f}, Below Lower Band: {close_relation['below_lower']}, Above Upper Band: {close_relation['above_upper']}")

    return entry_condition, close_relation

# Instantiate the minimum trade size for the trading pair
min_trade_size = get_min_trade_size(TRADE_SYMBOL)

# Initialize variables for tracking trade state
position_open = False
initial_investment = 0.0
btc_balance = 0.0
print("Trading Bot Initialized!")

# Starting the main trading loop
while True:
    current_local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nCurrent Local Time: {current_local_time}")

    # Fetch candle data
    candle_map = fetch_candles_in_parallel(['1m', '3m', '5m'])  
    usdc_balance = get_balance('USDC')
    current_btc_price = get_current_btc_price()

    # Instant backtesting on the last 100 candles of 1m timeframe
    model, backtest_mae, last_predictions, actuals = None, None, None, None
    if "1m" in candle_map:
        model, backtest_mae, last_predictions, actuals = backtest_model(candle_map["1m"])
        print(f"Backtest MAE: {backtest_mae:.2f}")

    # Forecast next price using the ML model
    adjusted_forecasted_price = None
    if model is not None:
        forecasted_prices = forecast_next_price(model, num_steps=1)
        if "1m" in candle_map:
            closes = [candle['close'] for candle in candle_map["1m"]]
            min_threshold, max_threshold, _, _, _, _, _ = calculate_thresholds(closes)
            adjusted_forecasted_price = np.clip(forecasted_prices[-1], min_threshold, max_threshold)
            print(f"Forecasted Price: {adjusted_forecasted_price:.2f}")

    # Calculate volume details
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

    # Find specific support and resistance levels
    support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_btc_price)
    
    # Forecast potential volumes based on conditions
    forecasted_price = forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_btc_price)

    # Initialize conditions status
    conditions_status = {
        "volume_bullish_1m": False,
        "ML_Forecasted_Price_over_Current_Close": False,
        "current_close_below_average_threshold_5m": False,
        "dip_confirmed_1m": False,
        "dip_confirmed_3m": False,
        "dip_confirmed_5m": False,  # Added for dip confirmation
    }

    # Check conditions for each timeframe
    for timeframe in ['1m', '3m', '5m']:  # Check all timeframes
        if timeframe in candle_map:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = closes[-1]

            # Calculate thresholds
            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=5, maximum_percentage=5, range_distance=0.05
            )

            # Timestamp extraction for min and max threshold
            min_threshold_time = candle_map[timeframe][np.nanargmin(closes)]['time'] if closes else None
            max_threshold_time = candle_map[timeframe][np.nanargmax(closes)]['time'] if closes else None

            print(f"Minimum Threshold: {min_threshold:.2f} at {datetime.datetime.fromtimestamp(min_threshold_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Maximum Threshold: {max_threshold:.2f} at {datetime.datetime.fromtimestamp(max_threshold_time).strftime('%Y-%m-%d %H:%M:%S')}")

            # Calculate distances to thresholds
            distance_to_min = current_close - min_threshold
            distance_to_max = max_threshold - current_close
            total_distance = abs(distance_to_min) + abs(distance_to_max)

            if total_distance > 0:
                percent_distance_to_min = (abs(distance_to_min) / total_distance) * 100
                percent_distance_to_max = (abs(distance_to_max) / total_distance) * 100
            else:
                percent_distance_to_min = 0.0
                percent_distance_to_max = 0.0

            print(f"Distance to Min Threshold {timeframe}: {distance_to_min:.2f} ({percent_distance_to_min:.2f}%)")
            print(f"Distance to Max Threshold {timeframe}: {distance_to_max:.2f} ({percent_distance_to_max:.2f}%)")

            # Calculate distances for sine wave
            distances_to_min_sine, distances_to_max_sine, current_sine = scale_to_sine(closes)
            print(f"Distance to Min of Sine {timeframe}: {distances_to_min_sine:.2f}")
            print(f"Distance to Max of Sine {timeframe}: {distances_to_max_sine:.2f}")
            print(f"Current Sine value for {timeframe}: {current_sine:.2f}\n")

            # Calculate Bollinger Bands and check relations
            band_entry_condition, close_relation = calculate_entry_probability(candle_map[timeframe])
            if band_entry_condition == 1:
                print(f"Best reversal opportunity detected below lower BB in {timeframe}.")
            elif band_entry_condition == -1:
                print(f"Best reversal opportunity detected above upper BB in {timeframe}.")
            print(f"BB Conditions: {close_relation}")

            # Major reversals
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_btc_price, min_threshold, max_threshold)
            if timeframe == '1m':
                conditions_status["dip_confirmed_1m"] = closest_type == 'DIP'
            elif timeframe == '3m':
                conditions_status["dip_confirmed_3m"] = closest_type == 'DIP'
            elif timeframe == '5m':
                conditions_status["dip_confirmed_5m"] = closest_type == 'DIP'

            # Calculate Stochastic RSI values
            stoch_k, stoch_d = calculate_stochastic_rsi(closes, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3)
            current_stoch_k = stoch_k[-1]
            current_stoch_d = stoch_d[-1]

            # Market Sentiment Analysis
            negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
            market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers)
            print(f"Market Sentiment: {market_sentiment}")

            # Update conditions
            conditions_status["volume_bullish_1m"] = buy_volume['1m'] > sell_volume['1m']
            conditions_status["current_close_below_average_threshold_5m"] = current_close < avg_mtf

            # Check if current close is below the ML forecasted price
            if adjusted_forecasted_price is not None:
                conditions_status["ML_Forecasted_Price_over_Current_Close"] = current_close < adjusted_forecasted_price

            # Check trading signals
            if market_sentiment == "Predominantly Positive":
                if closest_type == 'DIP':
                    print("Signal: BUY (DIP Reversal & Predominantly Positive)")
                elif closest_type == 'TOP':
                    print("Signal: SELL (TOP Reversal & Predominantly Positive)")
            elif market_sentiment == "Predominantly Negative":
                if closest_type == 'DIP':
                    print("Signal: SELL (DIP Reversal & Predominantly Negative)")
                elif closest_type == 'TOP':
                    print("Signal: BUY (TOP Reversal & Predominantly Negative)")

            if closest_reversal is not None:
                print(f"Most Recent Major Reversal Type: {closest_type}")
                print(f"Last Major Reversal Found at Price: {closest_reversal:.2f}")
            else:
                print("No Major Reversal Found")

            projected_price = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price:.2f}" if projected_price is not None else "No projection available")
            print(f"Current Close: {closes[-1]:.2f}")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")
            print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.2f}%")
            print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.2f}%")
            print(f"Status: {volume_ratios[timeframe]['status']}")

    print()

    true_conditions_count = sum(int(status) for status in conditions_status.values())
    false_conditions_count = len(conditions_status) - true_conditions_count
    print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
    print("Conditions Details:")
    for cond, status in conditions_status.items():
        print(f"{cond}: {'True' if status else 'False'}")

    # Check all conditions before executing an entry trade
    if not position_open:
        all_conditions_met = all(value for key, value in conditions_status.items() if key not in ["current_close_below_average_threshold_5m"])
        if all_conditions_met:
            if usdc_balance > 0:  
                amount_to_invest = usdc_balance
                quantity_to_buy = amount_to_invest / current_btc_price
                
                if quantity_to_buy >= min_trade_size:
                    btc_order = buy_btc(quantity_to_buy) 
                    if btc_order:
                        initial_investment = amount_to_invest
                        btc_balance += quantity_to_buy  
                        print(f"New position opened with {amount_to_invest} USDC at price {current_btc_price:.2f}.")
                        position_open = True 
                else:
                    print(f"Cannot place order: Quantity {quantity_to_buy:.6f} is less than minimum trade size {min_trade_size:.6f}.")
        else:
            print("Conditions not met for placing a trade.")

    # Exit Logic
    if position_open and check_exit_condition(initial_investment, btc_balance):
        try:
            sell_order = client.order_market_sell(
                symbol='BTCUSDC',
                quantity=btc_balance
            )
            print(f"Market sell order executed: {sell_order}")
            print(f"Position closed. Sold BTC for USDC: {btc_balance * current_btc_price:.2f}")

            exit_usdc_balance = get_balance('USDC')
            usdc_diff = exit_usdc_balance - initial_investment
            profit_percentage = (usdc_diff / initial_investment) * 100 if initial_investment > 0 else 0.0

            print(f"Trade log: Time: {current_local_time}, Entry Price: {current_btc_price:.2f}, Exit Balance: {exit_usdc_balance:.2f}, Profit Percentage: {profit_percentage:.2f}%, Profit Amount: {usdc_diff:.2f}")

            position_open = False
            initial_investment = 0.0 
            btc_balance = 0.0 
        except BinanceAPIException as e:
            print(f"Error executing sell order: {e.message}")

    # Clean up references and collect garbage
    del candle_map
    gc.collect()  

    print()
    print(f"Current USDC balance: {usdc_balance:.15f}")
    print(f"Current BTC balance: {btc_balance:.15f} BTC")
    print(f"Current BTC price: {current_btc_price:.15f}")
    print()

    time.sleep(5)  # Wait before the next iteration