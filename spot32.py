import pandas as pd
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

# Load credentials from a file
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
                "status": "Bullish" if ratio > 50 else "Bearish" if ratio < 50 else "Neutral",
                "intensity": buy_volume[timeframe] / total_volume if total_volume > 0 else 0
            }
        else:
            volume_ratio[timeframe] = {
                "buy_ratio": 0,
                "sell_ratio": 0,
                "status": "No Activity",
                "intensity": 0
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
    
    if np.all(np.isnan(sine_wave)):
        return 0, 0, np.nan

    current_sine = np.nan_to_num(sine_wave)[-1]
    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0

    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_spectral_analysis(prices):
    """Calculate the FFT of the closing prices and analyze frequencies."""
    fft_result = np.fft.fft(prices)
    power_spectrum = np.abs(fft_result)**2
    frequencies = np.fft.fftfreq(len(prices))

    negative_freqs = frequencies[frequencies < 0]
    negative_powers = power_spectrum[frequencies < 0]

    positive_freqs = frequencies[frequencies >= 0]
    positive_powers = power_spectrum[frequencies >= 0]

    return negative_freqs, negative_powers, positive_freqs, positive_powers

def harmonic_sine_analysis(close_prices):
    """Analyzes the harmonic sine characteristics of the closing prices."""
    sine_wave, _ = talib.HT_SINE(np.array(close_prices))
    current_sine = np.nan_to_num(sine_wave)[-1]

    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    fft_result = np.fft.fft(close_prices)
    power_spectrum = np.abs(fft_result)**2
    frequencies = np.fft.fftfreq(len(close_prices))

    negative_freqs = frequencies[frequencies < 0]
    positive_freqs = frequencies[frequencies > 0]

    significant_negative_freq = negative_freqs[power_spectrum[frequencies < 0].argsort()[-25:]]
    significant_positive_freq = positive_freqs[power_spectrum[frequencies > 0].argsort()[-25:]]

    total_negative_power = np.sum(power_spectrum[frequencies < 0])
    total_positive_power = np.sum(power_spectrum[frequencies > 0])
    predominant_frequency = "Negative" if total_positive_power > total_negative_power else "Positive"

    results = {
        "current_sine": current_sine,
        "sine_wave_min": sine_wave_min,
        "sine_wave_max": sine_wave_max,
        "significant_negative_freq": significant_negative_freq,
        "significant_positive_freq": significant_positive_freq,
        "predominant_frequency": predominant_frequency
    }

    return results

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

    return [level[0] for level in significant_support], [level[0] for level in significant_resistance]

def calculate_stochastic_rsi(close_prices, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI."""
    rsi = talib.RSI(np.array(close_prices), timeperiod=length_rsi)
    min_rsi = talib.MIN(rsi, timeperiod=length_stoch)
    max_rsi = talib.MAX(rsi, timeperiod=length_stoch)
    
    stoch_k = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    stoch_k_smooth = talib.EMA(stoch_k, timeperiod=smooth_k)
    stoch_d = talib.EMA(stoch_k_smooth, timeperiod=smooth_d)

    return stoch_k_smooth, stoch_d

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

def analyze_volume_for_reversals(candle_map, current_close):
    """Analyze volume for major reversals based on recent candles."""
    volume_analysis = {
        'DIP': {'price': None, 'volume': None, 'type': 'Support', 'intensity': None},
        'TOP': {'price': None, 'volume': None, 'type': 'Resistance', 'intensity': None}
    }

    total_buy_volume = 0
    total_sell_volume = 0
    
    # Analyze volumes for reversals and compute total volumes
    for timeframe in candle_map:
        candles = candle_map[timeframe]
        for candle in candles:
            if candle['close'] < current_close:  # Check potential dip
                volume_analysis['DIP']['price'] = candle['close']
                volume_analysis['DIP']['volume'] = candle['volume']
                volume_analysis['DIP']['intensity'] = candle['volume']
            if candle['close'] > current_close:  # Check potential top
                volume_analysis['TOP']['price'] = candle['close']
                volume_analysis['TOP']['volume'] = candle['volume']
                volume_analysis['TOP']['intensity'] = candle['volume']
            total_buy_volume += candle['volume'] if candle['close'] > candle['open'] else 0
            total_sell_volume += candle['volume'] if candle['close'] < candle['open'] else 0

    # Calculate intensity based on total buy/sell volumes
    if volume_analysis['DIP']['volume']:
        volume_analysis['DIP']['intensity'] = volume_analysis['DIP']['volume'] / total_buy_volume * 100 if total_buy_volume > 0 else 0
    if volume_analysis['TOP']['volume']:
        volume_analysis['TOP']['intensity'] = volume_analysis['TOP']['volume'] / total_sell_volume * 100 if total_sell_volume > 0 else 0

    return volume_analysis, total_buy_volume, total_sell_volume

def calculate_smi(candle_df, length=14):
    """Calculate Stochastic Momentum Index (SMI) based on close prices."""
    if len(candle_df) < length:
        return pd.DataFrame({'smi': [None] * len(candle_df)})

    high_max = candle_df['high'].rolling(window=length).max()
    low_min = candle_df['low'].rolling(window=length).min()
    rsi = talib.RSI(candle_df['close'].values, timeperiod=length)

    stoch_value = ((rsi - low_min) / (high_max - low_min)) * 100
    smi = pd.Series(stoch_value).rolling(window=length).mean()  # SMI calculation

    return pd.DataFrame({'smi': smi})

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
    candle_map = fetch_candles_in_parallel(['1m', '3m', '5m', '1d'])  # Adding daily candles for macro analysis
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
        closes = [candle['close'] for candle in candle_map["1m"]]
        min_threshold, max_threshold, _, _, _, _, _ = calculate_thresholds(closes)
        adjusted_forecasted_price = np.clip(forecasted_prices[-1], min_threshold, max_threshold)
        print(f"Forecasted Price: {adjusted_forecasted_price:.2f}")

    # Calculate volume details
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

    # Analyze volume for reversals
    volume_analysis, total_buy_volume, total_sell_volume = analyze_volume_for_reversals(candle_map, current_btc_price)
    print(f"Volume Analysis for Reversals: ")
    print(f"  DIP Price: {volume_analysis['DIP']['price']}")
    print(f"  DIP Volume: {volume_analysis['DIP']['volume']}")
    print(f"  DIP Intensity: {volume_analysis['DIP']['intensity']}")
    print(f"  TOP Price: {volume_analysis['TOP']['price']}")
    print(f"  TOP Volume: {volume_analysis['TOP']['volume']}")
    print(f"  TOP Intensity: {volume_analysis['TOP']['intensity']}")
    print(f"Total Buy Volume: {total_buy_volume:.2f}, Total Sell Volume: {total_sell_volume:.2f}")

    # Find specific support and resistance levels
    support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_btc_price)

    # Adding macro reversals - Fixed unpacking error
    last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map['1d'], current_btc_price, min_threshold, max_threshold)
    
    # Safe printing handling for NoneTypes
    last_bottom_print = last_bottom if last_bottom is not None else "None"
    last_top_print = last_top if last_top is not None else "None"

    print(f"Last Major Macro Bottom: {last_bottom_print}, Last Major Macro Top: {last_top_print}")

    # Initialize conditions status
    conditions_status = {
        "volume_bullish_1m": False,
        "dist_to_min_less_than_max_on_sine_1m": False,
        "dist_to_min_less_than_max_on_sine_3m": False,
        "dist_to_min_less_than_max_on_sine_5m": False,
        "current_close_below_average_threshold_5m": False,
        "dip_confirmed_1m": False,
        "dip_confirmed_3m": False,
        "dip_confirmed_5m": False,
    }

    # Variable to track major reversal types
    last_bottom = None
    last_top = None

    # Check conditions for each timeframe
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = closes[-1]

            # Calculate thresholds
            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_btc_price, min_threshold, max_threshold)
            conditions_status[f"dip_confirmed_{timeframe}"] = closest_type == 'DIP'
            print(f"Dip Confirmed on {timeframe} TF: {'True' if conditions_status[f'dip_confirmed_{timeframe}'] else 'False'}")

            # Check distances to minimum and maximum
            distances_to_min, distances_to_max, current_sine = scale_to_sine(closes)

            # New Percentage Calculation
            percent_distance_min = ((current_close - min_threshold) / (max_threshold - min_threshold)) * 100 if (max_threshold - min_threshold) != 0 else 0
            percent_distance_max = ((max_threshold - current_close) / (max_threshold - min_threshold)) * 100 if (max_threshold - min_threshold) != 0 else 0

            # Clipping to ensure proper bounds
            percent_distance_min_symmetrical = np.clip(percent_distance_min, 0, 100)
            percent_distance_max_symmetrical = np.clip(percent_distance_max, 0, 100)

            # Update the new sine conditions
            conditions_status[f"dist_to_min_less_than_max_on_sine_{timeframe}"] = distances_to_min < distances_to_max

            # Calculate harmonic sine data
            sine_data = harmonic_sine_analysis(closes)

            # Print calculated distances
            print(f"Distance to Min Threshold: {percent_distance_min_symmetrical:.2f}%, Distance to Max Threshold: {percent_distance_max_symmetrical:.2f}%")
            print(f"Predominant Frequency: {sine_data['predominant_frequency']}")
            print(f"Distance to Min on {timeframe}: {distances_to_min:.2f}%, Distance to Max: {distances_to_max:.2f}%")
            print(f"Current Close: {current_close:.2f}, Current Sine: {current_sine:.2f}")

            # Get SMI data and calculate distance ratios
            smi_df = calculate_smi(pd.DataFrame(candle_map[timeframe]))  # Convert current candles to a DataFrame for SMI calculation
            current_smi = smi_df['smi'].iloc[-1] if not smi_df['smi'].isnull().all() else None

            # Calculate thresholds for SMI
            smi_min = smi_df['smi'].min()
            smi_max = smi_df['smi'].max()

            # Calculate distances and percentages for SMI
            if current_smi is not None and smi_max - smi_min != 0:
                smi_dist_to_min = ((current_smi - smi_min) / (smi_max - smi_min)) * 100
                smi_dist_to_max = ((smi_max - current_smi) / (smi_max - smi_min)) * 100
            else:
                smi_dist_to_min = smi_dist_to_max = 0

            # Print SMI details
            print(f"Current SMI ({timeframe}): {current_smi:.2f} (Dist to Min: {smi_dist_to_min:.2f}%, Dist to Max: {smi_dist_to_max:.2f}%)")

            negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
            market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers)
            print(f"Market Sentiment: {market_sentiment}")

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

            conditions_status["volume_bullish_1m"] = buy_volume['1m'] > sell_volume['1m']
            conditions_status["current_close_below_average_threshold_5m"] = current_close < avg_mtf

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

    # Adjust the overall conditions status
    true_conditions_count = sum(int(status) for status in conditions_status.values())
    false_conditions_count = len(conditions_status) - true_conditions_count
    print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
    print("Conditions Details:")
    for cond, status in conditions_status.items():
        print(f"{cond}: {'True' if status else 'False'}")

    # Check all conditions before executing an entry trade
    if not position_open:
        all_conditions_met = all(conditions_status.values())
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