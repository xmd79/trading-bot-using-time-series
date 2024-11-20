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

def scale_to_sine(close_prices, adjustment=False):
    sine_wave, _ = talib.HT_SINE(np.array(close_prices))
    current_sine = np.nan_to_num(sine_wave)[-1]
    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)

    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0

    if adjustment:
        if dist_from_close_to_min >= dist_from_close_to_max:
            adjustment_factor = (dist_from_close_to_min + dist_from_close_to_max) / 2
            sine_wave_adjusted = sine_wave - (sine_wave_max - sine_wave_min) * adjustment_factor / 200

            sine_wave_min_adjusted = np.nanmin(sine_wave_adjusted)
            sine_wave_max_adjusted = np.nanmax(sine_wave_adjusted)

            dist_from_close_to_min = ((current_sine - sine_wave_min_adjusted) / (sine_wave_max_adjusted - sine_wave_min_adjusted)) * 100 if (sine_wave_max_adjusted - sine_wave_min_adjusted) != 0 else 0
            dist_from_close_to_max = ((sine_wave_max_adjusted - current_sine) / (sine_wave_max_adjusted - sine_wave_min_adjusted)) * 100 if (sine_wave_max_adjusted - sine_wave_min_adjusted) != 0 else 0

    return dist_from_close_to_min, dist_from_close_to_max

def log_trade(timestamp, btc_price, usdc_balance, operation, profit_percentage, usdc_diff):
    print(f"Trade: Timestamp: {timestamp}, BTC Price: {btc_price:.2f}, USDC Balance: {usdc_balance:.2f}, "
          f"Operation: {operation}, Profit Percentage: {profit_percentage:.2f}%, USDC Difference: {usdc_diff:.2f}")

def calculate_45_degree_projection(last_bottom, last_top):
    angle_distance = last_top - last_bottom
    projected_price_45 = last_top + angle_distance
    return projected_price_45 

def calculate_golden_ratio_projection(last_close, angle_projection):
    phi = 1.618
    projected_price_golden = last_close + (angle_projection - last_close) * phi
    return projected_price_golden 

def forecast_and_adjust_price(forecasted_price, min_threshold, max_threshold):
    return np.clip(forecasted_price, min_threshold, max_threshold)

def evaluate_forecast(current_price, support, resistance):
    if current_price < support:
        return support  
    elif current_price > resistance:
        return resistance  
    else:
        return current_price  

def calculate_stochastic_rsi(close_prices, high_prices, low_prices, length_rsi, length_stoch, smooth_k, smooth_d, lower_band, upper_band):
    rsi = talib.RSI(np.array(close_prices), timeperiod=length_rsi)
    stoch_k, stoch_d = talib.STOCHF(np.array(high_prices), np.array(low_prices), np.array(close_prices), fastk_period=length_stoch, fastd_period=smooth_d)
    return stoch_k, stoch_d

def determine_market_trend(last_bottom, last_top, last_major_reversal_type):
    if last_major_reversal_type == 'DIP':
        return "Upcycle"
    elif last_major_reversal_type == 'TOP':
        return "Downcycle"
    else:
        return None  

def get_min_trade_size(symbol):
    """Fetch the minimum trade size for a symbol from Binance."""
    exchange_info = client.get_symbol_info(symbol)
    for filter in exchange_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            return float(filter['minQty'])
    return 0.0

def calculate_tsi(close_prices, long_length=25, short_length=5, signal_length=14):
    """Calculate True Strength Index (TSI)."""
    close_prices = np.array(close_prices)
    
    # Calculate momentum
    momentum = close_prices - np.roll(close_prices, 1)

    # Get the smoothed momentum values
    smoothed_momentum = talib.EMA(momentum, timeperiod=short_length) - talib.EMA(momentum, timeperiod=long_length)
    
    # Calculate TSI = 100 * ( Smoothed Momentum / Smoothed Price )
    price_momentum = talib.EMA(close_prices, timeperiod=short_length) - talib.EMA(close_prices, timeperiod=long_length)
    tsi = 100 * (smoothed_momentum / price_momentum)

    # Get the signal for TSI
    tsi_signal = talib.EMA(tsi, timeperiod=signal_length)

    return tsi, tsi_signal

# Instantiate the minimum trade size for the trading pair
min_trade_size = get_min_trade_size(TRADE_SYMBOL)

# Initialize variables for tracking trade state
position_open = False
initial_investment = 0.0
btc_balance = 0.0

# Starting main trading loop
while True:  
    current_local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Local Time: {current_local_time}")

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
            min_threshold, max_threshold, _, _, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(closes)
            adjusted_forecasted_price = forecast_and_adjust_price(forecasted_prices[-1], min_threshold, max_threshold)
            print(f"Forecasted Price: {adjusted_forecasted_price:.2f}")

    # Calculate volume details
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

    # Calculate TSI for all timeframes and print the requested values 
    tsi_values = {}
    tsi_signals = {}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            closes = [candle['close'] for candle in candle_map[timeframe]]
            tsi_values[timeframe], tsi_signals[timeframe] = calculate_tsi(closes)

            if tsi_values[timeframe].size > 0:
                current_tsi_value = tsi_values[timeframe][-1]
                min_tsi_value = np.nanmin(tsi_values[timeframe])
                max_tsi_value = np.nanmax(tsi_values[timeframe])

                dist_to_min_tsi = ((current_tsi_value - min_tsi_value) / (max_tsi_value - min_tsi_value)) * 100 if (max_tsi_value - min_tsi_value) != 0 else 0
                dist_to_max_tsi = ((max_tsi_value - current_tsi_value) / (max_tsi_value - min_tsi_value)) * 100 if (max_tsi_value - min_tsi_value) != 0 else 0

                print(f"Current TSI Value for {timeframe}: {current_tsi_value:.2f}")
                print(f"Current TSI Signal for {timeframe}: {tsi_signals[timeframe][-1]:.2f}")
                print(f"Distance to TSI Min for {timeframe}: {dist_to_min_tsi:.2f}% | Distance to TSI Max for {timeframe}: {dist_to_max_tsi:.2f}%")

    # MTF TSI Calculation
    if all(timeframe in tsi_values for timeframe in ['1m', '3m', '5m']):
        avg_tsi_value = np.mean([tsi_values[tf][-1] for tf in ['1m', '3m', '5m']])
        avg_tsi_signal = np.mean([tsi_signals[tf][-1] for tf in ['1m', '3m', '5m']])
        
        # Calculate minimum and maximum TSI values from individual timeframes
        min_mtf_tsi = min([tsi_values[tf][-1] for tf in ['1m', '3m', '5m']])
        max_mtf_tsi = max([tsi_values[tf][-1] for tf in ['1m', '3m', '5m']])

        # Calculate distances to min and max
        dist_to_min_mtf_tsi = ((avg_tsi_value - min_mtf_tsi) / (max_mtf_tsi - min_mtf_tsi)) * 100 if (max_mtf_tsi - min_mtf_tsi) != 0 else 0
        dist_to_max_mtf_tsi = ((max_mtf_tsi - avg_tsi_value) / (max_mtf_tsi - min_mtf_tsi)) * 100 if (max_mtf_tsi - min_mtf_tsi) != 0 else 0

        print(f"MTF TSI Value: {avg_tsi_value:.2f}")
        print(f"MTF TSI Signal: {avg_tsi_signal:.2f}")
        print(f"Distance to MTF TSI Min: {dist_to_min_mtf_tsi:.2f}% | Distance to MTF TSI Max: {dist_to_max_mtf_tsi:.2f}%")

    # Enhance conditions status to include additional checks
    conditions_status = {
        "current_close_above_min_threshold": False,
        "dip_condition_met": False,
        "volume_bullish_1m": False,
        "dist_to_min_less_than_max_1m": False,
        "dist_to_min_less_than_max_5m": False,
        "current_close_below_average_threshold": False,
        "forecast_price_condition_met": False,
        "mtf_bullish_signal": False,  # New flag for MTF bullish signal
    }

    mtf_bullish_signals = []

    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = closes[-1]
            highs = [candle['high'] for candle in candle_map[timeframe]]
            lows = [candle['low'] for candle in candle_map[timeframe]]

            # Calculate thresholds and other indicators
            min_threshold, max_threshold, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(
                closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
            )

            # Get stochastic RSI values
            stoch_k, stoch_d = calculate_stochastic_rsi(closes, highs, lows, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3, lower_band=20, upper_band=80)
            current_stoch_k = stoch_k[-1]
            current_stoch_d = stoch_d[-1]

            # Print Stochastic RSI details
            print(f"Current Stochastic K: {current_stoch_k:.2f}")
            print(f"Current Stochastic D: {current_stoch_d:.2f}")

            # Determine if the MTF bullish signal condition is met
            is_bullish_volume = volume_ratios[timeframe]['status'] == 'Bullish'
            is_bearish_volume = volume_ratios[timeframe]['status'] == 'Bearish'

            # Generate signals and calculations for each timeframe
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_btc_price, min_threshold, max_threshold)
            
            if closest_type == 'DIP':
                if current_close < max_threshold and is_bullish_volume:
                    mtf_bullish_signals.append(True)
                elif current_close > min_threshold and is_bearish_volume:
                    mtf_bullish_signals.append(False)
            elif closest_type == 'TOP':
                if current_close > min_threshold and is_bearish_volume:
                    mtf_bullish_signals.append(False)
                elif current_close < max_threshold and is_bullish_volume:
                    mtf_bullish_signals.append(True)

            # Check conditions
            conditions_status["current_close_above_min_threshold"] = current_close > min_threshold
            conditions_status["forecast_price_condition_met"] = (adjusted_forecasted_price is not None) and (
                (closest_type == 'DIP' and current_close < adjusted_forecasted_price) or
                (closest_type == 'TOP' and current_close > adjusted_forecasted_price)
            )
            conditions_status["current_close_below_average_threshold"] = current_close < avg_mtf

            if closest_reversal is not None:
                print(f"Most Recent Major Reversal Type: {closest_type}")
                print(f"Last Major Reversal Found at Price: {closest_reversal:.2f}")

            print(f"Projected Price Using 45-Degree Angle: {calculate_45_degree_projection(last_bottom, last_top):.2f}"
                  if last_bottom is not None and last_top is not None else "No 45-Degree Projection")
            print(f"Current Close: {closes[-1]:.2f}")
            print(f"Minimum Threshold: {min_threshold:.2f}")
            print(f"Maximum Threshold: {max_threshold:.2f}")
            print(f"Average MTF: {avg_mtf:.2f}")
            print(f"Momentum Signal: {momentum_signal:.2f}")
            print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.2f}%")
            print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.2f}%")
            print(f"Status: {volume_ratios[timeframe]['status']}")

            market_trend = determine_market_trend(last_bottom, last_top, closest_type)
            print(f"Market Trend: {market_trend}" if market_trend else "Market Trend: Undefined")

            print()  # Extra line for spacing between timeframes

    # Overall conditions summary
    true_conditions_count = sum(int(status) for status in conditions_status.values())
    false_conditions_count = len(conditions_status) - true_conditions_count
    print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
    print("Conditions Details:")
    for cond, status in conditions_status.items():
        print(f"{cond}: {'True' if status else 'False'}")

    # Check all conditions before executing an entry trade
    if not position_open:
        # Ensure all conditions are true
        all_conditions_met = all(conditions_status.values())
        if all_conditions_met:
            if usdc_balance > 0:
                amount_to_invest = usdc_balance
                quantity_to_buy = amount_to_invest / current_btc_price
                
                # Check if the calculated quantity is greater than the minimum trade size
                if quantity_to_buy >= min_trade_size:
                    btc_order = buy_btc(quantity_to_buy)  # Place the order with the valid quantity
                    if btc_order:
                        initial_investment = amount_to_invest
                        btc_balance += quantity_to_buy  # track the BTC amount purchased
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
    print(f"Current USDC balance: {usdc_balance:.2f}")
    print(f"Current BTC balance: {btc_balance:.4f} BTC")
    print(f"Current BTC price: {current_btc_price:.2f}")
    
    time.sleep(5)  # Wait before the next iteration