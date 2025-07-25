import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from decimal import Decimal, getcontext
import requests
import logging
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"  # Futures use USDC
LEVERAGE = 20
RISK_PERCENTAGE = Decimal('0.02')  # 2% risk per trade
STOP_LOSS_PERCENTAGE = Decimal('0.02')  # 2% stop-loss
TAKE_PROFIT_PERCENTAGE = Decimal('0.04')  # 4% take-profit
QUANTITY_PRECISION = Decimal('0.000001')  # Binance quantity precision for BTCUSDC
MINIMUM_BALANCE = Decimal('10.0')  # Minimum USDC balance to place trades

# Load credentials from file
try:
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
except FileNotFoundError:
    logging.error("credentials.txt not found.")
    print("Error: credentials.txt not found.")
    exit(1)
except IndexError:
    logging.error("credentials.txt is incorrectly formatted.")
    print("Error: credentials.txt is incorrectly formatted.")
    exit(1)

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})
client.API_URL = 'https://fapi.binance.com'  # Futures API endpoint

# Set leverage for the symbol
try:
    client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
    logging.info(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
    print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
except BinanceAPIException as e:
    logging.error(f"Error setting leverage: {e.message}")
    print(f"Error setting leverage: {e.message}")

# Utility Functions
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1200):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=1200, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
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
            if e.code == -1003:  # Rate limit error
                logging.warning(f"Rate limit exceeded for {timeframe}. Waiting 60 seconds.")
                print(f"Rate limit exceeded for {timeframe}. Waiting 60 seconds.")
                time.sleep(60)
            else:
                logging.error(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
                print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
        except requests.exceptions.Timeout as e:
            logging.error(f"Read Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Read Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            logging.error(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch candles for {timeframe} after {retries} attempts. Skipping timeframe.")
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts. Skipping timeframe.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                return price
            logging.warning(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            if e.code == -1003:  # Rate limit error
                logging.warning(f"Rate limit exceeded fetching price. Waiting 60 seconds.")
                print(f"Rate limit exceeded fetching price. Waiting 60 seconds.")
                time.sleep(60)
            else:
                logging.error(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}/{retries}): {e.message}")
                print(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}/{retries}): {e.message}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            logging.error(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            print(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    return Decimal('0.0')

def get_balance(asset='USDC'):
    try:
        account = client.futures_account()
        if 'assets' not in account:
            logging.error("No 'assets' key in futures account response.")
            print("No 'assets' key in futures account response. Full response:")
            print(account)
            return Decimal('0.0')
        for asset_info in account['assets']:
            if asset_info.get('asset') == asset:
                wallet = Decimal(str(asset_info.get('walletBalance', '0.0')))
                logging.info(f"{asset} wallet balance: {wallet:.25f}")
                print(f"{asset} wallet balance: {wallet:.25f}")
                return wallet
        logging.warning(f"{asset} not found in futures account balances.")
        print(f"{asset} not found in futures account balances.")
        return Decimal('0.0')
    except BinanceAPIException as e:
        logging.error(f"Binance API exception while fetching {asset} balance: {e.message}")
        print(f"Binance API exception while fetching {asset} balance: {e.message}")
    except Exception as e:
        logging.error(f"Unexpected error fetching {asset} balance: {e}")
        print(f"Unexpected error fetching {asset} balance: {e}")
    return Decimal('0.0')

def get_position():
    try:
        positions = client.futures_position_information(symbol=TRADE_SYMBOL)
        if not positions:
            logging.warning(f"No position data returned for {TRADE_SYMBOL}. Assuming no open position.")
            print(f"No position data returned for {TRADE_SYMBOL}. Assuming no open position.")
            return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0')}
        position = positions[0]
        quantity = Decimal(str(position['positionAmt']))
        entry_price = Decimal(str(position['entryPrice']))
        return {
            "quantity": quantity,
            "entry_price": entry_price,
            "side": "LONG" if quantity > Decimal('0') else "SHORT" if quantity < Decimal('0') else "NONE",
            "unrealized_pnl": Decimal(str(position['unrealizedProfit']))
        }
    except BinanceAPIException as e:
        logging.error(f"Error fetching position info: {e.message}")
        print(f"Error fetching position info: {e.message}")
        return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0')}

# Trading Functions
def calculate_quantity(balance, price):
    if price <= Decimal('0') or balance < MINIMUM_BALANCE:
        logging.warning(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}). Cannot calculate quantity.")
        print(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}). Cannot calculate quantity.")
        return Decimal('0.0')
    quantity = (balance * RISK_PERCENTAGE * LEVERAGE) / price
    return quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')

def place_order(signal, quantity, current_price):
    try:
        if quantity <= Decimal('0'):
            logging.warning(f"Invalid quantity {quantity:.25f}. Skipping order.")
            print(f"Invalid quantity {quantity:.25f}. Skipping order.")
            return
        if signal == "LONG":
            order = client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="BUY",
                type="MARKET",
                quantity=str(quantity)
            )
            logging.info(f"Placed LONG order: {quantity:.25f} BTC at market price ~{current_price:.25f}")
            print(f"Placed LONG order: {quantity:.25f} BTC at market price ~{current_price:.25f}")
            stop_loss_price = (current_price * (Decimal('1') - STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
            take_profit_price = (current_price * (Decimal('1') + TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
            client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="SELL",
                type="STOP_MARKET",
                quantity=str(quantity),
                stopPrice=str(stop_loss_price)
            )
            client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="SELL",
                type="TAKE_PROFIT_MARKET",
                quantity=str(quantity),
                stopPrice=str(take_profit_price)
            )
            logging.info(f"Placed SL: {stop_loss_price:.25f}, TP: {take_profit_price:.25f}")
            print(f"Placed SL: {stop_loss_price:.25f}, TP: {take_profit_price:.25f}")
        elif signal == "SHORT":
            order = client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="SELL",
                type="MARKET",
                quantity=str(quantity)
            )
            logging.info(f"Placed SHORT order: {quantity:.25f} BTC at market price ~{current_price:.25f}")
            print(f"Placed SHORT order: {quantity:.25f} BTC at market price ~{current_price:.25f}")
            stop_loss_price = (current_price * (Decimal('1') + STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
            take_profit_price = (current_price * (Decimal('1') - TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
            client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="BUY",
                type="STOP_MARKET",
                quantity=str(quantity),
                stopPrice=str(stop_loss_price)
            )
            client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="BUY",
                type="TAKE_PROFIT_MARKET",
                quantity=str(quantity),
                stopPrice=str(take_profit_price)
            )
            logging.info(f"Placed SL: {stop_loss_price:.25f}, TP: {take_profit_price:.25f}")
            print(f"Placed SL: {stop_loss_price:.25f}, TP: {take_profit_price:.25f}")
    except BinanceAPIException as e:
        logging.error(f"Error placing order: {e.message}")
        print(f"Error placing order: {e.message}")

def close_position(position, current_price):
    if position["side"] == "NONE" or position["quantity"] == Decimal('0'):
        return
    try:
        quantity = abs(position["quantity"]).quantize(QUANTITY_PRECISION)
        side = "SELL" if position["side"] == "LONG" else "BUY"
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side=side,
            type="MARKET",
            quantity=str(quantity)
        )
        logging.info(f"Closed {position['side']} position: {quantity:.25f} BTC at market price ~{current_price:.25f}")
        print(f"Closed {position['side']} position: {quantity:.25f} BTC at market price ~{current_price:.25f}")
    except BinanceAPIException as e:
        logging.error(f"Error closing position: {e.message}")
        print(f"Error closing position: {e.message}")

# Analysis Functions
def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in candle_map:
        buy_volume[timeframe] = []
        sell_volume[timeframe] = []
        for candle in candle_map[timeframe]:
            total_volume = Decimal(str(candle["volume"]))
            close_price = Decimal(str(candle["close"]))
            open_price = Decimal(str(candle["open"]))
            high_price = Decimal(str(candle["high"]))
            low_price = Decimal(str(candle["low"]))
            if high_price == low_price:
                buy_vol = total_volume / Decimal('2')
                sell_vol = total_volume / Decimal('2')
            else:
                price_range = high_price - low_price
                bullish_strength = (close_price - low_price) / price_range if price_range > Decimal('0') else Decimal('0.5')
                bearish_strength = (high_price - close_price) / price_range if price_range > Decimal('0') else Decimal('0.5')
                buy_vol = total_volume * bullish_strength
                sell_vol = total_volume * bearish_strength
            buy_volume[timeframe].append(buy_vol)
            sell_volume[timeframe].append(sell_vol)
    return buy_volume, sell_volume

def calculate_volume_ratio(buy_volume, sell_volume):
    volume_ratio = {}
    for timeframe in buy_volume:
        if not buy_volume[timeframe] or not sell_volume[timeframe]:
            volume_ratio[timeframe] = {"buy_ratio": Decimal('0'), "sell_ratio": Decimal('0'), "status": "Insufficient Data"}
            logging.warning(f"Insufficient volume data for {timeframe}")
            print(f"Insufficient volume data for {timeframe}")
            continue
        buy_vols = np.array([float(v) for v in buy_volume[timeframe][-3:]], dtype=np.float64)
        sell_vols = np.array([float(v) for v in sell_volume[timeframe][-3:]], dtype=np.float64)
        if len(buy_vols) >= 3:
            buy_ema = Decimal(str(talib.EMA(buy_vols, timeperiod=3)[-1]))
            sell_ema = Decimal(str(talib.EMA(sell_vols, timeperiod=3)[-1]))
        else:
            buy_ema = sum(buy_volume[timeframe][-3:]) / Decimal(str(len(buy_volume[timeframe][-3:]))) if buy_volume[timeframe] else Decimal('0')
            sell_ema = sum(sell_volume[timeframe][-3:]) / Decimal(str(len(sell_volume[timeframe][-3:]))) if sell_volume[timeframe] else Decimal('0')
        total_ema = buy_ema + sell_ema
        if total_ema > Decimal('0'):
            buy_ratio = (buy_ema / total_ema) * Decimal('100')
            sell_ratio = Decimal('100') - buy_ratio
            status = "Bullish" if buy_ratio >= Decimal('50') else "Bearish"
        else:
            buy_ratio = Decimal('0')
            sell_ratio = Decimal('0')
            status = "No Activity"
        volume_ratio[timeframe] = {"buy_ratio": buy_ratio, "sell_ratio": sell_ratio, "status": status}
        logging.info(f"{timeframe} - Smoothed Buy Ratio: {buy_ratio:.25f}%, Sell Ratio: {sell_ratio:.25f}%, Status: {status}")
        print(f"{timeframe} - Smoothed Buy Ratio: {buy_ratio:.25f}%, Sell Ratio: {sell_ratio:.25f}%, Status: {status}")
    return volume_ratio

def calculate_linear_regression_forecast(closes, min_threshold, max_threshold, window=360):
    closes_np = np.array([float(x) for x in closes[-window:] if not np.isnan(x) and x > 0], dtype=np.float64)
    if len(closes_np) < 10:  # Minimum data points for stable regression
        logging.warning("Insufficient data for linear regression forecast.")
        print("Insufficient data for linear regression forecast.")
        return Decimal('0'), "NONE"

    # Prepare data for regression
    X = np.arange(len(closes_np)).reshape(-1, 1)
    y = closes_np

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    
    # Calculate regression channel bounds
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_residuals = np.std(residuals)
    upper_channel = y_pred + 2 * std_residuals
    lower_channel = y_pred - 2 * std_residuals

    # Determine last major reversal
    current_price = Decimal(str(closes_np[-1]))
    min_threshold = Decimal(str(min_threshold))
    max_threshold = Decimal(str(max_threshold))
    threshold_proximity = Decimal('0.005') * (max_threshold - min_threshold)  # 0.5% of range for tighter detection
    
    # Check for dip or top
    is_dip = abs(current_price - min_threshold) <= threshold_proximity and current_price <= min_threshold
    is_top = abs(current_price - max_threshold) <= threshold_proximity and current_price >= max_threshold
    
    # Determine cycle direction and forecast price
    if is_dip:
        cycle_direction = "UP (Reversal)"
        # Forecast incoming reversal price near max threshold
        forecast_price = max_threshold - Decimal('0.0005') * (max_threshold - min_threshold)  # 0.05% below max
    elif is_top:
        cycle_direction = "DOWN (Reversal)"
        # Forecast incoming reversal price near min threshold
        forecast_price = min_threshold + Decimal('0.0005') * (max_threshold - min_threshold)  # 0.05% above min
    else:
        # Use regression slope for cycle direction
        cycle_direction = "UP" if slope >= 0 else "DOWN"
        # Forecast next price within channel, biased toward opposite threshold
        next_x = np.array([[len(closes_np)]])
        forecast = model.predict(next_x)[0]
        forecast_price = Decimal(str(forecast))
        # Bias forecast toward max/min threshold based on cycle direction
        if cycle_direction == "UP":
            forecast_price = min(max_threshold, max(forecast_price, Decimal(str(upper_channel[-1]))))
        else:
            forecast_price = max(min_threshold, min(forecast_price, Decimal(str(lower_channel[-1]))))

    logging.info(f"Linear Regression: Cycle Direction: {cycle_direction}, Forecast Price: {forecast_price:.25f}, Min Threshold: {min_threshold:.25f}, Max Threshold: {max_threshold:.25f}, Channel Upper: {upper_channel[-1]:.2f}, Channel Lower: {lower_channel[-1]:.2f}")
    print(f"Linear Regression: Cycle Direction: {cycle_direction}, Forecast Price: {forecast_price:.25f}, Min Threshold: {min_threshold:.25f}, Max Threshold: {max_threshold:.25f}, Channel Upper: {upper_channel[-1]:.2f}, Channel Lower: {lower_channel[-1]:.2f}")

    return forecast_price, cycle_direction

def calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=Decimal('0.05')):
    close_prices = np.array([float(x) for x in close_prices if not np.isnan(x) and x > 0], dtype=np.float64)
    if len(close_prices) == 0:
        logging.warning("No valid close prices for threshold calculation.")
        print("No valid close prices for threshold calculation.")
        return None, None, None, None, None, None, None
    min_close = Decimal(str(np.nanmin(close_prices)))
    max_close = Decimal(str(np.nanmax(close_prices)))
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = Decimal(str(np.nanmin(momentum)))
    max_momentum = Decimal(str(np.nanmax(momentum)))
    min_percentage_custom = Decimal(str(minimum_percentage)) / Decimal('100')
    max_percentage_custom = Decimal(str(maximum_percentage)) / Decimal('100')
    min_threshold = min(min_close - (max_close - min_close) * min_percentage_custom, Decimal(str(close_prices[-1])))
    max_threshold = max(max_close + (max_close - min_close) * max_percentage_custom, Decimal(str(close_prices[-1])))
    range_price = [Decimal(str(x)) for x in np.linspace(float(close_prices[-1]) * (1 - float(range_distance)), float(close_prices[-1]) * (1 + float(range_distance)), num=50)]
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < float(min_threshold), float(min_threshold), close_prices)
        filtered_close = np.where(filtered_close > float(max_threshold), float(max_threshold), close_prices)
    avg_mtf = Decimal(str(np.nanmean(filtered_close)))
    current_momentum = Decimal(str(momentum[-1]))
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * Decimal('100') if max_momentum != min_momentum else Decimal('0')
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * Decimal('100') if max_momentum != min_momentum else Decimal('0')
    percent_to_min_combined = (Decimal(str(minimum_percentage)) + percent_to_min_momentum) / Decimal('2')
    percent_to_max_combined = (Decimal(str(maximum_percentage)) + percent_to_max_momentum) / Decimal('2')
    momentum_signal = percent_to_max_combined - percent_to_min_combined
    logging.info(f"Momentum Signal: {momentum_signal:.25f}")
    logging.info(f"Minimum Threshold: {min_threshold:.25f}")
    logging.info(f"Maximum Threshold: {max_threshold:.25f}")
    logging.info(f"Average MTF: {avg_mtf:.25f}")
    print(f"Momentum Signal: {momentum_signal:.25f}")
    print(f"Minimum Threshold: {min_threshold:.25f}")
    print(f"Maximum Threshold: {max_threshold:.25f}")
    print(f"Average MTF: {avg_mtf:.25f}")
    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum

# Main Analysis Loop
def main():
    timeframes = ["1m", "3m", "5m"]
    logging.info("Futures Analysis Bot Initialized!")
    print("Futures Analysis Bot Initialized!")
    
    try:
        while True:
            current_local_time = datetime.datetime.now()
            current_local_time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Current Time: {current_local_time_str}")
            print(f"\nCurrent Time: {current_local_time_str}")

            candle_map = fetch_candles_in_parallel(timeframes)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candle data available. Retrying in 60 seconds.")
                print("No candle data available. Retrying in 60 seconds.")
                time.sleep(60)
                continue

            current_price = get_current_price()
            if current_price <= Decimal('0'):
                logging.warning(f"Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")
                print(f"Warning: Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")
                time.sleep(60)
                continue

            usdc_balance = get_balance('USDC')
            position = get_position()

            # Initialize conditions
            conditions_long = {
                "volume_bullish_1m": False,
                "volume_bullish_3m": False,
                "volume_bullish_5m": False,
                "momentum_positive_1m": False,
                "forecast_above_close": False,
                "near_dip_1m": False,
                "near_dip_3m": False,
                "near_dip_5m": False
            }
            conditions_short = {
                "volume_bearish_1m": False,
                "volume_bearish_3m": False,
                "volume_bearish_5m": False,
                "momentum_negative_1m": False,
                "forecast_below_close": False,
                "near_top_1m": False,
                "near_top_3m": False,
                "near_top_5m": False
            }

            # Volume analysis
            buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
            volume_ratio = calculate_volume_ratio(buy_volume, sell_volume)

            # Reversal and advanced analysis
            for timeframe in timeframes:
                if not candle_map.get(timeframe):
                    logging.warning(f"No data for {timeframe}. Skipping analysis.")
                    print(f"No data for {timeframe}. Skipping analysis.")
                    continue
                closes = [candle["close"] for candle in candle_map[timeframe]]
                volumes = [candle["volume"] for candle in candle_map[timeframe]]
                current_close = Decimal(str(closes[-1])) if closes else Decimal('0')

                # Threshold calculation
                (min_threshold, max_threshold, avg_mtf, momentum_signal, 
                 range_price, percent_to_min_momentum, percent_to_max_momentum) = calculate_thresholds(closes)

                if min_threshold is None or max_threshold is None:
                    logging.warning(f"No valid thresholds for {timeframe}. Skipping proximity analysis.")
                    print(f"No valid thresholds for {timeframe}. Skipping proximity analysis.")
                    conditions_long[f"near_dip_{timeframe}"] = True
                    conditions_short[f"near_top_{timeframe}"] = False
                    continue

                # Calculate normalized distance percentages
                reversal_range = max_threshold - min_threshold
                if reversal_range > Decimal('0'):
                    dist_to_min = abs(current_close - min_threshold)
                    dist_to_max = abs(current_close - max_threshold)
                    percent_to_min = (dist_to_min / reversal_range) * Decimal('100')
                    percent_to_max = (dist_to_max / reversal_range) * Decimal('100')
                    total_percent = percent_to_min + percent_to_max
                    if total_percent > Decimal('0'):
                        percent_to_min = (percent_to_min / total_percent) * Decimal('100')
                        percent_to_max = (percent_to_max / total_percent) * Decimal('100')
                    else:
                        percent_to_min = Decimal('50')
                        percent_to_max = Decimal('50')
                else:
                    percent_to_min = Decimal('50')
                    percent_to_max = Decimal('50')
                    logging.warning(f"No valid reversal range for {timeframe}. Setting percentages to 50%.")

                # Debug proximity condition
                logging.info(f"{timeframe} - Before Proximity Condition: Percent to Min: {percent_to_min:.2f}%, Percent to Max: {percent_to_max:.2f}%")
                print(f"{timeframe} - Before Proximity Condition: Percent to Min: {percent_to_min:.2f}%, Percent to Max: {percent_to_max:.2f}%")

                # Determine proximity using normalized percentages
                if dist_to_min <= dist_to_max:
                    conditions_long[f"near_dip_{timeframe}"] = True
                    conditions_short[f"near_top_{timeframe}"] = False
                else:
                    conditions_long[f"near_dip_{timeframe}"] = False
                    conditions_short[f"near_top_{timeframe}"] = True

                # Print proximity details
                logging.info(f"{timeframe} - Current Close: {current_close:.25f}, Nearest Dip: {min_threshold:.25f}, Nearest Top: {max_threshold:.25f}")
                print(f"{timeframe} - Current Close: {current_close:.25f}, Nearest Dip: {min_threshold:.25f}, Nearest Top: {max_threshold:.25f}")
                logging.info(f"{timeframe} - Percent to Min: {percent_to_min:.2f}%, Percent to Max: {percent_to_max:.2f}%")
                print(f"{timeframe} - Percent to Min: {percent_to_min:.2f}%, Percent to Max: {percent_to_max:.2f}%")
                logging.info(f"{timeframe} - Near Dip: {conditions_long[f'near_dip_{timeframe}']}, Near Top: {conditions_short[f'near_top_{timeframe}']}")
                print(f"{timeframe} - Near Dip: {conditions_long[f'near_dip_{timeframe}']}, Near Top: {conditions_short[f'near_top_{timeframe}']}")

                # Validate symmetry
                if conditions_long[f"near_dip_{timeframe}"] == conditions_short[f"near_top_{timeframe}"]:
                    logging.error(f"Symmetry violation in {timeframe}: near_dip_{timeframe} and near_top_{timeframe} are both {conditions_long[f'near_dip_{timeframe}']}")
                    print(f"Symmetry violation in {timeframe}: near_dip_{timeframe} and near_top_{timeframe} are both {conditions_long[f'near_dip_{timeframe}']}")

                # Volume conditions
                buy_vol = buy_volume.get(timeframe, [Decimal('0')])[-1]
                sell_vol = sell_volume.get(timeframe, [Decimal('0')])[-1]
                price_trend = Decimal(str(closes[-1])) - Decimal(str(closes[-2])) if len(closes) >= 2 else Decimal('0')
                if buy_vol >= sell_vol or price_trend > Decimal('0'):
                    conditions_long[f"volume_bullish_{timeframe}"] = True
                    conditions_short[f"volume_bearish_{timeframe}"] = False
                else:
                    conditions_long[f"volume_bullish_{timeframe}"] = False
                    conditions_short[f"volume_bearish_{timeframe}"] = True
                logging.info(f"Volume Bullish ({timeframe}): {buy_vol:.25f}, Bearish: {sell_vol:.25f}, Bullish Condition: {conditions_long[f'volume_bullish_{timeframe}']}, Bearish Condition: {conditions_short[f'volume_bearish_{timeframe}']}")
                print(f"Volume Bullish ({timeframe}): {buy_vol:.25f}, Bearish: {sell_vol:.25f}, Bullish Condition: {conditions_long[f'volume_bullish_{timeframe}']}, Bearish Condition: {conditions_short[f'volume_bearish_{timeframe}']}")

                # 1m-specific analysis (momentum and linear regression)
                if timeframe == "1m":
                    print("\n--- 1m Timeframe Analysis (Momentum, Linear Regression Forecast) ---")
                    valid_closes = np.array([float(c) for c in closes if not np.isnan(c) and c > 0], dtype=np.float64)
                    if len(valid_closes) >= 14:
                        momentum = talib.MOM(valid_closes, timeperiod=14)
                        if len(momentum) > 0 and not np.isnan(momentum[-1]):
                            current_momentum = Decimal(str(momentum[-1]))
                            if current_momentum >= Decimal('0'):
                                conditions_long["momentum_positive_1m"] = True
                                conditions_short["momentum_negative_1m"] = False
                            else:
                                conditions_long["momentum_positive_1m"] = False
                                conditions_short["momentum_negative_1m"] = True
                            logging.info(f"1m Momentum: {current_momentum:.25f} - {'Positive' if conditions_long['momentum_positive_1m'] else 'Negative'}")
                            print(f"1m Momentum: {current_momentum:.25f} - {'Positive' if conditions_long['momentum_positive_1m'] else 'Negative'}")
                    else:
                        logging.warning("1m Momentum: Insufficient data")
                        print("1m Momentum: Insufficient data")
                        conditions_long["momentum_positive_1m"] = True
                        conditions_short["momentum_negative_1m"] = False

                    if len(valid_closes) >= 10:
                        forecast_price, cycle_direction = calculate_linear_regression_forecast(closes, min_threshold, max_threshold)
                        logging.info(f"Linear Regression Forecast Price (1m): {forecast_price:.25f}, Cycle Direction: {cycle_direction}")
                        print(f"Linear Regression Forecast Price (1m): {forecast_price:.25f}, Cycle Direction: {cycle_direction}")
                        if forecast_price >= current_close:
                            conditions_long["forecast_above_close"] = True
                            conditions_short["forecast_below_close"] = False
                        else:
                            conditions_long["forecast_above_close"] = False
                            conditions_short["forecast_below_close"] = True
                    else:
                        conditions_long["forecast_above_close"] = True
                        conditions_short["forecast_below_close"] = False

            # Log condition pairs for debugging
            condition_pairs = [
                ("volume_bullish_1m", "volume_bearish_1m"),
                ("volume_bullish_3m", "volume_bearish_3m"),
                ("volume_bullish_5m", "volume_bearish_5m"),
                ("momentum_positive_1m", "momentum_negative_1m"),
                ("forecast_above_close", "forecast_below_close"),
                ("near_dip_1m", "near_top_1m"),
                ("near_dip_3m", "near_top_3m"),
                ("near_dip_5m", "near_top_5m")
            ]
            logging.info("Condition Pairs Status:")
            print("Condition Pairs Status:")
            for long_cond, short_cond in condition_pairs:
                logging.info(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]}")
                print(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]}")

            # Evaluate signals with strict rules
            long_signal = all(conditions_long.values()) and not any(conditions_short.values())
            short_signal = all(conditions_short.values()) and not any(conditions_long.values())

            # Print condition states
            logging.info("Trade Signal Status:")
            print("\nTrade Signal Status:")
            logging.info(f"LONG Signal: {'Active' if long_signal else 'Inactive'}")
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'}")
            logging.info(f"SHORT Signal: {'Active' if short_signal else 'Inactive'}")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'}")
            logging.info("\nLong Conditions Status:")
            print("\nLong Conditions Status:")
            for condition, status in conditions_long.items():
                logging.info(f"{condition}: {'True' if status else 'False'}")
                print(f"{condition}: {'True' if status else 'False'}")
            logging.info("\nShort Conditions Status:")
            print("\nShort Conditions Status:")
            for condition, status in conditions_short.items():
                logging.info(f"{condition}: {'True' if status else 'False'}")
                print(f"{condition}: {'True' if status else 'False'}")

            # Calculate condition summary
            long_true = sum(1 for val in conditions_long.values() if val)
            long_false = len(conditions_long) - long_true
            short_true = sum(1 for val in conditions_short.values() if val)
            short_false = len(conditions_short) - short_true
            logging.info(f"\nLong Conditions Summary: {long_true} True, {long_false} False")
            print(f"\nLong Conditions Summary: {long_true} True, {long_false} False")
            logging.info(f"Short Conditions Summary: {short_true} True, {short_false} False")
            print(f"Short Conditions Summary: {short_true} True, {short_false} False")

            # Determine final signal
            signal = "NO_SIGNAL"
            if long_signal:
                signal = "LONG"
            elif short_signal:
                signal = "SHORT"
            if long_signal and short_signal:
                signal = "NO_SIGNAL"
                logging.warning("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                print("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
            logging.info(f"Final Signal: {signal}")
            print(f"Final Signal: {signal}")

            # Execute trades
            if usdc_balance < MINIMUM_BALANCE:
                logging.warning(f"Insufficient USDC balance ({usdc_balance:.25f}) to place trades. Minimum required: {MINIMUM_BALANCE:.25f}")
                print(f"Insufficient USDC balance ({usdc_balance:.25f}) to place trades. Minimum required: {MINIMUM_BALANCE:.25f}")
            elif signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                place_order(signal, quantity, current_price)
            elif (signal == "LONG" and position["side"] == "SHORT") or (signal == "SHORT" and position["side"] == "LONG"):
                close_position(position, current_price)
                quantity = calculate_quantity(usdc_balance, current_price)
                place_order(signal, quantity, current_price)

            # Position status
            if position["side"] != "NONE":
                print("\nCurrent Position Status:")
                print(f"Position Side: {position['side']}")
                print(f"Quantity: {position['quantity']:.25f} BTC")
                print(f"Entry Price: {position['entry_price']:.25f} USDC")
                print(f"Current Price: {current_price:.25f} USDC")
                print(f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC")
                current_balance = usdc_balance + position['unrealized_pnl']
                print(f"Current Total Balance: {current_balance:.25f} USDC")
            else:
                print(f"\nNo open position. USDC Balance: {usdc_balance:.25f}")

            print(f"\nCurrent USDC Balance: {usdc_balance:.25f}")
            print(f"Current Position: {position['side']}, Quantity: {position['quantity']:.25f} BTC")
            print(f"Current Price: {current_price:.25f}\n")

            # Cleanup
            del candle_map
            gc.collect()
            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Shutting down bot...")
        print("Shutting down bot...")
        position = get_position()
        if position["side"] != "NONE":
            close_position(position, get_current_price())
        logging.info("Bot shutdown complete.")
        print("Bot shutdown complete.")
        exit(0)
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        print(f"Unexpected error in main loop: {e}")
        time.sleep(5)

if __name__ == "__main__":
    main()