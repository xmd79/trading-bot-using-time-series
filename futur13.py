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

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set Decimal precision
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"
LEVERAGE = 100
STOP_LOSS_PERCENTAGE = Decimal('0.10')  # 10% stop-loss
TAKE_PROFIT_PERCENTAGE = Decimal('0.10')  # 10% take-profit
QUANTITY_PRECISION = Decimal('0.000001')  # Binance quantity precision for BTCUSDC
MINIMUM_BALANCE = Decimal('10.0')  # Minimum USDC balance to place trades
TIMEFRAMES = ["1m", "3m", "5m"]
LOOKBACK_PERIODS = {"1m": 500, "3m": 500, "5m": 500}  # Use 500 candles for all timeframes
PRICE_TOLERANCE = Decimal('0.01')  # Tolerance for matching thresholds

# Load credentials
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

# Initialize Binance client
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})
client.API_URL = 'https://fapi.binance.com'  # Futures API endpoint

# Set leverage
try:
    client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
    logging.info(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
    print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
except BinanceAPIException as e:
    logging.error(f"Error setting leverage: {e.message}")
    print(f"Error setting leverage: {e.message}")

# Reversal Detection Function
def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold):
    """
    Detect the most recent major reversal (DIP or TOP) using provided min and max thresholds.
    Finds the most recent candles with lows/highs closest to thresholds within tolerance,
    sorts by timestamp, and confirms based on proximity to current close.
    Returns reversal type and logs normalized percentage distances.
    """
    if len(candles) < 3:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        return "NONE"

    # Use lookback of 500 candles, or available candles if fewer
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]

    # Create arrays, filtering out invalid data
    closes = np.array([float(c['close']) for c in recent_candles if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)
    lows = np.array([float(c['low']) for c in recent_candles if not np.isnan(c['low']) and c['low'] > 0], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles if not np.isnan(c['high']) and c['high'] > 0], dtype=np.float64)
    times = np.array([c['time'] for c in recent_candles if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)

    # Validate array lengths
    if len(closes) < 3 or len(times) < 3 or len(lows) < 3 or len(highs) < 3:
        logging.warning(f"Insufficient valid data (closes: {len(closes)}, times: {len(times)}, highs: {len(highs)}, lows: {len(lows)}) for reversal detection in {timeframe}.")
        print(f"Insufficient valid data (closes: {len(closes)}, times: {len(times)}, highs: {len(highs)}, lows: {len(lows)}) for reversal detection in {timeframe}.")
        return "NONE"

    if len(closes) != len(times):
        logging.error(f"Array length mismatch in {timeframe}: closes={len(closes)}, times={len(times)}")
        print(f"Array length mismatch in {timeframe}: closes={len(closes)}, times={len(times)}")
        return "NONE"

    # Current close
    current_close = Decimal(str(closes[-1]))

    # Find the most recent candles closest to min_threshold and max_threshold within tolerance
    min_time = 0
    max_time = 0
    closest_min_diff = Decimal('infinity')
    closest_max_diff = Decimal('infinity')
    closest_min_price = min_threshold
    closest_max_price = max_threshold

    for i, candle in enumerate(recent_candles):
        low_price = Decimal(str(candle['low']))
        high_price = Decimal(str(candle['high']))
        candle_time = candle['time']

        # Check if low is within tolerance of min_threshold
        min_diff = abs(low_price - min_threshold)
        if min_diff <= PRICE_TOLERANCE and min_diff < closest_min_diff:
            closest_min_diff = min_diff
            min_time = candle_time
            closest_min_price = low_price

        # Check if high is within tolerance of max_threshold
        max_diff = abs(high_price - max_threshold)
        if max_diff <= PRICE_TOLERANCE and max_diff < closest_max_diff:
            closest_max_diff = max_diff
            max_time = candle_time
            closest_max_price = high_price

    # Log the closest matches found
    if min_time > 0:
        logging.info(f"{timeframe} - Found closest low {closest_min_price:.2f} (target: {min_threshold:.2f}, diff: {closest_min_diff:.2f}) at time {datetime.datetime.fromtimestamp(min_time)}")
        print(f"{timeframe} - Found closest low {closest_min_price:.2f} (target: {min_threshold:.2f}, diff: {closest_min_diff:.2f}) at time {datetime.datetime.fromtimestamp(min_time)}")
    else:
        logging.warning(f"{timeframe} - No low found within {PRICE_TOLERANCE:.2f} of min threshold {min_threshold:.2f} over {len(recent_candles)} candles.")
        print(f"{timeframe} - No low found within {PRICE_TOLERANCE:.2f} of min threshold {min_threshold:.2f} over {len(recent_candles)} candles.")

    if max_time > 0:
        logging.info(f"{timeframe} - Found closest high {closest_max_price:.2f} (target: {max_threshold:.2f}, diff: {closest_max_diff:.2f}) at time {datetime.datetime.fromtimestamp(max_time)}")
        print(f"{timeframe} - Found closest high {closest_max_price:.2f} (target: {max_threshold:.2f}, diff: {closest_max_diff:.2f}) at time {datetime.datetime.fromtimestamp(max_time)}")
    else:
        logging.warning(f"{timeframe} - No high found within {PRICE_TOLERANCE:.2f} of max threshold {max_threshold:.2f} over {len(recent_candles)} candles.")
        print(f"{timeframe} - No high found within {PRICE_TOLERANCE:.2f} of max threshold {max_threshold:.2f} over {len(recent_candles)} candles.")

    # Calculate normalized percentage distances
    price_range = max_threshold - min_threshold
    if price_range > Decimal('0'):
        min_pct = ((current_close - min_threshold) / price_range * Decimal('100')).quantize(Decimal('0.01'))
        max_pct = ((max_threshold - current_close) / price_range * Decimal('100')).quantize(Decimal('0.01'))
    else:
        min_pct = Decimal('50.00')
        max_pct = Decimal('50.00')
    logging.info(f"{timeframe} - Normalized Distance: {min_pct:.2f}% from min threshold, {max_pct:.2f}% from max threshold")
    print(f"{timeframe} - Normalized Distance: {min_pct:.2f}% from min threshold, {max_pct:.2f}% from max threshold")

    # Sort reversals by recency
    reversals = []
    if min_time > 0:
        reversals.append({"type": "DIP", "price": closest_min_price, "time": min_time})
    if max_time > 0:
        reversals.append({"type": "TOP", "price": closest_max_price, "time": max_time})

    if not reversals:
        logging.info(f"{timeframe} - No valid reversals detected over {len(recent_candles)} candles.")
        print(f"{timeframe} - No valid reversals detected over {len(recent_candles)} candles.")
        return "NONE"

    reversals.sort(key=lambda x: x["time"], reverse=True)  # Most recent first

    # Select the most recent reversal
    most_recent = reversals[0]
    if most_recent["time"] == 0:
        logging.info(f"{timeframe} - No valid reversal detected over {len(recent_candles)} candles.")
        print(f"{timeframe} - No valid reversal detected over {len(recent_candles)} candles.")
        return "NONE"

    # Confirm reversal based on proximity to current close
    dist_to_min = abs(current_close - closest_min_price)
    dist_to_max = abs(current_close - closest_max_price)
    reversal_type = most_recent["type"]

    # If the most recent reversal is DIP and closer to current close, confirm DIP
    if reversal_type == "DIP" and dist_to_min <= dist_to_max:
        confirmed_type = "DIP"
    # If the most recent reversal is TOP and closer to current close, confirm TOP
    elif reversal_type == "TOP" and dist_to_max < dist_to_min:
        confirmed_type = "TOP"
    # Otherwise, choose the closer threshold
    else:
        confirmed_type = "DIP" if dist_to_min <= dist_to_max else "TOP"

    logging.info(f"{timeframe} - Confirmed reversal: {confirmed_type} at price {most_recent['price']:.2f}, time {datetime.datetime.fromtimestamp(most_recent['time'])}, distance to current close: {min(dist_to_min, dist_to_max):.2f}")
    print(f"{timeframe} - Confirmed reversal: {confirmed_type} at price {most_recent['price']:.2f}, time {datetime.datetime.fromtimestamp(most_recent['time'])}, distance to current close: {min(dist_to_min, dist_to_max):.2f}")
    return confirmed_type

# Utility Functions
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=500):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=500, retries=5, delay=5):
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
            retry_after = e.response.headers.get('Retry-After', 60) if e.response else 60
            if e.code == -1003:
                logging.warning(f"Rate limit exceeded for {timeframe}. Waiting {retry_after} seconds.")
                print(f"Rate limit exceeded for {timeframe}. Waiting {retry_after} seconds.")
                time.sleep(int(retry_after))
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
            retry_after = e.response.headers.get('Retry-After', 60) if e.response else 60
            if e.code == -1003:
                logging.warning(f"Rate limit exceeded fetching price. Waiting {retry_after} seconds.")
                print(f"Rate limit exceeded fetching price. Waiting {retry_after} seconds.")
                time.sleep(int(retry_after))
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
            print("No 'assets' key in futures account response.")
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
            return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0'), "initial_balance": Decimal('0.0'), "sl_price": Decimal('0.0'), "tp_price": Decimal('0.0')}
        position = positions[0]
        quantity = Decimal(str(position['positionAmt']))
        entry_price = Decimal(str(position['entryPrice']))
        return {
            "quantity": quantity,
            "entry_price": entry_price,
            "side": "LONG" if quantity > Decimal('0') else "SHORT" if quantity < Decimal('0') else "NONE",
            "unrealized_pnl": Decimal(str(position['unrealizedProfit'])),
            "initial_balance": Decimal('0.0'),
            "sl_price": Decimal('0.0'),
            "tp_price": Decimal('0.0')
        }
    except BinanceAPIException as e:
        logging.error(f"Error fetching position info: {e.message}")
        print(f"Error fetching position info: {e.message}")
        return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0'), "initial_balance": Decimal('0.0'), "sl_price": Decimal('0.0'), "tp_price": Decimal('0.0')}

def check_open_orders():
    try:
        orders = client.futures_get_open_orders(symbol=TRADE_SYMBOL)
        for order in orders:
            logging.info(f"Open order: {order['type']} at {order['stopPrice']}")
            print(f"Open order: {order['type']} at {order['stopPrice']}")
        return len(orders)
    except BinanceAPIException as e:
        logging.error(f"Error checking open orders: {e.message}")
        print(f"Error checking open orders: {e.message}")
        return 0

# Trading Functions
def calculate_quantity(balance, price):
    if price <= Decimal('0') or balance < MINIMUM_BALANCE:
        logging.warning(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}). Cannot calculate quantity.")
        print(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}). Cannot calculate quantity.")
        return Decimal('0.0')
    quantity = (balance * LEVERAGE) / price
    return quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')

def place_order(signal, quantity, current_price, initial_balance):
    try:
        if quantity <= Decimal('0'):
            logging.warning(f"Invalid quantity {quantity:.25f}. Skipping order.")
            print(f"Invalid quantity {quantity:.25f}. Skipping order.")
            return None
        position = get_position()
        position["initial_balance"] = initial_balance
        if signal == "LONG":
            order = client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="BUY",
                type="MARKET",
                quantity=str(quantity)
            )
            tp_price = (current_price * (Decimal('1') + TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
            sl_price = (current_price * (Decimal('1') - STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
            
            position["sl_price"] = sl_price
            position["tp_price"] = tp_price
            position["side"] = "LONG"
            position["quantity"] = quantity
            position["entry_price"] = current_price
            
            logging.info(f"Placed LONG order: {quantity:.25f} BTC at market price ~{current_price:.25f}")
            print(f"\n=== TRADE ENTERED ===")
            print(f"Side: LONG")
            print(f"Quantity: {quantity:.25f} BTC")
            print(f"Entry Price: ~{current_price:.25f} USDC")
            print(f"Initial USDC Balance: {initial_balance:.25f}")
            print(f"Stop-Loss Price: {sl_price:.25f} (-12% ROI)")
            print(f"Take-Profit Price: {tp_price:.25f} (+4% ROI)")
            print(f"===================\n")
        elif signal == "SHORT":
            order = client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side="SELL",
                type="MARKET",
                quantity=str(quantity)
            )
            tp_price = (current_price * (Decimal('1') - TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
            sl_price = (current_price * (Decimal('1') + STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
            
            position["sl_price"] = sl_price
            position["tp_price"] = tp_price
            position["side"] = "SHORT"
            position["quantity"] = -quantity  # Negative for SHORT
            position["entry_price"] = current_price
            
            logging.info(f"Placed SHORT order: {quantity:.25f} BTC at market price ~{current_price:.25f}")
            print(f"\n=== TRADE ENTERED ===")
            print(f"Side: SHORT")
            print(f"Quantity: {quantity:.25f} BTC")
            print(f"Entry Price: ~{current_price:.25f} USDC")
            print(f"Initial USDC Balance: {initial_balance:.25f}")
            print(f"Stop-Loss Price: {sl_price:.25f} (-12% ROI)")
            print(f"Take-Profit Price: {tp_price:.25f} (+4% ROI)")
            print(f"===================\n")
        
        # Verify no unintended open orders
        open_orders = check_open_orders()
        if open_orders > 0:
            logging.warning(f"Unexpected open orders ({open_orders}) detected after placing {signal} order.")
            print(f"Warning: Unexpected open orders ({open_orders}) detected after placing {signal} order.")
            
        return position
    except BinanceAPIException as e:
        logging.error(f"Error placing order: {e.message}")
        print(f"Error placing order: {e.message}")
        return None

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

def calculate_thresholds(candles):
    lookback = min(len(candles), LOOKBACK_PERIODS[candles[0]['timeframe']])
    highs = np.array([float(c['high']) for c in candles[-lookback:] if not np.isnan(c['high']) and c['high'] > 0], dtype=np.float64)
    lows = np.array([float(c['low']) for c in candles[-lookback:] if not np.isnan(c['low']) and c['low'] > 0], dtype=np.float64)
    closes = np.array([float(c['close']) for c in candles[-lookback:] if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)

    if len(closes) == 0 or len(highs) == 0 or len(lows) == 0:
        logging.warning(f"No valid OHLC data for threshold calculation. Candles processed: {len(candles)}")
        print(f"No valid OHLC data for threshold calculation. Candles processed: {len(candles)}")
        return Decimal('0'), Decimal('0'), Decimal('0')

    current_close = Decimal(str(closes[-1]))

    min_price = Decimal(str(np.min(lows)))
    max_price = Decimal(str(np.max(highs)))

    if min_price >= current_close:
        min_threshold = current_close * Decimal('0.995')
    else:
        min_threshold = min_price

    if max_price <= current_close:
        max_threshold = current_close * Decimal('1.005')
    else:
        max_threshold = max_price

    middle_threshold = (min_threshold + max_threshold) / Decimal('2')

    logging.info(f"Thresholds calculated over {len(highs)} candles: Minimum: {min_threshold:.25f}, Middle: {middle_threshold:.25f}, Maximum: {max_threshold:.25f}, Current Close: {current_close:.25f}")
    print(f"Thresholds calculated over {len(highs)} candles: Minimum: {min_threshold:.25f}, Middle: {middle_threshold:.25f}, Maximum: {max_threshold:.25f}, Current Close: {current_close:.25f}")
    return min_threshold, middle_threshold, max_threshold

# Main Analysis Loop
def main():
    timeframes = TIMEFRAMES
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

            # Check if SL or TP is hit for an open position
            if position["side"] != "NONE" and position["sl_price"] > Decimal('0') and position["tp_price"] > Decimal('0'):
                if position["side"] == "LONG":
                    if current_price <= position["sl_price"]:
                        logging.info(f"Stop-Loss triggered for LONG at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                        print(f"Stop-Loss triggered for LONG at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                        close_position(position, current_price)
                        position = get_position()  # Refresh position
                    elif current_price >= position["tp_price"]:
                        logging.info(f"Take-Profit triggered for LONG at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        print(f"Take-Profit triggered for LONG at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        close_position(position, current_price)
                        position = get_position()  # Refresh position
                elif position["side"] == "SHORT":
                    if current_price >= position["sl_price"]:
                        logging.info(f"Stop-Loss triggered for SHORT at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                        print(f"Stop-Loss triggered for SHORT at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                        close_position(position, current_price)
                        position = get_position()  # Refresh position
                    elif current_price <= position["tp_price"]:
                        logging.info(f"Take-Profit triggered for SHORT at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        print(f"Take-Profit triggered for SHORT at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        close_position(position, current_price)
                        position = get_position()  # Refresh position

            conditions_long = {
                "volume_bullish_1m": False,
                "volume_bullish_3m": False,
                "volume_bullish_5m": False,
                "momentum_positive_1m": False,
                "dip_confirmation_1m": False,
                "dip_confirmation_3m": False,
                "dip_confirmation_5m": False,
                "below_middle_1m": False,
                "below_middle_3m": False,
                "below_middle_5m": False
            }
            conditions_short = {
                "volume_bearish_1m": False,
                "volume_bearish_3m": False,
                "volume_bearish_5m": False,
                "momentum_negative_1m": False,
                "top_confirmation_1m": False,
                "top_confirmation_3m": False,
                "top_confirmation_5m": False,
                "above_middle_1m": False,
                "above_middle_3m": False,
                "above_middle_5m": False
            }

            buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
            volume_ratio = calculate_volume_ratio(buy_volume, sell_volume)

            for timeframe in timeframes:
                if not candle_map.get(timeframe):
                    logging.warning(f"No data for {timeframe}. Skipping analysis.")
                    print(f"No data for {timeframe}. Skipping analysis.")
                    continue
                candles = candle_map[timeframe]
                closes = [candle["close"] for candle in candles]
                current_close = Decimal(str(closes[-1])) if closes else Decimal('0')

                min_threshold, middle_threshold, max_threshold = calculate_thresholds(candles)

                if min_threshold == Decimal('0') or max_threshold == Decimal('0'):
                    logging.warning(f"No valid thresholds for {timeframe}. Skipping threshold and reversal analysis.")
                    print(f"No valid thresholds for {timeframe}. Skipping threshold and reversal analysis.")
                    conditions_long[f"below_middle_{timeframe}"] = True
                    conditions_short[f"above_middle_{timeframe}"] = False
                    conditions_long[f"dip_confirmation_{timeframe}"] = True
                    conditions_short[f"top_confirmation_{timeframe}"] = False
                    continue

                # Check if current close is below or above middle threshold
                if current_close < middle_threshold:
                    conditions_long[f"below_middle_{timeframe}"] = True
                    conditions_short[f"above_middle_{timeframe}"] = False
                else:
                    conditions_long[f"below_middle_{timeframe}"] = False
                    conditions_short[f"above_middle_{timeframe}"] = True

                logging.info(f"{timeframe} - Current Close: {current_close:.25f}, Below Middle: {conditions_long[f'below_middle_{timeframe}']}, Above Middle: {conditions_short[f'above_middle_{timeframe}']}")
                print(f"{timeframe} - Current Close: {current_close:.25f}, Below Middle: {conditions_long[f'below_middle_{timeframe}']}, Above Middle: {conditions_short[f'above_middle_{timeframe}']}")

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

                # Reversal detection using min and max thresholds
                reversal_type = detect_recent_reversal(candles, timeframe, min_threshold, max_threshold)
                if reversal_type == "DIP":
                    conditions_long[f"dip_confirmation_{timeframe}"] = True
                    conditions_short[f"top_confirmation_{timeframe}"] = False
                elif reversal_type == "TOP":
                    conditions_long[f"dip_confirmation_{timeframe}"] = False
                    conditions_short[f"top_confirmation_{timeframe}"] = True
                else:
                    conditions_long[f"dip_confirmation_{timeframe}"] = True  # Default to allow LONG if no clear reversal
                    conditions_short[f"top_confirmation_{timeframe}"] = False
                logging.info(f"{timeframe} - Dip Confirmation: {conditions_long[f'dip_confirmation_{timeframe}']}, Top Confirmation: {conditions_short[f'top_confirmation_{timeframe}']}")
                print(f"{timeframe} - Dip Confirmation: {conditions_long[f'dip_confirmation_{timeframe}']}, Top Confirmation: {conditions_short[f'top_confirmation_{timeframe}']}")

                if timeframe == "1m":
                    print("\n--- 1m Timeframe Analysis (Momentum) ---")
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

            condition_pairs = [
                ("volume_bullish_1m", "volume_bearish_1m"),
                ("volume_bullish_3m", "volume_bearish_3m"),
                ("volume_bullish_5m", "volume_bearish_5m"),
                ("momentum_positive_1m", "momentum_negative_1m"),
                ("dip_confirmation_1m", "top_confirmation_1m"),
                ("dip_confirmation_3m", "top_confirmation_3m"),
                ("dip_confirmation_5m", "top_confirmation_5m"),
                ("below_middle_1m", "above_middle_1m"),
                ("below_middle_3m", "above_middle_3m"),
                ("below_middle_5m", "above_middle_5m")
            ]
            logging.info("Condition Pairs Status:")
            print("Condition Pairs Status:")
            for long_cond, short_cond in condition_pairs:
                logging.info(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]}")
                print(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]}")

            long_signal = all(conditions_long.values()) and not any(conditions_short.values())
            short_signal = all(conditions_short.values()) and not any(conditions_long.values())

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

            long_true = sum(1 for val in conditions_long.values() if val)
            long_false = len(conditions_long) - long_true
            short_true = sum(1 for val in conditions_short.values() if val)
            short_false = len(conditions_short) - short_true
            logging.info(f"\nLong Conditions Summary: {long_true} True, {long_false} False")
            print(f"\nLong Conditions Summary: {long_true} True, {long_false} False")
            logging.info(f"Short Conditions Summary: {short_true} True, {short_false} False")
            print(f"Short Conditions Summary: {short_true} True, {short_false} False")

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

            if usdc_balance < MINIMUM_BALANCE:
                logging.warning(f"Insufficient USDC balance ({usdc_balance:.25f}) to place trades. Minimum required: {MINIMUM_BALANCE:.25f}")
                print(f"Insufficient USDC balance ({usdc_balance:.25f}) to place trades. Minimum required: {MINIMUM_BALANCE:.25f}")
            elif signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            elif (signal == "LONG" and position["side"] == "SHORT") or (signal == "SHORT" and position["side"] == "LONG"):
                close_position(position, current_price)
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)

            if position["side"] != "NONE":
                print("\nCurrent Position Status:")
                print(f"Position Side: {position['side']}")
                print(f"Quantity: {position['quantity']:.25f} BTC")
                print(f"Entry Price: {position['entry_price']:.25f} USDC")
                print(f"Current Price: {current_price:.25f} USDC")
                print(f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC")
                print(f"Stop-Loss Price: {position['sl_price']:.25f} USDC")
                print(f"Take-Profit Price: {position['tp_price']:.25f} USDC")
                current_balance = usdc_balance + position['unrealized_pnl']
                roi = ((current_balance - position['initial_balance']) / position['initial_balance'] * Decimal('100')).quantize(Decimal('0.01')) if position['initial_balance'] > Decimal('0') else Decimal('0')
                print(f"Current ROI: {roi:.2f}%")
                print(f"Initial USDC Balance: {position['initial_balance']:.25f}")
                print(f"Current Total Balance: {current_balance:.25f} USDC")
            else:
                print(f"\nNo open position. USDC Balance: {usdc_balance:.25f}")

            print(f"\nCurrent USDC Balance: {usdc_balance:.25f}")
            print(f"Current Position: {position['side']}, Quantity: {position['quantity']:.25f} BTC")
            print(f"Current Price: {current_price:.25f}\n")

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
        time.sleep(60)

if __name__ == "__main__":
    main()
