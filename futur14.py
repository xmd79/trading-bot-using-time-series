import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from decimal import Decimal, getcontext, ConversionSyntax
import requests
import logging
import uuid
import traceback

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
MAX_CANDLES = 1500  # Maximum candles for analysis
PRICE_TOLERANCE = Decimal('0.01')  # Tolerance for price matching
VOLUME_THRESHOLD = Decimal('1.5')  # Volume multiplier for significant zones
FORECAST_EXTENSION = Decimal('1.618')  # Fibonacci extension for price projection
MIN_CANDLES = 100  # Minimum candles for reliable analysis
LIQUIDITY_WINDOW = 10
CONSOLIDATION_THRESHOLD = Decimal('0.02')
DEFAULT_VALUE = Decimal('0.01')
VOLUME_CONFIRMATION_THRESHOLD = Decimal('2.0')

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
    print(f"Leverage Set: {LEVERAGE}x")
    print(f"Symbol: {TRADE_SYMBOL}")
except BinanceAPIException as e:
    logging.error(f"Error setting leverage: {e.message}")
    print(f"Error Setting Leverage: {e.message}")
    exit(1)

# Utility Functions
def safe_decimal(value, default=Decimal('0')):
    """Safely convert a value to Decimal, returning default on failure."""
    try:
        return Decimal(str(value)) if value is not None else default
    except (ConversionSyntax, TypeError, ValueError) as e:
        logging.warning(f"Decimal conversion failed for value '{value}': {e}")
        return default

def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=MAX_CANDLES):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=MAX_CANDLES, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = [{
                "time": safe_decimal(k[0], default=Decimal('0')) / 1000,
                "open": float(safe_decimal(k[1], default=DEFAULT_VALUE)),
                "high": float(safe_decimal(k[2], default=DEFAULT_VALUE)),
                "low": float(safe_decimal(k[3], default=DEFAULT_VALUE)),
                "close": float(safe_decimal(k[4], default=DEFAULT_VALUE)),
                "volume": float(safe_decimal(k[5], default=Decimal('1'))),
                "timeframe": timeframe
            } for k in klines]
            return candles
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', 60) if e.response else 60
            logging.error(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error Fetching Candles for Timeframe: {timeframe}")
            print(f"Attempt: {attempt + 1}/{retries}")
            print(f"Error Message: {e.message}")
            time.sleep(int(retry_after) if e.code == -1003 else delay * (attempt + 1))
        except requests.exceptions.Timeout as e:
            logging.error(f"Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Timeout Fetching Candles for Timeframe: {timeframe}")
            print(f"Attempt: {attempt + 1}/{retries}")
            print(f"Error Message: {e}")
            time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    print(f"Failed to Fetch Candles for Timeframe: {timeframe}")
    print(f"Attempts Made: {retries}")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            price = safe_decimal(ticker.get('price', '0'))
            if price > Decimal('0'):
                return price
            logging.warning(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
            print(f"Invalid Price: {price:.25f}")
            print(f"Attempt: {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', 60) if e.response else 60
            logging.error(f"Error fetching price (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error Fetching Price")
            print(f"Attempt: {attempt + 1}/{retries}")
            print(f"Error Message: {e.message}")
            time.sleep(int(retry_after) if e.code == -1003 else delay * (attempt + 1))
    logging.error(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    print(f"Failed to Fetch Valid Price for Symbol: {TRADE_SYMBOL}")
    print(f"Attempts Made: {retries}")
    return DEFAULT_VALUE

def get_balance(asset='USDC', retries=5, delay=5):
    for attempt in range(retries):
        try:
            account = client.futures_account()
            if 'assets' not in account:
                logging.error(f"No 'assets' key in futures account response (attempt {attempt + 1}/{retries})")
                print(f"No 'assets' key in futures account response")
                time.sleep(delay * (attempt + 1))
                continue
            for asset_info in account.get('assets', []):
                if asset_info.get('asset') == asset:
                    available_balance = safe_decimal(asset_info.get('availableBalance', '0.0'))
                    if available_balance >= Decimal('0'):
                        logging.info(f"{asset} available balance: {available_balance:.25f}")
                        print(f"{asset} Available Balance: {available_balance:.25f}")
                        return available_balance
                    logging.warning(f"Invalid {asset} available balance: {available_balance:.25f} (attempt {attempt + 1}/{retries})")
                    print(f"Invalid {asset} Available Balance: {available_balance:.25f}")
            logging.warning(f"{asset} not found in futures account balances (attempt {attempt + 1}/{retries})")
            print(f"{asset} not found in futures account balances")
            time.sleep(delay * (attempt + 1))
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', 60) if e.response else 60
            logging.error(f"Binance API exception fetching {asset} balance (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Binance API exception fetching {asset} balance: {e.message}")
            time.sleep(int(retry_after) if e.code == -1003 else delay * (attempt + 1))
        except Exception as e:
            logging.error(f"Unexpected error fetching {asset} balance (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching {asset} balance: {e}")
            time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch valid {asset} balance after {retries} attempts.")
    print(f"Failed to fetch valid {asset} balance after {retries} attempts")
    return Decimal('0.0')

def get_position():
    try:
        positions = client.futures_position_information(symbol=TRADE_SYMBOL)
        position = positions[0] if positions else {}
        return {
            "quantity": safe_decimal(position.get('positionAmt', '0')),
            "entry_price": safe_decimal(position.get('entryPrice', '0')),
            "side": "LONG" if safe_decimal(position.get('positionAmt', '0')) > 0 else "SHORT" if safe_decimal(position.get('positionAmt', '0')) < 0 else "NONE",
            "unrealized_pnl": safe_decimal(position.get('unrealizedProfit', '0')),
            "initial_balance": Decimal('0'),
            "sl_price": Decimal('0'),
            "tp_price": Decimal('0')
        }
    except BinanceAPIException as e:
        logging.error(f"Error fetching position info: {e.message}")
        print(f"Error Fetching Position Info")
        print(f"Error Message: {e.message}")
        return {"quantity": Decimal('0'), "entry_price": Decimal('0'), "side": "NONE", "unrealized_pnl": Decimal('0'), "initial_balance": Decimal('0'), "sl_price": Decimal('0'), "tp_price": Decimal('0')}

def check_open_orders():
    try:
        orders = client.futures_get_open_orders(symbol=TRADE_SYMBOL)
        for order in orders:
            logging.info(f"Open order: {order['type']} at {order['stopPrice']}")
            print(f"Open Order Type: {order['type']}")
            print(f"Open Order Stop Price: {order['stopPrice']}")
        return len(orders)
    except BinanceAPIException as e:
        logging.error(f"Error checking open orders: {e.message}")
        print(f"Error Checking Open Orders")
        print(f"Error Message: {e.message}")
        return 0

def calculate_quantity(balance, price):
    if price <= Decimal('0') or balance < MINIMUM_BALANCE:
        logging.warning(f"Insufficient balance {balance:.25f} USDC or invalid price {price:.25f}. Cannot calculate quantity.")
        print(f"Insufficient Balance: {balance:.25f} USDC")
        print(f"Invalid Price: {price:.25f}")
        print("Cannot Calculate Quantity")
        return Decimal('0')
    quantity = (balance * LEVERAGE) / price
    return quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')

def place_order(signal, quantity, current_price, initial_balance):
    try:
        if quantity <= Decimal('0'):
            logging.warning(f"Invalid quantity {quantity:.25f}. Skipping order.")
            print(f"Invalid Quantity: {quantity:.25f}")
            print("Skipping Order")
            return None
        position = get_position()
        position["initial_balance"] = initial_balance
        side = "BUY" if signal == "LONG" else "SELL"
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side=side,
            type="MARKET",
            quantity=str(quantity)
        )
        tp_price = (current_price * (Decimal('1') + TAKE_PROFIT_PERCENTAGE if signal == "LONG" else Decimal('1') - TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
        sl_price = (current_price * (Decimal('1') - STOP_LOSS_PERCENTAGE if signal == "LONG" else Decimal('1') + STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
        position.update({
            "sl_price": sl_price,
            "tp_price": tp_price,
            "side": signal,
            "quantity": quantity if signal == "LONG" else -quantity,
            "entry_price": current_price
        })
        logging.info(f"Placed {signal} order: {quantity:.25f} BTC at ~{current_price:.25f}, SL: {sl_price:.25f}, TP: {tp_price:.25f}")
        print("=== TRADE ENTERED ===")
        print(f"Trade Side: {signal}")
        print(f"Quantity: {quantity:.25f} BTC")
        print(f"Entry Price: {current_price:.25f} USDC")
        print(f"Initial USDC Balance: {initial_balance:.25f}")
        print(f"Stop-Loss Price: {sl_price:.25f}")
        print("Stop-Loss ROI: -10%")
        print(f"Take-Profit Price: {tp_price:.25f}")
        print("Take-Profit ROI: +10%")
        print("===================")
        open_orders = check_open_orders()
        if open_orders > 0:
            logging.warning(f"Unexpected open orders ({open_orders}) detected after placing {signal} order.")
            print(f"Unexpected Open Orders Detected: {open_orders}")
            print(f"After Placing Signal: {signal}")
        return position
    except BinanceAPIException as e:
        logging.error(f"Error placing {signal} order: {e.message}")
        print(f"Error Placing Order: {signal}")
        print(f"Error Message: {e.message}")
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
        logging.info(f"Closed {position['side']} position: {quantity:.25f} BTC at ~{current_price:.25f}")
        print(f"Closed Position Side: {position['side']}")
        print(f"Closed Quantity: {quantity:.25f} BTC")
        print(f"Closed Price: {current_price:.25f}")
    except BinanceAPIException as e:
        logging.error(f"Error closing position: {e.message}")
        print(f"Error Closing Position")
        print(f"Error Message: {e.message}")

# Analysis Functions
def calculate_volume_sr_zones(candles, timeframe):
    if len(candles) < MIN_CANDLES:
        logging.warning(f"Insufficient candles ({len(candles)}) for S/R in {timeframe}")
        print(f"{timeframe} S/R Insufficient Candles: {len(candles)}")
        return DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, False

    lows = np.array([float(c['low']) for c in candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in candles], dtype=np.float64)
    closes = np.array([float(c['close']) for c in candles], dtype=np.float64)

    if len(lows) == 0 or len(highs) == 0 or len(closes) == 0:
        logging.warning(f"No valid price data for S/R in {timeframe}")
        print(f"{timeframe} S/R No Valid Price Data")
        return DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, False

    current_close = safe_decimal(closes[-1], default=DEFAULT_VALUE)
    support = safe_decimal(np.min(lows), default=DEFAULT_VALUE)
    resistance = safe_decimal(np.max(highs), default=DEFAULT_VALUE)
    middle = (support + resistance) / Decimal('2')

    price_range = safe_decimal(np.max(highs[-LIQUIDITY_WINDOW:]) - np.min(lows[-LIQUIDITY_WINDOW:]), default=DEFAULT_VALUE)
    is_consolidating = price_range <= current_close * CONSOLIDATION_THRESHOLD

    if is_consolidating:
        support = current_close * (Decimal('1') - CONSOLIDATION_THRESHOLD / Decimal('2'))
        resistance = current_close * (Decimal('1') + CONSOLIDATION_THRESHOLD / Decimal('2'))
        middle = current_close

    support = support if support > 0 else DEFAULT_VALUE
    resistance = resistance if resistance > 0 else DEFAULT_VALUE
    middle = middle if middle > 0 else DEFAULT_VALUE

    logging.info(f"{timeframe} S/R - Support: {support:.25f}, Resistance: {resistance:.25f}, Middle: {middle:.25f}, Consolidating: {is_consolidating}")
    print(f"{timeframe} Support Price: {support:.25f}")
    print(f"{timeframe} Resistance Price: {resistance:.25f}")
    print(f"{timeframe} Middle Price: {middle:.25f}")
    print(f"{timeframe} Is Consolidating: {is_consolidating}")
    return support, middle, resistance, is_consolidating

def analyze_volume_zones(candles, timeframe, min_threshold, max_threshold, current_close):
    if len(candles) < LIQUIDITY_WINDOW:
        logging.warning(f"Insufficient candles ({len(candles)}) for volume analysis in {timeframe}")
        print(f"{timeframe} Volume Zone Insufficient Candles: {len(candles)}")
        return {
            "zone": "NONE",
            "volume_status": "NONE",
            "bullish_volume_pct": Decimal('50'),
            "bearish_volume_pct": Decimal('50'),
            "consolidation_price": current_close
        }

    recent_candles = candles[-LIQUIDITY_WINDOW:]
    volumes = np.array([float(c['volume']) for c in recent_candles], dtype=np.float64)
    lows = np.array([float(c['low']) for c in recent_candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles], dtype=np.float64)
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)

    avg_volume = np.mean(volumes) if len(volumes) > 0 else 1
    volume_std = np.std(volumes) if len(volumes) > 0 else 0
    is_volume_consolidating = volume_std < avg_volume * 0.5

    buy_volume = Decimal('0')
    sell_volume = Decimal('0')
    total_volume = sum(safe_decimal(v, default=Decimal('1')) for v in volumes) or Decimal('1')

    for candle, low, high, close in zip(recent_candles, lows, highs, closes):
        vol = safe_decimal(candle['volume'], default=Decimal('1'))
        price_range = safe_decimal(high - low, default=Decimal('0.01'))
        bullish_strength = (safe_decimal(close, default=DEFAULT_VALUE) - safe_decimal(low, default=DEFAULT_VALUE)) / price_range
        buy_vol = vol * bullish_strength
        sell_vol = vol * (Decimal('1') - bullish_strength)
        buy_volume += buy_vol
        sell_volume += sell_vol

    bullish_volume_pct = (buy_volume / total_volume * Decimal('100')).quantize(Decimal('0.01'))
    bearish_volume_pct = (sell_volume / total_volume * Decimal('100')).quantize(Decimal('0.01'))

    if bullish_volume_pct == bearish_volume_pct:
        bullish_volume_pct += Decimal('0.01')
        bearish_volume_pct -= Decimal('0.01')

    zone = "NONE"
    volume_status = "NONE"
    consolidation_price = current_close

    if any(abs(safe_decimal(low, default=DEFAULT_VALUE) - min_threshold) <= PRICE_TOLERANCE for low in lows):
        zone = "SUPPORT"
        volume_status = "EXHAUSTION" if volumes[-1] > avg_volume * float(VOLUME_THRESHOLD) else "CONSOLIDATION"
        consolidation_price = min_threshold
    elif any(abs(safe_decimal(high, default=DEFAULT_VALUE) - max_threshold) <= PRICE_TOLERANCE for high in highs):
        zone = "RESISTANCE"
        volume_status = "EXHAUSTION" if volumes[-1] > avg_volume * float(VOLUME_THRESHOLD) else "CONSOLIDATION"
        consolidation_price = max_threshold
    elif all(min_threshold < safe_decimal(low, default=DEFAULT_VALUE) < max_threshold and min_threshold < safe_decimal(high, default=DEFAULT_VALUE) < max_threshold for low, high in zip(lows, highs)):
        zone = "RANGE"
        volume_status = "CONSOLIDATION" if is_volume_consolidating else "ACCUMULATION"
        consolidation_price = safe_decimal(np.mean(closes), default=DEFAULT_VALUE)
    elif any(safe_decimal(low, default=DEFAULT_VALUE) < min_threshold for low in lows):
        zone = "BELOW_SUPPORT"
        volume_status = "BREAKDOWN" if volumes[-1] > avg_volume * float(VOLUME_THRESHOLD) else "RETEST"
        consolidation_price = safe_decimal(np.min(lows), default=DEFAULT_VALUE)
    elif any(safe_decimal(high, default=DEFAULT_VALUE) > max_threshold for high in highs):
        zone = "ABOVE_RESISTANCE"
        volume_status = "BREAKOUT" if volumes[-1] > avg_volume * float(VOLUME_THRESHOLD) else "RETEST"
        consolidation_price = safe_decimal(np.max(highs), default=DEFAULT_VALUE)

    logging.info(f"{timeframe} Volume Zone - Zone: {zone}, Status: {volume_status}, Bullish Volume: {bullish_volume_pct:.2f}%, Bearish Volume: {bearish_volume_pct:.2f}%, Consolidation Price: {consolidation_price:.25f}")
    print(f"{timeframe} Volume Zone: {zone}")
    print(f"{timeframe} Volume Status: {volume_status}")
    print(f"{timeframe} Bullish Volume Percentage: {bullish_volume_pct:.2f}%")
    print(f"{timeframe} Bearish Volume Percentage: {bearish_volume_pct:.2f}%")
    print(f"{timeframe} Consolidation Price: {consolidation_price:.25f}")
    return {
        "zone": zone,
        "volume_status": volume_status,
        "bullish_volume_pct": bullish_volume_pct,
        "bearish_volume_pct": bearish_volume_pct,
        "consolidation_price": consolidation_price
    }

def test_liquidity_levels(candles, timeframe, min_threshold, max_threshold, current_close):
    if len(candles) < LIQUIDITY_WINDOW:
        logging.warning(f"Insufficient candles ({len(candles)}) for liquidity in {timeframe}")
        print(f"{timeframe} Liquidity Insufficient Candles: {len(candles)}")
        return {
            "support_target": min_threshold,
            "resistance_target": max_threshold,
            "breakout_target": max_threshold * (Decimal('1') + FORECAST_EXTENSION),
            "breakdown_target": min_threshold * (Decimal('1') - FORECAST_EXTENSION),
            "support_volume_pct": Decimal('25'),
            "resistance_volume_pct": Decimal('25'),
            "breakout_volume_pct": Decimal('25'),
            "breakdown_volume_pct": Decimal('25')
        }

    recent_candles = candles[-LIQUIDITY_WINDOW:]
    volumes = np.array([float(c['volume']) for c in recent_candles], dtype=np.float64)
    lows = np.array([float(c['low']) for c in recent_candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles], dtype=np.float64)

    total_volume = sum(safe_decimal(v, default=Decimal('1')) for v in volumes) or Decimal('1')
    breakout_target = max_threshold * (Decimal('1') + FORECAST_EXTENSION)
    breakdown_target = min_threshold * (Decimal('1') - FORECAST_EXTENSION)

    support_volume = sum(safe_decimal(v, default=Decimal('1')) for v, l in zip(volumes, lows) if abs(safe_decimal(l, default=DEFAULT_VALUE) - min_threshold) <= PRICE_TOLERANCE)
    resistance_volume = sum(safe_decimal(v, default=Decimal('1')) for v, h in zip(volumes, highs) if abs(safe_decimal(h, default=DEFAULT_VALUE) - max_threshold) <= PRICE_TOLERANCE)
    breakout_volume = sum(safe_decimal(v, default=Decimal('1')) for v, h in zip(volumes, highs) if safe_decimal(h, default=DEFAULT_VALUE) > max_threshold)
    breakdown_volume = sum(safe_decimal(v, default=Decimal('1')) for v, l in zip(volumes, lows) if safe_decimal(l, default=DEFAULT_VALUE) < min_threshold)

    support_volume_pct = (support_volume / total_volume * Decimal('100')).quantize(Decimal('0.01')) or Decimal('25')
    resistance_volume_pct = (resistance_volume / total_volume * Decimal('100')).quantize(Decimal('0.01')) or Decimal('25')
    breakout_volume_pct = (breakout_volume / total_volume * Decimal('100')).quantize(Decimal('0.01')) or Decimal('25')
    breakdown_volume_pct = (breakdown_volume / total_volume * Decimal('100')).quantize(Decimal('0.01')) or Decimal('25')

    total_pct = support_volume_pct + resistance_volume_pct + breakout_volume_pct + breakdown_volume_pct
    if total_pct != Decimal('100'):
        support_volume_pct = (support_volume_pct / total_pct * Decimal('100')).quantize(Decimal('0.01'))
        resistance_volume_pct = (resistance_volume_pct / total_pct * Decimal('100')).quantize(Decimal('0.01'))
        breakout_volume_pct = (breakout_volume_pct / total_pct * Decimal('100')).quantize(Decimal('0.01'))
        breakdown_volume_pct = (breakdown_volume_pct / total_pct * Decimal('100')).quantize(Decimal('0.01'))

    logging.info(f"{timeframe} Liquidity - Support: {min_threshold:.25f} ({support_volume_pct:.2f}%), Resistance: {max_threshold:.25f} ({resistance_volume_pct:.2f}%), Breakout: {breakout_target:.25f} ({breakout_volume_pct:.2f}%), Breakdown: {breakdown_target:.25f} ({breakdown_volume_pct:.2f}%)")
    print(f"{timeframe} Support Target: {min_threshold:.25f}")
    print(f"{timeframe} Support Volume Percentage: {support_volume_pct:.2f}%")
    print(f"{timeframe} Resistance Target: {max_threshold:.25f}")
    print(f"{timeframe} Resistance Volume Percentage: {resistance_volume_pct:.2f}%")
    print(f"{timeframe} Breakout Target: {breakout_target:.25f}")
    print(f"{timeframe} Breakout Volume Percentage: {breakout_volume_pct:.2f}%")
    print(f"{timeframe} Breakdown Target: {breakdown_target:.25f}")
    print(f"{timeframe} Breakdown Volume Percentage: {breakdown_volume_pct:.2f}%")
    return {
        "support_target": min_threshold,
        "resistance_target": max_threshold,
        "breakout_target": breakout_target,
        "breakdown_target": breakdown_target,
        "support_volume_pct": support_volume_pct,
        "resistance_volume_pct": resistance_volume_pct,
        "breakout_volume_pct": breakout_volume_pct,
        "breakdown_volume_pct": breakdown_volume_pct
    }

def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, volume_zone):
    if len(candles) < MIN_CANDLES:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal in {timeframe}")
        print(f"{timeframe} Reversal Insufficient Candles: {len(candles)}")
        return "DIP", min_threshold, max_threshold

    lookback = min(len(candles), MAX_CANDLES)
    recent_candles = candles[-lookback:]
    valid_candles = [c for c in recent_candles if c['close'] > 0 and c['low'] > 0 and c['high'] > 0 and c['volume'] > 0]
    
    if len(valid_candles) < MIN_CANDLES:
        logging.warning(f"Insufficient valid candles ({len(valid_candles)}) for reversal in {timeframe}")
        print(f"{timeframe} Reversal Insufficient Valid Candles: {len(valid_candles)}")
        return "DIP", min_threshold, max_threshold

    closes = np.array([float(c['close']) for c in valid_candles], dtype=np.float64)
    lows = np.array([float(c['low']) for c in valid_candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in valid_candles], dtype=np.float64)
    volumes = np.array([float(c['volume']) for c in valid_candles], dtype=np.float64)
    times = np.array([c['time'] for c in valid_candles], dtype=np.float64)

    current_close = safe_decimal(closes[-1], default=DEFAULT_VALUE)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 1

    high_volume_mask = volumes > avg_volume * float(VOLUME_THRESHOLD)
    high_volume_lows = lows[high_volume_mask]
    high_volume_highs = highs[high_volume_mask]
    high_volume_times = times[high_volume_mask]
    high_volume_volumes = volumes[high_volume_mask]
    high_volume_indices = np.where(high_volume_mask)[0]

    reversals = []
    if len(high_volume_lows) > 0 and len(high_volume_highs) > 0:
        for i, idx in enumerate(high_volume_indices):
            if lows[idx] in high_volume_lows:
                reversals.append({
                    "type": "DIP",
                    "price": safe_decimal(lows[idx], default=DEFAULT_VALUE),
                    "time": times[idx],
                    "volume": safe_decimal(volumes[idx], default=Decimal('1')),
                    "index": idx
                })
            if highs[idx] in high_volume_highs:
                reversals.append({
                    "type": "TOP",
                    "price": safe_decimal(highs[idx], default=DEFAULT_VALUE),
                    "time": times[idx],
                    "volume": safe_decimal(volumes[idx], default=Decimal('1')),
                    "index": idx
                })
    else:
        logging.info(f"No high-volume candles in {timeframe}. Using nearest extremes.")
        print(f"{timeframe} Reversal No High-Volume Candles")
        print(f"{timeframe} Using Nearest Extremes")
        min_idx = np.argmin(lows)
        max_idx = np.argmax(highs)
        reversals = [
            {"type": "DIP", "price": safe_decimal(lows[min_idx], default=DEFAULT_VALUE), "time": times[min_idx], "volume": safe_decimal(volumes[min_idx], default=Decimal('1')), "index": min_idx},
            {"type": "TOP", "price": safe_decimal(highs[max_idx], default=DEFAULT_VALUE), "time": times[max_idx], "volume": safe_decimal(volumes[max_idx], default=Decimal('1')), "index": max_idx}
        ]

    if not reversals:
        logging.warning(f"No valid reversals detected in {timeframe}. Defaulting to DIP.")
        print(f"{timeframe} Reversal No Valid Reversals Detected")
        print(f"{timeframe} Defaulting to DIP")
        return "DIP", min_threshold, max_threshold

    reversals.sort(key=lambda x: (x["time"], x["volume"]), reverse=True)
    most_recent = reversals[0]

    price_range = max_threshold - min_threshold or Decimal('0.01')
    dist_to_min = abs(current_close - min_threshold)
    dist_to_max = abs(current_close - max_threshold)
    total_dist = dist_to_min + dist_to_max or Decimal('0.01')
    min_pct = (dist_to_min / total_dist * Decimal('100')).quantize(Decimal('0.01'))
    max_pct = (dist_to_max / total_dist * Decimal('100')).quantize(Decimal('0.01'))

    bullish_volume_pct = volume_zone.get("bullish_volume_pct", Decimal('50'))
    if bullish_volume_pct > Decimal('50'):
        min_pct = min(min_pct + Decimal('0.01'), Decimal('99.99'))
        max_pct = max(max_pct - Decimal('0.01'), Decimal('0.01'))
    else:
        max_pct = min(max_pct + Decimal('0.01'), Decimal('99.99'))
        min_pct = max(min_pct - Decimal('0.01'), Decimal('0.01'))

    reversal_price = most_recent["price"]
    confirmed_type = most_recent["type"]
    reversal_time = datetime.datetime.fromtimestamp(most_recent['time'])
    reversal_volume = most_recent["volume"]

    logging.info(f"{timeframe} Most Recent Reversal for Current Cycle - Type: {confirmed_type}, Price: {reversal_price:.25f}, Time: {reversal_time}, Volume: {reversal_volume:.25f}, Dist to Min: {min_pct:.2f}%, Dist to Max: {max_pct:.2f}%")
    print(f"{timeframe} Reversal Type Found Most Recent for Current Cycle: {confirmed_type}")
    print(f"{timeframe} Reversal Price: {reversal_price:.25f}")
    print(f"{timeframe} Reversal Time: {reversal_time}")
    print(f"{timeframe} Reversal Volume: {reversal_volume:.25f}")
    print(f"{timeframe} Distance to Min: {min_pct:.2f}%")
    print(f"{timeframe} Distance to Max: {max_pct:.2f}%")
    return confirmed_type, min_threshold, max_threshold

def detect_new_reversals(candles, timeframe, min_threshold, max_threshold, volume_zone, recent_reversal_type):
    if len(candles) < LIQUIDITY_WINDOW:
        logging.warning(f"Insufficient candles ({len(candles)}) for new reversals in {timeframe}")
        print(f"{timeframe} New Reversal Insufficient Candles: {len(candles)}")
        return {
            "new_reversal_type": "DIP",
            "new_reversal_price": min_threshold,
            "new_reversal_volume": Decimal('1'),
            "incoming_signal": "BREAKDOWN"
        }

    recent_candles = candles[-LIQUIDITY_WINDOW:]
    lows = np.array([float(c['low']) for c in recent_candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles], dtype=np.float64)
    volumes = np.array([float(c['volume']) for c in recent_candles], dtype=np.float64)
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    times = np.array([c['time'] for c in recent_candles], dtype=np.float64)

    avg_volume = np.mean(volumes) if len(volumes) > 0 else 1
    bullish_volume_pct = volume_zone.get("bullish_volume_pct", Decimal('50'))
    bearish_volume_pct = volume_zone.get("bearish_volume_pct", Decimal('50'))

    new_reversal_type = "DIP"
    new_reversal_price = safe_decimal(np.min(lows), default=min_threshold)
    new_reversal_volume = safe_decimal(volumes[np.argmin(lows)], default=Decimal('1'))
    incoming_signal = "BREAKDOWN"
    volume_confirmed = volumes[-1] > avg_volume * float(VOLUME_CONFIRMATION_THRESHOLD)

    if recent_reversal_type == "DIP":
        if volume_zone["zone"] == "ABOVE_RESISTANCE" and bullish_volume_pct > Decimal('60') and volume_confirmed:
            new_reversal_type = "TOP"
            new_reversal_price = safe_decimal(np.max(highs), default=max_threshold)
            new_reversal_volume = safe_decimal(volumes[np.argmax(highs)], default=Decimal('1'))
            incoming_signal = "BREAKUP"
            logging.info(f"{timeframe} New Reversal - Volume confirms BREAKUP after DIP. Setting to TOP.")
            print(f"{timeframe} New Reversal Type: TOP (Volume confirms BREAKUP after DIP)")
        elif bearish_volume_pct > Decimal('60') and volume_confirmed:
            new_reversal_type = "TOP"
            new_reversal_price = safe_decimal(np.max(highs), default=max_threshold)
            new_reversal_volume = safe_decimal(volumes[np.argmax(highs)], default=Decimal('1'))
            incoming_signal = "REVERSAL_TOP"
            logging.info(f"{timeframe} New Reversal - Volume confirms REVERSAL_TOP after DIP.")
            print(f"{timeframe} New Reversal Type: TOP (Volume confirms REVERSAL_TOP after DIP)")
        else:
            new_reversal_type = "DIP"
            incoming_signal = "BREAKDOWN"
            logging.info(f"{timeframe} New Reversal - No volume reversal. Continuing DIP with BREAKDOWN.")
            print(f"{timeframe} New Reversal Type: DIP (No volume reversal, BREAKDOWN incoming)")
    else:  # recent_reversal_type == "TOP"
        if volume_zone["zone"] == "BELOW_SUPPORT" and bearish_volume_pct > Decimal('60') and volume_confirmed:
            new_reversal_type = "DIP"
            new_reversal_price = safe_decimal(np.min(lows), default=min_threshold)
            new_reversal_volume = safe_decimal(volumes[np.argmin(lows)], default=Decimal('1'))
            incoming_signal = "BREAKDOWN"
            logging.info(f"{timeframe} New Reversal - Volume confirms BREAKDOWN after TOP. Setting to DIP.")
            print(f"{timeframe} New Reversal Type: DIP (Volume confirms BREAKDOWN after TOP)")
        elif bullish_volume_pct > Decimal('60') and volume_confirmed:
            new_reversal_type = "DIP"
            new_reversal_price = safe_decimal(np.min(lows), default=min_threshold)
            new_reversal_volume = safe_decimal(volumes[np.argmin(lows)], default=Decimal('1'))
            incoming_signal = "REVERSAL_DIP"
            logging.info(f"{timeframe} New Reversal - Volume confirms REVERSAL_DIP after TOP.")
            print(f"{timeframe} New Reversal Type: DIP (Volume confirms REVERSAL_DIP after TOP)")
        else:
            new_reversal_type = "TOP"
            new_reversal_price = safe_decimal(np.max(highs), default=max_threshold)
            new_reversal_volume = safe_decimal(volumes[np.argmax(highs)], default=Decimal('1'))
            incoming_signal = "BREAKUP"
            logging.info(f"{timeframe} New Reversal - No volume reversal. Continuing TOP with BREAKUP.")
            print(f"{timeframe} New Reversal Type: TOP (No volume reversal, BREAKUP incoming)")

    new_reversal_price = new_reversal_price if new_reversal_price > 0 else min_threshold
    new_reversal_volume = new_reversal_volume if new_reversal_volume > 0 else Decimal('1')

    logging.info(f"{timeframe} New Reversal - Type: {new_reversal_type}, Price: {new_reversal_price:.25f}, Volume: {new_reversal_volume:.25f}, Incoming Signal: {incoming_signal}")
    print(f"{timeframe} New Reversal Price: {new_reversal_price:.25f}")
    print(f"{timeframe} New Reversal Volume: {new_reversal_volume:.25f}")
    print(f"{timeframe} Incoming Signal: {incoming_signal}")
    return {
        "new_reversal_type": new_reversal_type,
        "new_reversal_price": new_reversal_price,
        "new_reversal_volume": new_reversal_volume,
        "incoming_signal": incoming_signal
    }

def analyze_threshold_break_intensity(candles, timeframe, min_threshold, max_threshold):
    if len(candles) < LIQUIDITY_WINDOW:
        logging.warning(f"Insufficient candles ({len(candles)}) for threshold break in {timeframe}")
        print(f"{timeframe} Threshold Break Insufficient Candles: {len(candles)}")
        return {
            "min_break_intensity": Decimal('0.01'),
            "max_break_intensity": Decimal('0.01'),
            "min_confirmed": False,
            "max_confirmed": False,
            "new_support": min_threshold,
            "new_resistance": max_threshold
        }

    recent_candles = candles[-LIQUIDITY_WINDOW:]
    volumes = np.array([float(c['volume']) for c in recent_candles], dtype=np.float64)
    lows = np.array([float(c['low']) for c in recent_candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles], dtype=np.float64)

    avg_volume = np.mean(volumes) if len(volumes) > 0 else 1
    latest_candle = recent_candles[-1]
    current_low = safe_decimal(latest_candle['low'], default=DEFAULT_VALUE)
    current_high = safe_decimal(latest_candle['high'], default=DEFAULT_VALUE)
    current_volume = safe_decimal(latest_candle['volume'], default=Decimal('1'))
    volume_intensity = current_volume / Decimal(str(avg_volume)) or Decimal('1')

    min_break_intensity = Decimal('0.01')
    max_break_intensity = Decimal('0.01')
    min_confirmed = False
    max_confirmed = False
    new_support = min_threshold
    new_resistance = max_threshold

    if abs(current_low - min_threshold) <= PRICE_TOLERANCE:
        min_confirmed = True
        min_break_intensity = volume_intensity
    elif current_low < min_threshold:
        min_break_intensity = volume_intensity * (min_threshold - current_low) / min_threshold
        if current_volume > avg_volume * float(VOLUME_THRESHOLD):
            new_support = current_low

    if abs(current_high - max_threshold) <= PRICE_TOLERANCE:
        max_confirmed = True
        max_break_intensity = volume_intensity
    elif current_high > max_threshold:
        max_break_intensity = volume_intensity * (current_high - max_threshold) / max_threshold
        if current_volume > avg_volume * float(VOLUME_THRESHOLD):
            new_resistance = current_high

    logging.info(f"{timeframe} Threshold Break - Min Intensity: {min_break_intensity:.2f}, Confirmed: {min_confirmed}, New Support: {new_support:.25f}, Max Intensity: {max_break_intensity:.2f}, Confirmed: {max_confirmed}, New Resistance: {new_resistance:.25f}")
    print(f"{timeframe} Min Break Intensity: {min_break_intensity:.2f}")
    print(f"{timeframe} Min Break Confirmed: {min_confirmed}")
    print(f"{timeframe} New Support Price: {new_support:.25f}")
    print(f"{timeframe} Max Break Intensity: {max_break_intensity:.2f}")
    print(f"{timeframe} Max Break Confirmed: {max_confirmed}")
    print(f"{timeframe} New Resistance Price: {new_resistance:.25f}")
    return {
        "min_break_intensity": min_break_intensity.quantize(Decimal('0.01')),
        "max_break_intensity": max_break_intensity.quantize(Decimal('0.01')),
        "min_confirmed": min_confirmed,
        "max_confirmed": max_confirmed,
        "new_support": new_support,
        "new_resistance": new_resistance
    }

def forecast_price_breakout(candles, timeframe, volume_ratio, reversal_type, liquidity_metrics, volume_zone, incoming_signal):
    if len(candles) < MIN_CANDLES:
        logging.warning(f"Insufficient candles ({len(candles)}) for forecast in {timeframe}")
        print(f"{timeframe} Forecast Insufficient Candles: {len(candles)}")
        return {"projected_price": DEFAULT_VALUE, "direction": "DOWN", "confidence": Decimal('0.5')}

    current_close = safe_decimal(candles[-1]['close'], default=DEFAULT_VALUE)
    min_threshold, middle_threshold, max_threshold, is_consolidating = calculate_volume_sr_zones(candles, timeframe)
    buy_ratio = volume_ratio.get("buy_ratio", Decimal('50'))

    volumes = np.array([float(c['volume']) for c in candles[-LIQUIDITY_WINDOW:]], dtype=np.float64)
    volume_std = np.std(volumes) if len(volumes) > 0 else 0
    mean_volume = np.mean(volumes) if len(volumes) > 0 else 1
    is_volume_consolidating = volume_std < mean_volume * 0.5

    price_range = max_threshold - min_threshold or Decimal('0.01')
    breakout_volume_pct = liquidity_metrics.get("breakout_volume_pct", Decimal('25'))
    breakdown_volume_pct = liquidity_metrics.get("breakdown_volume_pct", Decimal('25'))
    bullish_volume_pct = volume_zone.get("bullish_volume_pct", Decimal('50'))

    confidence = Decimal('0.6')
    projected_price = current_close
    direction = "DOWN"

    if incoming_signal in ["REVERSAL_TOP", "REVERSAL_DIP"]:
        confidence = Decimal('0.9') if buy_ratio > 50 else Decimal('0.7')
        if incoming_signal == "REVERSAL_TOP":
            projected_price = current_close - price_range * FORECAST_EXTENSION
            direction = "DOWN"
        else:  # REVERSAL_DIP
            projected_price = current_close + price_range * FORECAST_EXTENSION
            direction = "UP"
    elif incoming_signal == "BREAKUP":
        confidence = Decimal('0.8') + (Decimal('0.2') * (breakout_volume_pct / Decimal('100')))
        projected_price = current_close + price_range * FORECAST_EXTENSION
        direction = "UP"
    elif incoming_signal == "BREAKDOWN":
        confidence = Decimal('0.8') + (Decimal('0.2') * (breakdown_volume_pct / Decimal('100')))
        projected_price = current_close - price_range * FORECAST_EXTENSION
        direction = "DOWN"
    elif is_consolidating and is_volume_consolidating:
        confidence = Decimal('0.7')
        projected_price = middle_threshold
        direction = "SIDEWAYS"
    else:
        confidence += Decimal('0.3') * (bullish_volume_pct / Decimal('100'))
        projected_price = current_close + (price_range if buy_ratio >= 50 else -price_range)
        direction = "UP" if buy_ratio >= 50 else "DOWN"

    projected_price = projected_price.quantize(Decimal('0.01')) or DEFAULT_VALUE
    logging.info(f"{timeframe} Forecast - Projected Price: {projected_price:.25f}, Direction: {direction}, Confidence: {confidence:.2f}, Breakout Vol: {breakout_volume_pct:.2f}%, Breakdown Vol: {breakdown_volume_pct:.2f}%, Incoming Signal: {incoming_signal}")
    print(f"{timeframe} Forecast Projected Price: {projected_price:.25f}")
    print(f"{timeframe} Forecast Direction: {direction}")
    print(f"{timeframe} Forecast Confidence: {confidence:.2f}")
    print(f"{timeframe} Forecast Breakout Volume Percentage: {breakout_volume_pct:.2f}%")
    print(f"{timeframe} Forecast Breakdown Volume Percentage: {breakdown_volume_pct:.2f}%")
    print(f"{timeframe} Incoming Signal: {incoming_signal}")
    return {"projected_price": projected_price, "direction": direction, "confidence": confidence}

def calculate_buy_sell_volume(candles):
    buy_volume, sell_volume = [], []
    for candle in candles:
        total_volume = safe_decimal(candle["volume"], default=Decimal('1'))
        close_price = safe_decimal(candle["close"], default=DEFAULT_VALUE)
        high_price = safe_decimal(candle["high"], default=DEFAULT_VALUE)
        low_price = safe_decimal(candle["low"], default=DEFAULT_VALUE)
        price_range = high_price - low_price or Decimal('0.01')
        bullish_strength = (close_price - low_price) / price_range
        buy_vol = total_volume * bullish_strength
        sell_vol = total_volume * (Decimal('1') - bullish_strength)
        buy_volume.append(buy_vol)
        sell_volume.append(sell_vol)
    return buy_volume, sell_volume

def calculate_volume_ratio(candle_map, timeframe):
    candles = candle_map.get(timeframe, [])
    if not candles:
        logging.warning(f"Insufficient volume data for {timeframe}")
        print(f"{timeframe} Volume Ratio: Insufficient Data")
        return {
            "buy_ratio": Decimal('50'),
            "sell_ratio": Decimal('50'),
            "status": "Insufficient Data",
            "buy_volume_pct": Decimal('50'),
            "sell_volume_pct": Decimal('50')
        }

    buy_volume, sell_volume = calculate_buy_sell_volume(candles)
    total_volume = buy_volume[-1] + sell_volume[-1] or Decimal('1')
    buy_pct = (buy_volume[-1] / total_volume * Decimal('100')).quantize(Decimal('0.01')) or Decimal('50')
    sell_volume_pct = (sell_volume[-1] / total_volume * Decimal('100')).quantize(Decimal('0.01')) or Decimal('50')

    if buy_pct >= Decimal('100'):
        buy_pct = Decimal('99.99')
        sell_volume_pct = Decimal('0.01')
    elif sell_volume_pct >= Decimal('100'):
        sell_volume_pct = Decimal('99.99')
        buy_pct = Decimal('0.01')

    buy_vols = np.array([float(v) for v in buy_volume[-3:]], dtype=np.float64)
    sell_vols = np.array([float(v) for v in sell_volume[-3:]], dtype=np.float64)
    buy_ema = safe_decimal(talib.EMA(buy_vols, timeperiod=3)[-1] if len(buy_vols) >= 3 else np.mean(buy_vols), default=Decimal('0')) if len(buy_vols) > 0 else sum(buy_volume[-3:]) / Decimal('3')
    sell_ema = safe_decimal(talib.EMA(sell_vols, timeperiod=3)[-1] if len(sell_vols) >= 3 else np.mean(sell_vols), default=Decimal('0')) if len(sell_vols) > 0 else sum(sell_volume[-3:]) / Decimal('3')
    total_ema = buy_ema + sell_ema or Decimal('1')
    buy_ratio = (buy_ema / total_ema * Decimal('100')).quantize(Decimal('0.01')) or Decimal('50')
    sell_ratio = (Decimal('100') - buy_ratio).quantize(Decimal('0.01'))
    status = "Bullish" if buy_ratio >= Decimal('50') else "Bearish"

    logging.info(f"{timeframe} Buy/Sell Ratio - Smoothed Buy: {buy_ratio:.2f}%, Sell: {sell_ratio:.2f}%, Buy Vol: {buy_pct:.2f}%, Sell Vol: {sell_volume_pct:.2f}%, Status: {status}")
    print(f"{timeframe} Smoothed Buy Ratio: {buy_ratio:.2f}%")
    print(f"{timeframe} Smoothed Sell Ratio: {sell_ratio:.2f}%")
    print(f"{timeframe} Buy Volume Percentage: {buy_pct:.2f}%")
    print(f"{timeframe} Sell Volume Percentage: {sell_volume_pct:.2f}%")
    print(f"{timeframe} Volume Status: {status}")
    return {
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "status": status,
        "buy_volume_pct": buy_pct,
        "sell_volume_pct": sell_volume_pct
    }

# Main Loop
def main():
    logging.info("Futures Trading Bot Initialized!")
    print("Futures Trading Bot Initialized")

    try:
        while True:
            current_local_time = datetime.datetime.now()
            time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Current Time: {time_str}")
            print(f"Current Local Time: {time_str}")

            candle_map = fetch_candles_in_parallel(TIMEFRAMES)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candle data available. Retrying in 60 seconds.")
                print("No Candle Data Available")
                print("Retrying in 60 Seconds")
                time.sleep(60)
                continue

            current_price = get_current_price()
            if current_price <= Decimal('0'):
                logging.error(f"Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")
                print(f"Current Symbol: {TRADE_SYMBOL}")
                print(f"Current Price: {current_price:.25f}")
                print("API May Be Failing")
                time.sleep(60)
                continue

            usdc_balance = get_balance()
            position = get_position()

            if position["side"] != "NONE" and position["sl_price"] > 0 and position["tp_price"] > 0:
                if (position["side"] == "LONG" and current_price <= position["sl_price"]) or \
                   (position["side"] == "SHORT" and current_price >= position["sl_price"]):
                    logging.info(f"Stop-Loss triggered for {position['side']} at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                    print(f"Stop-Loss Triggered Side: {position['side']}")
                    print(f"Stop-Loss Triggered Price: {current_price:.25f}")
                    print(f"Stop-Loss Price: {position['sl_price']:.25f}")
                    close_position(position, current_price)
                    position = get_position()
                elif (position["side"] == "LONG" and current_price >= position["tp_price"]) or \
                     (position["side"] == "SHORT" and current_price <= position["tp_price"]):
                    logging.info(f"Take-Profit triggered for {position['side']} at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                    print(f"Take-Profit Triggered Side: {position['side']}")
                    print(f"Take-Profit Triggered Price: {current_price:.25f}")
                    print(f"Take-Profit Price: {position['tp_price']:.25f}")
                    close_position(position, current_price)
                    position = get_position()

            conditions_long = {f"volume_bullish_{tf}": False for tf in TIMEFRAMES}
            conditions_long.update({f"dip_confirmation_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"below_middle_{tf}": False for tf in TIMEFRAMES})
            conditions_long["momentum_positive_1m"] = False
            conditions_short = {f"volume_bearish_{tf}": False for tf in TIMEFRAMES}
            conditions_short.update({f"top_confirmation_{tf}": False for tf in TIMEFRAMES})
            conditions_short.update({f"above_middle_{tf}": False for tf in TIMEFRAMES})
            conditions_short["momentum_negative_1m"] = False

            volume_ratios = {}
            reversal_signals = {}
            liquidity_levels = {}
            volume_zones = {}
            new_reversals = {}

            for timeframe in TIMEFRAMES:
                print(f"\n=== {timeframe} Analysis ===")
                candles = candle_map.get(timeframe, [])
                if not candles:
                    logging.warning(f"No data for {timeframe}. Skipping timeframe.")
                    print(f"{timeframe} No Data Available")
                    continue

                closes = [candle["close"] for candle in candles]
                current_close = safe_decimal(closes[-1], default=DEFAULT_VALUE) if closes else DEFAULT_VALUE

                min_threshold, middle_threshold, max_threshold, _ = calculate_volume_sr_zones(candles, timeframe)
                if min_threshold == DEFAULT_VALUE or max_threshold == DEFAULT_VALUE:
                    logging.warning(f"No valid thresholds for {timeframe}. Skipping analysis.")
                    print(f"{timeframe} No Valid Thresholds")
                    continue

                volume_ratios[timeframe] = calculate_volume_ratio(candle_map, timeframe)
                volume_zones[timeframe] = analyze_volume_zones(candles, timeframe, min_threshold, max_threshold, current_close)
                liquidity_levels[timeframe] = test_liquidity_levels(candles, timeframe, min_threshold, max_threshold, current_close)
                reversal_signal, min_threshold, max_threshold = detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, volume_zones[timeframe])
                reversal_signals[timeframe] = reversal_signal
                new_reversals[timeframe] = detect_new_reversals(candles, timeframe, min_threshold, max_threshold, volume_zones[timeframe], reversal_signal)
                intensity = analyze_threshold_break_intensity(candles, timeframe, min_threshold, max_threshold)
                forecast = forecast_price_breakout(candles, timeframe, volume_ratios[timeframe], reversal_signal, liquidity_levels[timeframe], volume_zones[timeframe], new_reversals[timeframe]["incoming_signal"])

                conditions_long[f"below_middle_{timeframe}"] = current_close < middle_threshold
                conditions_short[f"above_middle_{timeframe}"] = not conditions_long[f"below_middle_{timeframe}"]
                print(f"{timeframe} Current Close Price: {current_close:.25f}")
                print(f"{timeframe} Below Middle: {conditions_long[f'below_middle_{timeframe}']}")
                print(f"{timeframe} Above Middle: {conditions_short[f'above_middle_{timeframe}']}")

                buy_vol = volume_ratios[timeframe]["buy_volume_pct"]
                sell_vol = volume_ratios[timeframe]["sell_volume_pct"]
                price_trend = safe_decimal(closes[-1], default=DEFAULT_VALUE) - safe_decimal(closes[-2], default=DEFAULT_VALUE) if len(closes) >= 2 else Decimal('0')
                conditions_long[f"volume_bullish_{timeframe}"] = buy_vol >= sell_vol or price_trend > 0
                conditions_short[f"volume_bearish_{timeframe}"] = not conditions_long[f"volume_bullish_{timeframe}"]
                print(f"{timeframe} Bullish Volume: {buy_vol:.2f}%")
                print(f"{timeframe} Bearish Volume: {sell_vol:.2f}%")
                print(f"{timeframe} Volume Bullish Condition: {conditions_long[f'volume_bullish_{timeframe}']}")
                print(f"{timeframe} Volume Bearish Condition: {conditions_short[f'volume_bearish_{timeframe}']}")

                conditions_long[f"dip_confirmation_{timeframe}"] = reversal_signal == "DIP" or new_reversals[timeframe]["new_reversal_type"] == "DIP"
                conditions_short[f"top_confirmation_{timeframe}"] = reversal_signal == "TOP" or new_reversals[timeframe]["new_reversal_type"] == "TOP"
                print(f"{timeframe} Dip Confirmation: {conditions_long[f'dip_confirmation_{timeframe}']}")
                print(f"{timeframe} Top Confirmation: {conditions_short[f'top_confirmation_{timeframe}']}")

                if timeframe == "1m":
                    print("\n=== 1m Momentum Analysis ===")
                    valid_closes = np.array([float(c) for c in closes if c > 0], dtype=np.float64)
                    if len(valid_closes) >= 14:
                        momentum = talib.MOM(valid_closes, timeperiod=14)
                        if len(momentum) > 0 and not np.isnan(momentum[-1]):
                            current_momentum = safe_decimal(momentum[-1], default=Decimal('0'))
                            conditions_long["momentum_positive_1m"] = current_momentum >= 0
                            conditions_short["momentum_negative_1m"] = not conditions_long["momentum_positive_1m"]
                            print(f"1m Momentum Value: {current_momentum:.25f}")
                            print(f"1m Momentum Status: {'Positive' if conditions_long['momentum_positive_1m'] else 'Negative'}")
                    else:
                        logging.warning("1m Momentum: Insufficient data")
                        print("1m Momentum Insufficient Data")
                        conditions_long["momentum_positive_1m"] = True
                        conditions_short["momentum_negative_1m"] = False

            # Ensure mutual exclusivity of condition pairs
            for tf in TIMEFRAMES:
                if conditions_long[f"volume_bullish_{tf}"]:
                    conditions_short[f"volume_bearish_{tf}"] = False
                if conditions_long[f"dip_confirmation_{tf}"]:
                    conditions_short[f"top_confirmation_{tf}"] = False
                if conditions_long[f"below_middle_{tf}"]:
                    conditions_short[f"above_middle_{tf}"] = False
                if conditions_long["momentum_positive_1m"]:
                    conditions_short["momentum_negative_1m"] = False

            print("\n=== Long Conditions ===")
            long_conditions_list = [
                "volume_bullish_1m", "volume_bullish_3m", "volume_bullish_5m",
                "dip_confirmation_1m", "dip_confirmation_3m", "dip_confirmation_5m",
                "below_middle_1m", "below_middle_3m", "below_middle_5m",
                "momentum_positive_1m"
            ]
            for cond in long_conditions_list:
                print(f"Condition {cond}: {conditions_long[cond]}")
                logging.info(f"Long Condition {cond}: {conditions_long[cond]}")

            print("\n=== Short Conditions ===")
            short_conditions_list = [
                "volume_bearish_1m", "volume_bearish_3m", "volume_bearish_5m",
                "top_confirmation_1m", "top_confirmation_3m", "top_confirmation_5m",
                "above_middle_1m", "above_middle_3m", "above_middle_5m",
                "momentum_negative_1m"
            ]
            for cond in short_conditions_list:
                print(f"Condition {cond}: {conditions_short[cond]}")
                logging.info(f"Short Condition {cond}: {conditions_short[cond]}")

            long_signal = all(conditions_long.values()) and not any(conditions_short.values())
            short_signal = all(conditions_short.values()) and not any(conditions_long.values())

            print("\n=== Trade Signal Status ===")
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'}")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'}")

            long_true = sum(1 for val in conditions_long.values() if val)
            long_false = len(conditions_long) - long_true
            short_true = sum(1 for val in conditions_short.values() if val)
            short_false = len(conditions_short) - short_true
            print(f"Long Conditions: {long_true} True, {long_false} False")
            print(f"Short Conditions: {short_true} True, {short_false} False")

            signal = "NO_SIGNAL"
            if long_signal:
                signal = "LONG"
            elif short_signal:
                signal = "SHORT"
            if long_signal and short_signal:
                signal = "NO_SIGNAL"
                logging.warning("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                print("Signal Conflict: Both LONG and SHORT Active")
                print("Setting Signal to NO_SIGNAL")
            print(f"Final Signal: {signal}")
            logging.info(f"Final Signal: {signal}")

            if usdc_balance < MINIMUM_BALANCE:
                logging.warning(f"Insufficient balance {usdc_balance:.25f} USDC. Minimum: {MINIMUM_BALANCE:.25f}")
                print(f"USDC Balance: {usdc_balance:.25f}")
                print(f"Minimum Required Balance: {MINIMUM_BALANCE:.25f}")
                print("Insufficient Balance to Trade")
            elif signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            elif (signal == "LONG" and position["side"] == "SHORT") or (signal == "SHORT" and position["side"] == "LONG"):
                close_position(position, current_price)
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)

            if position["side"] != "NONE":
                print("\n=== Current Position Status ===")
                print(f"Position Side: {position['side']}")
                print(f"Position Quantity: {abs(position['quantity']):.25f} BTC")
                print(f"Entry Price: {position['entry_price']:.25f} USDC")
                print(f"Current Price: {current_price:.25f} USDC")
                print(f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC")
                print(f"Stop-Loss Price: {position['sl_price']:.25f} USDC")
                print(f"Take-Profit Price: {position['tp_price']:.25f} USDC")
                current_balance = usdc_balance + position['unrealized_pnl']
                roi = ((current_balance - position['initial_balance']) / position['initial_balance'] * Decimal('100')).quantize(Decimal('0.01')) if position['initial_balance'] > 0 else Decimal('0')
                print(f"Current ROI: {roi:.2f}%")
                print(f"Initial USDC Balance: {position['initial_balance']:.25f}")
                print(f"Current Total Balance: {current_balance:.25f} USDC")
            else:
                print("\n=== Position Status ===")
                print(f"No Open Position")
                print(f"USDC Balance: {usdc_balance:.25f}")

            print(f"Current USDC Balance: {usdc_balance:.25f}")
            print(f"Current Position Side: {position['side']}")
            print(f"Current Position Quantity: {abs(position['quantity']):.25f} BTC")
            print(f"Current Market Price: {current_price:.25f}")

            del candle_map
            gc.collect()
            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Shutting down bot...")
        print("Shutting Down Bot")
        position = get_position()
        if position["side"] != "NONE":
            close_position(position, get_current_price())
        logging.info("Bot shutdown complete.")
        print("Bot Shutdown Complete")
        exit(0)
    except Exception as e:
        error_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logging.error(f"Unexpected error in main loop: {e}\nStack Trace:\n{error_trace}")
        print(f"Unexpected Error in Main Loop: {e}")
        print(f"Stack Trace:\n{error_trace}")
        time.sleep(5)

if __name__ == "__main__":
    main()