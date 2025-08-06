import asyncio
import datetime
import time
import concurrent.futures
import talib
import numpy as np
from decimal import Decimal, getcontext
import logging
from telegram.ext import Application
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

# Configure logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set Decimal precision
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"
LEVERAGE = 25
STOP_LOSS_PERCENTAGE = Decimal('0.05')  # 5% stop-loss
TAKE_PROFIT_PERCENTAGE = Decimal('0.05')  # 5% take-profit
QUANTITY_PRECISION = Decimal('0.000001')  # Binance quantity precision for BTCUSDC
MINIMUM_BALANCE = Decimal('1.0000')  # Minimum USDC balance to place trades
TIMEFRAMES = ["1m", "3m"]  # Timeframes for analysis
LOOKBACK_PERIODS = {"1m": 1500, "3m": 1500}
VOLUME_CONFIRMATION_RATIO = Decimal('1.5')  # Volume increase ratio for LONG
VOLUME_EXHAUSTION_RATIO = Decimal('0.5')  # Volume decrease ratio for SHORT
SUPPORT_RESISTANCE_TOLERANCE = Decimal('0.005')  # 0.5% tolerance for support/resistance
API_TIMEOUT = 60  # Timeout for Binance API requests
RECENT_LOOKBACK = {"1m": 100, "3m": 50}  # Recent candles for reversal check
TIMEFRAME_INTERVALS = {"1m": 60, "3m": 180}  # Timeframe intervals in seconds
TOLERANCE_FACTORS = {"1m": Decimal('0.10'), "3m": Decimal('0.08')}  # Dynamic tolerance factors
VOLUME_LOOKBACK = 5  # Lookback for volume trend analysis at reversal

# Global event loop for Telegram
telegram_loop = asyncio.new_event_loop()

# Load credentials
try:
    with open("credentials.txt", "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) != 4:
        logging.error(f"credentials.txt must contain exactly 4 non-empty lines, found {len(lines)}.")
        print(f"Error: credentials.txt must contain exactly 4 non-empty lines, found {len(lines)}.")
        print("Expected format:\nBinance API key\nBinance API secret\nTelegram bot token\nTelegram chat ID")
        exit(1)
    api_key, api_secret, telegram_token, telegram_chat_id = lines
    if not all([api_key, api_secret, telegram_token, telegram_chat_id]):
        logging.error("One or more credentials in credentials.txt are empty.")
        print("Error: One or more credentials in credentials.txt are empty.")
        exit(1)
except FileNotFoundError:
    logging.error("credentials.txt not found.")
    print("Error: credentials.txt not found.")
    exit(1)
except Exception as e:
    logging.error(f"Unexpected error reading credentials.txt: {e}")
    print(f"Unexpected error reading credentials.txt: {e}")
    exit(1)

# Initialize Binance client
try:
    client = BinanceClient(api_key, api_secret, requests_params={"timeout": API_TIMEOUT})
    client.API_URL = 'https://fapi.binance.com'  # Futures API endpoint
except Exception as e:
    logging.error(f"Failed to initialize Binance client: {e}")
    print(f"Failed to initialize Binance client: {e}")
    exit(1)

# Initialize and validate Telegram bot
async def validate_telegram_chat_id(app, chat_id):
    try:
        chat = await app.bot.get_chat(chat_id)
        if chat.type in ["private", "group", "supergroup", "channel"]:
            logging.info(f"Valid Telegram chat ID: {chat_id} (Type: {chat.type})")
            print(f"Valid Telegram chat ID: {chat_id} (Type: {chat.type})")
            return True
        else:
            logging.error(f"Invalid Telegram chat ID: {chat_id}. Chat type '{chat.type}' is not supported.")
            print(f"Error: Invalid Telegram chat ID: {chat_id}. Chat type '{chat.type}' is not supported.")
            return False
    except Exception as e:
        if "Forbidden" in str(e):
            logging.error(f"Telegram chat ID validation failed: {chat_id}. Cannot send messages to bots or invalid chat IDs.")
            print(f"Error: Telegram chat ID {chat_id} is invalid or belongs to a bot.")
        else:
            logging.error(f"Failed to validate Telegram chat ID {chat_id}: {e}")
            print(f"Error validating Telegram chat ID {chat_id}: {e}")
        return False

try:
    telegram_app = Application.builder().token(telegram_token).build()
    asyncio.set_event_loop(telegram_loop)
    telegram_loop.run_until_complete(telegram_app.bot.get_me())
    if not telegram_loop.run_until_complete(validate_telegram_chat_id(telegram_app, telegram_chat_id)):
        exit(1)
    logging.info("Telegram bot initialized successfully.")
    print("Telegram bot initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Telegram bot: {e}")
    print(f"Failed to initialize Telegram bot: {e}")
    exit(1)

# Set leverage
try:
    client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
    logging.info(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
    print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
except BinanceAPIException as e:
    logging.error(f"Error setting leverage: {e.message}")
    print(f"Error setting leverage: {e.message}")
    exit(1)

async def send_telegram_message(message, retries=3, base_delay=5):
    for attempt in range(retries):
        try:
            await telegram_app.bot.send_message(chat_id=telegram_chat_id, text=message, parse_mode='Markdown')
            logging.info(f"Telegram message sent: {message[:100]}...")
            print(f"Telegram message sent: {message[:100]}...")
            return True
        except Exception as e:
            if "Forbidden" in str(e):
                logging.error(f"Cannot send Telegram message: Invalid chat ID: {telegram_chat_id}")
                print(f"Error: Cannot send Telegram message. Invalid chat ID: {telegram_chat_id}")
                return False
            delay = base_delay * (2 ** attempt)
            logging.error(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
            print(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    logging.error(f"Failed to send Telegram message after {retries} attempts.")
    print(f"Failed to send Telegram message after {retries} attempts.")
    return False

def calculate_mtf_trend(candles, timeframe, min_threshold, max_threshold, buy_vol, sell_vol, lookback=50):
    if len(candles) < lookback:
        logging.warning(f"{timeframe} - MTF trend analysis skipped: Insufficient data")
        print(f"{timeframe} - MTF trend analysis skipped: Insufficient data")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, max_threshold
    
    recent_candles = candles[-lookback:]
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    
    if len(closes) == 0 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid close prices in {timeframe}.")
        print(f"Invalid close prices in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, max_threshold

    current_close = Decimal(str(closes[-1]))
    midpoint = (min_threshold + max_threshold) / Decimal('2')
    trend_bullish = current_close < midpoint
    cycle_status = "Up" if trend_bullish else "Down"
    trend = "BULLISH" if trend_bullish else "BEARISH"
    
    volume_ratio = buy_vol / sell_vol if sell_vol > Decimal('0') else Decimal('1.0')
    volume_confirmed = volume_ratio >= VOLUME_CONFIRMATION_RATIO if trend == "BULLISH" else Decimal('1.0') / volume_ratio >= VOLUME_CONFIRMATION_RATIO
    
    cycle_target = current_close * Decimal('1.01') if trend_bullish else current_close * Decimal('0.99')
    price_range = max_threshold - min_threshold
    tolerance = price_range * SUPPORT_RESISTANCE_TOLERANCE
    if trend == "BULLISH":
        cycle_target = max(cycle_target, current_close * Decimal('1.01'), min_threshold * Decimal('1.02'))
        cycle_target = min(cycle_target, current_close * Decimal('1.10'), max_threshold * Decimal('0.98'))
        if abs(cycle_target - max_threshold) <= tolerance:
            cycle_status = "Approaching Resistance"
    else:
        cycle_target = min(cycle_target, current_close * Decimal('0.99'), max_threshold * Decimal('0.98'))
        cycle_target = max(cycle_target, current_close * Decimal('0.90'), min_threshold * Decimal('1.02'))
        if abs(cycle_target - min_threshold) <= tolerance:
            cycle_status = "Approaching Support"

    logging.info(f"{timeframe} - MTF Trend: {trend}, Cycle: {cycle_status}, Cycle Target: {cycle_target:.25f}")
    print(f"{timeframe} - MTF Trend: {trend}, Cycle Status: {cycle_status}, Cycle Target: {cycle_target:.25f}")
    return trend, min_threshold, max_threshold, cycle_status, trend_bullish, not trend_bullish, volume_ratio, volume_confirmed, 0.0, cycle_target

def calculate_buy_sell_volume(candles, timeframe, reversal_type, trend_bullish, min_idx, max_idx):
    if not candles or len(candles) < VOLUME_LOOKBACK:
        logging.warning(f"No candles or insufficient candles ({len(candles)}) for volume analysis in {timeframe}.")
        print(f"No candles or insufficient candles ({len(candles)}) for volume analysis in {timeframe}.")
        return Decimal('0'), Decimal('0'), "BEARISH", False, False
    
    # Calculate buy and sell volume for logging
    buy_volume = sum(Decimal(str(c["volume"])) for c in candles if Decimal(str(c["close"])) > Decimal(str(c["open"])))
    sell_volume = sum(Decimal(str(c["volume"])) for c in candles if Decimal(str(c["close"])) < Decimal(str(c["open"])))
    volume_mood = "BULLISH" if buy_volume > sell_volume else "BEARISH"
    
    # Calculate average volume over the full lookback period (1500 candles)
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    full_volumes = np.array([float(c["volume"]) for c in candles[-lookback:]], dtype=np.float64)
    avg_full_volume = np.mean(full_volumes) if len(full_volumes) > 0 else 0.0
    
    # Initialize volume flags
    volume_increasing = False
    volume_exhausted = False
    
    # Volume analysis based on reversal type and trend
    if reversal_type == "DIP" and trend_bullish:
        # LONG: Check volume at dip (argmin) for increasing volume (accumulation)
        start_idx = max(0, min_idx - VOLUME_LOOKBACK // 2)
        end_idx = min(len(candles), start_idx + VOLUME_LOOKBACK)
        reversal_volumes = np.array([float(c["volume"]) for c in candles[start_idx:end_idx]], dtype=np.float64)
        avg_reversal_volume = np.mean(reversal_volumes) if len(reversal_volumes) > 0 else 0.0
        if avg_reversal_volume > 0 and avg_full_volume > 0:
            volume_increasing = (avg_reversal_volume / avg_full_volume) >= float(VOLUME_CONFIRMATION_RATIO)
            volume_exhausted = not volume_increasing  # Mutually exclusive
    elif reversal_type == "TOP" and not trend_bullish:
        # SHORT: Check volume at top (argmax) for decreasing volume (distribution)
        start_idx = max(0, max_idx - VOLUME_LOOKBACK // 2)
        end_idx = min(len(candles), start_idx + VOLUME_LOOKBACK)
        reversal_volumes = np.array([float(c["volume"]) for c in candles[start_idx:end_idx]], dtype=np.float64)
        avg_reversal_volume = np.mean(reversal_volumes) if len(reversal_volumes) > 0 else 0.0
        if avg_reversal_volume > 0 and avg_full_volume > 0:
            volume_exhausted = (avg_reversal_volume / avg_full_volume) <= float(VOLUME_EXHAUSTION_RATIO)
            volume_increasing = not volume_exhausted  # Mutually exclusive
    else:
        # Default case: set both to False if conditions don't match
        volume_increasing = False
        volume_exhausted = False
    
    logging.info(
        f"{timeframe} - Buy Volume: {buy_volume:.25f}, Sell Volume: {sell_volume:.25f}, "
        f"Volume Mood: {volume_mood}, Avg Full Volume: {avg_full_volume:.2f}, "
        f"Volume Increasing: {volume_increasing}, Volume Exhausted: {volume_exhausted}"
    )
    print(
        f"{timeframe} - Buy Volume: {buy_volume:.25f}, Sell Volume: {sell_volume:.25f}, "
        f"Volume Mood: {volume_mood}, Avg Full Volume: {avg_full_volume:.2f}, "
        f"Volume Increasing: {volume_increasing}, Volume Exhausted: {volume_exhausted}"
    )
    return buy_volume, sell_volume, volume_mood, volume_increasing, volume_exhausted

def calculate_support_resistance(candles, timeframe):
    if len(candles) < 10:
        logging.warning(f"Insufficient candles ({len(candles)}) for support/resistance in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for support/resistance in {timeframe}.")
        return [], []
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    
    min_candle = min(recent_candles, key=lambda x: x['low'])
    max_candle = max(recent_candles, key=lambda x: x['high'])
    support_levels = [{"price": Decimal(str(min_candle['low'])), "touches": 1}]
    resistance_levels = [{"price": Decimal(str(max_candle['high'])), "touches": 1}]
    
    logging.info(f"{timeframe} - Support: {support_levels[0]['price']:.25f}, Resistance: {resistance_levels[0]['price']:.25f}")
    print(f"{timeframe} - Support: {support_levels[0]['price']:.25f}, Resistance: {resistance_levels[0]['price']:.25f}")
    return support_levels, resistance_levels

def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, higher_tf_tops=None):
    if len(candles) < 3:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        return "TOP", 0, Decimal('0'), "TOP", 0, Decimal('0'), 0, 0
    
    lookback = min(len(candles), RECENT_LOOKBACK[timeframe])
    recent_candles = candles[-lookback:]
    
    price_range = max_threshold - min_threshold
    tolerance = price_range * TOLERANCE_FACTORS[timeframe]
    
    lows = np.array([float(c['low']) for c in recent_candles])
    min_idx = np.argmin(lows)
    min_candle = recent_candles[min_idx]
    min_time = min_candle['time']
    closest_min_price = Decimal(str(min_candle['low']))
    
    highs = np.array([float(c['high']) for c in recent_candles])
    max_idx = np.argmax(highs)
    max_candle = recent_candles[max_idx]
    max_time = max_candle['time']
    closest_max_price = Decimal(str(max_candle['high']))
    
    current_close = Decimal(str(recent_candles[-1]['close']))
    
    # Determine reversal based on price proximity to current close
    diff_to_high = abs(current_close - closest_max_price)
    diff_to_low = abs(current_close - closest_min_price)
    
    most_recent_reversal = "TOP" if diff_to_high < diff_to_low else "DIP"
    most_recent_time = max_time if most_recent_reversal == "TOP" else min_time
    most_recent_price = closest_max_price if most_recent_reversal == "TOP" else closest_min_price
    
    # Override for 1m if higher timeframe signals a top
    if timeframe == "1m" and higher_tf_tops and any(higher_tf_tops.values()):
        most_recent_reversal = "TOP"
        most_recent_time = max_time
        most_recent_price = closest_max_price
    
    logging.info(
        f"{timeframe} - Reversal: {most_recent_reversal} at {most_recent_price:.25f}, "
        f"Diff to High: {diff_to_high:.25f}, Diff to Low: {diff_to_low:.25f}"
    )
    print(
        f"{timeframe} - Reversal: {most_recent_reversal} at {most_recent_price:.25f}, "
        f"Diff to High: {diff_to_high:.25f}, Diff to Low: {diff_to_low:.25f}"
    )
    return most_recent_reversal, min_time, closest_min_price, "TOP", max_time, closest_max_price, min_idx, max_idx

def calculate_thresholds(candles, timeframe_ranges=None):
    if not candles:
        logging.warning("No candles provided for threshold calculation.")
        print("No candles provided for threshold calculation.")
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
    
    timeframe = candles[0]['timeframe']
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    min_candle = min(recent_candles, key=lambda x: x['low'])
    max_candle = max(recent_candles, key=lambda x: x['high'])
    
    min_threshold = Decimal(str(min_candle['low']))
    max_threshold = Decimal(str(max_candle['high']))
    price_range = max_threshold - min_threshold
    
    if timeframe_ranges and timeframe == "1m":
        higher_tf = "3m"
        if higher_tf in timeframe_ranges and price_range > timeframe_ranges[higher_tf]:
            max_threshold = min_threshold + timeframe_ranges[higher_tf]
            price_range = max_threshold - min_threshold
    
    middle_threshold = (min_threshold + max_threshold) / Decimal('2')
    
    logging.info(f"{timeframe} - Min: {min_threshold:.25f}, Mid: {middle_threshold:.25f}, Max: {max_threshold:.25f}")
    print(f"{timeframe} - Min: {min_threshold:.25f}, Mid: {middle_threshold:.25f}, Max: {max_threshold:.25f}")
    return min_threshold, middle_threshold, max_threshold, price_range

def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1500):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=1500, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = [{
                "time": k[0] / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "timeframe": timeframe
            } for k in klines]
            if candles and (time.time() - candles[-1]["time"]) > TIMEFRAME_INTERVALS[timeframe] * 2:
                logging.warning(f"Stale data for {timeframe}: latest candle is too old.")
                print(f"Stale data for {timeframe}: latest candle is too old.")
                raise Exception("Stale candle data")
            logging.info(f"Fetched {len(candles)} candles for {timeframe}")
            print(f"Fetched {len(candles)} candles for {timeframe}")
            return candles
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    return []

def get_current_price(retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                logging.info(f"Current {TRADE_SYMBOL} price: {price:.25f}")
                print(f"Current {TRADE_SYMBOL} price: {price:.25f}")
                return price
            logging.warning(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Binance API Error fetching price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    return Decimal('0.0')

def get_balance(asset='USDC', retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            account = client.futures_account()
            for asset_info in account.get('assets', []):
                if asset_info.get('asset') == asset:
                    wallet = Decimal(str(asset_info.get('walletBalance', '0.0')))
                    logging.info(f"{asset} wallet balance: {wallet:.25f}")
                    print(f"{asset} wallet balance: {wallet:.25f}")
                    return wallet
            logging.warning(f"{asset} not found in futures account balances.")
            return Decimal('0.0')
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Binance API exception while fetching {asset} balance (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching balance (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch {asset} balance after {retries} attempts.")
    print(f"Failed to fetch {asset} balance after {retries} attempts.")
    return Decimal('0.0')

def get_position(retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            positions = client.futures_position_information(symbol=TRADE_SYMBOL)
            position = positions[0] if positions else {}
            quantity = Decimal(str(position.get('positionAmt', '0.0')))
            return {
                "quantity": quantity,
                "entry_price": Decimal(str(position.get('entryPrice', '0.0'))),
                "side": "LONG" if quantity > Decimal('0') else "SHORT" if quantity < Decimal('0') else "NONE",
                "unrealized_pnl": Decimal(str(position.get('unrealizedProfit', '0.0'))),
                "initial_balance": Decimal('0.0'),
                "sl_price": Decimal('0.0'),
                "tp_price": Decimal('0.0')
            }
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Error fetching position info (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching position (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch position info after {retries} attempts.")
    print(f"Failed to fetch position info after {retries} attempts.")
    return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0'), "initial_balance": Decimal('0.0'), "sl_price": Decimal('0.0'), "tp_price": Decimal('0.0')}

def check_open_orders(retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            orders = client.futures_get_open_orders(symbol=TRADE_SYMBOL)
            return len(orders)
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Error checking open orders (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error checking open orders (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to check open orders after {retries} attempts.")
    print(f"Failed to check open orders after {retries} attempts.")
    return 0

def calculate_quantity(balance, price):
    if price <= Decimal('0') or balance < MINIMUM_BALANCE:
        logging.warning(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}).")
        print(f"Warning: Insufficient balance {balance:.25f} USDC or invalid price {price:.25f}")
        return Decimal('0.0')
    quantity = (balance * Decimal(str(LEVERAGE))) / price
    quantity = quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')
    logging.info(f"Calculated quantity: {quantity:.25f} BTC for balance {balance:.25f} USDC at price {price:.25f}")
    print(f"Calculated quantity: {quantity:.25f} BTC for balance {balance:.25f} USDC at price {price:.25f}")
    return quantity

def place_order(signal, quantity, price, initial_balance, analysis_details, retries=5, base_delay=5):
    if check_open_orders() > 0:
        logging.warning(f"Cannot place {signal} order: existing open orders detected.")
        print(f"Cannot place {signal} order: existing open orders detected.")
        return None
    
    for attempt in range(retries):
        try:
            if quantity <= Decimal('0'):
                logging.warning(f"Invalid order quantity {quantity:.25f}. Skipping order.")
                print(f"Invalid order quantity {quantity:.25f}. Skipping order placement.")
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
                tp_price = (price * (Decimal('1') + TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
                sl_price = (price * (Decimal('1') - STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
                position.update({"sl_price": sl_price, "tp_price": tp_price, "side": "LONG", "quantity": quantity, "entry_price": price})
                message = (
                    f"*Trade Signal: LONG*\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Quantity: {quantity:.2f} BTC\n"
                    f"Entry Price: ~{price:.2f} USDC\n"
                    f"Initial Balance: {initial_balance:.2f} USDC\n"
                    f"Stop-Loss: {sl_price:.2f} (-5%)\n"
                    f"Take-Profit: {tp_price:.2f} (+5%)\n"
                    f"\n*Analysis Details*\n"
                )
                for tf in TIMEFRAMES:
                    details = analysis_details.get(tf, {})
                    message += (
                        f"\n*{tf} Timeframe*\n"
                        f"MTF Trend: {details.get('trend', 'N/A')}\n"
                        f"Cycle Status: {details.get('cycle_status', 'N/A')}\n"
                        f"Dominant Frequency: {details.get('dominant_freq', 0.0):.2f}\n"
                        f"Cycle Target: {details.get('cycle_target', Decimal('0')):.2f} USDC\n"
                        f"Volume Ratio: {details.get('volume_ratio', Decimal('0')):.2f}\n"
                        f"Volume Confirmed: {details.get('volume_confirmed', False)}\n"
                        f"Minimum Threshold: {details.get('min_threshold', Decimal('0')):.2f} USDC\n"
                        f"Middle Threshold: {details.get('middle_threshold', Decimal('0')):.2f} USDC\n"
                        f"Maximum Threshold: {details.get('max_threshold', Decimal('0')):.2f} USDC\n"
                        f"High-Low Range: {details.get('price_range', Decimal('0')):.2f} USDC\n"
                        f"Most Recent Reversal: {details.get('reversal_type', 'N/A')} at price {details.get('reversal_price', Decimal('0')):.2f}\n"
                        f"Support Level: {details.get('support_levels', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Resistance Level: {details.get('resistance_levels', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Buy Volume: {details.get('buy_volume', Decimal('0')):.2f}\n"
                        f"Sell Volume: {details.get('sell_volume', Decimal('0')):.2f}\n"
                        f"Volume Mood: {details.get('volume_mood', 'N/A')}\n"
                        f"Volume Increasing: {details.get('volume_increasing', False)}\n"
                        f"Volume Exhausted: {details.get('volume_exhausted', False)}\n"
                    )
                telegram_loop.run_until_complete(send_telegram_message(message))
                logging.info(f"Placed LONG order: {quantity:.25f} BTC at market price ~{price:.25f}")
                print(f"\n=== TRADE ENTERED ===\nSide: LONG\nQuantity: {quantity:.25f} BTC\nEntry Price: ~{price:.25f} USDC\nInitial USDC Balance: {initial_balance:.25f}\nStop-Loss Price: {sl_price:.25f} (-5% ROI)\nTake-Profit Price: {tp_price:.25f} (+5% ROI)\n===================\n")
            elif signal == "SHORT":
                order = client.futures_create_order(
                    symbol=TRADE_SYMBOL,
                    side="SELL",
                    type="MARKET",
                    quantity=str(quantity)
                )
                tp_price = (price * (Decimal('1') - TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
                sl_price = (price * (Decimal('1') + STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
                position.update({"sl_price": sl_price, "tp_price": tp_price, "side": "SHORT", "quantity": -quantity, "entry_price": price})
                message = (
                    f"*Trade Signal: SHORT*\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Quantity: {quantity:.2f} BTC\n"
                    f"Entry Price: ~{price:.2f} USDC\n"
                    f"Initial Balance: {initial_balance:.2f} USDC\n"
                    f"Stop-Loss: {sl_price:.2f} (-5%)\n"
                    f"Take-Profit: {tp_price:.2f} (+5%)\n"
                    f"\n*Analysis Details*\n"
                )
                for tf in TIMEFRAMES:
                    details = analysis_details.get(tf, {})
                    message += (
                        f"\n*{tf} Timeframe*\n"
                        f"MTF Trend: {details.get('trend', 'N/A')}\n"
                        f"Cycle Status: {details.get('cycle_status', 'N/A')}\n"
                        f"Dominant Frequency: {details.get('dominant_freq', 0.0):.2f}\n"
                        f"Cycle Target: {details.get('cycle_target', Decimal('0')):.2f} USDC\n"
                        f"Volume Ratio: {details.get('volume_ratio', Decimal('0')):.2f}\n"
                        f"Volume Confirmed: {details.get('volume_confirmed', False)}\n"
                        f"Minimum Threshold: {details.get('min_threshold', Decimal('0')):.2f} USDC\n"
                        f"Middle Threshold: {details.get('middle_threshold', Decimal('0')):.2f} USDC\n"
                        f"Maximum Threshold: {details.get('max_threshold', Decimal('0')):.2f} USDC\n"
                        f"High-Low Range: {details.get('price_range', Decimal('0')):.2f} USDC\n"
                        f"Most Recent Reversal: {details.get('reversal_type', 'N/A')} at price {details.get('reversal_price', Decimal('0')):.2f}\n"
                        f"Support Level: {details.get('support_levels', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Resistance Level: {details.get('resistance_levels', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Buy Volume: {details.get('buy_volume', Decimal('0')):.2f}\n"
                        f"Sell Volume: {details.get('sell_volume', Decimal('0')):.2f}\n"
                        f"Volume Mood: {details.get('volume_mood', 'N/A')}\n"
                        f"Volume Increasing: {details.get('volume_increasing', False)}\n"
                        f"Volume Exhausted: {details.get('volume_exhausted', False)}\n"
                    )
                telegram_loop.run_until_complete(send_telegram_message(message))
                logging.info(f"Placed SHORT order: {quantity:.25f} BTC at market price ~{price:.25f}")
                print(f"\n=== TRADE ENTERED ===\nSide: SHORT\nQuantity: {quantity:.25f} BTC\nEntry Price: ~{price:.25f} USDC\nInitial USDC Balance: {initial_balance:.25f}\nStop-Loss Price: {sl_price:.25f} (-5% ROI)\nTake-Profit Price: {tp_price:.25f} (+5% ROI)\n===================\n")
            return position
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Error placing order (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error placing order (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error placing order (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error placing order (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to place order after {retries} attempts.")
    print(f"Failed to place order after {retries} attempts.")
    return None

def close_position(position, price, retries=5, base_delay=5):
    if position["side"] == "NONE" or position["quantity"] == Decimal('0'):
        logging.info("No position to close.")
        print("No position to close.")
        return
    for attempt in range(retries):
        try:
            quantity = abs(position["quantity"]).quantize(QUANTITY_PRECISION)
            side = "SELL" if position["side"] == "LONG" else "BUY"
            order = client.futures_create_order(
                symbol=TRADE_SYMBOL,
                side=side,
                type="MARKET",
                quantity=str(quantity)
            )
            message = (
                f"*Position Closed*\n"
                f"Symbol: {TRADE_SYMBOL}\n"
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Side: {position['side']}\n"
                f"Quantity: {quantity:.25f} BTC\n"
                f"Exit Price: ~{price:.25f} USDC\n"
                f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC\n"
            )
            telegram_loop.run_until_complete(send_telegram_message(message))
            logging.info(f"Closed {position['side']} position: {quantity:.25f} BTC at market price ~{price:.25f}")
            print(f"Closed {position['side']} position: {quantity:.25f} BTC at market price ~{price:.25f}")
            return
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Error closing position (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error closing position (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error closing position (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error closing position (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to close position after {retries} attempts.")
    print(f"Failed to close position after {retries} attempts.")

def main():
    try:
        logging.info("Futures Analysis Bot Initialized!")
        print("Futures Analysis Bot Initialized!")
        
        while True:
            current_local_time = datetime.datetime.now()
            current_local_time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Current Time: {current_local_time_str}")
            print(f"\n=== Analysis for All Timeframes ===\nLocal Time: {current_local_time_str}")
            
            current_price = get_current_price()
            if current_price <= Decimal('0'):
                logging.warning(f"Failed to fetch valid {TRADE_SYMBOL} price. Retrying in 60 seconds.")
                print(f"Warning: Failed to fetch valid {TRADE_SYMBOL} price. Retrying in 60 seconds.")
                time.sleep(60)
                continue
            
            candle_map = fetch_candles_in_parallel(TIMEFRAMES)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candle data available. Retrying in 60 seconds.")
                print("No candle data available. Retrying in 60 seconds.")
                time.sleep(60)
                continue
            
            usdc_balance = get_balance('USDC')
            position = get_position()
            
            if position["side"] != "NONE" and position["sl_price"] > Decimal('0'):
                if position["side"] == "LONG" and current_price <= position["sl_price"]:
                    message = (
                        f"*Stop-Loss Triggered: LONG*\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.2f} USDC\n"
                        f"Stop-Loss Price: {position['sl_price']:.2f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                elif position["side"] == "LONG" and current_price >= position["tp_price"]:
                    message = (
                        f"*Take-Profit Triggered: LONG*\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.2f} USDC\n"
                        f"Take-Profit Price: {position['tp_price']:.2f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                elif position["side"] == "SHORT" and current_price >= position["sl_price"]:
                    message = (
                        f"*Stop-Loss Triggered: SHORT*\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.2f} USDC\n"
                        f"Stop-Loss Price: {position['sl_price']:.2f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                elif position["side"] == "SHORT" and current_price <= position["tp_price"]:
                    message = (
                        f"*Take-Profit Triggered: SHORT*\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.2f} USDC\n"
                        f"Take-Profit Price: {position['tp_price']:.2f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                position = get_position()
            
            # Define conditions for LONG and SHORT signals
            conditions_long = {
                "momentum_positive_1m": False  # Only 1m momentum for LONG
            }
            conditions_long.update({
                f"dip_confirmed_{tf}": False for tf in TIMEFRAMES
            })
            conditions_long.update({
                f"below_middle_{tf}": False for tf in TIMEFRAMES
            })
            conditions_long.update({
                f"volume_increasing_{tf}": False for tf in TIMEFRAMES
            })
            
            conditions_short = {
                "momentum_negative_1m": False  # Only 1m momentum for SHORT
            }
            conditions_short.update({
                f"top_confirmed_{tf}": False for tf in TIMEFRAMES
            })
            conditions_short.update({
                f"above_middle_{tf}": False for tf in TIMEFRAMES
            })
            conditions_short.update({
                f"volume_exhausted_{tf}": False for tf in TIMEFRAMES
            })
            
            timeframe_ranges = {tf: calculate_thresholds(candle_map.get(tf, []))[3] if candle_map.get(tf) else Decimal('0') for tf in TIMEFRAMES}
            
            higher_tf_tops = {}
            for tf in ["3m"]:
                if candle_map.get(tf):
                    min_th, _, max_th, _ = calculate_thresholds(candle_map[tf], timeframe_ranges)
                    reversal_type, _, _, _, _, _, _, _ = detect_recent_reversal(candle_map[tf], tf, min_th, max_th)
                    higher_tf_tops[tf] = (reversal_type == "TOP")
            
            analysis_details = {}
            
            for timeframe in TIMEFRAMES:
                print(f"\n--- {timeframe} Timeframe Analysis ---")
                if not candle_map.get(timeframe):
                    logging.warning(f"No data for {timeframe}. Skipping analysis.")
                    print(f"{timeframe} - No data available")
                    continue
                
                candles_tf = candle_map[timeframe]
                closes = np.array([c["close"] for c in candles_tf if c["close"] > 0], dtype=np.float64)
                
                min_threshold, middle_threshold, max_threshold, price_range = calculate_thresholds(candles_tf, timeframe_ranges)
                
                reversal_type, min_time, closest_min_price, _, max_time, closest_max_price, min_idx, max_idx = detect_recent_reversal(candles_tf, timeframe, min_threshold, max_threshold, higher_tf_tops if timeframe == "1m" else None)
                
                support_levels, resistance_levels = calculate_support_resistance(candles_tf, timeframe)
                
                buy_vol, sell_vol, volume_mood, volume_increasing, volume_exhausted = calculate_buy_sell_volume(
                    candles_tf, timeframe, reversal_type, (closes[-1] < (min_threshold + max_threshold) / 2), min_idx, max_idx
                )
                
                conditions_long[f"dip_confirmed_{timeframe}"] = reversal_type == "DIP"
                conditions_short[f"top_confirmed_{timeframe}"] = reversal_type == "TOP"
                conditions_long[f"volume_increasing_{timeframe}"] = volume_increasing
                conditions_short[f"volume_exhausted_{timeframe}"] = volume_exhausted
                
                current_close = Decimal(str(closes[-1])) if len(closes) > 0 else current_price
                conditions_long[f"below_middle_{timeframe}"] = current_close < middle_threshold
                conditions_short[f"above_middle_{timeframe}"] = current_close > middle_threshold
                
                trend, min_th, max_th, cycle_status, trend_bullish, trend_bearish, volume_ratio, volume_confirmed, dominant_freq, cycle_target = calculate_mtf_trend(
                    candles_tf, timeframe, min_threshold, max_threshold, buy_vol, sell_vol
                )
                
                if len(closes) >= 14 and timeframe == "1m":  # Only calculate momentum for 1m
                    momentum = talib.MOM(closes, timeperiod=14)
                    if len(momentum) > 0 and not np.isnan(momentum[-1]):
                        conditions_long[f"momentum_positive_1m"] = Decimal(str(momentum[-1])) >= Decimal('0')
                        conditions_short[f"momentum_negative_1m"] = not conditions_long[f"momentum_positive_1m"]
                
                analysis_details[timeframe] = {
                    "trend": trend,
                    "cycle_status": cycle_status,
                    "dominant_freq": dominant_freq,
                    "cycle_target": cycle_target,
                    "volume_ratio": volume_ratio,
                    "volume_confirmed": volume_confirmed,
                    "min_threshold": min_threshold,
                    "middle_threshold": middle_threshold,
                    "max_threshold": max_threshold,
                    "reversal_type": reversal_type,
                    "reversal_time": datetime.datetime.fromtimestamp(min_time if reversal_type == "DIP" else max_time) if (reversal_type == "DIP" and min_time) or (reversal_type == "TOP" and max_time) else None,
                    "reversal_price": closest_min_price if reversal_type == "DIP" else closest_max_price,
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "volume_mood": volume_mood,
                    "price_range": price_range,
                    "volume_increasing": volume_increasing,
                    "volume_exhausted": volume_exhausted
                }
            
            # Print condition pairs status
            print("\nCondition Pairs Status:")
            for tf in TIMEFRAMES:
                for cond_pair in [
                    ("dip_confirmed", "top_confirmed"),
                    ("below_middle", "above_middle"),
                    ("volume_increasing", "volume_exhausted")
                ]:
                    cond1, cond2 = cond_pair
                    status1 = conditions_long.get(f"{cond1}_{tf}", False)
                    status2 = conditions_short.get(f"{cond2}_{tf}", False)
                    print(f"{cond1}_{tf}: {status1}, {cond2}_{tf}: {status2}")
                if tf == "1m":
                    print(f"momentum_positive_1m: {conditions_long.get('momentum_positive_1m', False)}, momentum_negative_1m: {conditions_short.get('momentum_negative_1m', False)}")
            
            # Print long and short conditions status
            print("\nLong Conditions Status:")
            print(f"momentum_positive_1m: {conditions_long.get('momentum_positive_1m', False)}")
            for cond in ["dip_confirmed", "below_middle", "volume_increasing"]:
                for tf in TIMEFRAMES:
                    print(f"{cond}_{tf}: {conditions_long.get(f'{cond}_{tf}', False)}")
            
            print("\nShort Conditions Status:")
            print(f"momentum_negative_1m: {conditions_short.get('momentum_negative_1m', False)}")
            for cond in ["top_confirmed", "above_middle", "volume_exhausted"]:
                for tf in TIMEFRAMES:
                    print(f"{cond}_{tf}: {conditions_short.get(f'{cond}_{tf}', False)}")
            
            long_true_count = sum(1 for v in conditions_long.values() if v)
            short_true_count = sum(1 for v in conditions_short.values() if v)
            total_conditions_long = len(conditions_long)
            total_conditions_short = len(conditions_short)
            
            # Print summary
            print(f"\nSummary: LONG Conditions - True: {long_true_count}, False: {total_conditions_long - long_true_count}")
            print(f"Summary: SHORT Conditions - True: {short_true_count}, False: {total_conditions_short - short_true_count}")
            logging.info(f"Summary: LONG Conditions - True: {long_true_count}, False: {total_conditions_long - long_true_count}")
            logging.info(f"Summary: SHORT Conditions - True: {short_true_count}, False: {total_conditions_short - short_true_count}")
            
            long_signal = long_true_count == total_conditions_long
            short_signal = short_true_count == total_conditions_short
            
            print(f"\nTrade Signal Status:")
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'} (All conditions: {long_signal})")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'} (All conditions: {short_signal})")
            logging.info(f"LONG Signal: {'Active' if long_signal else 'Inactive'} (All conditions: {long_signal})")
            logging.info(f"SHORT Signal: {'Active' if short_signal else 'Inactive'} (All conditions: {short_signal})")
            
            if long_signal and short_signal:
                logging.warning("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                print("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                long_signal = False
                short_signal = False
            
            signal = "LONG" if long_signal else "SHORT" if short_signal else "NO_SIGNAL"
            print(f"\nTrade Signal Status: {signal}")
            logging.info(f"Trade Signal Status: {signal}")
            
            if signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                if quantity > Decimal('0'):
                    new_position = place_order(signal, quantity, current_price, usdc_balance, analysis_details)
                    if new_position:
                        position = new_position
                else:
                    logging.warning(f"Cannot place {signal} order: Insufficient quantity ({quantity:.25f} BTC).")
                    print(f"Cannot place {signal} order: Insufficient quantity ({quantity:.25f} BTC).")
            
            time.sleep(5)  # Wait 5 seconds before next analysis cycle
    
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
        print("Bot stopped by user.")
        telegram_loop.run_until_complete(send_telegram_message("Bot stopped by user."))
    except Exception as e:
        logging.error(f"Fatal error in main loop: {e}")
        print(f"Fatal error in main loop: {e}")
        telegram_loop.run_until_complete(send_telegram_message(f"*Fatal Error*\nSymbol: {TRADE_SYMBOL}\nError: {str(e)}"))
        exit(1)

if __name__ == "__main__":
    main()