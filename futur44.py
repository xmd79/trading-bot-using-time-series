import asyncio
import datetime
import time
import concurrent.futures
import talib
import gc
import numpy as np
from decimal import Decimal, getcontext
import requests
import logging
from scipy import fft
from telegram.ext import Application
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import uuid
import math

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set Decimal precision
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"
LEVERAGE = 75
STOP_LOSS_PERCENTAGE = Decimal('0.50')  # 50% stop-loss
TAKE_PROFIT_PERCENTAGE = Decimal('0.05')  # 5% take-profit
QUANTITY_PRECISION = Decimal('0.000001')  # Binance quantity precision for BTCUSDC
MINIMUM_BALANCE = Decimal('1.0000')  # Minimum USDC balance to place trades
TIMEFRAMES = ["1m", "3m", "5m"]
LOOKBACK_PERIODS = {"1m": 500, "3m": 500, "5m": 500}
PRICE_TOLERANCE = Decimal('0.01')  # 1% tolerance for reversal detection
VOLUME_CONFIRMATION_RATIO = Decimal('1.5')  # Buy/Sell volume ratio for reversal confirmation
SUPPORT_RESISTANCE_TOLERANCE = Decimal('0.005')  # 0.5% tolerance for support/resistance levels
API_TIMEOUT = 60  # Timeout for Binance API requests

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

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": API_TIMEOUT})
client.API_URL = 'https://fapi.binance.com'  # Futures API endpoint

# Initialize Telegram bot
try:
    telegram_app = Application.builder().token(telegram_token).build()
    asyncio.set_event_loop(telegram_loop)
    telegram_loop.run_until_complete(telegram_app.bot.get_me())  # Test Telegram bot connection
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

# Send Telegram message with retry logic and exponential backoff
async def send_telegram_message(message, retries=3, base_delay=5):
    for attempt in range(retries):
        try:
            await telegram_app.bot.send_message(chat_id=telegram_chat_id, text=message, parse_mode='Markdown')
            logging.info(f"Telegram message sent: {message[:100]}...")
            print(f"Telegram message sent: {message[:100]}...")
            return True
        except Exception as e:
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            logging.error(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
            print(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    logging.error(f"Failed to send Telegram message after {retries} attempts.")
    print(f"Failed to send Telegram message after {retries} attempts.")
    return False

# Enhanced MTF Trend Analysis with Binary Classification
def calculate_mtf_trend(candles, timeframe, min_threshold, max_threshold, buy_vol, sell_vol, lookback=50):
    if len(candles) < lookback:
        logging.warning(f"Insufficient candles ({len(candles)}) for MTF trend analysis in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for MTF trend analysis in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "PEAK", False, True, Decimal('1.0'), False, 0.0, min_threshold
    
    recent_candles = candles[-lookback:]
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    
    # Validate input data
    if len(closes) == 0 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid or non-positive close prices in {timeframe}.")
        print(f"Invalid or non-positive close prices in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "PEAK", False, True, Decimal('1.0'), False, 0.0, min_threshold
    
    # FFT for sinusoidal model with Hamming window
    window = np.hamming(len(closes))
    fft_result = fft.rfft(closes * window)
    frequencies = fft.rfftfreq(len(closes), d=1.0)
    amplitudes = np.abs(fft_result)
    phases = np.angle(fft_result)
    
    # Identify dominant frequency and phase
    idx = np.argsort(amplitudes)[::-1][1:3]  # Skip DC component, take top 2
    dominant_freq = frequencies[idx[0]] if idx.size > 0 else 0.0
    dominant_phase = phases[idx[0]] if idx.size > 0 else 0.0
    
    # Force binary cycle classification (UP or DOWN)
    current_close = Decimal(str(closes[-1]))
    midpoint = (min_threshold + max_threshold) / Decimal('2')
    trend_bullish = current_close < midpoint and dominant_phase < 0
    trend_bearish = not trend_bullish
    cycle_status = "Up" if trend_bullish else "Down"
    trend = "BULLISH" if trend_bullish else "BEARISH"
    
    # Volume confirmation
    volume_ratio = buy_vol / sell_vol if sell_vol > Decimal('0') else Decimal('1.0')
    volume_confirmed = (volume_ratio >= VOLUME_CONFIRMATION_RATIO if trend == "BULLISH" else 
                       Decimal('1.0') / volume_ratio >= VOLUME_CONFIRMATION_RATIO if volume_ratio > Decimal('0') else False)
    
    # Calculate cycle target price
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[idx[:2]] = fft_result[idx[:2]]
    filtered_signal = fft.irfft(filtered_fft, n=len(closes))
    cycle_target = Decimal(str(filtered_signal[-1])) if len(filtered_signal) > 0 else current_close
    if trend == "BULLISH":
        cycle_target = max(cycle_target, current_close * Decimal('1.005'), max_threshold * Decimal('0.99'))
    else:
        cycle_target = min(cycle_target, current_close * Decimal('0.995'), min_threshold * Decimal('1.01'))
    
    logging.info(f"{timeframe} - MTF Trend: {trend}, Cycle: {cycle_status}, Dominant Freq: {dominant_freq:.6f}, "
                 f"Current Close: {current_close:.2f}, Cycle Target: {cycle_target:.2f}, "
                 f"Volume Ratio: {volume_ratio:.2f}, Volume Confirmed: {volume_confirmed}")
    print(f"{timeframe} - MTF Trend: {trend}")
    print(f"{timeframe} - Cycle Status: {cycle_status}")
    print(f"{timeframe} - Dominant Frequency: {dominant_freq:.6f}")
    print(f"{timeframe} - Cycle Target: {cycle_target:.2f}")
    print(f"{timeframe} - Volume Ratio: {volume_ratio:.2f}")
    print(f"{timeframe} - Volume Confirmed: {volume_confirmed}")
    
    return trend, min_threshold, max_threshold, cycle_status, trend_bullish, trend_bearish, volume_ratio, volume_confirmed, dominant_freq, cycle_target

# FFT-Based Target Price Function
def get_target(closes, n_components, timeframe, min_threshold, max_threshold, buy_vol, sell_vol):
    if len(closes) < 2 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid closes data for FFT analysis in {timeframe}.")
        print(f"Invalid closes data for FFT analysis in {timeframe}.")
        return (datetime.datetime.now(), Decimal('0'), Decimal('0'), Decimal('0'), 
                Decimal('0'), Decimal('0'), "Bearish", False, True, 0.0, "PEAK")
    
    # Apply Hamming window
    window = np.hamming(len(closes))
    fft_result = fft.rfft(closes * window)
    frequencies = fft.rfftfreq(len(closes), d=1.0)
    amplitudes = np.abs(fft_result)
    phases = np.angle(fft_result)
    
    # Select top N components (excluding DC)
    idx = np.argsort(amplitudes)[::-1][1:n_components+1]
    dominant_freq = frequencies[idx[0]] if idx.size > 0 else 0.0
    
    # Forecast future prices
    forecast_steps = 5  # Number of future steps to forecast
    extended_n = len(closes) + forecast_steps
    extended_t = np.arange(extended_n)
    
    # Initialize forecast arrays
    fastest_forecast = np.zeros(extended_n)
    average_forecast = np.zeros(extended_n)
    reversal_forecast = np.zeros(extended_n)
    
    # Reconstruct signal for fastest target (top 3 components)
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[idx[:3]] = fft_result[idx[:3]]
    fastest_signal = fft.irfft(filtered_fft, n=extended_n)
    fastest_forecast = fastest_signal
    
    # Reconstruct signal for average target (top 2 components)
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[idx[:2]] = fft_result[idx[:2]]
    average_signal = fft.irfft(filtered_fft, n=extended_n)
    average_forecast = average_signal
    
    # Reconstruct signal for reversal target (top 1 component)
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[idx[0]] = fft_result[idx[0]]
    reversal_signal = fft.irfft(filtered_fft, n=extended_n)
    reversal_forecast = reversal_signal
    
    # Calculate trend direction based on forecast slope
    forecast_prices = fastest_forecast[-forecast_steps:]
    slope = np.polyfit(np.arange(forecast_steps), forecast_prices, 1)[0]
    is_bullish_target = slope > 0
    is_bearish_target = not is_bullish_target
    phase_status = "DIP" if is_bullish_target else "PEAK"
    market_mood = "Bullish" if is_bullish_target else "Bearish"
    
    # Calculate target prices
    current_close = Decimal(str(closes[-1]))
    volume_ratio = buy_vol / sell_vol if sell_vol > Decimal('0') else Decimal('1.0')
    volume_exhausted = (volume_ratio < Decimal('0.5') if is_bullish_target else 
                        volume_ratio > Decimal('2.0') if sell_vol > Decimal('0') else False)
    
    fastest_target = Decimal(str(fastest_forecast[-1])) if len(fastest_forecast) > 0 else current_close
    fastest_volume_factor = (Decimal('1.0') + (volume_ratio - Decimal('1.0')) * Decimal('0.1') if is_bullish_target 
                           else Decimal('1.0') / (volume_ratio + Decimal('0.1')) if volume_ratio > Decimal('0') else Decimal('1.0'))
    fastest_target *= fastest_volume_factor
    
    average_target = Decimal(str(average_forecast[-1])) if len(average_forecast) > 0 else current_close
    average_volume_factor = (Decimal('1.0') + (volume_ratio - Decimal('1.0')) * Decimal('0.2') if is_bullish_target 
                           else Decimal('1.0') / (volume_ratio + Decimal('0.2')) if volume_ratio > Decimal('0') else Decimal('1.0'))
    average_target *= average_volume_factor
    
    reversal_target = Decimal(str(reversal_forecast[-1])) if len(reversal_forecast) > 0 else current_close
    reversal_volume_factor = (Decimal('1.0') + (volume_ratio - Decimal('1.0')) * Decimal('0.3') if is_bullish_target and not volume_exhausted 
                            else Decimal('1.0') / (volume_ratio + Decimal('0.3')) if volume_ratio > Decimal('0') and not volume_exhausted else Decimal('1.0'))
    reversal_target *= reversal_volume_factor
    
    # Ensure targets are within bounds
    if is_bullish_target:
        fastest_target = max(fastest_target, current_close * Decimal('1.005'), max_threshold * Decimal('0.995'))
        average_target = max(average_target, fastest_target * Decimal('1.01'), max_threshold * Decimal('0.99'))
        reversal_target = max(reversal_target, average_target * Decimal('1.015'), max_threshold * Decimal('0.985'))
    else:
        fastest_target = min(fastest_target, current_close * Decimal('0.995'), min_threshold * Decimal('1.005'))
        average_target = min(average_target, fastest_target * Decimal('0.99'), min_threshold * Decimal('1.01'))
        reversal_target = min(reversal_target, average_target * Decimal('0.985'), min_threshold * Decimal('1.015'))
    
    stop_loss = current_close * (Decimal('1') - STOP_LOSS_PERCENTAGE) if is_bullish_target else current_close * (Decimal('1') + STOP_LOSS_PERCENTAGE)
    
    logging.info(f"FFT Analysis - {timeframe}: Market Mood: {market_mood}, Phase: {phase_status}, "
                 f"Current Close: {current_close:.2f}, Fastest Target: {fastest_target:.2f}, "
                 f"Average Target: {average_target:.2f}, Reversal Target: {reversal_target:.2f}, "
                 f"Stop Loss: {stop_loss:.2f}, Dominant Freq: {dominant_freq:.6f}, "
                 f"Forecast Slope: {slope:.6f}, Volume Ratio: {volume_ratio:.2f}, Volume Exhausted: {volume_exhausted}")
    print(f"{timeframe} - FFT Phase: {phase_status}")
    print(f"{timeframe} - Dominant Frequency: {dominant_freq:.6f}")
    print(f"{timeframe} - Forecast Slope: {slope:.6f}")
    print(f"{timeframe} - Fastest Target: {fastest_target:.2f}")
    print(f"{timeframe} - Average Target: {average_target:.2f}")
    print(f"{timeframe} - Reversal Target: {reversal_target:.2f}")
    print(f"{timeframe} - Stop Loss: {stop_loss:.2f}")
    print(f"{timeframe} - Volume Ratio: {volume_ratio:.2f}")
    print(f"{timeframe} - Volume Exhausted: {volume_exhausted}")
    
    return (datetime.datetime.now(), current_close, stop_loss, fastest_target, 
            average_target, reversal_target, market_mood, 
            is_bullish_target, is_bearish_target, dominant_freq, phase_status)

# Volume Analysis Functions
def calculate_volume(candles):
    if not candles:
        logging.warning("No candles provided for volume calculation.")
        print("No candles provided for volume calculation.")
        return Decimal('0')
    total_volume = sum(Decimal(str(candle["volume"])) for candle in candles)
    logging.info(f"Total Volume: {total_volume:.2f}")
    print(f"Total Volume: {total_volume:.2f}")
    return total_volume

def calculate_buy_sell_volume(candles, timeframe):
    if not candles:
        logging.warning(f"No candles provided for volume analysis in {timeframe}.")
        print(f"No candles provided for volume analysis in {timeframe}.")
        return Decimal('0'), Decimal('0'), "BEARISH"
    
    try:
        buy_volume = sum(Decimal(str(c["volume"])) for c in candles if Decimal(str(c["close"])) > Decimal(str(c["open"])))
        sell_volume = sum(Decimal(str(c["volume"])) for c in candles if Decimal(str(c["close"])) < Decimal(str(c["open"])))
        volume_mood = "BULLISH" if buy_volume >= sell_volume else "BEARISH"
        logging.info(f"{timeframe} - Buy Volume: {buy_volume:.2f}, Sell Volume: {sell_volume:.2f}, Mood: {volume_mood}")
        print(f"{timeframe} - Buy Volume: {buy_volume:.2f}")
        print(f"{timeframe} - Sell Volume: {sell_volume:.2f}")
        print(f"{timeframe} - Volume Mood: {volume_mood}")
        return buy_volume, sell_volume, volume_mood
    except Exception as e:
        logging.error(f"Error in calculate_buy_sell_volume for {timeframe}: {e}")
        print(f"Error in calculate_buy_sell_volume for {timeframe}: {e}")
        return Decimal('0'), Decimal('0'), "BEARISH"

# Support and Resistance Analysis
def calculate_support_resistance(candles, timeframe):
    if not candles or len(candles) < 10:
        logging.warning(f"Insufficient candles ({len(candles)}) for support/resistance analysis in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for support/resistance analysis in {timeframe}.")
        return [], []
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    
    min_candle = min(recent_candles, key=lambda x: x['low'])
    max_candle = max(recent_candles, key=lambda x: x['high'])
    support_levels = [{"price": Decimal(str(min_candle['low'])), "touches": 1}]
    resistance_levels = [{"price": Decimal(str(max_candle['high'])), "touches": 1}]
    
    logging.info(f"{timeframe} - Support Level: Price: {support_levels[0]['price']:.2f}, Touches: {support_levels[0]['touches']}")
    print(f"{timeframe} - Support Level: Price: {support_levels[0]['price']:.2f}, Touches: {support_levels[0]['touches']}")
    logging.info(f"{timeframe} - Resistance Level: Price: {resistance_levels[0]['price']:.2f}, Touches: {resistance_levels[0]['touches']}")
    print(f"{timeframe} - Resistance Level: Price: {resistance_levels[0]['price']:.2f}, Touches: {resistance_levels[0]['touches']}")
    
    return support_levels, resistance_levels

# Reversal Detection Function
def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold):
    if len(candles) < 3:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        return "PEAK", 0, Decimal('0'), "PEAK", 0, Decimal('0')
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    
    current_close = Decimal(str(recent_candles[-1]['close']))
    tolerance = current_close * PRICE_TOLERANCE
    
    min_time = 0
    max_time = 0
    closest_min_diff = Decimal('infinity')
    closest_max_diff = Decimal('infinity')
    closest_min_price = min_threshold
    closest_max_price = max_threshold
    min_volume_confirmed = False
    max_volume_confirmed = False
    
    for i, candle in enumerate(recent_candles):
        low_price = Decimal(str(candle['low']))
        high_price = Decimal(str(candle['high']))
        candle_time = candle['time']
        
        min_diff = abs(low_price - min_threshold)
        if min_diff <= tolerance and min_diff < closest_min_diff:
            closest_min_diff = min_diff
            min_time = candle_time
            closest_min_price = low_price
            min_volume_confirmed = True
        
        max_diff = abs(high_price - max_threshold)
        if max_diff <= tolerance and max_diff < closest_max_diff:
            closest_max_diff = max_diff
            max_time = candle_time
            closest_max_price = high_price
            max_volume_confirmed = True
    
    most_recent_reversal = "PEAK"
    most_recent_time = 0
    most_recent_price = Decimal('0')
    
    if min_time > 0 and max_time > 0:
        if min_time > max_time and min_volume_confirmed:
            most_recent_reversal = "DIP"
            most_recent_time = min_time
            most_recent_price = closest_min_price
        elif max_volume_confirmed:
            most_recent_reversal = "PEAK"
            most_recent_time = max_time
            most_recent_price = closest_max_price
    elif min_time > 0 and min_volume_confirmed:
        most_recent_reversal = "DIP"
        most_recent_time = min_time
        most_recent_price = closest_min_price
    elif max_time > 0 and max_volume_confirmed:
        most_recent_reversal = "PEAK"
        most_recent_time = max_time
        most_recent_price = closest_max_price
    
    logging.info(f"{timeframe} - Reversal: {most_recent_reversal} at price {most_recent_price:.2f}, time {datetime.datetime.fromtimestamp(most_recent_time) if most_recent_time else 'N/A'}")
    print(f"{timeframe} - Reversal: {most_recent_reversal} at price {most_recent_price:.2f}, time {datetime.datetime.fromtimestamp(most_recent_time) if most_recent_time else 'N/A'}")
    
    return most_recent_reversal, most_recent_time, most_recent_price, "PEAK", max_time, closest_max_price

# Utility Functions with Exponential Backoff
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=500):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=500, retries=5, base_delay=5):
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
                logging.info(f"Current {TRADE_SYMBOL} price: {price:.2f}")
                print(f"Current {TRADE_SYMBOL} price: {price:.2f}")
                return price
            logging.warning(f"Invalid price {price:.2f} on attempt {attempt + 1}/{retries}")
            print(f"Invalid price {price:.2f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Binance API Error fetching price (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Binance API Error fetching price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching price (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching price (attempt {attempt + 1}/{retries}): {e}")
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
                    logging.info(f"{asset} wallet balance: {wallet:.2f}")
                    print(f"{asset} wallet balance: {wallet:.2f}")
                    return wallet
            logging.warning(f"{asset} not found in futures account balances.")
            print(f"{asset} not found in futures account balances.")
            return Decimal('0.0')
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Binance API exception while fetching {asset} balance (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Binance API exception while fetching {asset} balance (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching balance (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching balance (attempt {attempt + 1}/{retries}): {e}")
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
            print(f"Error fetching position info (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching position (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching position (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch position info after {retries} attempts.")
    print(f"Failed to fetch position info after {retries} attempts.")
    return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0'), "initial_balance": Decimal('0.0'), "sl_price": Decimal('0.0'), "tp_price": Decimal('0.0')}

def check_open_orders(retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            orders = client.futures_get_open_orders(symbol=TRADE_SYMBOL)
            for order in orders:
                logging.info(f"Open order: {order['type']} at {order['stopPrice']}")
                print(f"Open order: {order['type']} at {order['stopPrice']}")
            return len(orders)
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Error checking open orders (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error checking open orders (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error checking open orders (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error checking open orders (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to check open orders after {retries} attempts.")
    print(f"Failed to check open orders after {retries} attempts.")
    return 0

# Trading Functions
def calculate_quantity(balance, price):
    if price <= Decimal('0') or balance < MINIMUM_BALANCE:
        logging.warning(f"Insufficient balance ({balance:.2f} USDC) or invalid price ({price:.2f}).")
        print(f"Insufficient balance ({balance:.2f} USDC) or invalid price ({price:.2f}).")
        return Decimal('0.0')
    quantity = (balance * Decimal(str(LEVERAGE))) / price
    quantity = quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')
    logging.info(f"Calculated quantity: {quantity:.6f} BTC for balance {balance:.2f} USDC at price {price:.2f}")
    print(f"Calculated quantity: {quantity:.6f} BTC for balance {balance:.2f} USDC at price {price:.2f}")
    return quantity

def place_order(signal, quantity, price, initial_balance, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            if quantity <= Decimal('0'):
                logging.warning(f"Invalid quantity {quantity:.6f}. Skipping order.")
                print(f"Invalid quantity {quantity:.6f}. Skipping order.")
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
                    f"Quantity: {quantity:.6f} BTC\n"
                    f"Entry Price: ~{price:.2f} USDC\n"
                    f"Initial Balance: {initial_balance:.2f} USDC\n"
                    f"Stop-Loss: {sl_price:.2f} (-50%)\n"
                    f"Take-Profit: {tp_price:.2f} (+5%)\n"
                )
                telegram_loop.run_until_complete(send_telegram_message(message))
                logging.info(f"Placed LONG order: {quantity:.6f} BTC at market price ~{price:.2f}")
                print(f"\n=== TRADE ENTERED ===")
                print(f"Side: LONG")
                print(f"Quantity: {quantity:.6f} BTC")
                print(f"Entry Price: ~{price:.2f} USDC")
                print(f"Initial USDC Balance: {initial_balance:.2f}")
                print(f"Stop-Loss Price: {sl_price:.2f} (-50% ROI)")
                print(f"Take-Profit Price: {tp_price:.2f} (+5% ROI)")
                print(f"===================\n")
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
                    f"Quantity: {quantity:.6f} BTC\n"
                    f"Entry Price: ~{price:.2f} USDC\n"
                    f"Initial Balance: {initial_balance:.2f} USDC\n"
                    f"Stop-Loss: {sl_price:.2f} (-50%)\n"
                    f"Take-Profit: {tp_price:.2f} (+5%)\n"
                )
                telegram_loop.run_until_complete(send_telegram_message(message))
                logging.info(f"Placed SHORT order: {quantity:.6f} BTC at market price ~{price:.2f}")
                print(f"\n=== TRADE ENTERED ===")
                print(f"Side: SHORT")
                print(f"Quantity: {quantity:.6f} BTC")
                print(f"Entry Price: ~{price:.2f} USDC")
                print(f"Initial USDC Balance: {initial_balance:.2f}")
                print(f"Stop-Loss Price: {sl_price:.2f} (-50% ROI)")
                print(f"Take-Profit Price: {tp_price:.2f} (+5% ROI)")
                print(f"===================\n")
            
            if check_open_orders() > 0:
                logging.warning(f"Unexpected open orders detected after placing {signal} order.")
                print(f"Warning: Unexpected open orders detected after placing {signal} order.")
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
                f"Quantity: {quantity:.6f} BTC\n"
                f"Exit Price: ~{price:.2f} USDC\n"
                f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
            )
            telegram_loop.run_until_complete(send_telegram_message(message))
            logging.info(f"Closed {position['side']} position: {quantity:.6f} BTC at market price ~{price:.2f}")
            print(f"Closed {position['side']} position: {quantity:.6f} BTC at market price ~{price:.2f}")
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

# Threshold Calculation
def calculate_thresholds(candles):
    if not candles:
        logging.warning("No candles provided for threshold calculation.")
        print("No candles provided for threshold calculation.")
        return Decimal('0'), Decimal('0'), Decimal('0')
    
    lookback = min(len(candles), LOOKBACK_PERIODS[candles[0]['timeframe']])
    recent_candles = candles[-lookback:]
    min_candle = min(recent_candles, key=lambda x: x['low'])
    max_candle = max(recent_candles, key=lambda x: x['high'])
    
    min_threshold = Decimal(str(min_candle['low']))
    max_threshold = Decimal(str(max_candle['high']))
    middle_threshold = (min_threshold + max_threshold) / Decimal('2')
    
    timeframe = candles[0]['timeframe']
    logging.info(f"{timeframe} - Thresholds: Min: {min_threshold:.2f}, Mid: {middle_threshold:.2f}, Max: {max_threshold:.2f}")
    print(f"{timeframe} - Minimum Threshold: {min_threshold:.2f}")
    print(f"{timeframe} - Middle Threshold: {middle_threshold:.2f}")
    print(f"{timeframe} - Maximum Threshold: {max_threshold:.2f}")
    
    return min_threshold, middle_threshold, max_threshold

# Main Analysis Loop
def main():
    try:
        logging.info("Futures Analysis Bot Initialized!")
        print("Futures Analysis Bot Initialized!")
        
        while True:
            current_local_time = datetime.datetime.now()
            current_local_time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Current Time: {current_local_time_str}")
            print(f"\nCurrent Time: {current_local_time_str}")
            
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
            
            # Check stop-loss/take-profit
            if position["side"] != "NONE" and position["sl_price"] > Decimal('0'):
                if position["side"] == "LONG":
                    if current_price <= position["sl_price"]:
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
                        position = get_position()
                    elif current_price >= position["tp_price"]:
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
                        position = get_position()
                elif position["side"] == "SHORT":
                    if current_price >= position["sl_price"]:
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
                        position = get_position()
                    elif current_price <= position["tp_price"]:
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
            
            conditions_long = {f"trend_bullish_{tf}": False for tf in TIMEFRAMES}
            conditions_long.update({f"momentum_positive_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"fft_bullish_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"volume_bullish_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"dip_confirmed_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"below_middle_{tf}": False for tf in TIMEFRAMES})
            
            conditions_short = {f"trend_bearish_{tf}": True for tf in TIMEFRAMES}
            conditions_short.update({f"momentum_negative_{tf}": True for tf in TIMEFRAMES})
            conditions_short.update({f"fft_bearish_{tf}": True for tf in TIMEFRAMES})
            conditions_short.update({f"volume_bearish_{tf}": True for tf in TIMEFRAMES})
            conditions_short.update({f"peak_confirmed_{tf}": True for tf in TIMEFRAMES})
            conditions_short.update({f"above_middle_{tf}": True for tf in TIMEFRAMES})
            
            volume_data = {tf: {"buy_volume": Decimal('0'), "sell_volume": Decimal('0'), "volume_mood": "BEARISH"} for tf in TIMEFRAMES}
            for tf in TIMEFRAMES:
                if candle_map.get(tf):
                    volume_data[tf]["buy_volume"], volume_data[tf]["sell_volume"], volume_data[tf]["volume_mood"] = calculate_buy_sell_volume(candle_map[tf], tf)
            
            fft_results = {}
            min_max_distances = []
            recent_extremes = []
            analysis_details = {tf: {} for tf in TIMEFRAMES}
            
            print(f"\n=== Analysis for All Timeframes ===")
            print(f"Current Price: {current_price:.2f} USDC")
            print(f"Local Time: {current_local_time_str}")
            
            for timeframe in TIMEFRAMES:
                if not candle_map.get(timeframe):
                    logging.warning(f"No data for {timeframe}. Skipping analysis.")
                    print(f"\n{timeframe} - No data available")
                    fft_results[timeframe] = {
                        "time": current_local_time_str,
                        "current_price": Decimal('0'),
                        "fastest_target": Decimal('0'),
                        "average_target": Decimal('0'),
                        "reversal_target": Decimal('0'),
                        "market_mood": "Bearish",
                        "min_threshold": Decimal('0'),
                        "middle_threshold": Decimal('0'),
                        "max_threshold": Decimal('0'),
                        "fft_phase": "PEAK",
                        "dominant_freq": 0.0
                    }
                    continue
                
                candles_tf = candle_map[timeframe]
                closes = np.array([c["close"] for c in candles_tf if c["close"] > 0], dtype=np.float64)
                
                min_threshold, middle_threshold, max_threshold = calculate_thresholds(candles_tf)
                reversal_type, min_time, min_price, peak_type, max_time, max_price = detect_recent_reversal(candles_tf, timeframe, min_threshold, max_threshold)
                support_levels, resistance_levels = calculate_support_resistance(candles_tf, timeframe)
                
                buy_vol = volume_data[timeframe]["buy_volume"]
                sell_vol = volume_data[timeframe]["sell_volume"]
                volume_mood = volume_data[timeframe]["volume_mood"]
                
                conditions_long[f"volume_bullish_{timeframe}"] = buy_vol > sell_vol
                conditions_short[f"volume_bearish_{timeframe}"] = sell_vol > buy_vol
                
                conditions_long[f"dip_confirmed_{timeframe}"] = reversal_type == "DIP" and min_time >= max_time
                conditions_short[f"peak_confirmed_{timeframe}"] = reversal_type == "PEAK" and max_time >= min_time
                
                current_close = Decimal(str(closes[-1])) if len(closes) > 0 else current_price
                conditions_long[f"below_middle_{timeframe}"] = current_close < middle_threshold
                conditions_short[f"above_middle_{timeframe}"] = current_close > middle_threshold
                
                print(f"\n--- {timeframe} Timeframe Analysis ---")
                trend, min_th, max_th, cycle_status, trend_bullish, trend_bearish, volume_ratio, volume_confirmed, dominant_freq, cycle_target = calculate_mtf_trend(
                    candles_tf, timeframe, min_threshold, max_threshold, buy_vol, sell_vol
                )
                conditions_long[f"trend_bullish_{timeframe}"] = trend_bullish
                conditions_short[f"trend_bearish_{timeframe}"] = trend_bearish
                
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
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "volume_mood": volume_mood
                }
                
                lookback = min(len(candles_tf), LOOKBACK_PERIODS[timeframe])
                recent_candles = candles_tf[-lookback:]
                min_candle = min(recent_candles, key=lambda x: x['low'])
                max_candle = max(recent_candles, key=lambda x: x['high'])
                recent_extremes.append({
                    "timeframe": timeframe,
                    "lowest_low": Decimal(str(min_candle['low'])),
                    "lowest_low_time": min_candle['time'],
                    "highest_high": Decimal(str(max_candle['high'])),
                    "highest_high_time": max_candle['time']
                })
                
                if len(closes) >= 2:
                    fft_data = get_target(
                        closes, n_components=5, timeframe=timeframe, min_threshold=min_threshold, max_threshold=max_threshold,
                        buy_vol=buy_vol, sell_vol=sell_vol
                    )
                    fft_results[timeframe] = {
                        "time": fft_data[0].strftime('%Y-%m-%d %H:%M:%S'),
                        "current_price": fft_data[1],
                        "fastest_target": fft_data[3],
                        "average_target": fft_data[4],
                        "reversal_target": fft_data[5],
                        "market_mood": fft_data[6],
                        "min_threshold": min_threshold,
                        "middle_threshold": middle_threshold,
                        "max_threshold": max_threshold,
                        "fft_phase": fft_data[10],
                        "dominant_freq": fft_data[9]
                    }
                    conditions_long[f"fft_bullish_{timeframe}"] = fft_data[7]
                    conditions_short[f"fft_bearish_{timeframe}"] = fft_data[8]
                
                if len(closes) >= 14:
                    momentum = talib.MOM(closes, timeperiod=14)
                    if len(momentum) > 0 and not np.isnan(momentum[-1]):
                        conditions_long[f"momentum_positive_{timeframe}"] = Decimal(str(momentum[-1])) >= Decimal('0')
                        conditions_short[f"momentum_negative_{timeframe}"] = not conditions_long[f"momentum_positive_{timeframe}"]
                
                min_max_distances.append({
                    "timeframe": timeframe,
                    "min_distance": abs(current_close - min_threshold),
                    "max_distance": abs(current_close - max_threshold),
                    "min_threshold": min_threshold,
                    "max_threshold": max_threshold
                })
            
            print(f"\n--- Multi-Timeframe Min/Max Comparison ---")
            closest_min_tf = min(min_max_distances, key=lambda x: x["min_distance"])
            closest_max_tf = min(min_max_distances, key=lambda x: x["max_distance"])
            print(f"Closest to Min Threshold: {closest_min_tf['timeframe']} (Distance: {closest_min_tf['min_distance']:.2f}, Min: {closest_min_tf['min_threshold']:.2f})")
            print(f"Closest to Max Threshold: {closest_max_tf['timeframe']} (Distance: {closest_max_tf['max_distance']:.2f}, Max: {closest_max_tf['max_threshold']:.2f})")
            logging.info(f"Closest to Min Threshold: {closest_min_tf['timeframe']} (Distance: {closest_min_tf['min_distance']:.2f})")
            logging.info(f"Closest to Max Threshold: {closest_max_tf['timeframe']} (Distance: {closest_max_tf['max_distance']:.2f})")
            
            most_recent_extreme = max(recent_extremes, key=lambda x: max(x["lowest_low_time"], x["highest_high_time"]))
            most_recent_time = max(most_recent_extreme["lowest_low_time"], most_recent_extreme["highest_high_time"])
            most_recent_type = "LOW" if most_recent_extreme["lowest_low_time"] >= most_recent_extreme["highest_high_time"] else "HIGH"
            most_recent_price = most_recent_extreme["lowest_low"] if most_recent_type == "LOW" else most_recent_extreme["highest_high"]
            print(f"\n--- Most Recent Low vs High ---")
            print(f"Most Recent Extreme: {most_recent_type} in {most_recent_extreme['timeframe']} at {most_recent_price:.2f} "
                  f"(Time: {datetime.datetime.fromtimestamp(most_recent_time)})")
            logging.info(f"Most Recent Extreme: {most_recent_type} in {most_recent_extreme['timeframe']} at {most_recent_price:.2f} "
                         f"(Time: {datetime.datetime.fromtimestamp(most_recent_time)})")
            
            condition_pairs = [
                ("trend_bullish_1m", "trend_bearish_1m"),
                ("trend_bullish_3m", "trend_bearish_3m"),
                ("trend_bullish_5m", "trend_bearish_5m"),
                ("momentum_positive_1m", "momentum_negative_1m"),
                ("momentum_positive_3m", "momentum_negative_3m"),
                ("momentum_positive_5m", "momentum_negative_5m"),
                ("fft_bullish_1m", "fft_bearish_1m"),
                ("fft_bullish_3m", "fft_bearish_3m"),
                ("fft_bullish_5m", "fft_bearish_5m"),
                ("volume_bullish_1m", "volume_bearish_1m"),
                ("volume_bullish_3m", "volume_bearish_3m"),
                ("volume_bullish_5m", "volume_bearish_5m"),
                ("dip_confirmed_1m", "peak_confirmed_1m"),
                ("dip_confirmed_3m", "peak_confirmed_3m"),
                ("dip_confirmed_5m", "peak_confirmed_5m"),
                ("below_middle_1m", "above_middle_1m"),
                ("below_middle_3m", "above_middle_3m"),
                ("below_middle_5m", "above_middle_5m")
            ]
            logging.info("Condition Pairs Status:")
            print("\nCondition Pairs Status:")
            symmetry_valid = True
            for long_cond, short_cond in condition_pairs:
                if conditions_long[long_cond] == conditions_short[short_cond]:
                    conditions_long[long_cond] = False
                    conditions_short[short_cond] = True
                    logging.warning(f"Enforced symmetry: Set {long_cond}=False, {short_cond}=True due to conflict")
                    print(f"Enforced symmetry: Set {long_cond}=False, {short_cond}=True due to conflict")
                    symmetry_valid = False
                logging.info(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]} {'✓' if conditions_long[long_cond] != conditions_short[short_cond] else '✗'}")
                print(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]} {'✓' if conditions_long[long_cond] != conditions_short[short_cond] else '✗'}")
            
            long_signal = all(conditions_long.values()) and symmetry_valid
            short_signal = all(conditions_short.values()) and symmetry_valid
            
            logging.info("Trade Signal Status:")
            print("\nTrade Signal Status:")
            logging.info(f"LONG Signal: {'Active' if long_signal else 'Inactive'} (All conditions: {all(conditions_long.values())})")
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'} (All conditions: {all(conditions_long.values())})")
            logging.info(f"SHORT Signal: {'Active' if short_signal else 'Inactive'} (All conditions: {all(conditions_short.values())})")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'} (All conditions: {all(conditions_short.values())})")
            
            if long_signal and short_signal:
                logging.warning("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                print("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                long_signal = False
                short_signal = False
            
            signal = "LONG" if long_signal else "SHORT" if short_signal else "NO_SIGNAL"
            if signal in ["LONG", "SHORT"]:
                message = (
                    f"*Signal Triggered: {signal}*\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {current_local_time_str}\n"
                    f"Current Price: {current_price:.2f} USDC\n"
                    f"\n*Analysis Details*\n"
                )
                for tf in TIMEFRAMES:
                    details = analysis_details.get(tf, {})
                    fft_data = fft_results.get(tf, {})
                    message += (
                        f"\n*{tf} Timeframe*\n"
                        f"MTF Trend: {details.get('trend', 'N/A')}\n"
                        f"Cycle Status: {details.get('cycle_status', 'N/A')}\n"
                        f"Dominant Frequency: {details.get('dominant_freq', 0.0):.6f}\n"
                        f"Cycle Target: {details.get('cycle_target', Decimal('0')):.2f} USDC\n"
                        f"Volume Ratio: {details.get('volume_ratio', Decimal('0')):.2f}\n"
                        f"Volume Confirmed: {details.get('volume_confirmed', False)}\n"
                        f"Minimum Threshold: {details.get('min_threshold', Decimal('0')):.2f} USDC\n"
                        f"Middle Threshold: {details.get('middle_threshold', Decimal('0')):.2f} USDC\n"
                        f"Maximum Threshold: {details.get('max_threshold', Decimal('0')):.2f} USDC\n"
                        f"Reversal Type: {details.get('reversal_type', 'N/A')}\n"
                        f"Support Level: {details.get('support_levels', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Resistance Level: {details.get('resistance_levels', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Buy Volume: {details.get('buy_volume', Decimal('0')):.2f}\n"
                        f"Sell Volume: {details.get('sell_volume', Decimal('0')):.2f}\n"
                        f"Volume Mood: {details.get('volume_mood', 'N/A')}\n"
                        f"FFT Phase: {fft_data.get('fft_phase', 'N/A')}\n"
                        f"FFT Fastest Target: {fft_data.get('fastest_target', Decimal('0')):.2f} USDC\n"
                        f"FFT Average Target: {fft_data.get('average_target', Decimal('0')):.2f} USDC\n"
                        f"FFT Reversal Target: {fft_data.get('reversal_target', Decimal('0')):.2f} USDC\n"
                        f"FFT Market Mood: {fft_data.get('market_mood', 'N/A')}\n"
                    )
                telegram_loop.run_until_complete(send_telegram_message(message))
            
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
            
            logging.info(f"Final Signal: {signal}")
            print(f"Final Signal: {signal}")
            
            if usdc_balance < MINIMUM_BALANCE:
                logging.warning(f"Insufficient USDC balance ({usdc_balance:.2f}) to place trades.")
                print(f"Insufficient USDC balance ({usdc_balance:.2f}) to place trades.")
            elif signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            elif (signal == "LONG" and position["side"] == "SHORT") or (signal == "SHORT" and position["side"] == "LONG"):
                close_position(position, current_price)
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            
            if position["side"] != "NONE":
                current_balance = usdc_balance + position['unrealized_pnl']
                roi = ((current_balance - position['initial_balance']) / position['initial_balance'] * Decimal('100')).quantize(Decimal('0.01')) if position['initial_balance'] > Decimal('0') else Decimal('0')
                print(f"\nCurrent Position Status:")
                print(f"Position Side: {position['side']}")
                print(f"Quantity: {position['quantity']:.6f} BTC")
                print(f"Entry Price: {position['entry_price']:.2f} USDC")
                print(f"Current Price: {current_price:.2f} USDC")
                print(f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC")
                print(f"Stop-Loss Price: {position['sl_price']:.2f} USDC")
                print(f"Take-Profit Price: {position['tp_price']:.2f} USDC")
                print(f"Current ROI: {roi:.2f}%")
                print(f"Initial USDC Balance: {position['initial_balance']:.2f}")
                print(f"Current Total Balance: {current_balance:.2f} USDC")
            else:
                print(f"\nNo open position. USDC Balance: {usdc_balance:.2f}")
            
            del candle_map
            gc.collect()
            time.sleep(5)
            
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
        print("Bot stopped by user.")
    except Exception as e:
        logging.error(f"Fatal error in main loop: {e}")
        print(f"Fatal error in main loop: {e}")
        position = get_position()
        if position["side"] != "NONE":
            current_price = get_current_price()
            if current_price > Decimal('0'):
                close_position(position, current_price)
            else:
                logging.error("Failed to fetch price during error handling. Position not closed.")
                print("Failed to fetch price during error handling. Position not closed.")
        raise
    finally:
        try:
            position = get_position()
            if position["side"] != "NONE":
                current_price = get_current_price()
                if current_price > Decimal('0'):
                    close_position(position, current_price)
            telegram_loop.run_until_complete(telegram_app.shutdown())
            telegram_loop.close()
            logging.info("Cleanup completed.")
            print("Cleanup completed.")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
