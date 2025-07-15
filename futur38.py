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
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})
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

# Send Telegram message with retry logic
async def send_telegram_message(message, retries=3, delay=5):
    for attempt in range(retries):
        try:
            await telegram_app.bot.send_message(chat_id=telegram_chat_id, text=message, parse_mode='Markdown')
            logging.info(f"Telegram message sent: {message}")
            print(f"Telegram message sent: {message}")
            return True
        except Exception as e:
            logging.error(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
            print(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    logging.error(f"Failed to send Telegram message after {retries} attempts.")
    print(f"Failed to send Telegram message after {retries} attempts.")
    return False

# Enhanced MTF Trend Analysis with Binary Classification
def calculate_mtf_trend(candles, timeframe, min_threshold, max_threshold, buy_volume, sell_volume, lookback=50):
    if len(candles) < lookback:
        logging.warning(f"Insufficient candles ({len(candles)}) for MTF trend analysis in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for MTF trend analysis in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "PEAK", False, True
    
    recent_candles = candles[-lookback:]
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    
    # Validate input data
    if len(closes) == 0 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid or non-positive close prices in {timeframe}.")
        print(f"Invalid or non-positive close prices in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "PEAK", False, True
    
    # FFT for sinusoidal model
    fft_result = fft.rfft(closes)
    frequencies = fft.rfftfreq(len(closes))
    amplitudes = np.abs(fft_result)
    phases = np.angle(fft_result)
    
    # Identify dominant frequency and phase
    idx = np.argsort(amplitudes)[::-1][1:3]  # Skip DC component, take top 2
    dominant_freq = frequencies[idx[0]] if idx.size > 0 else 0.0
    dominant_phase = phases[idx[0]] if idx.size > 0 else 0.0
    
    # Force binary cycle classification (UP or DOWN)
    current_close = Decimal(str(closes[-1]))
    midpoint = (min_threshold + max_threshold) / Decimal('2')
    trend_bullish = current_close >= midpoint or dominant_phase < 0
    trend_bearish = not trend_bullish
    cycle_status = "DIP" if trend_bullish else "PEAK"
    trend = "BULLISH" if trend_bullish else "BEARISH"
    
    # Volume confirmation
    buy_vol = sum(buy_volume[-lookback:]) / lookback if buy_volume else Decimal('0')
    sell_vol = sum(sell_volume[-lookback:]) / lookback if sell_volume else Decimal('0')
    volume_confirmed = buy_vol >= VOLUME_CONFIRMATION_RATIO * sell_vol if trend == "BULLISH" else sell_vol >= VOLUME_CONFIRMATION_RATIO * buy_vol
    
    # Threshold-based confirmation
    trend_confirmed = volume_confirmed
    
    # Calculate cycle target price
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[idx[:2]] = fft_result[idx[:2]]
    filtered_signal = fft.irfft(filtered_fft)
    cycle_target = Decimal(str(filtered_signal[-1])) if len(filtered_signal) > 0 else Decimal('0')
    
    logging.info(f"{timeframe} - MTF Trend: {trend}, Cycle: {cycle_status}, Dominant Freq: {dominant_freq:.6f}, Current Close: {current_close:.25f}, Cycle Target: {cycle_target:.25f}, Volume Confirmed: {volume_confirmed}, Trend Bullish: {trend_bullish}, Trend Bearish: {trend_bearish}")
    print(f"{timeframe} - MTF Trend: {trend}")
    print(f"{timeframe} - Cycle Status: {cycle_status}")
    print(f"{timeframe} - Dominant Frequency: {dominant_freq:.6f}")
    print(f"{timeframe} - Cycle Target: {cycle_target:.25f}")
    print(f"{timeframe} - Volume Confirmation: {volume_confirmed}")
    print(f"{timeframe} - Trend Bullish: {trend_bullish}, Trend Bearish: {trend_bearish}")
    
    return trend, min_threshold, max_threshold, cycle_status, trend_bullish, trend_bearish

# FFT-Based Target Price Function
def get_target(closes, n_components, timeframe, min_threshold, max_threshold):
    if len(closes) < 2 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid closes data for FFT analysis in {timeframe}.")
        print(f"Invalid closes data for FFT analysis in {timeframe}.")
        return (datetime.datetime.now(), Decimal('0'), Decimal('0'), Decimal('0'), 
                Decimal('0'), Decimal('0'), "Bearish", False, True, 0.0, "PEAK")
    
    # FFT analysis
    fft_result = fft.rfft(closes)
    frequencies = fft.rfftfreq(len(closes))
    amplitudes = np.abs(fft_result)
    phases = np.angle(fft_result)
    
    # Identify dominant frequencies
    idx = np.argsort(amplitudes)[::-1][1:n_components+1]
    dominant_freq = frequencies[idx[0]] if idx.size > 0 else 0.0
    dominant_phase = phases[idx[0]] if idx.size > 0 else 0.0
    
    # Force binary phase status (UP or DOWN)
    current_close = Decimal(str(closes[-1]))
    midpoint = (min_threshold + max_threshold) / Decimal('2')
    is_bullish_target = current_close >= midpoint or dominant_phase < 0
    is_bearish_target = not is_bullish_target
    phase_status = "DIP" if is_bullish_target else "PEAK"
    market_mood = "Bullish" if is_bullish_target else "Bearish"
    
    # Reconstruct signal for fastest target
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[idx[:n_components]] = fft_result[idx[:n_components]]
    filtered_signal = fft.irfft(filtered_fft)
    fastest_target = Decimal(str(filtered_signal[-1])) if len(filtered_signal) > 0 else Decimal('0')
    
    # Most significant target (single dominant frequency)
    single_freq_fft = np.zeros_like(fft_result, dtype=complex)
    single_freq_fft[idx[0]] = fft_result[idx[0]]
    single_freq_signal = fft.irfft(single_freq_fft)
    most_significant_target = Decimal(str(single_freq_signal[-1])) if len(single_freq_signal) > 0 else Decimal('0')
    
    # Most extreme target (forecast forward)
    forecast_steps = 10
    extended_signal = np.zeros(len(closes) + forecast_steps)
    extended_signal[:len(closes)] = closes
    for i in range(forecast_steps):
        extended_fft = fft.rfft(extended_signal[:len(closes) + i])
        extended_signal[len(closes) + i] = fft.irfft(extended_fft)[-1] if len(fft.irfft(extended_fft)) > 0 else closes[-1]
    forecast_prices = extended_signal[len(closes):]
    most_extreme_target = Decimal(str(np.max(forecast_prices) if is_bullish_target else np.min(forecast_prices)))
    
    # Energy scaling
    energy = np.sum(amplitudes[idx[:n_components]] ** 2)
    energy_scale = Decimal(str(1 + energy / len(closes)))
    
    if is_bullish_target:
        fastest_target = min(fastest_target * energy_scale, max_threshold * Decimal('0.995'))
        fastest_target = max(fastest_target, current_close * Decimal('1.005'))
        most_significant_target = min(most_significant_target * energy_scale, max_threshold * Decimal('0.995'))
        most_significant_target = max(most_significant_target, current_close * Decimal('1.005'))
        most_extreme_target = min(most_extreme_target, max_threshold * Decimal('0.995'))
        most_extreme_target = max(most_extreme_target, current_close * Decimal('1.005'))
    else:
        fastest_target = max(fastest_target / energy_scale, min_threshold * Decimal('1.005'))
        fastest_target = min(fastest_target, current_close * Decimal('0.995'))
        most_significant_target = max(most_significant_target / energy_scale, min_threshold * Decimal('1.005'))
        most_significant_target = min(most_significant_target, current_close * Decimal('0.995'))
        most_extreme_target = max(most_extreme_target, min_threshold * Decimal('1.005'))
        most_extreme_target = min(most_extreme_target, current_close * Decimal('0.995'))
    
    stop_loss = current_close + Decimal(str(3 * np.std(closes))) if is_bearish_target else current_close - Decimal(str(3 * np.std(closes)))
    
    logging.info(f"FFT Analysis - Timeframe: {timeframe}, Market Mood: {market_mood}, Phase: {phase_status}, "
                 f"Current Close: {current_close:.25f}, Fastest Target: {fastest_target:.25f}, "
                 f"Most Significant Target: {most_significant_target:.25f}, Most Extreme Target: {most_extreme_target:.25f}, "
                 f"Stop Loss: {stop_loss:.25f}, Dominant Freq: {dominant_freq:.6f}, Energy: {energy:.2f}, "
                 f"Bullish Target: {is_bullish_target}, Bearish Target: {is_bearish_target}")
    print(f"{timeframe} - FFT Phase: {phase_status}")
    print(f"{timeframe} - Dominant Frequency: {dominant_freq:.6f}")
    print(f"{timeframe} - Energy Scale: {energy_scale:.2f}")
    print(f"{timeframe} - Fastest Target: {fastest_target:.25f}")
    print(f"{timeframe} - Most Significant Target: {most_significant_target:.25f}")
    print(f"{timeframe} - Most Extreme Target: {most_extreme_target:.25f}")
    print(f"{timeframe} - Bullish Target: {is_bullish_target}, Bearish Target: {is_bearish_target}")
    
    return (datetime.datetime.now(), current_close, stop_loss, fastest_target, 
            most_significant_target, most_extreme_target, market_mood, 
            is_bullish_target, is_bearish_target, dominant_freq, phase_status)

# Volume Analysis Functions
def calculate_volume(candles):
    if not candles:
        logging.warning("No candles provided for volume calculation.")
        print("No candles provided for volume calculation.")
        return Decimal('0')
    total_volume = sum(Decimal(str(candle["volume"])) for candle in candles)
    logging.info(f"Total Volume: {total_volume:.25f}")
    print(f"Total Volume: {total_volume:.25f}")
    return total_volume

def calculate_buy_sell_volume(candle_map):
    if not candle_map or not any(candle_map.values()):
        logging.warning("Empty or invalid candle_map for volume analysis.")
        print("Empty or invalid candle_map for volume analysis.")
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), {"1m": "BEARISH", "3m": "BEARISH", "5m": "BEARISH"}
    
    try:
        buy_volume_1m = sum(Decimal(str(candle["volume"])) for candle in candle_map.get("1m", []) if Decimal(str(candle["close"])) > Decimal(str(candle["open"])))
        sell_volume_1m = sum(Decimal(str(candle["volume"])) for candle in candle_map.get("1m", []) if Decimal(str(candle["close"])) < Decimal(str(candle["open"])))
        buy_volume_3m = sum(Decimal(str(candle["volume"])) for candle in candle_map.get("3m", []) if Decimal(str(candle["close"])) > Decimal(str(candle["open"])))
        sell_volume_3m = sum(Decimal(str(candle["volume"])) for candle in candle_map.get("3m", []) if Decimal(str(candle["close"])) < Decimal(str(candle["open"])))
        buy_volume_5m = sum(Decimal(str(candle["volume"])) for candle in candle_map.get("5m", []) if Decimal(str(candle["close"])) > Decimal(str(candle["open"])))
        sell_volume_5m = sum(Decimal(str(candle["volume"])) for candle in candle_map.get("5m", []) if Decimal(str(candle["close"])) < Decimal(str(candle["open"])))
        
        volume_moods = {}
        for tf in ["1m", "3m", "5m"]:
            buy_vol = locals()[f"buy_volume_{tf}"]
            sell_vol = locals()[f"sell_volume_{tf}"]
            volume_moods[tf] = "BULLISH" if buy_vol >= sell_vol else "BEARISH"
            logging.info(f"{tf} - Buy Volume: {buy_vol:.25f}, Sell Volume: {sell_vol:.25f}, Mood: {volume_moods[tf]}")
            print(f"{tf} - Buy Volume: {buy_vol:.25f}")
            print(f"{tf} - Sell Volume: {sell_vol:.25f}")
            print(f"{tf} - Volume Mood: {volume_moods[tf]}")
        
        return buy_volume_1m, sell_volume_1m, buy_volume_3m, sell_volume_3m, buy_volume_5m, sell_volume_5m, volume_moods
    except Exception as e:
        logging.error(f"Error in calculate_buy_sell_volume: {e}")
        print(f"Error in calculate_buy_sell_volume: {e}")
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), {"1m": "BEARISH", "3m": "BEARISH", "5m": "BEARISH"}

# Support and Resistance Analysis
def calculate_support_resistance(candles, timeframe):
    if not candles or len(candles) < 10:
        logging.warning(f"Insufficient candles ({len(candles)}) for support/resistance analysis in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for support/resistance analysis in {timeframe}.")
        return [], []
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    
    lows = np.array([float(candle['low']) for candle in recent_candles], dtype=np.float64)
    highs = np.array([float(candle['high']) for candle in recent_candles], dtype=np.float64)
    
    # Use histogram for support/resistance
    bins = 50
    low_hist, low_edges = np.histogram(lows, bins=bins)
    high_hist, high_edges = np.histogram(highs, bins=bins)
    
    support_levels = []
    for i in np.argsort(low_hist)[-3:][::-1]:
        if low_hist[i] >= 3:
            price = Decimal(str((low_edges[i] + low_edges[i+1]) / 2))
            support_levels.append({"price": price, "touches": int(low_hist[i])})
    
    resistance_levels = []
    for i in np.argsort(high_hist)[-3:][::-1]:
        if high_hist[i] >= 3:
            price = Decimal(str((high_edges[i] + high_edges[i+1]) / 2))
            resistance_levels.append({"price": price, "touches": int(high_hist[i])})
    
    logging.info(f"{timeframe} - Support Levels: {[f'Price: {level['price']:.25f}, Touches: {level['touches']}' for level in support_levels]}")
    print(f"{timeframe} - Support Levels: {[f'Price: {level['price']:.25f}, Touches: {level['touches']}' for level in support_levels]}")
    logging.info(f"{timeframe} - Resistance Levels: {[f'Price: {level['price']:.25f}, Touches: {level['touches']}' for level in resistance_levels]}")
    print(f"{timeframe} - Resistance Levels: {[f'Price: {level['price']:.25f}, Touches: {level['touches']}' for level in resistance_levels]}")
    
    return support_levels, resistance_levels

# Reversal Detection Function
def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, buy_volume, sell_volume):
    if len(candles) < 3:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        return "PEAK"
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    recent_buy_volume = buy_volume[-lookback:] if buy_volume else [Decimal('0')] * lookback
    recent_sell_volume = sell_volume[-lookback:] if sell_volume else [Decimal('0')] * lookback
    
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    lows = np.array([float(c['low']) for c in recent_candles], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles], dtype=np.float64)
    times = np.array([c['time'] for c in recent_candles], dtype=np.float64)
    
    if len(closes) < 3 or len(times) < 3 or len(lows) < 3 or len(highs) < 3:
        logging.warning(f"Insufficient valid data for reversal detection in {timeframe}.")
        print(f"Insufficient valid data for reversal detection in {timeframe}.")
        return "PEAK"
    
    current_close = Decimal(str(closes[-1]))
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
        buy_vol = recent_buy_volume[i]
        sell_vol = recent_sell_volume[i]
        
        min_diff = abs(low_price - min_threshold)
        if min_diff <= tolerance and min_diff < closest_min_diff:
            if buy_vol >= VOLUME_CONFIRMATION_RATIO * sell_vol:
                closest_min_diff = min_diff
                min_time = candle_time
                closest_min_price = low_price
                min_volume_confirmed = True
        
        max_diff = abs(high_price - max_threshold)
        if max_diff <= tolerance and max_diff < closest_max_diff:
            if sell_vol >= VOLUME_CONFIRMATION_RATIO * buy_vol:
                closest_max_diff = max_diff
                max_time = candle_time
                closest_max_price = high_price
                max_volume_confirmed = True
    
    if min_time > 0:
        logging.info(f"{timeframe} - Found closest low {closest_min_price:.25f} at time {datetime.datetime.fromtimestamp(min_time)}")
        print(f"{timeframe} - Found closest low {closest_min_price:.25f} at time {datetime.datetime.fromtimestamp(min_time)}")
    else:
        logging.warning(f"{timeframe} - No low found within {tolerance:.25f} of min threshold {min_threshold:.25f}.")
        print(f"{timeframe} - No low found within {tolerance:.25f} of min threshold {min_threshold:.25f}.")
    
    if max_time > 0:
        logging.info(f"{timeframe} - Found closest high {closest_max_price:.25f} at time {datetime.datetime.fromtimestamp(max_time)}")
        print(f"{timeframe} - Found closest high {closest_max_price:.25f} at time {datetime.datetime.fromtimestamp(max_time)}")
    else:
        logging.warning(f"{timeframe} - No high found within {tolerance:.25f} of max threshold {max_threshold:.25f}.")
        print(f"{timeframe} - No high found within {tolerance:.25f} of max threshold {max_threshold:.25f}.")
    
    price_range = max_threshold - min_threshold
    if price_range > Decimal('0'):
        min_pct = ((current_close - min_threshold) / price_range * Decimal('100')).quantize(Decimal('0.01'))
        max_pct = ((max_threshold - current_close) / price_range * Decimal('100')).quantize(Decimal('0.01'))
    else:
        min_pct = Decimal('50.00')
        max_pct = Decimal('50.00')
        logging.warning(f"{timeframe} - Zero price range detected. Defaulting normalized distances to 50%.")
        print(f"{timeframe} - Zero price range detected. Defaulting normalized distances to 50%.")
    logging.info(f"{timeframe} - Normalized Distance: {min_pct:.2f}% from min threshold, {max_pct:.2f}% from max threshold")
    print(f"{timeframe} - Normalized Distance: {min_pct:.2f}% from min threshold, {max_pct:.2f}% from max threshold")
    
    reversals = []
    if min_time > 0 and min_volume_confirmed:
        reversals.append({"type": "DIP", "price": closest_min_price, "time": min_time})
    if max_time > 0 and max_volume_confirmed:
        reversals.append({"type": "PEAK", "price": closest_max_price, "time": max_time})
    
    if not reversals:
        logging.info(f"{timeframe} - No valid reversals detected over {len(recent_candles)} candles. Defaulting to PEAK.")
        print(f"{timeframe} - No valid reversals detected over {len(recent_candles)} candles. Defaulting to PEAK.")
        return "PEAK"
    
    reversals.sort(key=lambda x: x["time"], reverse=True)
    most_recent = reversals[0]
    dist_to_min = abs(current_close - closest_min_price)
    dist_to_max = abs(current_close - closest_max_price)
    reversal_type = "DIP" if dist_to_min <= dist_to_max else "PEAK"
    
    logging.info(f"{timeframe} - Confirmed reversal: {reversal_type} at price {most_recent['price']:.25f}, time {datetime.datetime.fromtimestamp(most_recent['time'])}")
    print(f"{timeframe} - Confirmed reversal: {reversal_type} at price {most_recent['price']:.25f}, time {datetime.datetime.fromtimestamp(most_recent['time'])}")
    return reversal_type

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
            logging.info(f"Fetched {len(candles)} candles for {timeframe}")
            print(f"Fetched {len(candles)} candles for {timeframe}")
            return candles
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', '60') if e.response else '60'
            try:
                retry_after_int = int(retry_after)
            except (ValueError, TypeError):
                retry_after_int = 60
            if e.code == -1003:
                logging.warning(f"Rate limit exceeded for {timeframe}. Waiting {retry_after_int} seconds.")
                print(f"Rate limit exceeded for {timeframe}. Waiting {retry_after_int} seconds.")
                time.sleep(retry_after_int)
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
    logging.error(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                logging.info(f"Current {TRADE_SYMBOL} price: {price:.25f}")
                print(f"Current {TRADE_SYMBOL} price: {price:.25f}")
                return price
            logging.warning(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', '60') if e.response else '60'
            try:
                retry_after_int = int(retry_after)
            except (ValueError, TypeError):
                retry_after_int = 60
            logging.error(f"Binance API Error fetching price (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Binance API Error fetching price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after_int if e.code == -1003 else delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            logging.error(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            print(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts. Falling back to zero.")
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts. Falling back to zero.")
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
        return Decimal('0.0')
    except Exception as e:
        logging.error(f"Unexpected error fetching {asset} balance: {e}")
        print(f"Unexpected error fetching {asset} balance: {e}")
        return Decimal('0.0')

def get_position():
    try:
        positions = client.futures_position_information(symbol=TRADE_SYMBOL)
        if not positions:
            logging.warning(f"No position data returned for {TRADE_SYMBOL}.")
            print(f"No position data returned for {TRADE_SYMBOL}.")
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
        logging.warning(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}).")
        print(f"Insufficient balance ({balance:.25f} USDC) or invalid price ({price:.25f}).")
        return Decimal('0.0')
    quantity = (balance * Decimal(str(LEVERAGE))) / price
    quantity = quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')
    logging.info(f"Calculated quantity: {quantity:.25f} BTC for balance {balance:.25f} USDC at price {price:.25f}")
    print(f"Calculated quantity: {quantity:.25f} BTC for balance {balance:.25f} USDC at price {price:.25f}")
    return quantity

def place_order(signal, quantity, price, initial_balance):
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
            tp_price = (price * (Decimal('1') + TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'))
            sl_price = (price * (Decimal('1') - STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'))
            
            position["sl_price"] = sl_price
            position["tp_price"] = tp_price
            position["side"] = "LONG"
            position["quantity"] = quantity
            position["entry_price"] = price
            
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
            
            logging.info(f"Placed LONG order: {quantity:.25f} BTC at market price ~{price:.25f}")
            print(f"\n=== TRADE ENTERED ===")
            print(f"Side: LONG")
            print(f"Quantity: {quantity:.25f} BTC")
            print(f"Entry Price: ~{price:.25f} USDC")
            print(f"Initial USDC Balance: {initial_balance:.25f}")
            print(f"Stop-Loss Price: {sl_price:.25f} (-50% ROI)")
            print(f"Take-Profit Price: {tp_price:.25f} (+5% ROI)")
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
            
            position["sl_price"] = sl_price
            position["tp_price"] = tp_price
            position["side"] = "SHORT"
            position["quantity"] = -quantity
            position["entry_price"] = price
            
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
            
            logging.info(f"Placed SHORT order: {quantity:.25f} BTC at market price ~{price:.25f}")
            print(f"\n=== TRADE ENTERED ===")
            print(f"Side: SHORT")
            print(f"Quantity: {quantity:.25f} BTC")
            print(f"Entry Price: ~{price:.25f} USDC")
            print(f"Initial USDC Balance: {initial_balance:.25f}")
            print(f"Stop-Loss Price: {sl_price:.25f} (-50% ROI)")
            print(f"Take-Profit Price: {tp_price:.25f} (+5% ROI)")
            print(f"===================\n")
        
        open_orders = check_open_orders()
        if open_orders > 0:
            logging.warning(f"Unexpected open orders ({open_orders}) detected after placing {signal} order.")
            print(f"Warning: Unexpected open orders ({open_orders}) detected after placing {signal} order.")
        return position
    except BinanceAPIException as e:
        logging.error(f"Error placing order: {e.message}")
        print(f"Error placing order: {e.message}")
        return None

def close_position(position, price):
    if position["side"] == "NONE" or position["quantity"] == Decimal('0'):
        logging.info("No position to close.")
        print("No position to close.")
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
        
        logging.info(f"Closed {position['side']} position: {quantity:.25f} BTC at market price ~{price:.25f}")
        print(f"Closed {position['side']} position: {quantity:.25f} BTC at market price ~{price:.25f}")
    except BinanceAPIException as e:
        logging.error(f"Error closing position: {e.message}")
        print(f"Error closing position: {e.message}")

# Analysis Functions
def calculate_thresholds(candles):
    if not candles:
        logging.warning("No candles provided for threshold calculation.")
        print("No candles provided for threshold calculation.")
        return Decimal('0'), Decimal('0')
    
    lookback = min(len(candles), LOOKBACK_PERIODS[candles[0]['timeframe']])
    highs = np.array([float(c['high']) for c in candles[-lookback:]], dtype=np.float64)
    lows = np.array([float(c['low']) for c in candles[-lookback:]], dtype=np.float64)
    
    if len(highs) == 0 or len(lows) == 0 or np.any(np.isnan(highs)) or np.any(np.isnan(lows)):
        logging.warning(f"No valid OHLC data for threshold calculation.")
        print(f"No valid OHLC data for threshold calculation.")
        return Decimal('0'), Decimal('0')
    
    # Use histogram for thresholds
    bins = 50
    low_hist, low_edges = np.histogram(lows, bins=bins)
    high_hist, high_edges = np.histogram(highs, bins=bins)
    
    min_idx = np.argmax(low_hist)
    max_idx = np.argmax(high_hist)
    
    min_threshold = Decimal(str((low_edges[min_idx] + low_edges[min_idx+1]) / 2))
    max_threshold = Decimal(str((high_edges[max_idx] + high_edges[max_idx+1]) / 2))
    
    timeframe = candles[0]['timeframe']
    logging.info(f"{timeframe} - Thresholds calculated: Minimum: {min_threshold:.25f}, Maximum: {max_threshold:.25f}")
    print(f"{timeframe} - Minimum Threshold: {min_threshold:.25f}")
    print(f"{timeframe} - Maximum Threshold: {max_threshold:.25f}")
    
    return min_threshold, max_threshold

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
            
            # Fetch current price
            current_price = get_current_price()
            if current_price <= Decimal('0'):
                logging.warning(f"Failed to fetch valid {TRADE_SYMBOL} price. Retrying in 60 seconds.")
                print(f"Warning: Failed to fetch valid {TRADE_SYMBOL} price. Retrying in 60 seconds.")
                time.sleep(60)
                continue
            
            # Fetch candles
            candle_map = fetch_candles_in_parallel(timeframes)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candle data available. Retrying in 60 seconds.")
                print("No candle data available. Retrying in 60 seconds.")
                time.sleep(60)
                continue
            
            usdc_balance = get_balance('USDC')
            position = get_position()
            
            # Check for stop-loss or take-profit triggers
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
                        logging.info(f"Stop-Loss triggered for LONG at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                        print(f"Stop-Loss triggered for LONG at {current_price:.25f} (SL: {position['sl_price']:.25f})")
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
                        logging.info(f"Take-Profit triggered for LONG at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        print(f"Take-Profit triggered for LONG at {current_price:.25f} (TP: {position['tp_price']:.25f})")
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
                        logging.info(f"Stop-Loss triggered for SHORT at {current_price:.25f} (SL: {position['sl_price']:.25f})")
                        print(f"Stop-Loss triggered for SHORT at {current_price:.25f} (SL: {position['sl_price']:.25f})")
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
                        logging.info(f"Take-Profit triggered for SHORT at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        print(f"Take-Profit triggered for SHORT at {current_price:.25f} (TP: {position['tp_price']:.25f})")
                        close_position(position, current_price)
                        position = get_position()
            
            # Initialize conditions
            conditions_long = {
                "trend_bullish_1m": False, "trend_bullish_3m": False, "trend_bullish_5m": False,
                "momentum_positive_1m": False, "momentum_positive_3m": False, "momentum_positive_5m": False,
                "fft_bullish_1m": False, "fft_bullish_3m": False, "fft_bullish_5m": False
            }
            conditions_short = {
                "trend_bearish_1m": True, "trend_bearish_3m": True, "trend_bearish_5m": True,
                "momentum_negative_1m": True, "momentum_negative_3m": True, "momentum_negative_5m": True,
                "fft_bearish_1m": True, "fft_bearish_3m": True, "fft_bearish_5m": True
            }
            
            buy_volume = {}
            sell_volume = {}
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
            
            buy_vol_1m, sell_vol_1m, buy_vol_3m, sell_vol_3m, buy_vol_5m, sell_vol_5m, volume_moods = calculate_buy_sell_volume(candle_map)
            
            fft_results = {}
            print(f"\n=== FFT Analysis for All Timeframes ===")
            print(f"Current Price: {current_price:.25f} USDC")
            print(f"Local Time: {current_local_time_str}")
            
            # Compare min/max across timeframes
            min_max_distances = []
            
            for timeframe in timeframes:
                if not candle_map.get(timeframe):
                    logging.warning(f"No data for {timeframe}. Skipping analysis.")
                    print(f"\n{timeframe} - No data available")
                    fft_results[timeframe] = {
                        "time": current_local_time_str,
                        "current_price": Decimal('0'),
                        "fastest_target": Decimal('0'),
                        "most_significant_target": Decimal('0'),
                        "most_extreme_target": Decimal('0'),
                        "market_mood": "Bearish"
                    }
                    conditions_long[f"trend_bullish_{timeframe}"] = False
                    conditions_short[f"trend_bearish_{timeframe}"] = True
                    conditions_long[f"fft_bullish_{timeframe}"] = False
                    conditions_short[f"fft_bearish_{timeframe}"] = True
                    conditions_long[f"momentum_positive_{timeframe}"] = False
                    conditions_short[f"momentum_negative_{timeframe}"] = True
                    continue
                
                candles_tf = candle_map[timeframe]
                closes = np.array([candle["close"] for candle in candles_tf if not np.isnan(candle["close"]) and candle["close"] > 0], dtype=np.float64)
                
                # Calculate dynamic thresholds
                min_threshold, max_threshold = calculate_thresholds(candles_tf)
                reversal_type = detect_recent_reversal(candles_tf, timeframe, min_threshold, max_threshold, buy_volume[timeframe], sell_volume[timeframe])
                
                # Support/Resistance Analysis
                print(f"\n--- {timeframe} Timeframe Support/Resistance Analysis ---")
                support_levels, resistance_levels = calculate_support_resistance(candles_tf, timeframe)
                
                # MTF Trend Analysis
                print(f"\n--- {timeframe} Timeframe MTF Trend Analysis ---")
                trend, min_threshold, max_threshold, cycle_status, trend_bullish, trend_bearish = calculate_mtf_trend(
                    candles_tf, timeframe, min_threshold, max_threshold, buy_volume[timeframe], sell_volume[timeframe]
                )
                conditions_long[f"trend_bullish_{timeframe}"] = trend_bullish
                conditions_short[f"trend_bearish_{timeframe}"] = trend_bearish
                
                # FFT Analysis
                print(f"\n--- {timeframe} Timeframe FFT Analysis ---")
                if len(closes) >= 2:
                    (current_time, entry_price, stop_loss, fastest_target, 
                     most_significant_target, most_extreme_target, market_mood, 
                     is_bullish_target, is_bearish_target, dominant_freq, phase_status) = get_target(
                        closes, n_components=5, timeframe=timeframe, min_threshold=min_threshold, max_threshold=max_threshold
                    )
                    fft_results[timeframe] = {
                        "time": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "current_price": entry_price,
                        "fastest_target": fastest_target,
                        "most_significant_target": most_significant_target,
                        "most_extreme_target": most_extreme_target,
                        "market_mood": market_mood
                    }
                    print(f"{timeframe} - Local Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{timeframe} - Current Price: {entry_price:.25f}")
                    print(f"{timeframe} - FFT Fastest Target: {fastest_target:.25f}")
                    print(f"{timeframe} - FFT Most Significant Target: {most_significant_target:.25f}")
                    print(f"{timeframe} - FFT Most Extreme Target: {most_extreme_target:.25f}")
                    print(f"{timeframe} - Market Mood: {market_mood}")
                    
                    conditions_long[f"fft_bullish_{timeframe}"] = is_bullish_target
                    conditions_short[f"fft_bearish_{timeframe}"] = is_bearish_target
                else:
                    logging.warning(f"{timeframe} - FFT Data Status: Insufficient data")
                    print(f"{timeframe} - FFT Data Status: Insufficient data")
                    fft_results[timeframe] = {
                        "time": current_local_time_str,
                        "current_price": Decimal('0'),
                        "fastest_target": Decimal('0'),
                        "most_significant_target": Decimal('0'),
                        "most_extreme_target": Decimal('0'),
                        "market_mood": "Bearish"
                    }
                    conditions_long[f"fft_bullish_{timeframe}"] = False
                    conditions_short[f"fft_bearish_{timeframe}"] = True
                
                # Momentum Analysis
                print(f"\n--- {timeframe} Timeframe Momentum Analysis ---")
                if len(closes) >= 14:
                    momentum = talib.MOM(closes, timeperiod=14)
                    if len(momentum) > 0 and not np.isnan(momentum[-1]):
                        current_momentum = Decimal(str(momentum[-1]))
                        conditions_long[f"momentum_positive_{timeframe}"] = current_momentum >= Decimal('0')
                        conditions_short[f"momentum_negative_{timeframe}"] = not conditions_long[f"momentum_positive_{timeframe}"]
                        logging.info(f"{timeframe} Momentum: {current_momentum:.25f}, Positive: {conditions_long[f'momentum_positive_{timeframe}']}, Negative: {conditions_short[f'momentum_negative_{timeframe}']}")
                        print(f"{timeframe} Momentum: {current_momentum:.25f}")
                        print(f"{timeframe} Momentum Positive: {conditions_long[f'momentum_positive_{timeframe}']}, Negative: {conditions_short[f'momentum_negative_{timeframe}']}")
                else:
                    logging.warning(f"{timeframe} Momentum: Insufficient data")
                    print(f"{timeframe} Momentum: Insufficient data")
                    conditions_long[f"momentum_positive_{timeframe}"] = False
                    conditions_short[f"momentum_negative_{timeframe}"] = True
                
                # Store min/max distances for comparison
                current_close = Decimal(str(closes[-1])) if len(closes) > 0 else current_price
                min_max_distances.append({
                    "timeframe": timeframe,
                    "min_distance": abs(current_close - min_threshold),
                    "max_distance": abs(current_close - max_threshold),
                    "min_threshold": min_threshold,
                    "max_threshold": max_threshold
                })
            
            # Compare min/max across timeframes
            print(f"\n--- Multi-Timeframe Min/Max Comparison ---")
            closest_min_tf = min(min_max_distances, key=lambda x: x["min_distance"])
            closest_max_tf = min(min_max_distances, key=lambda x: x["max_distance"])
            print(f"Closest to Min Threshold: {closest_min_tf['timeframe']} (Distance: {closest_min_tf['min_distance']:.25f}, Min: {closest_min_tf['min_threshold']:.25f}")
            print(f"Closest to Max Threshold: {closest_max_tf['timeframe']} (Distance: {closest_max_tf['max_distance']:.25f}, Max: {closest_max_tf['max_threshold']:.25f}")
            logging.info(f"Closest to Min Threshold: {closest_min_tf['timeframe']} (Distance: {closest_min_tf['min_distance']:.25f})")
            logging.info(f"Closest to Max Threshold: {closest_max_tf['timeframe']} (Distance: {closest_max_tf['max_distance']:.25f})")
            
            # Enforce strict condition symmetry
            condition_pairs = [
                ("trend_bullish_1m", "trend_bearish_1m"),
                ("trend_bullish_3m", "trend_bearish_3m"),
                ("trend_bullish_5m", "trend_bearish_5m"),
                ("momentum_positive_1m", "momentum_negative_1m"),
                ("momentum_positive_3m", "momentum_negative_3m"),
                ("momentum_positive_5m", "momentum_negative_5m"),
                ("fft_bullish_1m", "fft_bearish_1m"),
                ("fft_bullish_3m", "fft_bearish_3m"),
                ("fft_bullish_5m", "fft_bearish_5m")
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
            
            # Signal generation: Require all conditions to be True
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
            logging.info(f"Final Signal: {signal}")
            print(f"Final Signal: {signal}")
            
            # Send Telegram notification for signals
            if signal in ["LONG", "SHORT"]:
                telegram_message = (
                    f"*Signal Triggered: {signal}*\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {current_local_time_str}\n"
                    f"Current Price: {current_price:.2f} USDC\n"
                    f"\n*FFT Analysis*\n"
                )
                for tf in timeframes:
                    fft_data = fft_results.get(tf, {})
                    telegram_message += (
                        f"{tf} Timeframe:\n"
                        f"  - Local Time: {fft_data.get('time', 'N/A')}\n"
                        f"  - Current Price: {fft_data.get('current_price', Decimal('0')):.2f} USDC\n"
                        f"  - FFT Fastest Target: {fft_data.get('fastest_target', Decimal('0')):.2f} USDC\n"
                        f"  - FFT Most Significant Target: {fft_data.get('most_significant_target', Decimal('0')):.2f} USDC\n"
                        f"  - FFT Most Extreme Target: {fft_data.get('most_extreme_target', Decimal('0')):.2f} USDC\n"
                        f"  - Market Mood: {fft_data.get('market_mood', 'Bearish')}\n"
                    )
                telegram_loop.run_until_complete(send_telegram_message(telegram_message))
            
            # Execute trades
            if usdc_balance < MINIMUM_BALANCE:
                logging.warning(f"Insufficient USDC balance ({usdc_balance:.25f}) to place trades.")
                print(f"Insufficient USDC balance ({usdc_balance:.25f}) to place trades.")
            elif signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            elif (signal == "LONG" and position["side"] == "SHORT") or (signal == "SHORT" and position["side"] == "LONG"):
                close_position(position, current_price)
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            
            if position["side"] != "NONE":
                print(f"\nCurrent Position Status:")
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
            current_price = get_current_price()
            if current_price > Decimal('0'):
                close_position(position, current_price)
            else:
                logging.error("Failed to fetch price during shutdown. Position not closed.")
                print("Failed to fetch price during shutdown. Position not closed.")
        logging.info("Bot shutdown complete.")
        print("Bot shutdown complete.")
        telegram_loop.close()
        exit(0)
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        print(f"Unexpected error in main loop: {e}")
        position = get_position()
        if position["side"] != "NONE":
            current_price = get_current_price()
            if current_price > Decimal('0'):
                close_position(position, current_price)
            else:
                logging.error("Failed to fetch price during error handling. Position not closed.")
                print(f"Failed to fetch price during error handling. Position not closed.")
        time.sleep(60)
    finally:
        telegram_loop.close()

if __name__ == "__main__":
    main()