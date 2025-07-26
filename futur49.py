import asyncio
import datetime
import time
import concurrent.futures
import talib
import gc
import numpy as np
from decimal import Decimal, getcontext
import logging
from scipy import fft
from telegram.ext import Application
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set Decimal precision
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"
LEVERAGE = 25
PNL_PERCENTAGE = Decimal('0.05')  # 5% PNL for SL/TP
QUANTITY_PRECISION = Decimal('0.000001')  # Binance quantity precision for BTCUSDC
MINIMUM_BALANCE = Decimal('1.0000')  # Minimum USDC balance to place trades
TIMEFRAMES = ["1m", "3m", "5m"]
LOOKBACK_PERIODS = {"1m": 1500, "3m": 1500, "5m": 1500}
RECENT_LOOKBACK = {"1m": 100, "3m": 50, "5m": 30}  # Recent candles for short-term reversal check
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

# Enhanced MTF Trend Analysis with Dynamic Thresholds
def calculate_mtf_trend(candles, timeframe, min_threshold, max_threshold, buy_vol, sell_vol, lookback=50):
    if len(candles) < lookback:
        logging.warning(f"Insufficient candles ({len(candles)}) for MTF trend analysis in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for MTF trend analysis in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, min_threshold
    
    recent_candles = candles[-lookback:]
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    
    # Validate input data
    if len(closes) == 0 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid or non-positive close prices in {timeframe}.")
        print(f"Invalid or non-positive close prices in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, min_threshold
    
    current_close = Decimal(str(closes[-1]))
    rangehilo = max_threshold - min_threshold
    
    # Dynamic volume confirmation ratio based on rangehilo
    try:
        volume_ratio = buy_vol / sell_vol if sell_vol > Decimal('0') else Decimal('1.0')
        volume_confirmation_ratio = Decimal('1.5') * (rangehilo / current_close) if current_close > Decimal('0') else Decimal('1.5')
        volume_confirmation_ratio = min(max(volume_confirmation_ratio, Decimal('0.01') * rangehilo), Decimal('0.50') * rangehilo)
    except Exception as e:
        logging.error(f"Error calculating volume ratio for {timeframe}: {e}")
        volume_ratio = Decimal('1.0')
        volume_confirmation_ratio = Decimal('1.5')
    
    # FFT for sinusoidal model with Hamming window
    try:
        # Normalize closes by subtracting mean
        closes_mean = np.mean(closes)
        closes_normalized = closes - closes_mean
        
        window = np.hamming(len(closes_normalized))
        fft_result = fft.rfft(closes_normalized * window)
        frequencies = fft.rfftfreq(len(closes_normalized), d=1.0)
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Identify dominant frequency and phase
        idx = np.argsort(amplitudes)[::-1][1:3]  # Skip DC component, take top 2
        dominant_freq = frequencies[idx[0]] if idx.size > 0 else 0.0
        dominant_phase = phases[idx[0]] if idx.size > 0 else 0.0
        
        # Convert frequency to cycles per second
        timeframe_seconds = {"1m": 60, "3m": 180, "5m": 300}
        if timeframe in timeframe_seconds:
            dominant_freq = dominant_freq / timeframe_seconds[timeframe]
        
        # Ensure non-zero frequency
        if abs(dominant_freq) < 1e-10:
            dominant_freq = 0.001
    except Exception as e:
        logging.error(f"FFT calculation failed for {timeframe}: {e}")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, volume_ratio, False, 0.0, current_close
    
    # Determine trend and cycle status
    midpoint = (min_threshold + max_threshold) / Decimal('2')
    trend_bullish = False  # Enforced bearish
    trend_bearish = True   # Enforced bearish
    cycle_status = "Down"
    trend = "BEARISH"
    
    # Apply sign to dominant frequency
    dominant_freq_signed = -dominant_freq if trend_bullish else dominant_freq
    
    # Volume confirmation
    volume_confirmed = (Decimal('1.0') / volume_ratio >= volume_confirmation_ratio if volume_ratio > Decimal('0') else False)
    
    # Calculate cycle target price
    try:
        filtered_fft = np.zeros_like(fft_result, dtype=complex)
        filtered_fft[idx[:2]] = fft_result[idx[:2]]
        filtered_signal = fft.irfft(filtered_fft, n=len(closes_normalized))
        # Restore mean and ensure positive, realistic target
        filtered_signal = filtered_signal + closes_mean
        cycle_target = Decimal(str(filtered_signal[-1])) if len(filtered_signal) > 0 else current_close
        cycle_target = max(cycle_target, current_close * Decimal('0.98'))  # Prevent negative or too low
        cycle_target = min(cycle_target, current_close * Decimal('1.02'), min_threshold * Decimal('1.01'))  # Bound upper limit
        cycle_target = cycle_target.quantize(Decimal('0.01'))
    except Exception as e:
        logging.error(f"Error computing cycle target for {timeframe}: {e}")
        cycle_target = current_close
    
    logging.info(f"{timeframe} - MTF Trend: {trend}, Cycle: {cycle_status}, Dominant Freq: {dominant_freq_signed:.6f}, "
                 f"Current Close: {current_close:.2f}, Cycle Target: {cycle_target:.2f}, "
                 f"Volume Ratio: {volume_ratio:.2f}, Volume Confirmed: {volume_confirmed}, Rangehilo: {rangehilo:.2f}")
    print(f"{timeframe} - MTF Trend: {trend}")
    print(f"{timeframe} - Cycle Status: {cycle_status}")
    print(f"{timeframe} - Dominant Frequency: {dominant_freq_signed:.6f}")
    print(f"{timeframe} - Cycle Target: {cycle_target:.2f}")
    print(f"{timeframe} - Volume Ratio: {volume_ratio:.2f}")
    print(f"{timeframe} - Volume Confirmed: {volume_confirmed}")
    print(f"{timeframe} - Rangehilo: {rangehilo:.2f}")
    
    return trend, min_threshold, max_threshold, cycle_status, trend_bullish, trend_bearish, volume_ratio, volume_confirmed, dominant_freq_signed, cycle_target

# FFT-Based Target Price Function
def get_target(closes, n_components, timeframe, min_th, max_th, buy_vol, sell_vol):
    try:
        if len(closes) < 2 or np.any(np.isnan(closes)) or np.any(closes <= 0):
            logging.warning(f"Invalid closes data for FFT analysis in {timeframe}.")
            return (datetime.datetime.now(), Decimal('0'), Decimal('0.0'), Decimal('0.0'), 
                    Decimal('0.0'), Decimal('0'), "Bearish", False, True, 0.0, "TOP")
        
        # Normalize closes by subtracting the mean
        closes_mean = np.mean(closes)
        closes_normalized = closes - closes_mean
        
        window = np.hamming(len(closes_normalized))
        fft_result = fft.rfft(closes_normalized * window)
        frequencies = fft.rfftfreq(len(closes_normalized), d=1.0)
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Select top components, excluding DC (index 0)
        idx = np.argsort(amplitudes)[::-1][1:n_components+1]
        dominant_freq = frequencies[idx[0]] if idx.size > 0 else 0.0
        
        extended_n = len(closes) + 10
        current_close = Decimal(str(closes[-1]))
        
        def reconstruct(components):
            filt_fft = np.zeros_like(fft_result, dtype=complex)
            filt_fft[components] = fft_result[components]
            signal = fft.irfft(filt_fft, n=extended_n)
            # Restore mean and ensure positive values
            signal = signal + closes_mean
            target = Decimal(str(signal[-1])) if len(signal) > 0 else current_close
            # Bound the target to realistic values
            target = max(target, current_close * Decimal('0.98'))
            target = min(target, current_close * Decimal('1.02'), min_th * Decimal('1.01'))
            return target.quantize(Decimal('0.01'))
        
        # Compute targets with different numbers of components
        fastest_target = reconstruct(idx[:3])  # Top 3 components
        average_target = reconstruct(idx[:2])  # Top 2 components
        reversal_target = reconstruct([idx[0]])  # Single dominant component
        
        # Calculate slope for trend confirmation
        last_vals = []
        for i in range(5):
            t = len(closes) + i
            val = sum(
                2 * amplitudes[j] * np.cos(2 * np.pi * frequencies[j] * t + phases[j])
                for j in idx[:3]
            )
            last_vals.append(float(val) + closes_mean)
        slope = np.polyfit(np.array(range(len(last_vals))), np.array(last_vals, dtype=float), deg=1)[0]
        
        is_bullish = False  # Enforced bearish
        is_bearish = True
        market_mood = "Bearish"
        phase_status = "TOP"
        
        # Ensure targets are ordered and realistic
        values = sorted([fastest_target, average_target, reversal_target])
        targets = list(reversed(values))  # Highest to lowest
        if current_close <= targets[0]:
            targets = [
                current_close * Decimal('0.995'),
                current_close * Decimal('0.99'),
                current_close * Decimal('0.985')
            ]
        
        fastest_target, average_target, reversal_target = targets
        stop_loss = current_close * Decimal('1.05')  # Bearish SL
        
        logging.info(f"FFT {timeframe}: Mood: {market_mood}, Phase: {phase_status}, "
                     f"Close: {current_close:.2f}, Fastest: {fastest_target:.2f}, "
                     f"Average: {average_target:.2f}, Reversal: {reversal_target:.2f}, Slope: {slope:.6f}")
        print(f"{timeframe} - FFT Fastest Target: {fastest_target:.2f} USDC")
        print(f"{timeframe} - FFT Average Target: {average_target:.2f} USDC")
        print(f"{timeframe} - FFT Reversal Target: {reversal_target:.2f} USDC")
        
        return (datetime.datetime.now(), current_close, stop_loss, fastest_target, 
                average_target, reversal_target, market_mood, is_bullish, is_bearish, 
                dominant_freq, phase_status)
    except Exception as e:
        logging.error(f"Error in get_target for {timeframe}: {e}")
        return (datetime.datetime.now(), Decimal('0'), Decimal('0'), Decimal('0'), 
                Decimal('0'), Decimal('0'), "Bearish", False, True, 0.0, "TOP")

# Volume Analysis Functions
def calculate_volume(candles):
    try:
        if not candles:
            logging.warning("No candles provided for volume calculation.")
            print("No candles provided for volume calculation.")
            return Decimal('0')
        total_volume = sum(Decimal(str(candle["volume"])) for candle in candles)
        logging.info(f"Total Volume: {total_volume:.2f}")
        print(f"Total Volume: {total_volume:.2f}")
        return total_volume
    except Exception as e:
        logging.error(f"Error in calculate_volume: {e}")
        return Decimal('0')

def calculate_buy_sell_volume(candles, timeframe):
    try:
        if not candles:
            logging.warning(f"No candles provided for volume analysis in {timeframe}.")
            print(f"No candles provided for volume analysis in {timeframe}.")
            return Decimal('0'), Decimal('0'), "BEARISH"
        
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

# Support and Resistance Analysis with Dynamic Tolerance
def calculate_support_resistance(candles, timeframe, rangehilo, current_close):
    try:
        if not candles or len(candles) < 10:
            logging.warning(f"Insufficient candles ({len(candles)}) for support/resistance analysis in {timeframe}.")
            print(f"Insufficient candles ({len(candles)}) for support/resistance analysis in {timeframe}.")
            return [], []
        
        lookback = min(len(candles), LOOKBACK_PERSec) if timeframe in LOOKBACK_PERIODS else 1500
        recent_candles = candles[-lookback:]
        
        min_candle = min(recent_candles, key=lambda x: x['low'])
        max_candle = max(recent_candles, key=lambda x: x['high'])
        
        # Dynamic support/resistance tolerance
        sr_tolerance = Decimal('0.005') * rangehilo if rangehilo > Decimal('0') else Decimal('0.005') * current_close
        sr_tolerance = min(max(sr_tolerance, Decimal('0.01') * rangehilo), Decimal('0.50') * rangehilo)
        
        support_levels = [{"price": Decimal(str(min_candle['low'])), "touches": 1}]
        resistance_levels = [{"price": Decimal(str(max_candle['high'])), "touches": 1}]
        
        logging.info(f"{timeframe} - Support Level: Price: {support_levels[0]['price']:.2f}, Touches: {support_levels[0]['touches']}, Tolerance: {sr_tolerance:.4f}")
        print(f"{timeframe} - Support Level: Price: {support_levels[0]['price']:.2f}, Touches: {support_levels[0]['touches']}")
        logging.info(f"{timeframe} - Resistance Level: Price: {resistance_levels[0]['price']:.2f}, Touches: {resistance_levels[0]['touches']}, Tolerance: {sr_tolerance:.4f}")
        print(f"{timeframe} - Resistance Level: Price: {resistance_levels[0]['price']:.2f}, Touches: {resistance_levels[0]['touches']}")
        
        return support_levels, resistance_levels
    except Exception as e:
        logging.error(f"Error in calculate_support_resistance for {timeframe}: {e}")
        print(f"Error in calculate_support_resistance for {timeframe}: {e}")
        return [], []

# Reversal Detection Function with Dynamic Tolerance
def detect_reversal(candles, timeframe, min_threshold, max_threshold, higher_tf_tops=None):
    try:
        if len(candles) < 3:
            logging.warning(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
            print(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
            return "TOP", 0, Decimal('0'), "TOP", 0, Decimal('0')
        
        lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
        recent_candles = candles[-lookback:]
        rangehilo = max_threshold - min_threshold
        current_close = Decimal(str(recent_candles[-1]['close']))
        
        # Dynamic tolerance based on rangehilo
        tolerance = Decimal('0.10') * rangehilo if rangehilo > Decimal('0') else Decimal('0.10') * current_close
        tolerance = min(max(tolerance, Decimal('0.01') * rangehilo), Decimal('0.50') * rangehilo)
        if timeframe == "1m":
            tolerance *= Decimal('1.5')
        
        lows = np.array([float(c['low']) for c in recent_candles], dtype=np.float64)
        min_idx = np.argmin(lows)
        min_candle = recent_candles[min_idx]
        min_time = min_candle['time']
        closest_min_price = Decimal(str(min_candle['low']))
        min_volume_confirmed = abs(closest_min_price - min_threshold) <= tolerance
        
        highs = np.array([float(c['high']) for c in recent_candles], dtype=np.float64)
        max_idx = np.argmax(highs)
        max_candle = recent_candles[max_idx]
        max_time = max_candle['time']
        closest_max_price = Decimal(str(max_candle['high']))
        max_volume_confirmed = abs(closest_max_price - max_threshold) <= tolerance
        
        short_lookback = min(len(candles), RECENT_LOOKBACK[timeframe])
        recent_short_candles = candles[-short_lookback:]
        short_lows = np.array([float(c['low']) for c in recent_short_candles], dtype=np.float64)
        short_highs = np.array([float(c['high']) for c in recent_short_candles], dtype=np.float64)
        short_min_idx = np.argmin(short_lows) if len(short_lows) > 0 else 0
        short_max_idx = np.argmax(short_highs) if len(short_highs) > 0 else 0
        short_min_time = recent_short_candles[short_min_idx]['time'] if short_min_idx < len(recent_short_candles) else 0
        short_max_time = recent_short_candles[short_max_idx]['time'] if short_max_idx < len(recent_short_candles) else 0
        short_min_price = Decimal(str(recent_short_candles[short_min_idx]['low'])) if short_min_idx < len(recent_short_candles) else closest_min_price
        short_max_price = Decimal(str(recent_short_candles[short_max_idx]['high'])) if short_max_idx < len(recent_short_candles) else closest_max_price
        short_min_confirmed = abs(short_min_price - min_threshold) <= tolerance
        short_max_confirmed = abs(short_max_price - max_threshold) <= tolerance
        
        if timeframe == "1m" and higher_tf_tops:
            for tf, top_confirmed in higher_tf_tops.items():
                if top_confirmed and short_max_confirmed:
                    max_time = max(max_time, short_max_time)
                    closest_max_price = short_max_price
                    max_volume_confirmed = True
                    logging.info(f"{timeframe} - Forced top confirmation due to {tf} top_confirmed=True")
                    print(f"{timeframe} - Forced top confirmation due to {tf} top_confirmed=True")
        
        most_recent_reversal = "TOP"
        most_time = max_time
        most_price = closest_max_price
        
        if short_min_time > short_max_time and short_min_confirmed:
            most_recent_reversal = "DIP"
            most_time = short_min_time
            most_price = short_min_price
        elif min_time > max_time and min_volume_confirmed:
            most_recent_reversal = "DIP"
            most_time = min_time
            most_price = closest_min_price
        elif short_max_confirmed:
            most_recent_reversal = "TOP"
            most_time = short_max_time
            most_price = short_max_price
        elif max_volume_confirmed:
            most_recent_reversal = "TOP"
            most_time = max_time
            most_price = closest_max_price
        
        logging.info(f"{timeframe} - Reversal: {most_recent_reversal} at price {most_price:.2f}, time {datetime.datetime.fromtimestamp(most_time) if most_time else 'N/A'}, "
                     f"Rangehilo: {rangehilo:.2f}, Tolerance: {tolerance:.2f}, "
                     f"Min Price: {closest_min_price:.2f} (Confirmed: {min_volume_confirmed}), "
                     f"Max Price: {closest_max_price:.2f} (Confirmed: {max_volume_confirmed})")
        print(f"{timeframe} - Reversal: {most_recent_reversal} at price {most_price:.2f}, time {datetime.datetime.fromtimestamp(most_time) if most_time else 'N/A'}")
        print(f"{timeframe} - Rangehilo: {rangehilo:.2f}, Tolerance: {tolerance:.2f}")
        
        return most_recent_reversal, min_time, closest_min_price, "TOP", max_time, closest_max_price
    except Exception as e:
        logging.error(f"Error in detect_reversal for {timeframe}: {e}")
        return "TOP", 0, Decimal('0'), "TOP", 0, Decimal('0')

# Threshold Calculation with Range Validation
def calculate_thresholds(candles, timeframe_ranges=None):
    try:
        if not candles:
            logging.warning("No candles provided for threshold calculation.")
            print("No candles provided for threshold calculation.")
            return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
            
        lookback = min(len(candles), LOOKBACK_PERIODS[candles[0]['timeframe']])
        recent_candles = candles[-lookback:]
        min_candle = min(recent_candles, key=lambda x: x['low'])
        max_candle = max(recent_candles, key=lambda x: x['high'])
        
        min_threshold = Decimal(str(min_candle['low']))
        max_threshold = Decimal(str(max_candle['high']))
        rangehilo = max_threshold - min_threshold
        
        timeframe = candles[0]['timeframe']
        if timeframe_ranges and timeframe != "5m":
            next_tf = "3m" if timeframe == "1m" else "5m"
            if next_tf in timeframe_ranges and rangehilo > timeframe_ranges[next_tf]:
                logging.warning(f"{timeframe} rangehilo ({rangehilo:.2f}) exceeds {next_tf} range ({timeframe_ranges[next_tf]:.2f}). Capping thresholds.")
                print(f"{timeframe} rangehilo ({rangehilo:.2f}) exceeds {next_tf} range ({timeframe_ranges[next_tf]:.2f}). Capping thresholds.")
                max_threshold = min_threshold + timeframe_ranges[next_tf]
                rangehilo = max_threshold - min_threshold
        
        middle_threshold = (min_threshold + max_threshold) / Decimal('2')
        
        logging.info(f"{timeframe} - Thresholds: Min: {min_threshold:.2f}, Mid: {middle_threshold:.2f}, Max: {max_threshold:.2f}, Rangehilo: {rangehilo:.2f}")
        print(f"{timeframe} - Minimum Threshold: {min_threshold:.2f}")
        print(f"{timeframe} - Middle Threshold: {middle_threshold:.2f}")
        print(f"{timeframe} - Maximum Threshold: {max_threshold:.2f}")
        print(f"{timeframe} - Rangehilo: {rangehilo:.2f}")
        
        return min_threshold, middle_threshold, max_threshold, rangehilo
    except Exception as e:
        logging.error(f"Error in calculate_thresholds: {e}")
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')

# Utility Functions with Exponential Backoff
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1500):
    try:
        def fetch_candles(timeframe):
            return get_candles(symbol, timeframe, limit)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_candles, timeframes))
        return dict(zip(timeframes, results))
    except Exception as e:
        logging.error(f"Error in fetch_candles_in_parallel: {e}")
        return {}

def get_candles(symbol, timeframe, limit=1500, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = [
                {
                    "time": k[0] / 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "timeframe": timeframe
                }
                for k in klines
            ]
            if candles and (time.time() - candles[-1]["time"]) > 60 * int(timeframe.replace("m", "")) * 2:
                logging.warning(f"Stale data for {timeframe}: latest candle is too old.")
                print(f"Stale data for {timeframe}: latest candle is too old.")
                raise Exception("Stale candle data")
            logging.info(f"Fetched {len(candles)} candles for {timeframe}")
            print(f"Fetched {len(candles)} candles for {timeframe}")
            return candles
        except BinanceAPIException as e:
            retry_after = float(e.response.headers.get('Retry-After', '60')) if e.response else base_delay
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

def get_current_price(symbol=TRADE_SYMBOL, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=symbol)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                logging.info(f"Current {symbol} price: {price:.2f}")
                print(f"Current {symbol} price: {price:.2f}")
                return price
            logging.warning(f"Invalid price {price:.2f} on attempt {attempt + 1}/{retries}")
            print(f"Invalid price {price:.2f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            retry_after = float(e.response.headers.get('Retry-After', '60')) if e.response else base_delay
            logging.error(f"Binance API Error fetching price {symbol} (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Binance API Error fetching price {symbol} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching price {symbol} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching price {symbol} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch valid {symbol} price after {retries} attempts.")
    print(f"Failed to fetch valid {symbol} price after {retries} attempts.")
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
            print(f"Warning: {asset} not found in futures account balances.")
            return Decimal('0.0')
        except Exception as e:
            logging.error(f"Unexpected error fetching {asset} balance (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching {asset} balance (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch {asset} balance after {retries} attempts.")
    print(f"Failed to fetch {asset} balance after {retries} attempts.")
    return Decimal('0.0')

def get_position(symbol=TRADE_SYMBOL, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            positions = client.futures_position_information(symbol=symbol)
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
            retry_after = float(e.response.headers.get('Retry-After', '60')) if e.response else base_delay
            logging.error(f"Error fetching position info {symbol} (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error fetching position info {symbol} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error fetching position {symbol} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error fetching position {symbol} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to fetch position info {symbol} after {retries} attempts.")
    print(f"Failed to fetch position info {symbol} after {retries} attempts.")
    return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0'), "initial_balance": Decimal('0.0'), "sl_price": Decimal('0.0'), "tp_price": Decimal('0.0')}

def check_open_orders(symbol=TRADE_SYMBOL, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            orders = client.futures_get_open_orders(symbol=symbol)
            for order in orders:
                logging.info(f"Open order: {order['type']} at {order['stopPrice']}")
                print(f"Open order: {order['type']} at {order['stopPrice']}")
            return len(orders)
        except BinanceAPIException as e:
            retry_after = float(e.response.headers.get('Retry-After', '60')) if e.response else base_delay
            logging.error(f"Error checking open orders {symbol} (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error checking open orders {symbol} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(retry_after if e.code == -1003 else base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Unexpected error checking open orders {symbol} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Unexpected error checking open orders {symbol} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to check open orders {symbol} after {retries} attempts.")
    print(f"Failed to check open orders {symbol} after {retries} attempts.")
    return 0

# Trading Functions
def calculate_quantity(balance, price):
    try:
        if price <= Decimal('0') or balance < MINIMUM_BALANCE:
            logging.warning(f"Insufficient balance ({balance:.2f}) USDC or invalid price ({price:.2f}).")
            print(f"Insufficient balance ({balance:.2f}) USDC or invalid price ({price:.2f}).")
            return Decimal('0.0')
        quantity = (balance * Decimal(str(LEVERAGE))) / price
        quantity = quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')
        logging.info(f"Calculated quantity: {quantity:.6f} BTC for balance {balance:.2f} USDC at price {price:.2f}")
        print(f"Calculated quantity: {quantity:.6f} BTC for balance {balance:.2f} USDC at price {price:.2f}")
        return quantity
    except Exception as e:
        logging.error(f"Error in calculate_quantity: {e}")
        return Decimal('0.0')

def place_order(signal, quantity, price, initial_balance, symbol=TRADE_SYMBOL, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            if quantity <= Decimal('0'):
                logging.warning(f"Invalid quantity {quantity:.6f}. Skipping order.")
                print(f"Invalid quantity {quantity:.6f}. Skipping order.")
                return None
            position = get_position(symbol=symbol)
            position["initial_balance"] = initial_balance
            price_movement = (PNL_PERCENTAGE * initial_balance) / (quantity * Decimal(str(LEVERAGE)))
            price_movement = price_movement.quantize(Decimal('0.01'))
            
            if signal == "LONG":
                order = client.futures_create_order(
                    symbol=symbol,
                    side="BUY",
                    type="MARKET",
                    quantity=str(quantity)
                )
                tp_price = (price + price_movement).quantize(Decimal('0.01'))
                sl_price = (price - price_movement).quantize(Decimal('0.01'))
                
                client.futures_create_order(
                    symbol=symbol,
                    side="SELL",
                    type="STOP_MARKET",
                    stopPrice=str(sl_price),
                    closePosition=True
                )
                
                client.futures_create_order(
                    symbol=symbol,
                    side="SELL",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=str(tp_price),
                    closePosition=True
                )
                
                position.update({
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "side": "LONG",
                    "quantity": quantity,
                    "entry_price": price
                })
                message = (
                    f"\n*Trade Signal: LONG*\n"
                    f"Symbol: {symbol}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Quantity: {quantity:.6f} BTC\n"
                    f"Entry Price: {price:.2f} USDC\n"
                    f"Initial Balance: {initial_balance:.2f} USDC\n"
                    f"Stop-Loss: {sl_price:.2f} (-5% ROI)\n"
                    f"Take-Profit: {tp_price:.2f} (+5% ROI)\n"
                )
                asyncio.run_coroutine_threadsafe(send_telegram_message(message), telegram_loop)
                logging.info(f"Placed LONG order: {quantity:.6f} BTC at market price {price:.2f}")
                print(f"\n=== TRADE ENTERED ===")
                print(f"Side: LONG")
                print(f"Quantity: {quantity:.6f} BTC")
                print(f"Entry Price: {price:.2f} USDC")
                print(f"Initial USDC: {initial_balance:.2f}")
                print(f"Stop-Loss: {sl_price:.2f} (-5% ROI)")
                print(f"Take-Profit: {tp_price:.2f} (+5% ROI)")
                print(f"===================\n")
            elif signal == "SHORT":
                order = client.futures_create_order(
                    symbol=symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=str(quantity)
                )
                tp_price = (price - price_movement).quantize(Decimal('0.01'))
                sl_price = (price + price_movement).quantize(Decimal('0.01'))
                
                client.futures_create_order(
                    symbol=symbol,
                    side="BUY",
                    type="STOP_MARKET",
                    stopPrice=str(sl_price),
                    closePosition=True
                )
                
                client.futures_create_order(
                    symbol=symbol,
                    side="BUY",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=str(tp_price),
                    closePosition=True
                )
                
                position.update({
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "side": "SHORT",
                    "quantity": -quantity,
                    "entry_price": price
                })
                message = (
                    f"\n*Trade Signal: SHORT*\n"
                    f"Symbol: {symbol}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Quantity: {quantity:.6f} BTC\n"
                    f"Entry Price: {price:.2f} USDC\n"
                    f"Initial Balance: {initial_balance:.2f} USDC\n"
                    f"Stop-Loss: {sl_price:.2f} (-5% ROI)\n"
                    f"Take-Profit: {tp_price:.2f} (+5% ROI)\n"
                )
                asyncio.run_coroutine_threadsafe(send_telegram_message(message), telegram_loop)
                logging.info(f"Placed SHORT order: {quantity:.6f} BTC at market price {price:.2f}")
                print(f"\n=== TRADE ENTERED ===")
                print(f"Side: SHORT")
                print(f"Quantity: {quantity:.6f} BTC")
                print(f"Entry Price: {price:.2f} USDC")
                print(f"Initial USDC: {initial_balance:.2f}")
                print(f"Stop-Loss: {sl_price:.2f} (-5% ROI)")
                print(f"Take-Profit: {tp_price:.2f} (+5% ROI)")
                print(f"===================\n")
                
            if check_open_orders() > 2:  # Expecting SL and TP orders
                logging.warning(f"Unexpected open orders detected after placing {signal} order.")
                print(f"Warning: Unexpected open orders detected after placing {signal} order.")
            return position
        except Exception as e:
            logging.error(f"Error placing order {signal} (attempt {attempt + 1}/{retries}): {e}")
            print(f"Error placing order {signal} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    logging.error(f"Failed to place order {signal} after {retries} attempts.")
    print(f"Failed to place order {signal} after {retries} attempts.")
    return None

def close_position(position, price, signal=None, retries=5, base_delay=5):
    try:
        if position["side"] == "NONE" or position["quantity"] == Decimal('0'):
            logging.info("No position to close.")
            print("No position to close.")
            return
        for attempt in range(retries):
            try:
                # Cancel existing orders
                client.futures_cancel_all_open_orders(symbol=TRADE_SYMBOL)
                
                quantity = abs(position["quantity"]).quantize(QUANTITY_PRECISION)
                side = "BUY" if position["side"] == "SHORT" else "SELL"
                order = client.futures_create_order(
                    symbol=TRADE_SYMBOL,
                    side=side,
                    type="MARKET",
                    quantity=str(quantity)
                )
                message = (
                    f"\n*Position Closed*\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Side: {position['side']}\n"
                    f"Quantity: {quantity:.6f} BTC\n"
                    f"Exit Price: {price:.2f} USDC\n"
                    f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
                )
                asyncio.run_coroutine_threadsafe(send_telegram_message(message), telegram_loop)
                logging.info(f"Closed {position['side']} position: {quantity:.6f} BTC at {price:.2f}")
                print(f"Closed {position['side']} position: {quantity:.6f} BTC at {price:.2f}")
                return
            except Exception as e:
                logging.error(f"Error closing position {position['side']} (attempt {attempt + 1}/{retries}): {e}")
                print(f"Error closing position {position['side']} (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        logging.error(f"Failed to close position {position['side']} after {retries} attempts.")
        print(f"Failed to close position {position['side']} after {retries} attempts.")
    except Exception as e:
        logging.error(f"Error in close_position: {e}")
        print(f"Error in close_position: {e}")

# Main Analysis Loop
def main():
    try:
        logging.info("Futures Analysis API Initialized!")
        print("Starting main loop...")
        
        while True:
            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Current Time: {time_str}")
            print(f"\n=== Analysis at {time_str} ===")
            
            current_price = get_current_price()
            if current_price <= Decimal('0'):
                logging.warning(f"Failed to fetch price. Retrying in 60 seconds.")
                print(f"Warning: Failed to fetch price. Retrying in 60 seconds.")
                time.sleep(60)
                continue
            
            print(f"Current Price: {current_price:.2f} USDC")
            
            candle_map = fetch_candles_in_parallel(TIMEFRAMES)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candle data available.")
                print("No candle data available.")
                time.sleep(60)
                continue
            
            usdc_balance = get_balance()
            position = get_position()
            
            # Initialize trade conditions
            conditions_long = {f"momentum_long_{tf}": False for tf in TIMEFRAMES}
            conditions_long.update({f"fft_long_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"volume_long_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"dip_confirmed_{tf}": False for tf in TIMEFRAMES})
            conditions_long.update({f"below_middle_{tf}": False for tf in TIMEFRAMES})
            
            conditions_short = {f"momentum_short_{tf}": False for tf in TIMEFRAMES}
            conditions_short.update({f"fft_short_{tf}": True for tf in TIMEFRAMES})  # Forced bearish
            conditions_short.update({f"volume_short_{tf}": False for tf in TIMEFRAMES})
            conditions_short.update({f"top_confirmed_{tf}": False for tf in TIMEFRAMES})
            conditions_short.update({f"above_middle_{tf}": False for tf in TIMEFRAMES})
            
            volume_data = {tf: {"buy": Decimal('0'), "sell": Decimal('0'), "mood": "BEARISH"} for tf in TIMEFRAMES}
            for tf in candle_map:
                if candle_map.get(tf):
                    volume_data[tf]["buy"], volume_data[tf]["sell"], volume_data[tf]["mood"] = calculate_buy_sell_volume(candle_map[tf], tf)
            
            fft_results = {}
            min_distances = []
            recent_extremes = []
            analysis_details = {tf: {} for tf in TIMEFRAMES}
            timeframe_ranges = {}
            
            for tf in TIMEFRAMES:
                if candle_map.get(tf):
                    min_th, mid_th, max_th, range_hilo = calculate_thresholds(candle_map[tf], timeframe_ranges)
                    timeframe_ranges[tf] = range_hilo
                else:
                    timeframe_ranges[tf] = Decimal('0')
            
            higher_tf_tops = {}
            for tf in ["3m", "5m"]:
                if not candle_map.get(tf):
                    continue
                candles_tf = candle_map[tf]
                min_th, mid_th, max_th, _ = calculate_thresholds(candles_tf, timeframe_ranges)
                reversal_type, _, _, _, max_time, _ = detect_reversal(candles_tf, tf, min_th, max_th)
                higher_tf_tops[tf] = (reversal_type == "TOP")
            
            for tf in TIMEFRAMES:
                print(f"\n--- {tf} Analysis ---")
                if not candle_map.get(tf):
                    logging.info(f"No data for {tf}. Skipping.")
                    fft_results[tf] = {
                        "time": time_str,
                        "current_price": Decimal('0'),
                        "fastest_target": Decimal('0'),
                        "average_target": Decimal('0'),
                        "reversal_target": Decimal('0'),
                        "mood": "Bearish",
                        "min_th": Decimal('0'),
                        "mid_th": Decimal('0'),
                        "max_th": Decimal('0'),
                        "fft_phase": "TOP",
                        "freq": 0.0
                    }
                    continue
                
                candles_tf = candle_map[tf]
                closes = np.array([c["close"] for c in candles_tf if c["close"] > 0], dtype=np.float64)
                
                min_th, mid_th, max_th, range_hilo = calculate_thresholds(candles_tf, timeframe_ranges)
                
                reversal_type, min_t, min_p, top_type, max_t, max_p = detect_reversal(candles_tf, tf, min_th, max_th, higher_tf_tops if tf == "1m" else None)
                
                current_close = Decimal(str(closes[-1])) if len(closes) > 0 else current_price
                support, resistance = calculate_support_resistance(candles_tf, tf, range_hilo, current_close)
                
                buy_vol = volume_data[tf]["buy"]
                sell_vol = volume_data[tf]["sell"]
                vol_mood = volume_data[tf]["mood"]
                
                conditions_long[f"volume_long_{tf}"] = buy_vol > sell_vol
                conditions_short[f"volume_short_{tf}"] = sell_vol >= buy_vol
                
                conditions_long[f"dip_confirmed_{tf}"] = reversal_type == "DIP" and min_t >= max_t
                conditions_short[f"top_confirmed_{tf}"] = reversal_type == "TOP" and max_t >= min_t
                
                conditions_long[f"below_middle_{tf}"] = current_close < mid_th
                conditions_short[f"above_middle_{tf}"] = current_close > mid_th
                
                trend, min_th, max_th, cycle_state, trend_bull, trend_bear, vol_ratio, vol_conf, freq_mood, cycle_target = calculate_mtf_trend(candles_tf, tf, min_th, max_th, buy_vol, sell_vol)
                
                analysis_details[tf] = {
                    "trend": trend,
                    "cycle_state": cycle_state,
                    "freq_mood": freq_mood,
                    "cycle_target": cycle_target,
                    "vol_ratio": vol_ratio,
                    "vol_conf": vol_conf,
                    "min_th": min_th,
                    "mid_th": mid_th,
                    "max_th": max_th,
                    "range_hilo": range_hilo,
                    "reversal_type": reversal_type,
                    "reversal_time": str(datetime.datetime.fromtimestamp(min_t if reversal_type == "DIP" else max_t)) if (min_t or max_t) else "N/A",
                    "reversal_price": min_p if reversal_type == "DIP" else max_p,
                    "support": support,
                    "resistance": resistance,
                    "buy_vol": buy_vol,
                    "sell_vol": sell_vol,
                    "vol_mood": vol_mood
                }
                
                lookback = min(len(candles_tf), LOOKBACK_PERIODS[tf])
                recent_candles = candles_tf[-lookback:]
                min_c = min(recent_candles, key=lambda x: x['low'])
                max_c = max(recent_candles, key=lambda x: x['high'])
                recent_extremes.append({
                    "tf": tf,
                    "lowest": Decimal(str(min_c['low'])),
                    "low_time": min_c['time'],
                    "highest": Decimal(str(max_c['high'])),
                    "high_time": max_c['time']
                })
                
                if len(closes) >= 2:
                    fft_data = get_target(closes, n_components=5, timeframe=tf, min_th=min_th, max_th=max_th, buy_vol=buy_vol, sell_vol=sell_vol)
                    fft_results[tf] = {
                        "time": fft_data[0].strftime('%Y-%m-%d %H:%M:%S'),
                        "current_price": fft_data[1],
                        "fastest_target": fft_data[3],
                        "average_target": fft_data[4],
                        "reversal_target": fft_data[5],
                        "mood": fft_data[6],
                        "min_th": min_th,
                        "mid_th": mid_th,
                        "max_th": max_th,
                        "fft_phase": fft_data[10],
                        "freq": fft_data[9]
                    }
                    print(f"{tf} - FFT Fastest: {fft_data[3]:.2f} USDC")
                    print(f"{tf} - FFT Average: {fft_data[4]:.2f} USDC")
                    print(f"{tf} - FFT Reversal: {fft_data[5]:.2f} USDC")
                    print(f"{tf} - FFT Mood: {fft_data[6]}")
                    print(f"{tf} - FFT Phase: {fft_data[10]}")
                
                if len(closes) >= 14:
                    momentum = talib.MOM(closes, timeperiod=5)
                    if len(momentum) > 0 and not np.isnan(momentum[-1]):
                        conditions_long[f"momentum_long_{tf}"] = Decimal(str(momentum[-1])) >= Decimal('0')
                        conditions_short[f"momentum_short_{tf}"] = not conditions_long[f"momentum_long_{tf}"]
                
                min_distances.append({
                    "tf": tf,
                    "min_dist": abs(current_close - min_th),
                    "max_dist": abs(current_close - max_th),
                    "min_th": float(min_th),
                    "max_th": float(max_th)
                })
            
            print(f"\n--- MTF Min/Max Comparison ---")
            closest_min = min(min_distances, key=lambda x: x["min_dist"]) if min_distances else {"tf": "N/A", "min_dist": Decimal('0'), "min_th": 0.0}
            closest_max = min(min_distances, key=lambda x: x["max_dist"]) if min_distances else {"tf": "N/A", "max_dist": Decimal('0'), "max_th": 0.0}
            print(f"Closest Min TF: {closest_min['tf']} (Dist: {closest_min['min_dist']:.2f}, Min: {closest_min['min_th']:.2f})")
            print(f"Closest Max TF: {closest_max['tf']} (Dist: {closest_max['max_dist']:.2f}, Max: {closest_max['max_th']:.2f})")
            logging.info(f"Closest Min TF: {closest_min['tf']} (Dist: {closest_min['min_dist']:.2f})")
            logging.info(f"Closest Max TF: {closest_max['tf']} (Dist: {closest_max['max_dist']:.2f})")

            most_recent = max(recent_extremes, key=lambda x: max(x["low_time"], x["high_time"])) if recent_extremes else {"tf": "N/A", "low_time": 0, "high_time": 0, "lowest": Decimal('0'), "highest": Decimal('0')}
            most_recent_t = max(most_recent['low_time'], most_recent['high_time'])
            most_type = "LOW" if most_recent['low_time'] >= most_recent['high_time'] else "HIGH"
            most_price = most_recent['lowest'] if most_type == "LOW" else most_recent['highest']
            print(f"\nRecent Extremes:")
            print(f"Most Recent: {most_type} in {most_recent['tf']} at {most_price:.2f} (Time: {datetime.datetime.fromtimestamp(most_recent_t) if most_recent_t else 'N/A'})")
            logging.info(f"Recent Extreme: {most_type} in {most_recent['tf']} @ {most_price:.2f} (Time: {datetime.datetime.fromtimestamp(most_recent_t) if most_recent_t else 'N/A'})")

            condition_pairs = [
                ("momentum_long_1m", "momentum_short_1m"),
                ("momentum_long_3m", "momentum_short_3m"),
                ("momentum_long_5m", "momentum_short_5m"),
                ("fft_long_1m", "fft_short_1m"),
                ("fft_long_3m", "fft_short_3m"),
                ("fft_long_5m", "fft_short_5m"),
                ("volume_long_1m", "volume_short_1m"),
                ("volume_long_3m", "volume_short_3m"),
                ("volume_long_5m", "volume_short_5m"),
                ("dip_confirmed_1m", "top_confirmed_1m"),
                ("dip_confirmed_3m", "top_confirmed_3m"),
                ("dip_confirmed_5m", "top_confirmed_5m"),
                ("below_middle_1m", "above_middle_1m"),
                ("below_middle_3m", "above_middle_3m"),
                ("below_middle_5m", "above_middle_5m")
            ]
            
            logging.info("Condition Pairs:")
            print("\nCondition Pairs:")
            symmetry_valid = True
            for long_cond, short_cond in condition_pairs:
                if conditions_long[long_cond] == conditions_short[short_cond]:
                    conditions_long[long_cond] = False
                    conditions_short[short_cond] = True
                    logging.info(f"Symmetry enforced: {long_cond} = False, {short_cond} = True")
                    print(f"Symmetry enforced: {long_cond} = False, {short_cond} = True")
                    symmetry_valid = False
                logging.info(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]}")
                print(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]}")

            long_signal = all(conditions_long.values()) and symmetry_valid
            short_signal = all(conditions_short.values()) and symmetry_valid
            
            logging.info("Trade Signals:")
            print("\nTrade Signals:")
            logging.info(f"Long Signal: {'Active' if long_signal else 'Inactive'}")
            print(f"Long Signal: {'Active' if long_signal else 'Inactive'}")
            logging.info(f"Short Signal: {'Active' if short_signal else 'Inactive'}")
            print(f"Short Signal: {'Active' if short_signal else 'Inactive'}")
            
            if long_signal and short_signal:
                logging.warning("Conflict: Both signals active. Setting to NO_SIGNAL.")
                print("Warning: Both signals active. Setting to NO_SIGNAL.")
                long_signal = False
                short_signal = False
            
            signal = "LONG" if long_signal else "SHORT" if short_signal else "NO_SIGNAL"
            if signal != "NO_SIGNAL":
                message = (
                    f"\n*Signal: {signal}*\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {time_str}\n"
                    f"Price: {current_price:.2f} USDC\n"
                    f"\nDetails:\n"
                )
                for tf in TIMEFRAMES:
                    details = analysis_details.get(tf, {})
                    fft_data = fft_results.get(tf, {})
                    rev_time = details.get('reversal_time', 'N/A')
                    message += (
                        f"\n*{tf} TF:*\n"
                        f"Trend: {details.get('trend', 'N/A')}\n"
                        f"Cycle: {details.get('cycle_state', 'N/A')}\n"
                        f"Freq: {details.get('freq_mood', 0.0):.6f}\n"
                        f"Cycle Target: {details.get('cycle_target', Decimal('0')):.2f} USDC\n"
                        f"Vol Ratio: {details.get('vol_ratio', Decimal('0')):.2f}\n"
                        f"Vol Conf: {details.get('vol_conf', False)}\n"
                        f"Min Th: {details.get('min_th', Decimal('0')):.2f} USDC\n"
                        f"Mid Th: {details.get('mid_th', Decimal('0')):.2f} USDC\n"
                        f"Max Th: {details.get('max_th', Decimal('0')):.2f} USDC\n"
                        f"Rangehilo: {details.get('range_hilo', Decimal('0')):.2f}\n"
                        f"Reversal: {details.get('reversal_type', 'N/A')} at {details.get('reversal_price', Decimal('0')):.2f}, time {rev_time}\n"
                        f"Support: {details.get('support', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Touches: {details.get('support', [{}])[0].get('touches', 0)}\n"
                        f"Resistance: {details.get('resistance', [{}])[0].get('price', Decimal('0')):.2f} USDC\n"
                        f"Touches: {details.get('resistance', [{}])[0].get('touches', 0)}\n"
                        f"Buy Vol: {details.get('buy_vol', Decimal('0')):.2f}\n"
                        f"Sell Vol: {details.get('sell_vol', Decimal('0')):.2f}\n"
                        f"Vol Mood: {details.get('vol_mood', 'N/A')}\n"
                        f"FFT Phase: {fft_data.get('fft_phase', 'N/A')}\n"
                        f"Fastest: {fft_data.get('fastest_target', Decimal('0')):.2f} USDC\n"
                        f"Average: {fft_data.get('average_target', Decimal('0')):.2f} USDC\n"
                        f"Reversal: {fft_data.get('reversal_target', Decimal('0')):.2f} USDC\n"
                    )
                asyncio.run_coroutine_threadsafe(send_telegram_message(message), telegram_loop)
            
            logging.info("\nLong Conditions:")
            print("\nLong Conditions:")
            for cond, val in conditions_long.items():
                logging.info(f"{cond}: {val}")
                print(f"{cond}: {val}")
            logging.info("\nShort Conditions:")
            print("\nShort Conditions:")
            for cond, val in conditions_short.items():
                logging.info(f"{cond}: {val}")
                print(f"{cond}: {val}")
            
            long_true = sum(val for val in conditions_long.values())
            long_false = len(conditions_long) - long_true
            short_true = sum(val for val in conditions_short.values())
            short_false = len(conditions_short) - short_true
            
            logging.info(f"\nSummary: Long True={long_true}, False={long_false}")
            print(f"\nSummary: Long True={long_true}, False={long_false}")
            logging.info(f"Summary: Short True={short_true}, False={short_false}")
            print(f"Summary: Short True={short_true}, False={short_false}")
            
            if position["side"] == "NONE" and signal in ["LONG", "SHORT"] and usdc_balance >= MINIMUM_BALANCE:
                quantity = calculate_quantity(usdc_balance, current_price)
                if quantity > Decimal('0'):
                    new_position = place_order(signal, quantity, current_price, usdc_balance)
                    if new_position:
                        position = new_position
            elif position["side"] != "NONE" and signal == ("SHORT" if position["side"] == "LONG" else "LONG"):
                close_position(position, current_price)
                position = get_position()
                if signal != "NO_SIGNAL" and usdc_balance >= MINIMUM_BALANCE:
                    quantity = calculate_quantity(usdc_balance, current_price)
                    if quantity > Decimal('0'):
                        new_position = place_order(signal, quantity, current_price, usdc_balance)
                        if new_position:
                            position = new_position
            
            gc.collect()
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Bot stopped.")
        print("Bot stopped.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        message = f"*Error*\nSymbol: {TRADE_SYMBOL}\nTime: {time_str}\nError: {str(e)}"
        asyncio.run_coroutine_threadsafe(send_telegram_message(message), telegram_loop)
        exit(1)

if __name__ == "__main__":
    main()
