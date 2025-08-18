import asyncio
import datetime
import time
import concurrent.futures
import numpy as np
from decimal import Decimal, getcontext
import logging
import traceback
from telegram.ext import Application
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
import talib
import sys
import scipy.signal as signal
import pandas as pd

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
TIMEFRAMES = ["1m", "3m", "5m"]  # Timeframes for analysis
LOOKBACK_PERIODS = {"1m": 1500, "3m": 1500, "5m": 1500}
VOLUME_CONFIRMATION_RATIO = Decimal('1.5')  # Volume increase ratio for LONG
VOLUME_EXHAUSTION_RATIO = Decimal('0.5')  # Volume decrease ratio for SHORT
SUPPORT_RESISTANCE_TOLERANCE = Decimal('0.005')  # 0.5% tolerance for support/resistance
API_TIMEOUT = 60  # Timeout for Binance API requests
TIMEFRAME_INTERVALS = {"1m": 60, "3m": 180, "5m": 300}  # Timeframe intervals in seconds
TOLERANCE_FACTORS = {"1m": Decimal('0.10'), "3m": Decimal('0.08'), "5m": Decimal('0.06')}  # Dynamic tolerance factors
VOLUME_LOOKBACK = 5  # Lookback for volume trend analysis at reversal
ML_LOOKBACK = 200  # Lookback period for ML forecasting
CYCLE_TARGET_PERCENTAGE = Decimal('0.02')  # 2% target for cycle completion
ATR_PERIOD = 14  # Period for ATR calculation
ATR_MULTIPLIER_SL = Decimal('1.5')  # ATR multiplier for stop-loss
ATR_MULTIPLIER_TP = Decimal('1.5')  # ATR multiplier for take-profit
ATR_THRESHOLD = Decimal('0.01')  # Minimum ATR percentage for trade signals
MOMENTUM_PERIOD = 14  # Period for momentum calculation
MOMENTUM_LOOKBACK = 3  # Number of recent momentum values to check for trend

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
        sys.exit(1)
    api_key, api_secret, telegram_token, telegram_chat_id = lines
    if not all([api_key, api_secret, telegram_token, telegram_chat_id]):
        logging.error("One or more credentials in credentials.txt are empty.")
        print("Error: One or more credentials in credentials.txt are empty.")
        sys.exit(1)
except FileNotFoundError:
    logging.error("credentials.txt not found.")
    print("Error: credentials.txt not found.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Unexpected error reading credentials.txt: {e}")
    print(f"Unexpected error reading credentials.txt: {e}")
    sys.exit(1)

# Initialize Binance client
try:
    client = BinanceClient(api_key, api_secret, requests_params={"timeout": API_TIMEOUT})
    client.API_URL = 'https://fapi.binance.com'  # Futures API endpoint
except Exception as e:
    logging.error(f"Failed to initialize Binance client: {e}")
    print(f"Failed to initialize Binance client: {e}")
    sys.exit(1)

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
        sys.exit(1)
    logging.info("Telegram bot initialized successfully.")
    print("Telegram bot initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Telegram bot: {e}")
    print(f"Failed to initialize Telegram bot: {e}")
    sys.exit(1)

# Set leverage
try:
    client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
    logging.info(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
    print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
except BinanceAPIException as e:
    logging.error(f"Error setting leverage: {e.message}")
    print(f"Error setting leverage: {e.message}")
    sys.exit(1)

async def send_telegram_message(message, retries=3, base_delay=5):
    escaped_message = (
        message.replace('_', r'\_')
               .replace('*', r'\*')
               .replace('[', r'\[')
               .replace(']', r'\]')
               .replace('(', r'\(')
               .replace(')', r'\)')
               .replace('~', r'\~')
               .replace('`', r'\`')
               .replace('>', r'\>')
               .replace('#', r'\#')
               .replace('+', r'\+')
               .replace('-', r'\-')
               .replace('=', r'\=')
               .replace('|', r'\|')
               .replace('{', r'\{')
               .replace('}', r'\}')
               .replace('.', r'\.')
               .replace('!', r'\!')
    )
    
    for attempt in range(retries):
        try:
            await telegram_app.bot.send_message(chat_id=telegram_chat_id, text=escaped_message, parse_mode='MarkdownV2')
            logging.info(f"Telegram message sent successfully: {message[:100]}...")
            print(f"Telegram message sent successfully: {message[:100]}...")
            return True
        except Exception as e:
            if "Can't parse entities" in str(e):
                try:
                    await telegram_app.bot.send_message(chat_id=telegram_chat_id, text=message, parse_mode=None)
                    logging.info(f"Telegram message sent successfully (plain text fallback): {message[:100]}...")
                    print(f"Telegram message sent successfully (plain text fallback): {message[:100]}...")
                    return True
                except Exception as fallback_e:
                    error_msg = f"Failed to send Telegram message (plain text fallback, attempt {attempt + 1}/{retries}): {str(fallback_e)}\n{traceback.format_exc()}"
                    logging.error(error_msg)
                    print(error_msg)
            else:
                error_msg = f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {str(e)}\n{traceback.format_exc()}"
                logging.error(error_msg)
                print(error_msg)
            if "Forbidden" in str(e):
                logging.error(f"Cannot send Telegram message: Invalid chat ID: {telegram_chat_id}")
                print(f"Error: Cannot send Telegram message. Invalid chat ID: {telegram_chat_id}")
                return False
            delay = base_delay * (2 ** attempt)
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    
    logging.error(f"Failed to send Telegram message after {retries} attempts: {message[:100]}...")
    print(f"Failed to send Telegram message after {retries} attempts: {message[:100]}...")
    return False

def clean_data(data, min_val=1e-10):
    """Replace NaN, Inf, and zero values with a small positive value"""
    if isinstance(data, list):
        data = np.array(data)
    
    # Replace NaN and Inf with a small positive value
    data = np.nan_to_num(data, nan=min_val, posinf=min_val, neginf=-min_val)
    
    # Replace zeros with a small positive value
    data = np.where(np.abs(data) < min_val, min_val * np.sign(data) if data.any() != 0 else min_val, data)
    
    return data

def compute_Hc(series, kind="random_walk", simplified=True):
    """
    Compute the Hurst exponent H and the constant c using the R/S method.
    
    Parameters:
    series (array-like): Time series data
    kind (str): Type of series - "price", "random_walk", or "change"
    simplified (bool): If True, use simplified method
    
    Returns:
    H (float): Hurst exponent
    c (float): Constant
    data (array): Data points used in calculation
    """
    if kind == "price":
        # Convert price series to random walk
        series = np.log(series)
        series = np.diff(series)
    elif kind == "change":
        # Already a series of changes
        pass
    else:  # random_walk
        # Assume it's already a random walk
        pass
    
    # Clean the data
    series = clean_data(series)
    
    # Calculate the range of cumulative deviations
    n = len(series)
    if n < 2:
        return 0.5, 0.0, []
    
    mean_series = np.mean(series)
    cum_dev = series - mean_series
    cum_dev = np.cumsum(cum_dev)
    r = np.max(cum_dev) - np.min(cum_dev)
    
    # Calculate the standard deviation
    s = np.std(series)
    
    if s == 0:
        return 0.5, 0.0, []
    
    # Calculate the R/S ratio
    rs = r / s
    
    # For simplified method, return H = log(R/S) / log(n/2)
    if simplified:
        if n/2 <= 0:
            return 0.5, 0.0, []
        H = np.log(rs) / np.log(n/2)
        return H, 0.0, [(n/2, rs)]
    
    # For full method, we would calculate R/S for multiple window sizes
    # and fit a line to log(R/S) vs log(window_size)
    # This is a simplified version that just returns the basic calculation
    return H, 0.0, [(n/2, rs)]

def analyze_frequency_spectrum(candles, timeframe, cycle_status, min_threshold, max_threshold, current_close):
    """
    Analyze the frequency spectrum to determine the current stage in the stationary circuit of energy flow.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - cycle_status: Current cycle status ("Up" or "Down")
    - min_threshold: Minimum price threshold
    - max_threshold: Maximum price threshold
    - current_close: Current closing price
    
    Returns:
    - dominant_freq: The predominant frequency
    - spectral_power: Power of the predominant frequency
    - intensity: Normalized intensity (0-1)
    - freq_range: Frequency range
    - degree: Phase degree (0-1)
    - stage: Current stage in the cycle ("Early", "Middle", "Late")
    - cycle_direction: Determined cycle direction ("Up" or "Down")
    - top_frequencies: List of top frequencies
    - ht_sine: Sine wave generated via Hilbert Transform
    - ht_sine_cycle_type: "Up" or "Down" based on HT_SINE analysis
    """
    closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
    volumes = np.array([float(c['volume']) for c in candles], dtype=np.float64)
    n = len(closes)
    
    if n < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0, "N/A", "N/A", [], np.zeros(n), "N/A"
    
    # Clean the data
    closes = clean_data(closes)
    volumes = clean_data(volumes)
    
    # Center the data
    mean_close = np.mean(closes)
    mean_volume = np.mean(volumes)
    centered_closes = closes - mean_close
    centered_volumes = volumes - mean_volume
    
    # Perform FFT
    fft_result = fft(centered_closes)
    freqs = np.fft.fftfreq(n)
    
    # Calculate power spectrum (magnitude squared)
    power = np.abs(fft_result) ** 2
    
    # Skip DC component (index 0)
    indices = np.arange(1, n)
    
    # Get the middle threshold frequency (equilibrium point)
    middle_threshold = (min_threshold + max_threshold) / 2
    middle_idx = np.argmin(np.abs(closes - float(middle_threshold)))
    middle_freq = freqs[middle_idx]
    
    # Get min and max threshold frequencies
    min_idx = np.argmin(closes)
    max_idx = np.argmax(closes)
    min_freq = freqs[min_idx]
    max_freq = freqs[max_idx]
    
    # Determine if we're in an up or down cycle
    cycle_direction = "Up" if cycle_status == "Up" else "Down"
    
    # Get 16 frequencies for the current cycle direction
    if cycle_direction == "Up":
        # For up cycle, get frequencies from the lower half (negative frequencies)
        up_indices = indices[freqs[indices] < 0]
        if len(up_indices) > 16:
            # Sort by power and get top 16
            sorted_indices = up_indices[np.argsort(power[up_indices])[-16:]]
        else:
            sorted_indices = up_indices
        cycle_frequencies = list(freqs[sorted_indices])
    else:
        # For down cycle, get frequencies from the upper half (positive frequencies)
        down_indices = indices[freqs[indices] > 0]
        if len(down_indices) > 16:
            # Sort by power and get top 16
            sorted_indices = down_indices[np.argsort(power[down_indices])[-16:]]
        else:
            sorted_indices = down_indices
        cycle_frequencies = list(freqs[sorted_indices])
    
    # Combine all frequencies: 16 cycle frequencies + 3 threshold frequencies
    all_freqs = cycle_frequencies + [min_freq, middle_freq, max_freq]
    
    # Calculate weighted average based on power to determine dominant frequency
    weights = []
    for f in all_freqs:
        if f in freqs:
            idx = np.where(freqs == f)[0][0]
            weights.append(power[idx])
        else:
            weights.append(0.0)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
    dominant_freq = np.sum(np.array(all_freqs) * weights)
    
    # Find the index of the frequency closest to the dominant_freq
    idx = np.argmin(np.abs(freqs - dominant_freq))
    spectral_power = power[idx]
    
    # Calculate total power (excluding DC component)
    total_power = np.sum(power[indices])
    
    # Calculate intensity (normalized power)
    if total_power > 0:
        # Normalized power (0-1 scale)
        intensity = spectral_power / total_power
    else:
        intensity = 0.0
    
    # Calculate frequency range
    freq_range = np.max(freqs[indices]) - np.min(freqs[indices]) if len(indices) > 0 else 0.0
    
    # Calculate degree (phase of the predominant frequency)
    phase = np.angle(fft_result[idx])
    degree = (phase + np.pi) / (2 * np.pi)  # maps from [-pi, pi] to [0, 1]
    
    # Determine the stage in the stationary circuit
    cycle_position = (current_close - min_threshold) / (max_threshold - min_threshold) if max_threshold != min_threshold else 0.5
    
    if cycle_direction == "Up":
        if cycle_position < 0.33:
            stage = "Early"
        elif cycle_position < 0.66:
            stage = "Middle"
        else:
            stage = "Late"
    else:  # Down
        if cycle_position > 0.66:
            stage = "Early"
        elif cycle_position > 0.33:
            stage = "Middle"
        else:
            stage = "Late"
    
    # Generate sine wave using TALib's HT_SINE
    try:
        # Use TALib's HT_SINE function on the close prices
        ht_sine, _ = talib.HT_SINE(closes)
        
        # Handle any NaN values that might result from HT_SINE
        ht_sine = np.nan_to_num(ht_sine, nan=0.0)
        
        # Determine HT_SINE cycle type based on the slope of the last few values
        if len(ht_sine) >= 3:
            # Calculate the slope of the last 3 values
            recent_sine = ht_sine[-3:]
            x = np.array([0, 1, 2])
            y = recent_sine
            
            # Fit a line to the recent sine values
            slope = np.polyfit(x, y, 1)[0]
            
            # Determine cycle type based on slope
            ht_sine_cycle_type = "Up" if slope > 0 else "Down"
        else:
            # Not enough data points, use current cycle direction
            ht_sine_cycle_type = cycle_direction
    except Exception as e:
        logging.error(f"Error generating HT sine wave for {timeframe}: {e}")
        ht_sine = np.zeros(n)
        ht_sine_cycle_type = "N/A"
    
    # Store top frequencies for analysis
    sorted_indices = indices[np.argsort(power[indices])[-5:]]
    top_frequencies = [(freqs[i], power[i]) for i in sorted_indices]
    
    return dominant_freq, spectral_power, intensity, freq_range, degree, stage, cycle_direction, top_frequencies, ht_sine, ht_sine_cycle_type

def calculate_atr(candles, timeframe, period=ATR_PERIOD):
    if len(candles) < period:
        logging.warning(f"Insufficient candles ({len(candles)}) for ATR calculation in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for ATR calculation in {timeframe}.")
        return Decimal('0')
    
    try:
        high = np.array([float(c['high']) for c in candles[-period:]], dtype=np.float64)
        low = np.array([float(c['low']) for c in candles[-period:]], dtype=np.float64)
        close = np.array([float(c['close']) for c in candles[-period-1:-1]], dtype=np.float64)
        
        # Clean the data
        high = clean_data(high)
        low = clean_data(low)
        close = clean_data(close)
        
        if len(high) < period or len(low) < period or len(close) < period-1:
            logging.warning(f"Insufficient data for ATR calculation in {timeframe}.")
            print(f"Insufficient data for ATR calculation in {timeframe}.")
            return Decimal('0')
        
        atr = talib.ATR(high, low, close, timeperiod=period-1)
        if len(atr) > 0 and not np.isnan(atr[-1]):
            atr_value = Decimal(str(atr[-1]))
            logging.info(f"{timeframe} - ATR: {atr_value:.25f}")
            print(f"{timeframe} - ATR: {atr_value:.25f}")
            return atr_value
        else:
            logging.warning(f"Invalid ATR value in {timeframe}.")
            print(f"Invalid ATR value in {timeframe}.")
            return Decimal('0')
    except Exception as e:
        logging.error(f"Error calculating ATR for {timeframe}: {e}")
        print(f"Error calculating ATR for {timeframe}: {e}")
        return Decimal('0')

def hurst_cycle_analysis(series, timeframe, window_size=200, sampling_rate=20, preferred_kind='random_walk'):
    if len(series) < window_size:
        logging.warning(f"{timeframe} - Hurst cycle analysis skipped: Series length ({len(series)}) < window_size ({window_size})")
        return [], [], []
    
    # Clean the data
    series = clean_data(series)
    
    cycles = []
    peaks = []
    troughs = []
    length = len(series)
    kinds = [preferred_kind] + [k for k in ['price', 'random_walk', 'change'] if k != preferred_kind]
    
    for start in range(0, length - window_size + 1, sampling_rate):
        window = series[start:start + window_size]
        if len(window) < window_size:
            logging.warning(f"{timeframe} - Skipping window at index {start}: insufficient data points")
            continue
        
        if np.std(window) < 1e-10 or np.max(window) == np.min(window):
            logging.warning(f"{timeframe} - Skipping window at index {start}: zero range or standard deviation")
            continue
        
        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            logging.warning(f"{timeframe} - Skipping window at index {start}: contains NaN or Inf values")
            continue
        
        hurst_computed = False
        for kind in kinds:
            try:
                adjusted_window = window
                if kind == 'price' and np.any(window <= 0):
                    adjusted_window = window - np.min(window) + 1e-10
                
                H, c, data = compute_Hc(adjusted_window, kind=kind, simplified=True)
                cycles.append(H)
                logging.info(f"{timeframe} - Success at index {start} with kind='{kind}', Hurst={H:.25f}")
                hurst_computed = True
                break
            except Exception as e:
                logging.error(f"{timeframe} - Error computing Hurst in window at index {start} with kind='{kind}': {e}")
        
        if not hurst_computed:
            logging.warning(f"{timeframe} - Failed to compute Hurst for window at index {start}")
            continue
        
        max_idx = np.argmax(adjusted_window)
        min_idx = np.argmin(adjusted_window)
        peaks.append(start + max_idx)
        troughs.append(start + min_idx)
    
    if not cycles:
        logging.warning(f"{timeframe} - No valid windows processed for Hurst cycle analysis")
        return [], [], []
    
    return cycles, peaks, troughs

def calculate_mtf_trend(candles, timeframe, min_threshold, max_threshold, buy_vol, sell_vol, lookback=50):
    if len(candles) < lookback:
        logging.warning(f"{timeframe} - MTF trend analysis skipped: Insufficient data")
        print(f"{timeframe} - MTF trend analysis skipped: Insufficient data")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, max_threshold, np.zeros(lookback), "N/A"
    
    recent_candles = candles[-lookback:]
    closes = np.array([float(c['close']) for c in recent_candles], dtype=np.float64)
    
    # Clean the data
    closes = clean_data(closes)
    
    if len(closes) == 0 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f"Invalid close prices in {timeframe}.")
        print(f"Invalid close prices in {timeframe}.")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, max_threshold, np.zeros(lookback), "N/A"
    
    current_close = Decimal(str(closes[-1]))
    
    if max_threshold == min_threshold:
        logging.warning(f"{timeframe} - Min and max thresholds are equal, defaulting to BEARISH trend")
        return "BEARISH", min_threshold, max_threshold, "TOP", False, True, Decimal('1.0'), False, 0.0, max_threshold, np.zeros(lookback), "N/A"
    
    midpoint = (min_threshold + max_threshold) / Decimal('2')
    trend_bullish = current_close < midpoint
    cycle_status = "Up" if trend_bullish else "Down"
    trend = "BULLISH" if trend_bullish else "BEARISH"
    
    buy_vol = max(buy_vol, Decimal('0'))
    sell_vol = max(sell_vol, Decimal('0'))
    volume_ratio = buy_vol / sell_vol if sell_vol > Decimal('0') else Decimal('1.0')
    volume_confirmed = volume_ratio >= VOLUME_CONFIRMATION_RATIO if trend == "BULLISH" else Decimal('1.0') / volume_ratio >= VOLUME_CONFIRMATION_RATIO
    
    # Perform advanced frequency spectrum analysis
    dominant_freq, spectral_power, intensity, freq_range, degree, stage, cycle_direction, top_frequencies, ht_sine, ht_sine_cycle_type = analyze_frequency_spectrum(
        recent_candles, timeframe, cycle_status, min_threshold, max_threshold, current_close
    )
    
    # Log detailed frequency analysis with 25 decimal places
    logging.info(
        f"{timeframe} - Spectral Analysis: Dominant Freq: {dominant_freq:.25f}, "
        f"Power: {spectral_power:.25f}, Intensity: {intensity:.25f}, Range: {freq_range:.25f}, "
        f"Degree: {degree:.25f}, Stage: {stage}, Cycle Direction: {cycle_direction}, "
        f"HT_SINE Cycle Type: {ht_sine_cycle_type}"
    )
    print(
        f"{timeframe} - Spectral Analysis: Dominant Freq: {dominant_freq:.25f}, "
        f"Power: {spectral_power:.25f}, Intensity: {intensity:.25f}, Range: {freq_range:.25f}, "
        f"Degree: {degree:.25f}, Stage: {stage}, Cycle Direction: {cycle_direction}, "
        f"HT_SINE Cycle Type: {ht_sine_cycle_type}"
    )
    
    # Print top frequencies with 25 decimal places
    print(f"{timeframe} - Top Frequencies:")
    for i, (freq, power) in enumerate(top_frequencies):
        print(f"  {i+1}. Freq: {freq:.25f}, Power: {power:.25f}")
    
    price_range = max_threshold - min_threshold
    tolerance = price_range * SUPPORT_RESISTANCE_TOLERANCE
    
    if trend == "BULLISH":
        cycle_target = current_close * (Decimal('1') + CYCLE_TARGET_PERCENTAGE)
        cycle_target = min(cycle_target, max_threshold - tolerance)
    else:
        cycle_target = current_close * (Decimal('1') - CYCLE_TARGET_PERCENTAGE)
        cycle_target = max(cycle_target, min_threshold + tolerance)
    
    logging.info(f"{timeframe} - MTF Trend: {trend}, Cycle: {cycle_status}, Cycle Target: {cycle_target:.25f}, Dominant Freq: {dominant_freq:.25f}, HT_SINE Cycle: {ht_sine_cycle_type}")
    print(f"{timeframe} - MTF Trend: {trend}, Cycle Status: {cycle_status}, Cycle Target: {cycle_target:.25f}, Dominant Freq: {dominant_freq:.25f}, HT_SINE Cycle: {ht_sine_cycle_type}")
    
    return trend, min_threshold, max_threshold, cycle_status, trend_bullish, not trend_bullish, volume_ratio, volume_confirmed, dominant_freq, cycle_target, ht_sine, ht_sine_cycle_type

def calculate_buy_sell_volume(candles, timeframe, reversal_type, trend_bullish, min_idx, max_idx):
    if not candles or len(candles) < VOLUME_LOOKBACK:
        logging.warning(f"No candles or insufficient candles ({len(candles)}) for volume analysis in {timeframe}.")
        print(f"No candles or insufficient candles ({len(candles)}) for volume analysis in {timeframe}.")
        return Decimal('0'), Decimal('0'), "BEARISH", False, False
    
    buy_vol = sum(Decimal(str(c["volume"])) for c in candles if Decimal(str(c["close"])) > Decimal(str(c["open"])))
    sell_vol = sum(Decimal(str(c["volume"])) for c in candles if Decimal(str(c["close"])) < Decimal(str(c["open"])))
    volume_mood = "BULLISH" if buy_vol > sell_vol else "BEARISH"
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    full_volumes = np.array([float(c["volume"]) for c in candles[-lookback:]], dtype=np.float64)
    
    # Clean the data
    full_volumes = clean_data(full_volumes)
    
    avg_full_volume = np.mean(full_volumes) if len(full_volumes) > 0 else 0.0
    
    volume_increasing = False
    volume_decreasing = False
    
    if reversal_type == "DIP":
        start_idx = max(0, min_idx - VOLUME_LOOKBACK // 2)
        end_idx = min(len(candles), start_idx + VOLUME_LOOKBACK)
        if start_idx < end_idx:
            reversal_volumes = np.array([float(c["volume"]) for c in candles[start_idx:end_idx]], dtype=np.float64)
            reversal_volumes = clean_data(reversal_volumes)
            avg_reversal_volume = np.mean(reversal_volumes) if len(reversal_volumes) > 0 else 0.0
            if avg_reversal_volume > 0 and avg_full_volume > 0:
                volume_increasing = (avg_reversal_volume / avg_full_volume) >= float(VOLUME_CONFIRMATION_RATIO)
                volume_decreasing = not volume_increasing
    elif reversal_type == "TOP":
        start_idx = max(0, max_idx - VOLUME_LOOKBACK // 2)
        end_idx = min(len(candles), start_idx + VOLUME_LOOKBACK)
        if start_idx < end_idx:
            reversal_volumes = np.array([float(c["volume"]) for c in candles[start_idx:end_idx]], dtype=np.float64)
            reversal_volumes = clean_data(reversal_volumes)
            avg_reversal_volume = np.mean(reversal_volumes) if len(reversal_volumes) > 0 else 0.0
            if avg_reversal_volume > 0 and avg_full_volume > 0:
                volume_decreasing = (avg_reversal_volume / avg_full_volume) <= float(VOLUME_EXHAUSTION_RATIO)
                volume_increasing = not volume_decreasing
    else:
        start_idx = max(0, min_idx - VOLUME_LOOKBACK // 2)
        end_idx = min(len(candles), start_idx + VOLUME_LOOKBACK)
        if start_idx < end_idx:
            reversal_volumes = np.array([float(c["volume"]) for c in candles[start_idx:end_idx]], dtype=np.float64)
            reversal_volumes = clean_data(reversal_volumes)
            avg_reversal_volume = np.mean(reversal_volumes) if len(reversal_volumes) > 0 else 0.0
            if avg_reversal_volume > 0 and avg_full_volume > 0:
                volume_increasing = (avg_reversal_volume / avg_full_volume) >= float(VOLUME_CONFIRMATION_RATIO)
                volume_decreasing = not volume_increasing
    
    logging.info(
        f"{timeframe} - Buy Volume: {buy_vol:.25f}, Sell Volume: {sell_vol:.25f}, "
        f"Volume Mood: {volume_mood}, Avg Full Volume: {avg_full_volume:.25f}, "
        f"Volume Increasing: {volume_increasing}, Volume Decreasing: {volume_decreasing}"
    )
    print(
        f"{timeframe} - Buy Volume: {buy_vol:.25f}, Sell Volume: {sell_vol:.25f}, "
        f"Volume Mood: {volume_mood}, Avg Full Volume: {avg_full_volume:.25f}, "
        f"Volume Increasing: {volume_increasing}, Volume Decreasing: {volume_decreasing}"
    )
    
    return buy_vol, sell_vol, volume_mood, volume_increasing, volume_decreasing

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

def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, current_close, higher_tf_tops=None):
    if len(candles) < 3:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for reversal detection in {timeframe}.")
        return "TOP", 0, Decimal('0'), "TOP", 0, Decimal('0'), 0, 0
    
    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    price_range = max_threshold - min_threshold
    tolerance = price_range * TOLERANCE_FACTORS[timeframe]
    
    min_candle = min(recent_candles, key=lambda x: x['low'])
    max_candle = max(recent_candles, key=lambda x: x['high'])
    
    min_idx = 0
    max_idx = 0
    
    for i, c in enumerate(recent_candles):
        if c['low'] == float(min_threshold):
            min_idx = i
            break
    
    for i, c in enumerate(recent_candles):
        if c['high'] == float(max_threshold):
            max_idx = i
            break
    
    min_time = min_candle['time']
    max_time = max_candle['time']
    
    diff_to_high = abs(current_close - max_threshold)
    diff_to_low = abs(current_close - min_threshold)
    
    if diff_to_high < diff_to_low:
        most_recent_reversal = "TOP"
        most_recent_time = max_time
        most_recent_price = max_threshold
    else:
        most_recent_reversal = "DIP"
        most_recent_time = min_time
        most_recent_price = min_threshold
    
    logging.info(
        f"{timeframe} - Reversal: {most_recent_reversal} at {most_recent_price:.25f}, "
        f"Diff to High: {diff_to_high:.25f}, Diff to Low: {diff_to_low:.25f}"
    )
    print(
        f"{timeframe} - Reversal: {most_recent_reversal} at {most_recent_price:.25f}, "
        f"Diff to High: {diff_to_high:.25f}, Diff to Low: {diff_to_low:.25f}"
    )
    
    return most_recent_reversal, min_time, min_threshold, "TOP", max_time, max_threshold, min_idx, max_idx

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
        higher_tf_ranges = [timeframe_ranges[tf] for tf in ["3m", "5m"] if tf in timeframe_ranges]
        if higher_tf_ranges:
            min_higher_range = min(higher_tf_ranges)
            if price_range > min_higher_range:
                max_threshold = min_threshold + min_higher_range
                price_range = max_threshold - min_threshold
    
    middle_threshold = (min_threshold + max_threshold) / Decimal('2')
    
    logging.info(f"{timeframe} - Min: {min_threshold:.25f}, Mid: {middle_threshold:.25f}, Max: {max_threshold:.25f}")
    print(f"{timeframe} - Min: {min_threshold:.25f}, Mid: {middle_threshold:.25f}, Max: {max_threshold:.25f}")
    
    return min_threshold, middle_threshold, max_threshold, price_range

def calculate_percentage_distance(current_price, min_threshold, max_threshold):
    if max_threshold == min_threshold:
        return Decimal('0.0')
    return ((current_price - min_threshold) / (max_threshold - min_threshold)) * Decimal('100')

def generate_ml_forecast(candles, timeframe, forecast_periods=5):
    """
    Advanced ML forecast using Random Forest and Random Walk concepts for cycle prediction.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - forecast_periods: Number of periods to forecast (default: 5)
    
    Returns:
    - forecast_price: Predicted price after forecast_periods
    """
    if len(candles) < ML_LOOKBACK:
        logging.warning(f"Insufficient data ({len(candles)}) for ML forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for ML forecast in {timeframe}")
        return Decimal('0.0')
    
    try:
        # Extract data
        closes = np.array([float(c['close']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        highs = np.array([float(c['high']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        lows = np.array([float(c['low']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        volumes = np.array([float(c['volume']) for c in candles[-ML_LOOKBACK:]], dtype=np.float64)
        
        # Clean the data
        closes = clean_data(closes)
        highs = clean_data(highs)
        lows = clean_data(lows)
        volumes = clean_data(volumes)
        
        # Create DataFrame for easier feature engineering
        df = pd.DataFrame({
            'close': closes,
            'high': highs,
            'low': lows,
            'volume': volumes
        })
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate lagged returns
        for lag in range(1, 6):
            df[f'lagged_return_{lag}'] = df['returns'].shift(lag)
        
        # Calculate rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
        
        # Calculate technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Calculate ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate OBV
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Calculate Hurst exponent
        df['hurst'] = 0.5  # Default value
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    H, _, _ = compute_Hc(window_data, kind='price', simplified=True)
                    df.at[df.index[i], 'hurst'] = H
                except:
                    pass
        
        # Calculate FFT components
        df['fft_real'] = 0.0
        df['fft_imag'] = 0.0
        df['fft_power'] = 0.0
        
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    fft_result = fft(window_data - np.mean(window_data))
                    freqs = np.fft.fftfreq(len(window_data))
                    
                    # Get the dominant frequency (excluding DC component)
                    power = np.abs(fft_result[1:]) ** 2
                    if len(power) > 0:
                        dominant_idx = np.argmax(power) + 1
                        df.at[df.index[i], 'fft_real'] = np.real(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_imag'] = np.imag(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_power'] = power[dominant_idx - 1]
                except:
                    pass
        
        # Create target variable (future returns)
        df['target'] = df['close'].shift(-forecast_periods) / df['close'] - 1
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 50:
            logging.warning(f"Insufficient data after feature engineering in {timeframe}")
            print(f"Insufficient data after feature engineering in {timeframe}")
            return Decimal('0.0')
        
        # Define features and target
        feature_columns = [col for col in df.columns if col not in ['close', 'high', 'low', 'volume', 'target']]
        X = df[feature_columns].values
        y = df['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        # Get the most recent data point for prediction
        last_data = df.iloc[-1:][feature_columns].values
        last_data_scaled = scaler.transform(last_data)
        
        # Predict future return
        predicted_return = model.predict(last_data_scaled)[0]
        
        # Calculate forecast price
        current_close = Decimal(str(df['close'].iloc[-1]))
        forecast_price = current_close * (Decimal('1') + Decimal(str(predicted_return)))
        
        # Adjust forecast based on cycle direction
        # Get current cycle direction from the last few price movements
        recent_returns = df['returns'].iloc[-5:].values
        if len(recent_returns) >= 3:
            # Simple trend detection
            if np.mean(recent_returns) > 0:
                # Uptrend - ensure forecast is above current price
                forecast_price = max(forecast_price, current_close * Decimal('1.001'))
            else:
                # Downtrend - ensure forecast is below current price
                forecast_price = min(forecast_price, current_close * Decimal('0.999'))
        
        # Apply bounds based on recent price range
        recent_min = Decimal(str(df['close'].iloc[-20:].min()))
        recent_max = Decimal(str(df['close'].iloc[-20:].max()))
        
        # Don't let forecast go beyond reasonable bounds
        forecast_price = max(forecast_price, recent_min * Decimal('0.95'))
        forecast_price = min(forecast_price, recent_max * Decimal('1.05'))
        
        logging.info(f"{timeframe} - ML Forecast (Random Forest): {forecast_price:.25f}")
        print(f"{timeframe} - ML Forecast (Random Forest): {forecast_price:.25f}")
        
        return forecast_price
    except Exception as e:
        logging.error(f"Error generating ML forecast for {timeframe}: {e}")
        print(f"Error generating ML forecast for {timeframe}: {e}")
        return Decimal('0.0')

def generate_fft_forecast(candles, timeframe, forecast_periods=5):
    if len(candles) < 10:
        logging.warning(f"Insufficient data ({len(candles)}) for FFT forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for FFT forecast in {timeframe}")
        return Decimal('0.0')
    
    try:
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        if np.any(np.isnan(closes)) or np.any(closes <= 0):
            logging.warning(f"Invalid close prices in {timeframe} for FFT forecast.")
            print(f"Invalid close prices in {timeframe} for FFT forecast.")
            return Decimal('0.0')
        
        # Clean the data
        closes = clean_data(closes)
        
        current_close = Decimal(str(closes[-1])) if len(closes) > 0 else Decimal('0.0')
        
        mean_close = np.mean(closes)
        
        fft_result = fft(closes - mean_close)  # Center around zero
        freqs = np.fft.fftfreq(len(closes))
        
        magnitudes = np.abs(fft_result)
        
        # Skip DC component (index 0) when finding dominant frequencies
        if len(magnitudes) > 1:
            # Get both positive and negative frequencies
            positive_freq_indices = np.where(freqs > 0)[0]
            negative_freq_indices = np.where(freqs < 0)[0]
            
            # Find the most significant positive and negative frequencies
            if len(positive_freq_indices) > 0:
                pos_idx = positive_freq_indices[np.argmax(magnitudes[positive_freq_indices])]
                pos_freq = freqs[pos_idx]
                pos_amp = magnitudes[pos_idx] / len(closes) * 2
                pos_phase = np.angle(fft_result[pos_idx])
            else:
                pos_freq, pos_amp, pos_phase = 0, 0, 0
                
            if len(negative_freq_indices) > 0:
                neg_idx = negative_freq_indices[np.argmax(magnitudes[negative_freq_indices])]
                neg_freq = freqs[neg_idx]
                neg_amp = magnitudes[neg_idx] / len(closes) * 2
                neg_phase = np.angle(fft_result[neg_idx])
            else:
                neg_freq, neg_amp, neg_phase = 0, 0, 0
            
            # Determine which frequency is more dominant
            if pos_amp > neg_amp:
                dominant_freq = pos_freq
                dominant_amp = pos_amp
                dominant_phase = pos_phase
                cycle_direction = "Down"  # Positive frequency dominant suggests down cycle
            else:
                dominant_freq = neg_freq
                dominant_amp = neg_amp
                dominant_phase = neg_phase
                cycle_direction = "Up"    # Negative frequency dominant suggests up cycle
        else:
            dominant_freq, dominant_amp, dominant_phase, cycle_direction = 0, 0, 0, "N/A"
        
        # Generate forecast using the dominant frequency
        forecast = np.zeros(forecast_periods)
        future_t = np.arange(len(closes), len(closes) + forecast_periods)
        
        if dominant_amp > 0:
            forecast += dominant_amp * np.cos(2 * np.pi * dominant_freq * future_t + dominant_phase)
        
        forecast_price = Decimal(str(forecast[-1] + mean_close))
        
        # Use min and max thresholds to adjust/bound the forecast
        min_th, _, max_th, _ = calculate_thresholds(candles)
        amp_adjust = (max_th - min_th) / Decimal('2')
        mean_close_decimal = Decimal(str(mean_close))
        
        # Adjust forecast based on cycle direction
        if cycle_direction == "Up":
            # For up cycle, forecast should be higher than current price
            forecast_price = max(forecast_price, current_close)
            forecast_price = min(forecast_price, mean_close_decimal + amp_adjust)
        else:
            # For down cycle, forecast should be lower than current price
            forecast_price = min(forecast_price, current_close)
            forecast_price = max(forecast_price, mean_close_decimal - amp_adjust)
        
        logging.info(f"{timeframe} - FFT Forecast: {forecast_price:.25f} (Cycle Direction: {cycle_direction})")
        print(f"{timeframe} - FFT Forecast: {forecast_price:.25f} (Cycle Direction: {cycle_direction})")
        
        return forecast_price
    except Exception as e:
        logging.error(f"Error generating FFT forecast for {timeframe}: {e}")
        print(f"Error generating FFT forecast for {timeframe}: {e}")
        return Decimal('0.0')

def calculate_momentum_trend(candles, timeframe, period=MOMENTUM_PERIOD, lookback=MOMENTUM_LOOKBACK):
    """
    Calculate momentum and determine if it's increasing or decreasing over the last few values.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - period: Period for momentum calculation
    - lookback: Number of recent momentum values to check for trend
    
    Returns:
    - momentum_values: Array of momentum values
    - momentum_increasing: Boolean indicating if momentum is increasing
    - momentum_decreasing: Boolean indicating if momentum is decreasing
    """
    if len(candles) < period + lookback:
        logging.warning(f"Insufficient data ({len(candles)}) for momentum trend analysis in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for momentum trend analysis in {timeframe}")
        return np.zeros(lookback), False, False
    
    closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
    
    # Clean the data
    closes = clean_data(closes)
    
    # Calculate momentum using TALib
    momentum = talib.MOM(closes, timeperiod=period)
    
    # Get the last 'lookback' values
    if len(momentum) >= lookback:
        recent_momentum = momentum[-lookback:]
        
        # Check if momentum is increasing (each value > previous value)
        momentum_increasing = all(recent_momentum[i] > recent_momentum[i-1] for i in range(1, len(recent_momentum)))
        
        # Check if momentum is decreasing (each value < previous value)
        momentum_decreasing = all(recent_momentum[i] < recent_momentum[i-1] for i in range(1, len(recent_momentum)))
        
        logging.info(f"{timeframe} - Momentum Trend: Increasing={momentum_increasing}, Decreasing={momentum_decreasing}")
        print(f"{timeframe} - Momentum Trend: Increasing={momentum_increasing}, Decreasing={momentum_decreasing}")
        
        return recent_momentum, momentum_increasing, momentum_decreasing
    else:
        logging.warning(f"Insufficient momentum values ({len(momentum)}) for trend analysis in {timeframe}")
        print(f"Insufficient momentum values ({len(momentum)}) for trend analysis in {timeframe}")
        return np.zeros(lookback), False, False

def advanced_fft_forecast(candles, timeframe, min_threshold, max_threshold, current_close, forecast_periods=5):
    """
    Advanced FFT forecast using 360-degree unit circle, harmonic analysis, and cycle stage detection.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - min_threshold: Minimum price threshold
    - max_threshold: Maximum price threshold
    - current_close: Current closing price
    - forecast_periods: Number of periods to forecast (default: 5)
    
    Returns:
    - forecast_price: Predicted price after forecast_periods
    - cycle_stage: Current stage in the cycle ("Early", "Middle", "Late")
    - cycle_direction: Cycle direction ("Up" or "Down")
    - dominant_frequency: Most significant frequency component
    - phase_angle: Phase angle in degrees (0-360)
    - harmonic_strength: Strength of harmonic components
    """
    if len(candles) < 10:
        logging.warning(f"Insufficient data ({len(candles)}) for advanced FFT forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for advanced FFT forecast in {timeframe}")
        return Decimal('0.0'), "N/A", "N/A", 0.0, 0.0, 0.0
    
    try:
        # Extract close prices
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        
        # Clean the data
        closes = clean_data(closes)
        
        # Get last 5 close values for recent trend analysis
        last_5_closes = closes[-5:] if len(closes) >= 5 else closes
        
        # Center the data around zero
        mean_close = np.mean(closes)
        centered_closes = closes - mean_close
        
        # Perform FFT
        fft_result = fft(centered_closes)
        freqs = np.fft.fftfreq(len(closes))
        
        # Calculate power spectrum
        power = np.abs(fft_result) ** 2
        
        # Skip DC component (index 0)
        indices = np.arange(1, len(closes))
        
        # Find most negative and most positive frequencies
        negative_freq_indices = indices[freqs[indices] < 0]
        positive_freq_indices = indices[freqs[indices] > 0]
        
        # Get dominant negative and positive frequencies
        if len(negative_freq_indices) > 0:
            neg_idx = negative_freq_indices[np.argmax(power[negative_freq_indices])]
            dominant_neg_freq = freqs[neg_idx]
            dominant_neg_power = power[neg_idx]
            neg_phase = np.angle(fft_result[neg_idx])
        else:
            dominant_neg_freq, dominant_neg_power, neg_phase = 0, 0, 0
            
        if len(positive_freq_indices) > 0:
            pos_idx = positive_freq_indices[np.argmax(power[positive_freq_indices])]
            dominant_pos_freq = freqs[pos_idx]
            dominant_pos_power = power[pos_idx]
            pos_phase = np.angle(fft_result[pos_idx])
        else:
            dominant_pos_freq, dominant_pos_power, pos_phase = 0, 0, 0
        
        # Determine which frequency is more dominant
        if dominant_neg_power > dominant_pos_power:
            dominant_frequency = dominant_neg_freq
            dominant_phase = neg_phase
            cycle_direction = "Up"  # Negative frequency dominant suggests up cycle
        else:
            dominant_frequency = dominant_pos_freq
            dominant_phase = pos_phase
            cycle_direction = "Down"  # Positive frequency dominant suggests down cycle
        
        # Convert phase to degrees (0-360)
        phase_angle = (np.degrees(dominant_phase) + 360) % 360
        
        # Determine cycle stage based on phase angle
        if 0 <= phase_angle < 120:
            cycle_stage = "Early"
        elif 120 <= phase_angle < 240:
            cycle_stage = "Middle"
        else:
            cycle_stage = "Late"
        
        # Calculate harmonic strength (sum of power of top 5 harmonics)
        sorted_indices = indices[np.argsort(power[indices])[-5:]]
        harmonic_strength = np.sum(power[sorted_indices]) / np.sum(power[indices]) if len(indices) > 0 else 0
        
        # Generate forecast using dominant frequency and harmonics
        forecast = np.zeros(forecast_periods)
        future_t = np.arange(len(closes), len(closes) + forecast_periods)
        
        # Add contribution from dominant frequency
        if dominant_neg_power > 0 or dominant_pos_power > 0:
            dominant_amp = np.sqrt(max(dominant_neg_power, dominant_pos_power)) / len(closes) * 2
            forecast += dominant_amp * np.cos(2 * np.pi * dominant_frequency * future_t + dominant_phase)
        
        # Add contributions from harmonics
        for idx in sorted_indices:
            if idx in [neg_idx, pos_idx]:
                continue  # Skip the dominant frequency we already added
            freq = freqs[idx]
            amp = np.sqrt(power[idx]) / len(closes) * 2
            phase = np.angle(fft_result[idx])
            forecast += amp * np.cos(2 * np.pi * freq * future_t + phase)
        
        # Convert forecast back to price scale
        forecast_price = Decimal(str(forecast[-1] + mean_close))
        
        # Apply a directional bias based on cycle direction
        # This ensures the forecast moves in the direction of the cycle
        if cycle_direction == "Up":
            # For up cycle, ensure forecast is at least slightly above current price
            min_forecast = current_close * (Decimal('1') + Decimal('0.001'))  # 0.1% above current
            forecast_price = max(forecast_price, min_forecast)
        else:
            # For down cycle, ensure forecast is at least slightly below current price
            max_forecast = current_close * (Decimal('1') - Decimal('0.001'))  # 0.1% below current
            forecast_price = min(forecast_price, max_forecast)
        
        # Apply threshold bounds to keep forecast within reasonable range
        price_range = max_threshold - min_threshold
        tolerance = price_range * SUPPORT_RESISTANCE_TOLERANCE
        
        # Bound the forecast by the thresholds
        if forecast_price > max_threshold - tolerance:
            forecast_price = max_threshold - tolerance
        if forecast_price < min_threshold + tolerance:
            forecast_price = min_threshold + tolerance
        
        logging.info(
            f"{timeframe} - Advanced FFT Forecast: {forecast_price:.25f}, "
            f"Cycle Stage: {cycle_stage}, Direction: {cycle_direction}, "
            f"Dominant Freq: {dominant_frequency:.25f}, Phase: {phase_angle:.2f}°, "
            f"Harmonic Strength: {harmonic_strength:.25f}"
        )
        print(
            f"{timeframe} - Advanced FFT Forecast: {forecast_price:.25f}, "
            f"Cycle Stage: {cycle_stage}, Direction: {cycle_direction}, "
            f"Dominant Freq: {dominant_frequency:.25f}, Phase: {phase_angle:.2f}°, "
            f"Harmonic Strength: {harmonic_strength:.25f}"
        )
        
        return forecast_price, cycle_stage, cycle_direction, dominant_frequency, phase_angle, harmonic_strength
    except Exception as e:
        logging.error(f"Error in advanced FFT forecast for {timeframe}: {e}")
        print(f"Error in advanced FFT forecast for {timeframe}: {e}")
        return Decimal('0.0'), "N/A", "N/A", 0.0, 0.0, 0.0

def fetch_c_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1500):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    
    candle_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tf = {executor.submit(fetch_candles, tf): tf for tf in timeframes}
        for future in concurrent.futures.as_completed(future_to_tf):
            tf = future_to_tf[future]
            try:
                candles = future.result()
                if candles:
                    candle_map[tf] = candles
            except Exception as e:
                logging.error(f"Error fetching candles for {tf}: {e}")
                print(f"Error fetching candles for {tf}: {e}")
    
    return candle_map

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
            
            message = (
                f"Trade Signal: {signal}\n"
                f"Symbol: {TRADE_SYMBOL}\n"
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Quantity: {quantity:.2f} BTC\n"
                f"Entry Price: ~{price:.2f} USDC\n"
                f"Initial Balance: {initial_balance:.2f} USDC\n"
            )
            
            if signal == "LONG":
                order = client.futures_create_order(
                    symbol=TRADE_SYMBOL,
                    side="BUY",
                    type="MARKET",
                    quantity=str(quantity)
                )
                
                tp_price = (price * (Decimal('1') + TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'), rounding='ROUND_DOWN')
                sl_price = (price * (Decimal('1') - STOP_LOSS_PERCENTAGE)).quantize(Decimal('0.01'), rounding='ROUND_DOWN')
                
                position.update({"sl_price": sl_price, "tp_price": tp_price, "side": "LONG", "quantity": quantity, "entry_price": price})
                
                message += (
                    f"Stop-Loss: {sl_price:.2f} (-5%)\n"
                    f"Take-Profit: {tp_price:.2f} (+5%)\n"
                    f"\nAnalysis Details\n"
                )
            elif signal == "SHORT":
                order = client.futures_create_order(
                    symbol=TRADE_SYMBOL,
                    side="SELL",
                    type="MARKET",
                    quantity=str(quantity)
                )
                
                tp_price = (price * (Decimal('1') - TAKE_PROFIT_PERCENTAGE)).quantize(Decimal('0.01'), rounding='ROUND_DOWN')
                sl_price = (price * (Decimal('1') + STOP_LOFIT_PERCENTAGE)).quantize(Decimal('0.01'), rounding='ROUND_DOWN')
                
                position.update({"sl_price": sl_price, "tp_price": tp_price, "side": "SHORT", "quantity": -quantity, "entry_price": price})
                
                message += (
                    f"Stop-Loss: {sl_price:.2f} (+5%)\n"
                    f"Take-Profit: {tp_price:.2f} (-5%)\n"
                    f"\nAnalysis Details\n"
                )
            
            for tf in TIMEFRAMES:
                details = analysis_details.get(tf, {})
                perc_dist = calculate_percentage_distance(price, details.get('min_threshold', Decimal('0')), details.get('max_threshold', Decimal('0')))
                dist_to_min = price - details.get('min_threshold', Decimal('0'))
                dist_to_max = details.get('max_threshold', Decimal('0')) - price
                cycle_target = details.get('cycle_target', Decimal('0'))
                hurst_target = details.get('hurst_target', Decimal('0'))
                cycle_target = cycle_target if isinstance(cycle_target, Decimal) else Decimal('0')
                hurst_target = hurst_target if isinstance(hurst_target, Decimal) else Decimal('0')
                
                message += (
                    f"\n{tf} Timeframe\n"
                    f"MTF Trend: {details.get('trend', 'N/A')}\n"
                    f"Cycle Status: {details.get('cycle_status', 'N/A')}\n"
                    f"Dominant Frequency: {details.get('dominant_freq', 0.0):.25f}\n"
                    f"Cycle Target: {cycle_target:.25f} USDC\n"
                    f"Hurst Exponent: {details.get('hurst_exponent', 0.0):.25f}\n"
                    f"Hurst Cycle Type: {details.get('hurst_cycle_type', 'N/A')}\n"
                    f"Hurst Target Price: {hurst_target:.25f} USDC\n"
                    f"ML Forecast: {details.get('ml_forecast', Decimal('0')):.25f} USDC\n"
                    f"FFT Forecast: {details.get('fft_forecast', Decimal('0')):.25f} USDC\n"
                    f"Advanced FFT Forecast: {details.get('advanced_fft_forecast', Decimal('0')):.25f} USDC\n"
                    f"HT_SINE Cycle Type: {details.get('ht_sine_cycle_type', 'N/A')}\n"
                    f"Price Position: {perc_dist:.25f}% between min/max\n"
                    f"Dist to Min: {dist_to_min:.25f} USDC ({perc_dist:.25f}% of range)\n"
                    f"Dist to Max: {dist_to_max:.25f} USDC ({100 - perc_dist:.25f}% of range)\n"
                    f"Volume Ratio: {details.get('volume_ratio', Decimal('0')):.25f}\n"
                    f"Volume Confirmed: {details.get('volume_confirmed', False)}\n"
                    f"Minimum Threshold: {details.get('min_threshold', Decimal('0')):.25f} USDC\n"
                    f"Middle Threshold: {details.get('middle_threshold', Decimal('0')):.25f} USDC\n"
                    f"Maximum Threshold: {details.get('max_threshold', Decimal('0')):.25f} USDC\n"
                    f"High-Low Range: {details.get('price_range', Decimal('0')):.25f} USDC\n"
                    f"Most Recent Reversal: {details.get('reversal_type', 'N/A')} at price {details.get('reversal_price', Decimal('0')):.25f}\n"
                    f"Support Level: {details.get('support_levels', [{}])[0].get('price', Decimal('0')):.25f} USDC\n"
                    f"Resistance Level: {details.get('resistance_levels', [{}])[0].get('price', Decimal('0')):.25f} USDC\n"
                    f"Buy Volume: {details.get('buy_vol', Decimal('0')):.25f}\n"
                    f"Sell Volume: {details.get('sell_vol', Decimal('0')):.25f}\n"
                    f"Volume Mood: {details.get('volume_mood', 'N/A')}\n"
                    f"Volume Bullish (1m): {details.get('volume_bullish', False)}\n"
                    f"Volume Bearish (1m): {details.get('volume_bearish', False)}\n"
                )
            
            telegram_loop.run_until_complete(send_telegram_message(message))
            
            logging.info(f"Placed {signal} order: {quantity:.25f} BTC at market price ~{price:.25f}")
            print(f"\n=== TRADE ENTERED ===\nSide: {signal}\nQuantity: {quantity:.25f} BTC\nEntry Price: ~{price:.25f} USDC\nInitial Balance: {initial_balance:.25f} USDC\nStop-Loss: {sl_price:.25f}\nTake-Profit: {tp_price:.25f}\n===================\n")
            
            return position
        except BinanceAPIException as e:
            retry_after = int(e.response.headers.get('Retry-After', '60')) if e.response else 60
            logging.error(f"Error placing order (attempt {attempt + 1}/{retries}): {e.message}")
            print(f"Error placing order (attempt {attempt + 1}/{retries}: {e.message}")
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
                f"Position Closed\n"
                f"Symbol: {TRADE_SYMBOL}\n"
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Side: {position['side']}\n"
                f"Quantity: {quantity:.2f} BTC\n"
                f"Exit Price: ~{price:.2f} USDC\n"
                f"Unrealized PNL: {position['unrealized_pnl']:.2f} USDC\n"
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
        
        # Start the Telegram application
        telegram_loop.run_until_complete(telegram_app.initialize())
        
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
            
            candle_map = fetch_c_candles_in_parallel(TIMEFRAMES)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candle data available. Retrying in 60 seconds.")
                print("No candle data available. Retrying in 60 seconds.")
                time.sleep(60)
                continue
            
            usdc_balance = get_balance('USDC')
            position = get_position()
            
            print(f"\nCurrent Price: {current_price:.25f} USDC")
            print(f"USDC Balance: {usdc_balance:.25f}")
            print(f"Current Position: {position['side']} ({position['quantity']:.6f} BTC)")
            
            if position["side"] != "NONE" and position["sl_price"] > Decimal('0'):
                if position["side"] == "LONG" and current_price <= position["sl_price"]:
                    message = (
                        f"Stop-Loss Triggered: LONG\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.25f} USDC\n"
                        f"Stop-Loss Price: {position['sl_price']:.25f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                elif position["side"] == "LONG" and current_price >= position["tp_price"]:
                    message = (
                        f"Take-Profit Triggered: LONG\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.25f} USDC\n"
                        f"Take-Profit Price: {position['tp_price']:.25f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                elif position["side"] == "SHORT" and current_price >= position["sl_price"]:
                    message = (
                        f"Stop-Loss Triggered: SHORT\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.25f} USDC\n"
                        f"Stop-Loss Price: {position['sl_price']:.25f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                elif position["side"] == "SHORT" and current_price <= position["tp_price"]:
                    message = (
                        f"Take-Profit Triggered: SHORT\n"
                        f"Symbol: {TRADE_SYMBOL}\n"
                        f"Time: {current_local_time_str}\n"
                        f"Exit Price: {current_price:.25f} USDC\n"
                        f"Take-Profit Price: {position['tp_price']:.25f} USDC\n"
                        f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC\n"
                    )
                    telegram_loop.run_until_complete(send_telegram_message(message))
                    close_position(position, current_price)
                
                position = get_position()
            
            conditions_long = {
                "momentum_positive_1m": False,
                "ml_forecast_above_price_1m": False,
                "negative_dominant_freq_1m": False,
                "volume_bullish_1m": False,
            }
            
            conditions_short = {
                "momentum_negative_1m": False,
                "ml_forecast_below_price_1m": False,
                "positive_dominant_freq_1m": False,
                "volume_bearish_1m": False,
            }
            
            # Define required conditions (must all be true for signal)
            required_long_conditions = [
                "momentum_positive_1m",
                "ml_forecast_above_price_1m",
                "negative_dominant_freq_1m",
                "volume_bullish_1m",
            ]
            
            required_short_conditions = [
                "momentum_negative_1m",
                "ml_forecast_below_price_1m",
                "positive_dominant_freq_1m",
                "volume_bearish_1m",
            ]
            
            timeframe_ranges = {tf: calculate_thresholds(candle_map.get(tf, []))[3] if candle_map.get(tf) else Decimal('0') for tf in TIMEFRAMES}
            
            higher_tf_tops = {}
            for tf in ["3m", "5m"]:
                if candle_map.get(tf):
                    min_th, _, max_th, _ = calculate_thresholds(candle_map[tf], timeframe_ranges)
                    closes = np.array([c["close"] for c in candle_map[tf] if c["close"] > 0], dtype=np.float64)
                    current_close = Decimal(str(closes[-1])) if len(closes) > 0 else current_price
                    reversal_type, _, _, _, _, _, _, _ = detect_recent_reversal(candle_map[tf], tf, min_th, max_th, current_close)
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
                
                min_threshold, middle_threshold, max_threshold, price_range = calculate_thresholds(candle_map[timeframe], timeframe_ranges)
                
                current_close = Decimal(str(closes[-1])) if len(closes) > 0 else current_price
                reversal_type, min_time, closest_min_price, _, max_time, closest_max_price, min_idx, max_idx = detect_recent_reversal(
                    candles_tf, timeframe, min_threshold, max_threshold, current_close, higher_tf_tops if timeframe == "1m" else None
                )
                
                support_levels, resistance_levels = calculate_support_resistance(candles_tf, timeframe)
                
                buy_vol, sell_vol, volume_mood, volume_increasing, volume_decreasing = calculate_buy_sell_volume(
                    candles_tf, timeframe, reversal_type, (closes[-1] < (min_threshold + max_threshold) / 2), min_idx, max_idx
                )
                
                atr = calculate_atr(candles_tf, timeframe)
                
                hurst_cycles, hurst_peaks, hurst_troughs = hurst_cycle_analysis(closes, timeframe, window_size=200, sampling_rate=20)
                hurst_exponent = float(hurst_cycles[-1]) if hurst_cycles else 0.0
                hurst_cycle_type = "N/A"
                hurst_target = Decimal('0')
                
                if hurst_peaks and hurst_troughs:
                    latest_peak_idx = max(hurst_peaks)
                    latest_trough_idx = max(hurst_troughs)
                    hurst_cycle_type = "Up" if latest_trough_idx > latest_peak_idx else "Down"
                    
                    price_range = max_threshold - min_threshold
                    tolerance = price_range * SUPPORT_RESISTANCE_TOLERANCE
                    
                    if hurst_cycle_type == "Up":
                        hurst_target = current_close * (Decimal('1') + CYCLE_TARGET_PERCENTAGE)
                        hurst_target = min(hurst_target, max_threshold - tolerance)
                    else:
                        hurst_target = current_close * (Decimal('1') - CYCLE_TARGET_PERCENTAGE)
                        hurst_target = max(hurst_target, min_threshold + tolerance)
                
                # Add new volume conditions for 1m timeframe
                if timeframe == "1m":
                    conditions_long["volume_bullish_1m"] = (volume_mood == "BULLISH")
                    conditions_short["volume_bearish_1m"] = (volume_mood == "BEARISH")
                
                trend, min_th, max_th, cycle_status, trend_bullish, trend_bearish, volume_ratio, volume_confirmed, dominant_freq, cycle_target, ht_sine, ht_sine_cycle_type = calculate_mtf_trend(
                    candles_tf, timeframe, min_threshold, max_threshold, buy_vol, sell_vol
                )
                
                # Add new conditions for 1m timeframe based on predominant frequency
                if timeframe == "1m":
                    conditions_long["negative_dominant_freq_1m"] = dominant_freq < 0
                    conditions_short["positive_dominant_freq_1m"] = dominant_freq > 0
                
                # Calculate momentum trend for all timeframes
                momentum_values, momentum_increasing, momentum_decreasing = calculate_momentum_trend(
                    candles_tf, timeframe, period=MOMENTUM_PERIOD, lookback=MOMENTUM_LOOKBACK
                )
                
                ml_forecast = generate_ml_forecast(candles_tf, timeframe)
                
                if timeframe == "1m":
                    # Ensure ml_forecast conditions are exact opposites
                    conditions_long["ml_forecast_above_price_1m"] = ml_forecast > current_close
                    conditions_short["ml_forecast_below_price_1m"] = not conditions_long["ml_forecast_above_price_1m"]
                
                fft_forecast = generate_fft_forecast(candles_tf, timeframe)
                
                # Advanced FFT Forecast
                advanced_fft_forecast_price, fft_cycle_stage, fft_cycle_direction, fft_dominant_freq, fft_phase_angle, fft_harmonic_strength = advanced_fft_forecast(
                    candles_tf, timeframe, min_threshold, max_threshold, current_close
                )
                
                if len(closes) >= MOMENTUM_PERIOD and timeframe == "1m":
                    momentum = talib.MOM(closes, timeperiod=MOMENTUM_PERIOD)
                    if len(momentum) > 0 and not np.isnan(momentum[-1]):
                        conditions_long["momentum_positive_1m"] = Decimal(str(momentum[-1])) >= Decimal('0')
                        conditions_short["momentum_negative_1m"] = not conditions_long["momentum_positive_1m"]
                
                analysis_details[timeframe] = {
                    "trend": trend,
                    "cycle_status": cycle_status,
                    "dominant_freq": dominant_freq,
                    "cycle_target": cycle_target,
                    "hurst_exponent": hurst_exponent,
                    "hurst_cycle_type": hurst_cycle_type,
                    "hurst_target": hurst_target,
                    "ml_forecast": ml_forecast,
                    "fft_forecast": fft_forecast,
                    "advanced_fft_forecast": advanced_fft_forecast_price,
                    "ht_sine": ht_sine,
                    "ht_sine_cycle_type": ht_sine_cycle_type,
                    "momentum_values": momentum_values,
                    "momentum_increasing": momentum_increasing,
                    "momentum_decreasing": momentum_decreasing,
                    "perc_distance": calculate_percentage_distance(current_close, min_threshold, max_threshold),
                    "dist_to_min": current_close - min_threshold,
                    "dist_to_max": max_threshold - current_close,
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
                    "buy_vol": buy_vol,
                    "sell_vol": sell_vol,
                    "volume_mood": volume_mood,
                    "price_range": price_range,
                    "volume_increasing": volume_increasing,
                    "volume_decreasing": volume_decreasing,
                    "volume_bullish": (volume_mood == "BULLISH") if timeframe == "1m" else None,
                    "volume_bearish": (volume_mood == "BEARISH") if timeframe == "1m" else None,
                    "fft_cycle_stage": fft_cycle_stage,
                    "fft_cycle_direction": fft_cycle_direction,
                    "fft_dominant_freq": fft_dominant_freq,
                    "fft_phase_angle": fft_phase_angle,
                    "fft_harmonic_strength": fft_harmonic_strength
                }
                
                print(f"Trend: {trend}")
                print(f"Cycle Status: {cycle_status}")
                print(f"Cycle Target: {cycle_target:.25f}")
                print(f"Hurst Exponent: {hurst_exponent:.25f}")
                print(f"Hurst Cycle Type: {hurst_cycle_type}")
                print(f"Hurst Target Price: {hurst_target:.25f}")
                print(f"ML Forecast: {ml_forecast:.25f}")
                print(f"FFT Forecast: {fft_forecast:.25f}")
                print(f"Advanced FFT Forecast: {advanced_fft_forecast_price:.25f}")
                print(f"HT_SINE Cycle Type: {ht_sine_cycle_type}")
                print(f"Price Position: {analysis_details[timeframe]['perc_distance']:.25f}% between min/max")
                print(f"Dist to Min: {analysis_details[timeframe]['dist_to_min']:.25f} USDC ({analysis_details[timeframe]['perc_distance']:.25f}% of range)")
                print(f"Dist to Max: {analysis_details[timeframe]['dist_to_max']:.25f} USDC ({100 - analysis_details[timeframe]['perc_distance']:.25f}% of range)")
                print(f"Min Threshold: {min_threshold:.25f}")
                print(f"Mid Threshold: {middle_threshold:.25f}")
                print(f"Max Threshold: {max_threshold:.25f}")
                print(f"Reversal Type: {reversal_type}")
                print(f"Support: {support_levels[0]['price']:.25f}")
                print(f"Resistance: {resistance_levels[0]['price']:.25f}")
                print(f"Volume Mood: {volume_mood}")
                print(f"Volume Ratio: {volume_ratio:.25f}")
                
                # Print new volume conditions for 1m timeframe
                if timeframe == "1m":
                    print(f"Volume Bullish (1m): {conditions_long['volume_bullish_1m']}")
                    print(f"Volume Bearish (1m): {conditions_short['volume_bearish_1m']}")
            
            # Ensure symmetrical conditions for all pairs
            # For momentum_increasing/momentum_decreasing
            conditions_short["momentum_negative_1m"] = not conditions_long["momentum_positive_1m"]
            
            print("\nCondition Pairs Status:")
            for cond_pair in [
                ("negative_dominant_freq_1m", "positive_dominant_freq_1m"),
                ("volume_bullish_1m", "volume_bearish_1m")
            ]:
                cond1, cond2 = cond_pair
                status1 = conditions_long.get(cond1, False)
                status2 = conditions_short.get(cond2, False)
                print(f"{cond1}: {status1}, {cond2}: {status2}")
            
            print(f"momentum_positive_1m: {conditions_long.get('momentum_positive_1m', False)}, momentum_negative_1m: {conditions_short.get('momentum_negative_1m', False)}")
            print(f"ml_forecast_above_price_1m: {conditions_long.get('ml_forecast_above_price_1m', False)}")
            print(f"ml_forecast_below_price_1m: {conditions_short.get('ml_forecast_below_price_1m', False)}")
            
            print("\nLong Conditions Status:")
            for key, value in conditions_long.items():
                print(f"{key}: {value}")
            
            print("\nShort Conditions Status:")
            for key, value in conditions_short.items():
                print(f"{key}: {value}")
            
            # Count required conditions that are true
            required_long_true_count = sum(1 for cond in required_long_conditions if conditions_long.get(cond, False))
            required_short_true_count = sum(1 for cond in required_short_conditions if conditions_short.get(cond, False))
            
            total_required_long = len(required_long_conditions)
            total_required_short = len(required_short_conditions)
            
            print(f"\nSummary: LONG Required Conditions - True: {required_long_true_count}, False: {total_required_long - required_long_true_count}")
            print(f"Summary: SHORT Required Conditions - True: {required_short_true_count}, False: {total_required_short - required_short_true_count}")
            
            logging.info(f"Summary: LONG Required Conditions - True: {required_long_true_count}, False: {total_required_long - required_long_true_count}")
            logging.info(f"Summary: SHORT Required Conditions - True: {required_short_true_count}, False: {total_required_short - required_short_true_count}")
            
            # Only check required conditions for signal
            long_signal = required_long_true_count == total_required_long
            short_signal = required_short_true_count == total_required_short
            
            print(f"\nTrade Signal Status:")
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'} (Required conditions: {required_long_true_count}/{total_required_long})")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'} (Required conditions: {required_short_true_count}/{total_required_short})")
            
            logging.info(f"LONG Signal: {'Active' if long_signal else 'Inactive'} (Required conditions: {required_long_true_count}/{total_required_long})")
            logging.info(f"SHORT Signal: {'Active' if short_signal else 'Inactive'} (Required conditions: {required_short_true_count}/{total_required_short})")
            
            if long_signal and short_signal:
                logging.warning("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                print("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
                long_signal = False
                short_signal = False
            
            signal = "LONG" if long_signal else "SHORT" if short_signal else "NO_SIGNAL"
            
            print(f"\nTrade Signal Status: {signal}")
            logging.info(f"Trade Signal Status: {signal}")
            
            if signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                signal_message = (
                    f"Trade Signal Triggered: {signal}\n"
                    f"Symbol: {TRADE_SYMBOL}\n"
                    f"Time: {current_local_time_str}\n"
                    f"Current Price: {current_price:.25f} USDC\n"
                    f"Attempting to place order..."
                )
                telegram_loop.run_until_complete(send_telegram_message(signal_message))
                
                quantity = calculate_quantity(usdc_balance, current_price)
                if quantity > Decimal('0'):
                    new_position = place_order(signal, quantity, current_price, usdc_balance, analysis_details)
                    if new_position:
                        position = new_position
                    else:
                        logging.error(f"Failed to place {signal} order after retries.")
                        print(f"Failed to place {signal} order after retries.")
                else:
                    logging.warning(f"Cannot place {signal} order: Insufficient quantity ({quantity:.25f} BTC).")
                    print(f"Cannot place {signal} order: Insufficient quantity ({quantity:.25f} BTC).")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
        print("Bot stopped by user.")
        telegram_loop.run_until_complete(send_telegram_message("Bot stopped by user."))
    
    except Exception as e:
        error_message = (
            f"Fatal Error\n"
            f"Symbol: {TRADE_SYMBOL}\n"
            f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Error: {str(e)}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        logging.error(error_message)
        print(error_message)
        telegram_loop.run_until_complete(send_telegram_message(error_message))
    
    finally:
        try:
            if telegram_app.running:
                telegram_loop.run_until_complete(telegram_app.stop())
            telegram_loop.run_until_complete(telegram_app.shutdown())
            telegram_loop.close()
        except Exception as e:
            logging.error(f"Error during Telegram cleanup: {e}")
            print(f"Error during Telegram cleanup: {e}")

if __name__ == "__main__":
    main()
