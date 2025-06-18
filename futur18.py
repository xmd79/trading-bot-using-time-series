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
import scipy.fft as fftpack
from collections import deque

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
MINIMUM_BALANCE = Decimal('1.0000')  # Minimum USDC balance
TIMEFRAMES = ["1m", "3m", "5m"]
LOOKBACK_PERIODS = {"1m": 500, "3m": 500, "5m": 500}
PRICE_TOLERANCE = Decimal('0.01')  # 1% tolerance
VOLUME_CONFIRMATION_RATIO = Decimal('1.5')  # Buy/Sell volume ratio
PROXIMITY_THRESHOLD = Decimal('0.01')  # 1% proximity
VOLUME_SPIKE_MULTIPLIER = Decimal('2.5')  # Volume spike threshold

# State for quadrant tracking
quadrant_history = {tf: deque(maxlen=10) for tf in TIMEFRAMES}

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
client.API_URL = 'https://fapi.binance.com'  # Futures API

# Set leverage
try:
    client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
    logging.info(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
    print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
except BinanceAPIException as e:
    logging.error(f"Error setting leverage: {e.message}")
    print(f"Error setting leverage: {e.message}")
    exit(1)

# Quadrant Mapping Function
def map_price_to_quadrant(current_price, min_threshold, max_threshold, reversal_type, timeframe, prev_price=Decimal('0')):
    if min_threshold >= max_threshold or current_price < min_threshold:
        logging.warning(f"Invalid quadrant inputs: current={current_price:.25f}, min={min_threshold:.25f}, max={max_threshold:.25f}")
        return "Q1", "Q1", "Q2", Decimal('0')

    price_range = max_threshold - min_threshold
    segment_size = price_range / Decimal('4')

    q1_max = min_threshold + segment_size
    q2_max = min_threshold + 2 * segment_size
    q3_max = min_threshold + 3 * segment_size

    if current_price <= q1_max:
        current_quadrant = "Q1"
    elif current_price <= q2_max:
        current_quadrant = "Q2"
    elif current_price <= q3_max:
        current_quadrant = "Q3"
    else:
        current_quadrant = "Q4"

    # Update quadrant history before determining last quadrant
    last_quadrant = quadrant_history[timeframe][-1] if quadrant_history[timeframe] else "Q2"
    quadrant_history[timeframe].append(current_quadrant)

    # Transition logic
    if reversal_type == "DIP" and current_quadrant == "Q1":
        incoming_quadrant = "Q2"
    elif reversal_type == "TOP" and current_quadrant == "Q4":
        incoming_quadrant = "Q3"
    elif current_price > prev_price:
        incoming_quadrant = {"Q1": "Q2", "Q2": "Q3", "Q3": "Q4", "Q4": "Q3"}.get(current_quadrant, "Q2")
    else:
        incoming_quadrant = {"Q4": "Q3", "Q3": "Q2", "Q2": "Q1", "Q1": "Q2"}.get(current_quadrant, "Q3")

    # Degree mapping
    normalized_price = (current_price - min_threshold) / price_range if price_range > 0 else Decimal('0')
    if current_quadrant == "Q1":
        degrees = normalized_price * Decimal('90')
    elif current_quadrant == "Q2":
        degrees = Decimal('90') + (normalized_price - Decimal('0.25')) * Decimal('360')
    elif current_quadrant == "Q3":
        degrees = Decimal('180') + (normalized_price - Decimal('0.5')) * Decimal('360')
    else:
        degrees = Decimal('270') + (normalized_price - Decimal('0.75')) * Decimal('360')
    degrees = degrees % Decimal('360')

    logging.info(f"{timeframe} - Quadrant: Current={current_quadrant}, Last={last_quadrant}, Incoming={incoming_quadrant}, Degrees={degrees:.25f}")
    print(f"{timeframe} - Quadrant: Current={current_quadrant}, Last={last_quadrant}, Incoming={incoming_quadrant}, Degrees={degrees:.25f}")
    return current_quadrant, last_quadrant, incoming_quadrant, degrees

# FFT-Based Target Price Function
def get_target_price(closes, n_components, timeframe, reversal_type="NONE", min_threshold=Decimal('0'), middle_threshold=Decimal('0'), max_threshold=Decimal('0')):
    if len(closes) < 2 or np.any(np.isnan(closes)) or np.any(closes <= 0):
        logging.warning(f'Invalid closes data for FFT in {timeframe}.')
        print(f"Invalid closes data for FFT in {timeframe}.")
        return datetime.datetime.now(), Decimal('0'), Decimal('0'), Decimal('0'), "Neutral", False, True, 0.0, "Neutral"
    
    fft = fftpack.fft(closes)
    frequencies = fftpack.fftfreq(len(closes))
    amplitudes = np.abs(fft)
    
    idx_max = np.argmax(amplitudes[1:]) + 1
    predominant_freq = frequencies[idx_max]
    predominant_sign = "Positive" if fft[idx_max].real > 0 else "Negative"
    # Apply sign to frequency value
    signed_freq = -predominant_freq if predominant_sign == "Negative" else predominant_freq
    
    current_close = Decimal(str(closes[-1]))
    is_near_min = abs(current_close - min_threshold) <= min_threshold * PROXIMITY_THRESHOLD
    is_near_max = abs(current_close - max_threshold) <= max_threshold * PROXIMITY_THRESHOLD
    
    if reversal_type == "DIP" and is_near_min and predominant_sign != "Negative":
        logging.warning(f"DIP reversal near min_threshold ({min_threshold:.25f}), but frequency is {predominant_sign}")
        print(f"DIP reversal near min_threshold ({min_threshold:.25f}), but frequency is {predominant_sign}")
    elif reversal_type == "TOP" and is_near_max and predominant_sign != "Positive":
        logging.warning(f"TOP reversal near max_threshold ({max_threshold:.25f}), but frequency is {predominant_sign}")
        print(f"TOP reversal near max_threshold ({max_threshold:.25f}), but frequency is {predominant_sign}")
    
    idx = np.argsort(amplitudes)[::-1][:n_components]
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.ifft(filtered_fft).real
    
    if reversal_type == "DIP" and max_threshold > middle_threshold > current_close:
        target_price = middle_threshold + Decimal('0.75') * (max_threshold - middle_threshold)
        target_price = min(target_price, max_threshold * (Decimal('1') - PROXIMITY_THRESHOLD))
        is_bullish_target = True
        is_bearish_target = False
    elif reversal_type == "TOP" and min_threshold < middle_threshold < current_close:
        target_price = middle_threshold - Decimal('0.75') * (middle_threshold - min_threshold)
        target_price = max(target_price, min_threshold * (Decimal('1') + PROXIMITY_THRESHOLD))
        is_bullish_target = False
        is_bearish_target = True
    else:
        target_price = Decimal(str(filtered_signal[-1]))
        is_bullish_target = target_price >= middle_threshold
        is_bearish_target = not is_bullish_target
        logging.warning(f"Invalid reversal or thresholds in {timeframe}. Using FFT target: {target_price:.25f}")
        print(f"Invalid reversal or thresholds in {timeframe}. Using FFT target: {target_price:.25f}")
    
    market_mood = "Bullish" if is_bullish_target else "Bearish"
    stop_loss = current_close - Decimal(str(3 * np.std(closes)))
    fastest_target = target_price + Decimal('0.005') * (target_price - current_close) if is_bullish_target else target_price - Decimal('0.005') * (current_close - target_price)
    
    logging.info(f"FFT - {timeframe} - Mood: {market_mood}, Close: {current_close:.25f}, Target: {fastest_target:.25f}, SL: {stop_loss:.25f}, Freq: {signed_freq:.6f}")
    print(f"FFT - {timeframe} - Mood: {market_mood}, Close: {current_close:.25f}, Target: {fastest_target:.25f}, SL: {stop_loss:.25f}, Freq: {signed_freq:.6f}")
    return datetime.datetime.now(), current_close, stop_loss, fastest_target, market_mood, is_bullish_target, is_bearish_target, signed_freq, predominant_sign

# Double Pattern Detection
def detect_double_pattern(candles, timeframe, min_threshold, max_threshold, reversal_type, current_close, predominant_freq, predominant_sign, momentum, buy_volume, sell_volume):
    if len(candles) < 20:
        logging.warning(f"Insufficient candles ({len(candles)}) for pattern detection in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for pattern detection in {timeframe}.")
        return True, False, "D. BOTTOM", Decimal('60'), Decimal('40')

    lookback = min(len(candles), 20)
    recent_candles = candles[-lookback:]
    lows = np.array([Decimal(str(c['low'])) for c in recent_candles])
    highs = np.array([Decimal(str(c['high'])) for c in recent_candles])
    
    low_indices = np.argsort(lows)[:2]
    high_indices = np.argsort(highs)[-2:][::-1]
    
    last_ll = lows[low_indices[0]]
    second_ll = lows[low_indices[1]]
    last_hh = highs[high_indices[0]]
    second_hh = highs[high_indices[1]]
    
    avg_volume = sum(Decimal(str(c['volume'])) for c in recent_candles[-5:]) / Decimal('5') if recent_candles else Decimal('1')
    recent_buy_vols = buy_volume.get(timeframe, [Decimal('0')] * 3)[-3:]
    recent_sell_vols = sell_volume.get(timeframe, [Decimal('0')] * 3)[-3:]
    
    # Smoothed volume using EMA
    buy_vols = np.array([float(v) for v in recent_buy_vols], dtype=np.float64)
    sell_vols = np.array([float(v) for v in recent_sell_vols], dtype=np.float64)
    buy_ema = Decimal(str(talib.EMA(buy_vols, timeperiod=3)[-1])) if len(buy_vols) >= 3 else sum(recent_buy_vols) / Decimal(len(recent_buy_vols)) if recent_buy_vols else Decimal('0')
    sell_ema = Decimal(str(talib.EMA(sell_vols, timeperiod=3)[-1])) if len(sell_vols) >= 3 else sum(recent_sell_vols) / Decimal(len(recent_sell_vols)) if recent_sell_vols else Decimal('0')
    
    is_volume_spike_up = buy_ema > avg_volume * VOLUME_SPIKE_MULTIPLIER
    is_volume_spike_down = sell_ema > avg_volume * VOLUME_SPIKE_MULTIPLIER
    
    # Frequency and momentum weights
    freq_weight = Decimal('1.5') * Decimal(str(abs(predominant_freq * 100))) if predominant_freq else Decimal('1')
    if reversal_type == "DIP" and predominant_sign == "Negative":
        freq_weight *= Decimal('1.2')
    elif reversal_type == "TOP" and predominant_sign == "Positive":
        freq_weight *= Decimal('1.2')
    momentum_weight = Decimal(str(abs(momentum / 1000))) if momentum and momentum != 0 else Decimal('1')
    
    # Volume weights
    volume_weight_db = Decimal('1.5') if is_volume_spike_up and reversal_type == "DIP" else Decimal('1')
    volume_weight_dt = Decimal('1.5') if is_volume_spike_down and reversal_type == "TOP" else Decimal('1')
    
    # Calculate formation intensity scores
    db_score = Decimal('0')
    dt_score = Decimal('0')
    
    # Check D. BOTTOM conditions: current_close > last_ll >= second_ll > min_threshold
    if (current_close > last_ll >= second_ll > min_threshold and 
        abs(last_ll - min_threshold) <= min_threshold * PROXIMITY_THRESHOLD):
        db_distance = current_close - last_ll
        db_volume = buy_ema if is_volume_spike_up else Decimal('1')
        db_score = db_distance * db_volume * freq_weight * momentum_weight * volume_weight_db
        if is_volume_spike_up and predominant_sign == "Negative" and momentum >= 0:
            db_score *= Decimal('1.5')
    
    # Check D. TOP conditions: current_close < last_hh <= second_hh < max_threshold
    if (current_close < last_hh <= second_hh < max_threshold and 
        abs(last_hh - max_threshold) <= max_threshold * PROXIMITY_THRESHOLD):
        dt_distance = last_hh - current_close
        dt_volume = sell_ema if is_volume_spike_down else Decimal('1')
        dt_score = dt_distance * dt_volume * freq_weight * momentum_weight * volume_weight_dt
        if is_volume_spike_down and predominant_sign == "Positive" and momentum < 0:
            dt_score *= Decimal('1.5')
    
    # Calculate formation intensity ratio
    total_score = db_score + dt_score
    if total_score == Decimal('0'):
        # Fallback logic when no clear pattern
        dist_to_min = abs(current_close - min_threshold)
        dist_to_max = abs(current_close - max_threshold)
        vol_ratio = buy_ema / sell_ema if sell_ema > Decimal('0') else Decimal('2') if buy_ema > Decimal('0') else Decimal('1')
        momentum_bias = momentum >= Decimal('0') if momentum != 0 else True
        if dist_to_min <= dist_to_max or (vol_ratio > Decimal('1') or momentum_bias):
            db_score = Decimal('70')  # Favor D. BOTTOM
            dt_score = Decimal('30')
        else:
            db_score = Decimal('30')
            dt_score = Decimal('70')  # Favor D. TOP
        total_score = db_score + dt_score
    
    db_ratio = (db_score / total_score) * Decimal('100')
    dt_ratio = Decimal('100') - db_ratio

    # Ensure one pattern is predominant (>50%)
    if db_ratio >= Decimal('50'):
        double_bottom = True
        double_top = False
        pattern_type = "D. BOTTOM"
        if db_ratio < Decimal('60'):
            db_ratio = Decimal('60')
            dt_ratio = Decimal('40')
    else:
        double_bottom = False
        double_top = True
        pattern_type = "D. TOP"
        if dt_ratio < Decimal('60'):
            dt_ratio = Decimal('60')
            db_ratio = Decimal('40')

    logging.info(f"{timeframe} - Pattern: D. BOTTOM={double_bottom}, D. TOP={double_top}, Type={pattern_type}, D. BOTTOM Ratio={db_ratio:.2f}%, D. TOP Ratio={dt_ratio:.2f}%")
    print(f"{timeframe} - Pattern: D. BOTTOM={double_bottom}, D. TOP={double_top}, Type={pattern_type}, D. BOTTOM Ratio={db_ratio:.2f}%, D. TOP Ratio={dt_ratio:.2f}%")
    return double_bottom, double_top, pattern_type, db_ratio, dt_ratio

# Volume Analysis
def calculate_volume(candles):
    if not candles:
        logging.warning("No candles for volume.")
        print("No candles provided.")
        return Decimal('0')
    total_volume = sum(Decimal(str(c['volume'])) for c in candles)
    logging.info(f"Total Volume: {total_volume:.25f}")
    print(f"Total Volume: {total_volume:.2f}")
    return total_volume

def calculate_buy_sell_volume(candle_map):
    if not candle_map or not any(candle_map.values()):
        logging.warning("Invalid candle_map.")
        print("Invalid candle_map.")
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), {"1m": "NEUTRAL", "3m": "NEUTRAL", "5m": "NEUTRAL"}
    
    try:
        buy_volume_1m = sum(Decimal(str(c["volume"])) for c in candle_map.get("1m", []) if Decimal(str(c["close"])) > Decimal(str(c["open"])))
        sell_volume_1m = sum(Decimal(str(c["volume"])) for c in candle_map.get("1m", []) if Decimal(str(c["close"])) < Decimal(str(c["open"])))
        buy_volume_3m = sum(Decimal(str(c["volume"])) for c in candle_map.get("3m", []) if Decimal(str(c["close"])) > Decimal(str(c["open"])))
        sell_volume_3m = sum(Decimal(str(c["volume"])) for c in candle_map.get("3m", []) if Decimal(str(c["close"])) < Decimal(str(c["open"])))
        buy_volume_5m = sum(Decimal(str(c["volume"])) for c in candle_map.get("5m", []) if Decimal(str(c["close"])) > Decimal(str(c["open"])))
        sell_volume_5m = sum(Decimal(str(c["volume"])) for c in candle_map.get("5m", []) if Decimal(str(c["close"])) < Decimal(str(c["open"])))
        
        volume_moods = {}
        for tf in TIMEFRAMES:
            buy_vol = locals()[f"buy_volume_{tf}"]
            sell_vol = locals()[f"sell_volume_{tf}"]
            volume_moods[tf] = "BULLISH" if buy_vol >= sell_vol else "BEARISH"
            logging.info(f"{tf} - Buy: {buy_vol:.25f}, Sell: {sell_vol:.25f}, Mood: {volume_moods[tf]}")
            print(f"{tf} - Buy: {buy_vol:.25f}, Sell: {sell_vol:.25f}, Mood: {volume_moods[tf]}")
        
        return buy_volume_1m, sell_volume_1m, buy_volume_3m, sell_volume_3m, buy_volume_5m, sell_volume_5m, volume_moods
    except Exception as e:
        logging.error(f"Volume error: {e}")
        print(f"Volume error: {e}")
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), {"1m": "NEUTRAL", "3m": "NEUTRAL", "5m": "NEUTRAL"}

def calculate_buy_sell_volume_original(candle_map):
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
            volume_ratio[timeframe] = {"buy_ratio": Decimal('0'), "sell_ratio": Decimal('0'), "status": "No Data"}
            logging.warning(f"No volume data for {timeframe}")
            print(f"No volume data for {timeframe}")
            continue
        recent_buy = buy_volume[timeframe][-3:]
        recent_sell = sell_volume[timeframe][-3:]
        if len(recent_buy) < 3 or len(recent_sell) < 3:
            volume_ratio[timeframe] = {"buy_ratio": Decimal('0'), "sell_ratio": Decimal('0'), "status": "No Data"}
            logging.warning(f"Insufficient volume for {timeframe}")
            print(f"Insufficient volume for {timeframe}")
            continue
        buy_vols = np.array([float(v) for v in recent_buy], dtype=np.float64)
        sell_vols = np.array([float(v) for v in recent_sell], dtype=np.float64)
        if len(buy_vols) >= 3:
            buy_ema = Decimal(str(talib.EMA(buy_vols, timeperiod=3)[-1]))
            sell_ema = Decimal(str(talib.EMA(sell_vols, timeperiod=3)[-1]))
        else:
            buy_ema = sum(recent_buy) / Decimal(str(len(recent_buy))) if recent_buy else Decimal('0')
            sell_ema = sum(recent_sell) / Decimal(str(len(recent_sell))) if recent_sell else Decimal('0')
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
        logging.info(f"{timeframe} - Buy Ratio: {buy_ratio:.2f}%, Sell Ratio: {sell_ratio:.2f}%, Status: {status}")
        print(f"{timeframe} - Buy Ratio: {buy_ratio:.2f}%")
        print(f"{timeframe} - Sell Ratio: {sell_ratio:.2f}%")
        print(f"{timeframe} - Volume Status: {status}")
    return volume_ratio

# Reversal Detection
def detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, buy_volume, sell_volume):
    if len(candles) < 3:
        logging.warning(f"Insufficient candles ({len(candles)}) for reversal in {timeframe}.")
        print(f"Insufficient candles ({len(candles)}) for reversal in {timeframe}.")
        return "NONE"

    lookback = min(len(candles), LOOKBACK_PERIODS[timeframe])
    recent_candles = candles[-lookback:]
    recent_buy_volume = buy_volume.get(timeframe, [Decimal('0')] * lookback)[-lookback:]
    recent_sell_volume = sell_volume.get(timeframe, [Decimal('0')] * lookback)[-lookback:]

    closes = np.array([float(c['close']) for c in recent_candles if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)
    lows = np.array([float(c['low']) for c in recent_candles if not np.isnan(c['low']) and c['low'] > 0], dtype=np.float64)
    highs = np.array([float(c['high']) for c in recent_candles if not np.isnan(c['high']) and c['high'] > 0], dtype=np.float64)
    times = np.array([c['time'] for c in recent_candles if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)

    if len(closes) < 3 or len(times) < 3:
        logging.warning(f"Insufficient data for reversal in {timeframe}")
        print(f"Insufficient data for reversal in {timeframe}")
        return "NONE"

    current_close = Decimal(str(closes[-1]))
    tolerance = current_close * PRICE_TOLERANCE

    min_time = 0
    max_time = 0
    closest_min_diff = Decimal('inf')
    closest_max_diff = Decimal('inf')
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
        logging.info(f"{timeframe} - Low: {closest_min_price:.25f} at {datetime.datetime.fromtimestamp(min_time)}")
        print(f"{timeframe} - Low: {closest_min_price:.25f} at {datetime.datetime.fromtimestamp(min_time)}")
    else:
        logging.warning(f"{timeframe} - No low found within {tolerance:.25f}")
        print(f"{timeframe} - No low found within {tolerance:.25f}")

    if max_time > 0:
        logging.info(f"{timeframe} - High: {closest_max_price:.25f} at {datetime.datetime.fromtimestamp(max_time)}")
        print(f"{timeframe} - High: {closest_max_price:.25f} at {datetime.datetime.fromtimestamp(max_time)}")
    else:
        logging.warning(f"{timeframe} - No high found within {tolerance:.25f}")
        print(f"{timeframe} - No high found within {tolerance:.25f}")

    price_range = max_threshold - min_threshold
    if price_range > Decimal('0'):
        min_pct = ((current_close - min_threshold) / price_range * Decimal('100')).quantize(Decimal('0.01'))
        max_pct = ((max_threshold - current_close) / price_range * Decimal('100')).quantize(Decimal('0.01'))
    else:
        min_pct = Decimal('50.00')
        max_pct = Decimal('50.00')
        logging.warning(f"{timeframe} - Zero price range. Defaulting to 50%.")
        print(f"{timeframe} - Zero price range. Defaulting to 50%.")
    logging.info(f"{timeframe} - Distance: {min_pct:.2f}% from min, {max_pct:.2f}% from max")
    print(f"{timeframe} - Distance: {min_pct:.2f}% from min, {max_pct:.2f}% from max")

    reversals = []
    if min_time > 0 and min_volume_confirmed:
        reversals.append({"type": "DIP", "price": closest_min_price, "time": min_time})
    if max_time > 0 and max_volume_confirmed:
        reversals.append({"type": "TOP", "price": closest_max_price, "time": max_time})

    if not reversals:
        logging.info(f"{timeframe} - No reversals detected.")
        print(f"{timeframe} - No reversals detected.")
        return "NONE"

    reversals.sort(key=lambda x: x["time"], reverse=True)
    most_recent = reversals[0]
    dist_to_min = abs(current_close - closest_min_price)
    dist_to_max = abs(current_close - closest_max_price)
    reversal_type = most_recent["type"]

    if reversal_type == "DIP" and dist_to_min <= dist_to_max:
        confirmed_type = "DIP"
    elif reversal_type == "TOP" and dist_to_max < dist_to_min:
        confirmed_type = "TOP"
    else:
        confirmed_type = "DIP" if dist_to_min <= dist_to_max else "TOP"

    logging.info(f"{timeframe} - Confirmed Reversal: {confirmed_type} at {most_recent['price']:.25f}")
    print(f"{timeframe} - Confirmed Reversal: {confirmed_type} at {most_recent['price']:.25f}")
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
            logging.info(f"Fetched {len(candles)} candles for {timeframe}")
            print(f"Fetched {len(candles)} candles for {timeframe}")
            return candles
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', '60') if e.response else '60'
            retry_after_int = int(retry_after) if retry_after.isdigit() else 60
            if e.code == -1003:
                logging.warning(f"Rate limit for {timeframe}. Waiting {retry_after_int}s.")
                print(f"Rate limit for {timeframe}. Waiting {retry_after_int}s.")
                time.sleep(retry_after_int)
            else:
                logging.error(f"API error for {timeframe} (attempt {attempt + 1}): {e.message}")
                print(f"API error for {timeframe}: {e.message}")
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.Timeout as e:
            logging.error(f"Timeout for {timeframe} (attempt {attempt + 1}): {e}")
            print(f"Timeout for {timeframe}: {e}")
            time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch {timeframe} candles after {retries} attempts.")
    print(f"Failed to fetch {timeframe} candles.")
    return []

def get_current_price(symbol=TRADE_SYMBOL, retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=symbol)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                logging.info(f"Price: {price:.25f}")
                print(f"Price: {price:.25f}")
                return price
            logging.warning(f"Invalid price {price:.25f}")
            print(f"Invalid price: {price:.25f}")
        except BinanceAPIException as e:
            retry_after = e.response.headers.get('Retry-After', '60') if e.response else '60'
            retry_after_int = int(retry_after) if retry_after.isdigit() else 60
            logging.error(f"Price error (attempt {attempt + 1}): {e.message}")
            print(f"Price error: {e.message}")
            time.sleep(retry_after_int if e.code == -1003 else delay * (attempt + 1))
        except requests.exceptions.Timeout as e:
            logging.error(f"Price timeout (attempt {attempt + 1}): {e}")
            print(f"Price timeout: {e}")
            time.sleep(delay * (attempt + 1))
    logging.error(f"Failed to fetch price after {retries} attempts.")
    print(f"Failed to fetch price.")
    return Decimal('0.0')

def get_balance(asset='USDC'):
    try:
        account = client.futures_account()
        for asset_info in account.get('assets', []):
            if asset_info.get('asset') == asset:
                wallet = Decimal(str(asset_info.get('walletBalance', '0.0')))
                logging.info(f"{asset} balance: {wallet:.25f}")
                print(f"{asset} balance: {wallet:.25f}")
                return wallet
        logging.warning(f"{asset} not found.")
        print(f"{asset} not found.")
        return Decimal('0.0')
    except BinanceAPIException as e:
        logging.error(f"Balance error: {e.message}")
        print(f"Balance error: {e.message}")
    except Exception as e:
        logging.error(f"Balance error: {e}")
        print(f"Balance error: {e}")
    return Decimal('0.0')

def get_position(symbol=TRADE_SYMBOL):
    try:
        positions = client.futures_position_information(symbol=symbol)
        if not positions:
            logging.warning(f"No position for {symbol}.")
            print(f"No position for {symbol}.")
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
        logging.error(f"Position error: {e.message}")
        print(f"Position error: {e.message}")
        return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0'), "initial_balance": Decimal('0.0'), "sl_price": Decimal('0.0'), "tp_price": Decimal('0.0')}

def check_open_orders(symbol=TRADE_SYMBOL):
    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        for order in orders:
            logging.info(f"Open order: {order['type']} at {order['stopPrice']}")
            print(f"Open order: {order['type']} at {order['stopPrice']}")
        return len(orders)
    except BinanceAPIException as e:
        logging.error(f"Open orders error: {e.message}")
        print(f"Open orders error: {e.message}")
        return 0

# Trading Functions
def calculate_quantity(balance, price):
    if price <= Decimal('0') or balance < MINIMUM_BALANCE:
        logging.warning(f"Insufficient balance ({balance:.25f}) or price ({price:.25f}).")
        print(f"Insufficient balance ({balance:.25f}) or price ({price:.25f}).")
        return Decimal('0.0')
    quantity = (balance * Decimal(str(LEVERAGE))) / price
    quantity = quantity.quantize(QUANTITY_PRECISION, rounding='ROUND_DOWN')
    logging.info(f"Quantity: {quantity:.25f} BTC")
    print(f"Quantity: {quantity:.25f} BTC")
    return quantity

def place_order(signal, quantity, current_price, initial_balance):
    try:
        if quantity <= Decimal('0'):
            logging.warning(f"Invalid quantity {quantity:.25f}.")
            print(f"Invalid quantity {quantity:.25f}")
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
            
            logging.info(f"Placed LONG: {quantity:.25f} BTC at ~{current_price:.25f}")
            print(f"\n=== TRADE ===")
            print(f"Side: LONG")
            print(f"Quantity: {quantity:.25f} BTC")
            print(f"Entry: {current_price:.25f} USDC")
            print(f"Balance: {initial_balance:.25f}")
            print(f"Stop-Loss: {sl_price:.25f} (-50%)")
            print(f"Take-Profit: {tp_price:.25f} (+5%)")
            print(f"\n==================")
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
            position["quantity"] = -quantity
            position["entry_price"] = current_price
            
            logging.info(f"Placed SHORT: {quantity:.25f} BTC at ~{current_price:.25f}")
            print(f"\n=== TRADE ===")
            print(f"Side: SHORT")
            print(f"Quantity: {quantity:.25f} BTC")
            print(f"Entry Price: {current_price:.25f} USDC")
            print(f"Balance: {initial_balance:.25f}")
            print(f"Stop-Loss: {sl_price:.25f} (-50%)")
            print(f"Take-Profit: {tp_price:.25f} (+5%)")
            print(f"\n==================")
        
        open_orders = check_open_orders()
        if open_orders > 0:
            logging.warning(f"Unexpected orders after {signal}.")
            print(f"Warning: {open_orders} orders after {signal}.")
        return position
    except BinanceAPIException as e:
        logging.error(f"Order error: {e.message}")
        print(f"Order error: {e.message}")
        return None

def close_position(position, current_price):
    if position["side"] == "NONE" or position["quantity"] == Decimal('0'):
        logging.info("No position to close.")
        print("No position to close.")
        return
    try:
        quantity = abs(position['quantity']).quantize(QUANTITY_PRECISION)
        side = "SELL" if position["side"] == "LONG" else "BUY"
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side=side,
            type="MARKET",
            quantity=str(quantity)
        )
        logging.info(f"Closed {position['side']}: {quantity:.25f} at {current_price:.25f}")
        print(f"Closed {position['side']}: {quantity:.25f} at {current_price:.25f}")
    except Exception as e:
        logging.error(f"Close error: {e}")
        print(f"Close error: {e}")

# Analysis Functions
def calculate_thresholds(candles):
    if not candles:
        logging.warning("No candles for thresholds.")
        print("No candles provided.")
        return Decimal('0'), Decimal('0'), Decimal('0')
    lookback = min(len(candles), LOOKBACK_PERIODS[candles[0]['timeframe']])
    highs = np.array([float(c['high']) for c in candles[-lookback:] if not np.isnan(c['high']) and c['high'] > 0], dtype=np.float64)
    lows = np.array([float(c['low']) for c in candles[-lookback:] if not np.isnan(c['low']) and c['low'] > 0], dtype=np.float64)
    closes = np.array([float(c['close']) for c in candles[-lookback:] if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)

    if len(closes) == 0 or len(highs) == 0 or len(lows) == 0:
        logging.warning(f"No valid data. Candles: {len(candles)}")
        print(f"No valid data for {len(candles)} candles.")
        return Decimal('0'), Decimal('0'), Decimal('0')
    
    current_close = Decimal(str(closes[-1]))
    min_price = Decimal(str(np.min(lows)))
    max_price = Decimal(str(np.max(highs)))
    min_threshold = min_price if min_price < current_close else current_close * Decimal('0.995')
    max_threshold = max_price if max_price > current_close else current_close * Decimal('1.005')
    middle_threshold = (min_threshold + max_threshold) / Decimal('2')

    timeframe = candles[0]['timeframe']
    logging.info(f"Thresholds for {timeframe}: Min: {min_threshold:.25f}, Mid: {middle_threshold:.25f}, Max: {max_threshold:.25f}")
    print(f"{timeframe} - Min: {min_threshold:.2f}")
    print(f"{timeframe} - Mid: {middle_threshold:.2f}")
    print(f"{timeframe} - Max: {max_threshold:.25f}")
    print(f"{timeframe} - Close: {current_close:.25f}")
    return min_threshold, middle_threshold, max_threshold

# Main Loop
def main():
    global conditions_long, conditions_short
    timeframes = TIMEFRAMES
    logging.info("Bot initialized!")
    print("Bot initialized!")

    try:
        while True:
            current_time = datetime.datetime.now()
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Time: {current_time_str}")
            print(f"Time: {current_time_str}")

            candle_map = fetch_candles_in_parallel(timeframes)
            if not candle_map or not any(candle_map.values()):
                logging.warning("No candles. Retrying in 60s.")
                print("No candles. Retrying in 60s.")
                time.sleep(60)
                continue

            current_price = get_current_price()
            if current_price <= Decimal('0'):
                logging.warning(f"Price: {current_price:.25f}. API failing.")
                print(f"Warning: Price: {current_price:.25f}. API failing.")
                time.sleep(60)
                continue
            
            usdc_balance = get_balance('USDC')
            position = get_position()

            if position["side"] != "NONE" and position["sl_price"] > Decimal('0'):
                if position["side"] == "LONG":
                    if current_price <= position["sl_price"]:
                        logging.info(f"SL LONG at {current_price:.25f}")
                        print(f"SL LONG at {current_price:.25f}")
                        close_position(position, current_price)
                        position = get_position()
                    elif current_price >= position["tp_price"]:
                        logging.info(f"TP LONG at {current_price:.25f}")
                        print(f"TP LONG at {current_price:.25f}")
                        close_position(position, current_price)
                        position = get_position()
                elif position["side"] == "SHORT":
                    if current_price >= position["sl_price"]:
                        logging.info(f"SL SHORT at {current_price:.25f}")
                        print(f"SL SHORT at {current_price:.25f}")
                        close_position(position, current_price)
                        position = get_position()
                    elif current_price <= position["tp_price"]:
                        logging.info(f"TP SHORT at {current_price:.25f}")
                        print(f"TP SHORT at {current_price:.25f}")
                        close_position(position, current_price)
                        position = get_position()

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
                "below_middle_5m": False,
                "fft_bullish_1m": False,
                "fft_bullish_3m": False,
                "fft_bullish_5m": False,
                "double_bottom_1m": False,
                "double_bottom_3m": False,
                "double_bottom_5m": False,
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
                "above_middle_5m": False,
                "fft_bearish_1m": False,
                "fft_bearish_3m": False,
                "fft_bearish_5m": False,
                "double_top_1m": False,
                "double_top_3m": False,
                "double_top_5m": False,
            }
            
            buy_volume, sell_volume = calculate_buy_sell_volume_original(candle_map)
            volume_ratio = calculate_volume_ratio(buy_volume, sell_volume)
            buy_vol_1m, sell_vol_1m, buy_vol_3m, sell_vol_3m, buy_vol_5m, sell_vol_5m, volume_moods = calculate_buy_sell_volume(candle_map)

            for timeframe in timeframes:
                print(f"\n--- {timeframe} Analysis ---")
                
                if not candle_map.get(timeframe):
                    logging.warning(f"No data for {timeframe}.")
                    print(f"No data for {timeframe}.")
                    conditions_long.update({f"{k}_{timeframe}": True for k in ["volume_bullish", "below_middle", "fft_bullish", "dip_confirmation", "double_bottom"]})
                    conditions_short.update({f"{k}_{timeframe}": False for k in ["volume_bearish", "above_middle", "fft_bearish", "top_confirmation", "double_top"]})
                    if timeframe == "1m":
                        conditions_long["momentum_positive_1m"] = True
                        conditions_short["momentum_negative_1m"] = False
                    continue
                
                candles = candle_map[timeframe]
                closes = np.array([float(c['close']) for c in candles if not np.isnan(c['close']) and c['close'] > 0], dtype=np.float64)
                
                min_threshold, middle_threshold, max_threshold = calculate_thresholds(candles)
                reversal_type = detect_recent_reversal(candles, timeframe, min_threshold, max_threshold, buy_volume, sell_volume)
                
                current_close = Decimal(str(closes[-1])) if len(closes) > 0 else Decimal('0')
                prev_close = Decimal(str(closes[-2])) if len(closes) >= 2 else current_close
                current_quadrant, last_quadrant, incoming_quadrant, degrees = map_price_to_quadrant(
                    current_close, min_threshold, max_threshold, reversal_type, timeframe, prev_close
                )

                # FFT Analysis
                print(f"\n--- {timeframe} FFT ---")
                predominant_freq = 0.0
                predominant_sign = "Neutral"
                if len(closes) >= 2:
                    current_time, entry_price, stop_loss, fastest_target, market_mood, is_bullish_target, is_bearish_target, predominant_freq, predominant_sign = get_target_price(
                        closes, n_components=5, timeframe=timeframe, reversal_type=reversal_type, min_threshold=min_threshold, middle_threshold=middle_threshold, max_threshold=max_threshold
                    )
                    print(f"{timeframe} - Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{timeframe} - Mood: {market_mood}")
                    print(f"{timeframe} - Close: {entry_price:.25f}")
                    print(f"{timeframe} - Target: {fastest_target:.25f}")
                    print(f"{timeframe} - SL: {stop_loss:.25f}")
                    print(f"{timeframe} - Bullish: {is_bullish_target}")
                    print(f"{timeframe} - Bearish: {is_bearish_target}")
                    print(f"{timeframe} - Freq: {predominant_freq:.6f}, Sign: {predominant_sign}")
                    
                    conditions_long[f"fft_bullish_{timeframe}"] = is_bullish_target
                    conditions_short[f"fft_bearish_{timeframe}"] = is_bearish_target
                else:
                    logging.warning(f"{timeframe} - FFT: No data")
                    print(f"{timeframe} - FFT: No data")
                    conditions_long[f"fft_bullish_{timeframe}"] = True
                    conditions_short[f"fft_bearish_{timeframe}"] = False
                
                # Volume Analysis
                print(f"\n--- {timeframe} Volume ---")
                total_volume = calculate_volume(candles)
                
                if timeframe == "1m":
                    print(f"{timeframe} - Buy: {buy_vol_1m:.25f}")
                    print(f"{timeframe} - Sell: {sell_vol_1m:.25f}")
                    print(f"{timeframe} - Mood: {volume_moods['1m']}")
                elif timeframe == "3m":
                    print(f"{timeframe} - Buy: {buy_vol_3m:.25f}")
                    print(f"{timeframe} - Sell: {sell_vol_3m:.25f}")
                    print(f"{timeframe} - Mood: {volume_moods['3m']}")
                elif timeframe == "5m":
                    print(f"{timeframe} - Buy: {buy_vol_5m:.25f}")
                    print(f"{timeframe} - Sell: {sell_vol_5m:.25f}")
                    print(f"{timeframe} - Mood: {volume_moods['5m']}")

                # Price Analysis
                print(f"\n--- {timeframe} Price ---")
                
                if min_threshold == Decimal('0') or max_threshold == Decimal('0'):
                    logging.warning(f"No thresholds for {timeframe}.")
                    print(f"No thresholds for {timeframe}.")
                    conditions_long.update({f"{k}_{timeframe}": True for k in ["below_middle", "volume_bullish", "dip_confirmation", "double_bottom"]})
                    conditions_short.update({f"{k}_{timeframe}": False for k in ["above_middle", "volume_bearish", "top_confirmation", "double_top"]})
                    continue

                conditions_long[f"below_middle_{timeframe}"] = current_close <= middle_threshold
                conditions_short[f"above_middle_{timeframe}"] = not conditions_long[f"below_middle_{timeframe}"]
                logging.info(f"{timeframe} - Close: {current_close:.25f}, Below: {conditions_long[f'below_middle_{timeframe}']}")
                print(f"{timeframe} - Close: {current_close:.25f}")
                print(f"{timeframe} - Below Mid: {conditions_long[f'below_middle_{timeframe}']}")
                print(f"{timeframe} - Above Mid: {conditions_short[f'above_middle_{timeframe}']}")

                buy_vol = buy_volume.get(timeframe, [Decimal('0')])[-1]
                sell_vol = sell_volume.get(timeframe, [Decimal('0')])[-1]
                price_trend = Decimal(str(closes[-1])) - Decimal(str(closes[-2])) if len(closes) >= 2 else Decimal('0')
                conditions_long[f"volume_bullish_{timeframe}"] = buy_vol > sell_vol or (buy_vol == sell_vol and price_trend >= Decimal('0'))
                conditions_short[f"volume_bearish_{timeframe}"] = not conditions_long[f"volume_bullish_{timeframe}"]
                logging.info(f"{timeframe} - Bullish: {buy_vol:.25f}, Bearish: {sell_vol:.25f}")
                print(f"{timeframe} - Bullish Vol: {buy_vol:.25f}")
                print(f"{timeframe} - Bearish Vol: {sell_vol:.25f}")

                conditions_long[f"dip_confirmation_{timeframe}"] = reversal_type == "DIP" or reversal_type == "NONE"
                conditions_short[f"top_confirmation_{timeframe}"] = reversal_type == "TOP"
                logging.info(f"{timeframe} - Dip: {conditions_long[f'dip_confirmation_{timeframe}']}")
                print(f"{timeframe} - Dip: {conditions_long[f'dip_confirmation_{timeframe}']}")
                print(f"{timeframe} - Top: {conditions_short[f'top_confirmation_{timeframe}']}")

                # Momentum Analysis
                print(f"\n--- {timeframe} Momentum ---")
                current_momentum = Decimal('0')
                if len(closes) >= 14:
                    momentum = talib.MOM(closes, timeperiod=14)
                    if len(momentum) > 0 and not np.isnan(momentum[-1]):
                        current_momentum = Decimal(str(momentum[-1]))
                        if timeframe == "1m":
                            conditions_long[f"momentum_positive_1m"] = current_momentum >= Decimal('0')
                            conditions_short[f"momentum_negative_1m"] = not conditions_long[f"momentum_positive_1m"]
                            logging.info(f"1m Momentum: {current_momentum:.25f}, Positive: {conditions_long['momentum_positive_1m']}")
                            print(f"1m Momentum: {current_momentum:.25f}, Positive: {conditions_long['momentum_positive_1m']}")
                else:
                    logging.warning(f"{timeframe} Momentum: No data")
                    print(f"{timeframe} Momentum: No data")
                    if timeframe == "1m":
                        conditions_long["momentum_positive_1m"] = True
                        conditions_short["momentum_negative_1m"] = False
                        logging.info(f"1m Momentum: No data, Default Positive: True")
                        print(f"1m Momentum: No data, Default Positive: True")

                # Double Pattern Analysis
                print(f"\n--- {timeframe} Pattern ---")
                double_bottom, double_top, pattern_type, db_ratio, dt_ratio = detect_double_pattern(
                    candles, timeframe, min_threshold, max_threshold, reversal_type, current_close,
                    predominant_freq, predominant_sign, current_momentum, buy_volume, sell_volume
                )
                conditions_long[f"double_bottom_{timeframe}"] = double_bottom
                conditions_short[f"double_top_{timeframe}"] = double_top

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
                ("below_middle_5m", "above_middle_5m"),
                ("fft_bullish_1m", "fft_bearish_1m"),
                ("fft_bullish_3m", "fft_bearish_3m"),
                ("fft_bullish_5m", "fft_bearish_5m"),
                ("double_bottom_1m", "double_top_1m"),
                ("double_bottom_3m", "double_top_3m"),
                ("double_bottom_5m", "double_top_5m")
            ]
            logging.info("Condition Pairs:")
            print("\nCondition Pairs:")
            symmetry_valid = True
            for long_cond, short_cond in condition_pairs:
                valid = conditions_long[long_cond] != conditions_short[short_cond]
                logging.info(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]} {'✓' if valid else '✗'}")
                print(f"{long_cond}: {conditions_long[long_cond]}, {short_cond}: {conditions_short[short_cond]} {'✓' if valid else '✗'}")
                if not valid:
                    symmetry_valid = False
            logging.info(f"Symmetry: {'Valid' if symmetry_valid else 'Invalid'}")
            print(f"Symmetry: {'Valid' if symmetry_valid else 'Invalid'}")

            long_signal = all(
                conditions_long[key] for key in conditions_long if not key.startswith("double_bottom")
            ) and any(conditions_long[f"double_bottom_{tf}"] for tf in timeframes)
            short_signal = all(
                conditions_short[key] for key in conditions_short if not key.startswith("double_top")
            ) and any(conditions_short[f"double_top_{tf}"] for tf in timeframes)

            logging.info("Signals:")
            print("\nSignals:")
            logging.info(f"LONG: {'Active' if long_signal else 'Inactive'}")
            print(f"LONG: {'Active' if long_signal else 'Inactive'}")
            logging.info(f"SHORT: {'Active' if short_signal else 'Inactive'}")
            print(f"SHORT: {'Active' if short_signal else 'Inactive'}")
            if long_signal and short_signal:
                logging.warning("Conflict: Both signals active.")
                print("Conflict: Both signals active.")

            logging.info("\nLong Conditions:")
            print("\nLong Conditions:")
            for condition, status in conditions_long.items():
                logging.info(f"{condition}: {'True' if status else 'False'}")
                print(f"{condition}: {'True' if status else 'False'}")
            logging.info("\nShort Conditions:")
            print("\nShort Conditions:")
            for condition, status in conditions_short.items():
                logging.info(f"{condition}: {'True' if status else 'False'}")
                print(f"{condition}: {'True' if status else 'False'}")

            long_true = sum(1 for val in conditions_long.values() if val)
            long_false = len(conditions_long) - long_true
            short_true = sum(1 for val in conditions_short.values() if val)
            short_false = len(conditions_short) - short_true
            logging.info(f"\nLong: {long_true} True, {long_false} False")
            print(f"\nLong: {long_true} True, {long_false} False")
            logging.info(f"Short: {short_true} True, {short_false} False")
            print(f"Short: {short_true} True, {short_false} False")

            signal = "NO_SIGNAL"
            if long_signal and not short_signal:
                signal = "LONG"
            elif short_signal and not long_signal:
                signal = "SHORT"
            logging.info(f"Signal: {signal}")
            print(f"Signal: {signal}")

            if usdc_balance < MINIMUM_BALANCE:
                logging.warning(f"Low balance: {usdc_balance:.25f}")
                print(f"Low balance: {usdc_balance:.25f}")
            elif signal in ["LONG", "SHORT"] and position["side"] == "NONE":
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)
            elif (signal == "LONG" and position["side"] == "SHORT") or (signal == "SHORT" and position["side"] == "LONG"):
                close_position(position, current_price)
                quantity = calculate_quantity(usdc_balance, current_price)
                position = place_order(signal, quantity, current_price, usdc_balance)

            if position["side"] != "NONE":
                print("\nPosition:")
                print(f"Side: {position['side']}")
                print(f"Quantity: {position['quantity']:.25f} BTC")
                print(f"Entry: {position['entry_price']:.25f} USDC")
                print(f"Price: {current_price:.25f} USDC")
                print(f"PNL: {position['unrealized_pnl']:.25f} USDC")
                print(f"SL: {position['sl_price']:.25f} USDC")
                print(f"TP: {position['tp_price']:.25f} USDC")
                current_balance = usdc_balance + position['unrealized_pnl']
                roi = ((current_balance - position['initial_balance']) / position['initial_balance'] * Decimal('100')).quantize(Decimal('0.01')) if position['initial_balance'] > Decimal('0') else Decimal('0')
                print(f"ROI: {roi:.2f}%")
                print(f"Initial: {position['initial_balance']:.25f}")
                print(f"Balance: {current_balance:.25f} USDC")
            else:
                print(f"\nNo position. Balance: {usdc_balance:.25f}")

            print(f"\nBalance: {usdc_balance:.25f}")
            print(f"Position: {position['side']}, Quantity: {position['quantity']:.25f} BTC")
            print(f"Price: {current_price:.25f}\n")

            del candle_map
            gc.collect()
            time.sleep(5)

    except Exception as e:
        logging.error(f"Main error: {e}")
        print(f"Main error: {e}")
        position = get_position()
        if position["side"] != "NONE":
            close_position(position, get_current_price())
        time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Shutting down...")
        print("Shutting down...")
        position = get_position()
        if position["side"] != "NONE":
            close_position(position, get_current_price())
        logging.info("Shutdown complete.")
        print("Shutdown complete.")
        exit(0)

if __name__ == "__main__":
    main()
