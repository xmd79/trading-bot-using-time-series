import requests
import numpy as np
import talib
from binance.client import Client as BinanceClient
import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Binance client setup
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    return BinanceClient(api_key, api_secret)

client = get_binance_client()

TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Fetch OHLC data
def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 1000
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
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

candles = get_candles(TRADE_SYMBOL, timeframes)
candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]
    candle_map.setdefault(timeframe, []).append(candle)

# Get current price
def get_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("price key not found in API response")

# Get close prices
def get_close(timeframe):
    closes = [c['close'] for c in candle_map[timeframe] if not np.isnan(c['close'])]
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)
    return np.array(closes)

# New Feature: Calculate additional metrics
def calculate_ohlc_metrics(candles):
    closes = np.array([c['close'] for c in candles])
    opens = np.array([c['open'] for c in candles])
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    volumes = np.array([c['volume'] for c in candles])
    
    # Price change (percentage)
    price_changes = np.diff(closes) / closes[:-1] * 100
    
    # Momentum (rate of change)
    momentum = talib.ROC(closes, timeperiod=10)
    
    # On-Balance Volume (OBV)
    obv = talib.OBV(closes, volumes)
    
    # Volume-Weighted Average Price (VWAP)
    typical_price = (highs + lows + closes) / 3
    vwap = np.cumsum(volumes * typical_price) / np.cumsum(volumes)
    
    # Average True Range (ATR)
    atr = talib.ATR(highs, lows, closes, timeperiod=14)
    
    return {
        'price_changes': price_changes,
        'momentum': momentum,
        'obv': obv,
        'vwap': vwap,
        'atr': atr
    }

# New Feature: Cycle detection using FFT
def detect_dominant_cycle(closes, sample_rate=1):
    N = len(closes)
    yf = fft(closes - np.mean(closes))
    xf = fftfreq(N, 1 / sample_rate)[:N//2]
    power = np.abs(yf[:N//2]) ** 2
    dominant_freq = xf[np.argmax(power[1:]) + 1]  # Skip zero frequency
    cycle_period = 1 / dominant_freq if dominant_freq != 0 else np.inf
    return cycle_period

# Calculate thresholds
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    min_percentage_custom = minimum_percentage / 100
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])
    
    avg_mtf = np.nanmean(close_prices)
    return min_threshold, max_threshold, avg_mtf

# Detect reversals and SMA positions
def detect_reversal_and_sma(candles):
    closes = np.array([c['close'] for c in candles])
    sma7 = talib.SMA(closes, timeperiod=7)
    sma12 = talib.SMA(closes, timeperiod=12)
    sma26 = talib.SMA(closes, timeperiod=26)
    sma45 = talib.SMA(closes, timeperiod=45)

    smas = {'SMA7': sma7[-1], 'SMA12': sma12[-1], 'SMA26': sma26[-1], 'SMA45': sma45[-1]}
    current_close = closes[-1]
    
    position_messages = []
    for label, value in smas.items():
        if current_close < value:
            position_messages.append(f"Current close is below {label}: {value:.2f}")
        elif current_close > value:
            position_messages.append(f"Current close is above {label}: {value:.2f}")
        else:
            position_messages.append(f"Current close is equal to {label}: {value:.2f}")
    
    return position_messages

# Calculate volume-weighted support/resistance
def calculate_enforced_support_resistance(candles):
    closes = np.array([c['close'] for c in candles])
    volumes = np.array([c['volume'] for c in candles])
    weighted_sma = talib.SMA(closes * volumes, timeperiod=20) / talib.SMA(volumes, timeperiod=20)
    support = np.nanmin(weighted_sma)
    resistance = np.nanmax(weighted_sma)
    return support, resistance

# Scale to sine wave for forecasting
def scale_to_sine(thresholds, last_reversal, cycles=5):
    min_threshold, max_threshold, avg_threshold = thresholds
    num_points = 100
    t = np.linspace(0, 2 * np.pi * cycles, num_points)
    amplitude = (max_threshold - min_threshold) / 2
    baseline = min_threshold + amplitude if last_reversal == "dip" else max_threshold - amplitude
    sine_wave = baseline + amplitude * np.sin(t) if last_reversal == "dip" else baseline - amplitude * np.sin(t)
    return sine_wave

# New: MTF Signal Logic
def calculate_mtf_signal(candles, timeframe, close_prices, metrics, cycle_period):
    closes = np.array([c['close'] for c in candles])
    volumes = np.array([c['volume'] for c in candles])
    
    # Signal components
    score = 0
    signals = []
    
    # 1. Price-based signals (SMA and threshold proximity)
    current_close = closes[-1]
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close_prices)
    sma7 = talib.SMA(closes, timeperiod=7)[-1]
    sma12 = talib.SMA(closes, timeperiod=12)[-1]
    
    if current_close > sma7 > sma12:
        score += 2
        signals.append("Bullish SMA crossover")
    elif current_close < sma7 < sma12:
        score -= 2
        signals.append("Bearish SMA crossover")
    
    if abs(current_close - min_threshold) < abs(current_close - max_threshold):
        score -= 1
        signals.append("Near support")
    else:
        score += 1
        signals.append("Near resistance")
    
    # 2. Volume-based signals
    obv = metrics['obv']
    if len(obv) > 1:
        if obv[-1] > obv[-2]:
            score += 1
            signals.append("Rising OBV")
        else:
            score -= 1
            signals.append("Falling OBV")
    
    # 3. Volatility and momentum
    atr = metrics['atr'][-1]
    momentum = metrics['momentum'][-1]
    if momentum > 0:
        score += 1
        signals.append("Positive momentum")
    elif momentum < 0:
        score -= 1
        signals.append("Negative momentum")
    
    # 4. Cycle phase (time-based)
    time_since_start = len(closes) % int(cycle_period)
    cycle_phase = time_since_start / cycle_period  # 0 to 1
    if 0.25 <= cycle_phase <= 0.75:  # Mid-cycle (trending)
        score += 1 if score > 0 else -1
        signals.append("Mid-cycle trend")
    else:  # Near reversal
        signals.append("Near cycle reversal")
    
    # Determine cycle status
    timeframe_weight = {'1m': 0.5, '3m': 0.6, '5m': 0.7, '15m': 0.8, '30m': 0.9, '1h': 1.0,
                        '2h': 1.1, '4h': 1.2, '6h': 1.3, '8h': 1.4, '12h': 1.5, '1d': 2.0}
    weighted_score = score * timeframe_weight[timeframe]
    
    if weighted_score > 2:
        cycle_status = "Up"
        confidence = min(weighted_score / 5, 1.0)
    elif weighted_score < -2:
        cycle_status = "Down"
        confidence = min(abs(weighted_score) / 5, 1.0)
    else:
        cycle_status = "Neutral"
        confidence = 0.5
    
    return {
        'cycle_status': cycle_status,
        'confidence': confidence,
        'signals': signals,
        'weighted_score': weighted_score
    }

# Enhanced Plotting
def plot_market_trend(close_prices, min_threshold, max_threshold, support, resistance, timeframe, mtf_signal, cycle_period):
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices, label='Close Prices', color='blue')
    plt.axhline(y=min_threshold, label='Min Threshold (Support)', color='green', linestyle='--')
    plt.axhline(y=max_threshold, label='Max Threshold (Resistance)', color='red', linestyle='--')
    plt.axhline(y=support, label='SMA Volume Support', color='purple', linestyle='-')
    plt.axhline(y=resistance, label='SMA Volume Resistance', color='orange', linestyle='-')
    
    # Plot cycle period
    plt.axvline(x=len(close_prices) - cycle_period, color='black', linestyle=':', label=f'Cycle Period ({cycle_period:.1f})')
    
    plt.title(f'Market Trend and MTF Signals - Timeframe: {timeframe}\nCycle: {mtf_signal["cycle_status"]} (Confidence: {mtf_signal["confidence"]:.2f})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Main Analysis Loop
mtf_signals = {}
for timeframe in timeframes:
    close = get_close(timeframe)
    candles_tf = candle_map[timeframe]
    
    # Calculate new metrics
    metrics = calculate_ohlc_metrics(candles_tf)
    
    # Detect dominant cycle
    cycle_period = detect_dominant_cycle(close)
    
    # Calculate thresholds
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close, period=14, minimum_percentage=2, maximum_percentage=2)
    
    # Detect reversals
    dip_signal = detect_reversal_and_sma(candles_tf)
    current_close = close[-1]
    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - current_close))
    last_reversal = "dip" if "below" in dip_signal[0] else "top"
    
    # Calculate support/resistance
    support, resistance = calculate_enforced_support_resistance(candles_tf)
    
    # MTF signal calculation
    mtf_signal = calculate_mtf_signal(candles_tf, timeframe, close, metrics, cycle_period)
    mtf_signals[timeframe] = mtf_signal
    
    # Plot results
    plot_market_trend(close, min_threshold, max_threshold, support, resistance, timeframe, mtf_signal, cycle_period)
    
    # Output results
    print("=" * 50)
    print(f"Timeframe: {timeframe}")
    print(f"Current Price: {current_close:.2f}")
    print(f"Minimum Threshold: {min_threshold:.2f}")
    print(f"Maximum Threshold: {max_threshold:.2f}")
    print(f"Average MTF: {avg_mtf:.2f}")
    print(f"Dominant Cycle Period: {cycle_period:.1f} candles")
    print(f"Cycle Status: {mtf_signal['cycle_status']} (Confidence: {mtf_signal['confidence']:.2f})")
    print("Signals:")
    for signal in mtf_signal['signals']:
        print(f"  - {signal}")
    
    # Forecast next reversal
    sine_wave = scale_to_sine((min_threshold, max_threshold, avg_mtf), last_reversal)
    if closest_threshold == max_threshold:
        next_reversal_target = np.min(sine_wave)
        print(f"Expected Next Dip Price: {next_reversal_target:.2f}")
    else:
        next_reversal_target = np.max(sine_wave)
        print(f"Expected Next Top Price: {next_reversal_target:.2f}")
    print("=" * 50)

# Aggregate MTF Signals
def aggregate_mtf_signals(mtf_signals):
    total_score = sum(s['weighted_score'] for s in mtf_signals.values())
    up_count = sum(1 for s in mtf_signals.values() if s['cycle_status'] == 'Up')
    down_count = sum(1 for s in mtf_signals.values() if s['cycle_status'] == 'Down')
    total_timeframes = len(mtf_signals)
    
    avg_confidence = np.mean([s['confidence'] for s in mtf_signals.values()])
    dominant_cycle = 'Up' if up_count > down_count else 'Down' if down_count > up_count else 'Neutral'
    
    print("\nMTF Signal Summary:")
    print(f"Dominant Cycle: {dominant_cycle}")
    print(f"Up Signals: {up_count}/{total_timeframes}")
    print(f"Down Signals: {down_count}/{total_timeframes}")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Total Weighted Score: {total_score:.2f}")

aggregate_mtf_signals(mtf_signals)