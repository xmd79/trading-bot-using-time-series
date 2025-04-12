import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.fftpack as fftpack
import math
from decimal import Decimal, getcontext
import requests

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})

# Utility Functions (as provided)
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=100, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
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
            print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts. Skipping timeframe.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.get_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                return price
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            print(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    return Decimal('0.0')

# Trendline Analysis Function for High-Frequency MTF
def analyze_trendlines_mtf_high_freq(timeframes=['1h', '30m', '15m', '5m', '3m', '1m'], symbol=TRADE_SYMBOL, limit=200):
    """
    Analyzes trendlines across high-frequency timeframes (1h, 30m, 15m, 5m, 3m, 1m), identifies 
    double/triple bottom/top patterns, and compares major reversals (argmin/argmax) for the current cycle.
    
    Args:
        timeframes (list): List of timeframes (e.g., ['1h', '30m', '15m', '5m', '3m', '1m']).
        symbol (str): Trading pair (e.g., 'BTCUSDC').
        limit (int): Number of candles to fetch per timeframe (default 200 for more data).
    
    Returns:
        dict: Analysis results including trendlines, patterns, reversals, and trade signals.
    """
    # Fetch candles for all timeframes
    candle_data = fetch_candles_in_parallel(timeframes, symbol, limit)
    current_price = get_current_price()
    if current_price == Decimal('0.0'):
        return {"error": "Failed to fetch current price"}
    
    results = {}
    
    # Helper function to detect swing points
    def find_swings(candles, price_type='low', lookback=3):
        prices = [c[price_type] for c in candles]
        swings = []
        for i in range(lookback, len(prices) - lookback):
            if price_type == 'low' and prices[i] == min(prices[i-lookback:i+lookback+1]):
                swings.append((i, prices[i], candles[i]['time']))
            elif price_type == 'high' and prices[i] == max(prices[i-lookback:i+lookback+1]):
                swings.append((i, prices[i], candles[i]['time']))
        return swings
    
    # Helper function to fit trendline
    def fit_trendline(points, timestamps):
        if len(points) < 2:
            return None, None
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(points)
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        return slope, intercept
    
    # Helper function to detect double/triple patterns
    def detect_patterns(swings, price_type='low', tolerance=0.003):
        patterns = []
        prices = [p for _, p, _ in swings]
        times = [t for _, _, t in swings]
        for i in range(len(prices)):
            matches = [(i, prices[i], times[i])]
            for j in range(i + 1, len(prices)):
                if abs((prices[j] - prices[i]) / prices[i]) <= tolerance:
                    matches.append((j, prices[j], times[j]))
            if len(matches) == 2:
                patterns.append(("double_" + ("bottom" if price_type == 'low' else "top"), matches))
            elif len(matches) >= 3:
                patterns.append(("triple_" + ("bottom" if price_type == 'low' else "top"), matches))
        return patterns
    
    # Helper function to find major reversals
    def find_major_reversals(candles, timeframe):
        closes = np.array([c['close'] for c in candles])
        times = [c['time'] for c in candles]
        
        # Find argmin and argmax
        min_idx = np.argmin(closes)
        max_idx = np.argmax(closes)
        
        reversals = [
            ("dip", min_idx, closes[min_idx], times[min_idx]),
            ("top", max_idx, closes[max_idx], times[max_idx])
        ]
        
        # Sort by time to get the most recent
        reversals.sort(key=lambda x: x[3], reverse=True)
        return reversals[:2]  # Return last two major reversals
    
    # Process each timeframe
    for tf, candles in candle_data.items():
        if not candles:
            results[tf] = {"error": "No candle data"}
            continue
        
        # Find swing lows and highs
        lookback = 3 if tf in ['1m', '3m', '5m'] else 5  # Adjust lookback for smaller TFs
        swing_lows = find_swings(candles, 'low', lookback)
        swing_highs = find_swings(candles, 'high', lookback)
        
        # Fit trendlines
        low_points = [p for _, p, _ in swing_lows]
        low_times = [t for _, _, t in swing_lows]
        high_points = [p for _, p, _ in swing_highs]
        high_times = [t for _, _, t in swing_highs]
        
        support_slope, support_intercept = fit_trendline(low_points, low_times)
        resistance_slope, resistance_intercept = fit_trendline(high_points, high_times)
        
        # Detect double/triple patterns
        tolerance = 0.003 if tf in ['1m', '3m', '5m'] else 0.005  # Tighter tolerance for smaller TFs
        support_patterns = detect_patterns(swing_lows, 'low', tolerance)
        resistance_patterns = detect_patterns(swing_highs, 'high', tolerance)
        
        # Find major reversals
        reversals = find_major_reversals(candles, tf)
        
        # Determine trend direction
        latest_close = candles[-1]['close']
        support_value = support_slope * candles[-1]['time'] + support_intercept if support_slope is not None else None
        resistance_value = resistance_slope * candles[-1]['time'] + resistance_intercept if resistance_slope is not None else None
        
        trend = "neutral"
        if support_slope is not None and resistance_slope is not None:
            price_proximity_support = abs(latest_close - support_value) / latest_close if support_value else float('inf')
            price_proximity_resistance = abs(latest_close - resistance_value) / latest_close if resistance_value else float('inf')
            if support_slope > 0 and price_proximity_support < 0.01:
                trend = "bullish"
            elif resistance_slope < 0 and price_proximity_resistance < 0.01:
                trend = "bearish"
        
        # Store results for timeframe
        results[tf] = {
            "trend": trend,
            "support_trendline": {
                "slope": support_slope,
                "intercept": support_intercept,
                "value_at_latest": support_value
            },
            "resistance_trendline": {
                "slope": resistance_slope,
                "intercept": resistance_intercept,
                "value_at_latest": resistance_value
            },
            "patterns": {
                "support": support_patterns,
                "resistance": resistance_patterns
            },
            "reversals": reversals,
            "current_price": float(current_price),
            "latest_close": latest_close
        }
    
    # MTF Filtering and Signal Generation
    signals = {}
    for tf in timeframes:
        if tf not in results or "error" in results[tf]:
            signals[tf] = {"signal": "none", "reason": "No data"}
            continue
        
        tf_data = results[tf]
        higher_tf = timeframes[timeframes.index(tf) - 1] if timeframes.index(tf) > 0 else None
        
        # Initialize signal
        signal = "none"
        reason = []
        
        # Check trend alignment
        is_bullish = tf_data["trend"] == "bullish"
        is_bearish = tf_data["trend"] == "bearish"
        
        # Check higher timeframe alignment
        if higher_tf and higher_tf in results:
            htf_data = results[higher_tf]
            htf_bullish = htf_data["trend"] == "bullish"
            htf_bearish = htf_data["trend"] == "bearish"
            
            if is_bullish and htf_bullish:
                signal = "buy"
                reason.append(f"Aligned bullish trend on {tf} and {higher_tf}")
            elif is_bearish and htf_bearish:
                signal = "sell"
                reason.append(f"Aligned bearish trend on {tf} and {higher_tf}")
        
        # Check patterns
        for pattern_type, matches in tf_data["patterns"]["support"]:
            if "double_bottom" in pattern_type or "triple_bottom" in pattern_type:
                if not higher_tf or (higher_tf and results[higher_tf]["trend"] != "bearish"):
                    signal = "buy" if signal == "none" else signal
                    reason.append(f"{pattern_type} detected on {tf}")
        for pattern_type, matches in tf_data["patterns"]["resistance"]:
            if "double_top" in pattern_type or "triple_top" in pattern_type:
                if not higher_tf or (higher_tf and results[higher_tf]["trend"] != "bullish"):
                    signal = "sell" if signal == "none" else signal
                    reason.append(f"{pattern_type} detected on {tf}")
        
        # Compare reversals
        reversals = tf_data["reversals"]
        if reversals:
            most_recent = reversals[0]
            second_recent = reversals[1] if len(reversals) > 1 else None
            current_price_float = float(current_price)
            
            if most_recent[0] == "dip" and current_price_float > most_recent[2]:
                if not higher_tf or (higher_tf and results[higher_tf]["trend"] != "bearish"):
                    signal = "buy" if signal == "none" else signal
                    reason.append(f"Price above recent major dip ({most_recent[2]}) on {tf}")
            elif most_recent[0] == "top" and current_price_float < most_recent[2]:
                if not higher_tf or (higher_tf and results[higher_tf]["trend"] != "bullish"):
                    signal = "sell" if signal == "none" else signal
                    reason.append(f"Price below recent major top ({most_recent[2]}) on {tf}")
        
        # Volume confirmation for smaller timeframes
        if tf in ['1m', '3m', '5m'] and signal != "none":
            latest_volume = candles[-1]['volume']
            avg_volume = np.mean([c['volume'] for c in candles[-10:]])
            if latest_volume < avg_volume * 0.5:
                signal = "none"
                reason.append(f"Low volume on {tf} invalidates signal")
        
        signals[tf] = {
            "signal": signal,
            "reason": "; ".join(reason) if reason else "No clear signal"
        }
    
    # Combine results with signals
    results["signals"] = signals
    return results

# Example Usage
if __name__ == "__main__":
    timeframes = ['1h', '30m', '15m', '5m', '3m', '1m']
    analysis = analyze_trendlines_mtf_high_freq(timeframes, TRADE_SYMBOL, limit=200)
    for tf, data in analysis.items():
        if tf == "signals":
            print("\nSignals:")
            for tf_signal, signal_data in data.items():
                print(f"{tf_signal}: {signal_data['signal']} ({signal_data['reason']})")
        elif "error" not in data:
            print(f"\nTimeframe: {tf}")
            print(f"Trend: {data['trend']}")
            print(f"Support Trendline: Slope={data['support_trendline']['slope']}, Value={data['support_trendline']['value_at_latest']}")
            print(f"Resistance Trendline: Slope={data['resistance_trendline']['slope']}, Value={data['resistance_trendline']['value_at_latest']}")
            print(f"Patterns: Support={data['patterns']['support']}, Resistance={data['patterns']['resistance']}")
            print(f"Reversals: {data['reversals']}")
            print(f"Current Price: {data['current_price']}, Latest Close: {data['latest_close']}")