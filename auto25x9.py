"""
HFT Auto Trading Bot — KuCoin Futures Edition (CONCURRENT OPTIMIZED)
====================================================================
SIGNAL LOGIC:
  - STRICT 10/10: ALL conditions must be True simultaneously for a signal.
  - 1. Sine Scale (5m)        — dist_to_min < dist_to_max -> LONG
  - 2. Extrema Cycle (1m)     — Strictly Lowest Low vs Highest High (last 500)
  - 3. Momentum (1m)          — MOM > 0 -> LONG, < 0 -> SHORT
  - 4. Volume (1m)            — bullish% > bearish% -> LONG
  - 5. FFT Dominance (1m)     — neg spectral power dominant -> LONG
  - 6. ML Forecast (1m)       — Random Forest forecast_price > close -> LONG
  - 7. FFT Mood (1m)          — slope-projected next bar > close -> LONG (Bullish)
  - 8. SMA Stack (1m)         — close < SMA9 -> LONG / close > SMA9 -> SHORT
  - 9. SMA Stack (5m)         — close < SMA9 -> LONG / close > SMA9 -> SHORT
  - 10. LinReg (1m)           — Linear Regression forecast > close -> LONG

POSITION SIZING:
  - Only 10% of available balance is used per trade.
  - Remaining 90% stays untouched in the account.

25x Leverage
TP: 2.55% NET Profit (5.55% Gross ROE after 3.0% RT fee deduction)
SL: -99% ROE
"""

import time
import hmac
import json
import base64
import hashlib
import datetime
import os
import requests
import numpy as np
import talib
import gc
import logging
from decimal import Decimal
from scipy.fft import fft, fftfreq
import scipy.fftpack as fftpack
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# New ML Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Configure logging for the new ML forecast function
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR NEW ML LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def clean_data(arr):
    """Cleans NaN/0 values in an array by replacing them with the previous valid value."""
    arr = np.array(arr, dtype=np.float64)
    for i in range(len(arr)):
        if not np.isfinite(arr[i]) or arr[i] == 0:
            arr[i] = arr[i-1] if i > 0 else arr[0]
    return arr

def compute_Hc(data, kind='price', simplified=True):
    """Fallback for Hurst exponent calculation if 'hurst' package is not installed."""
    try:
        import hurst
        return hurst.compute_Hc(data, kind=kind, simplified=simplified)
    except ImportError:
        pass
    
    N = len(data)
    if N < 20:
        return 0.5, np.array([]), np.array([])
    
    data = np.array(data)
    R = np.max(data) - np.min(data)
    S = np.std(data)
    if S == 0:
        return 0.5, np.array([]), np.array([])
    H = np.log(R / S) / np.log(N / 2)
    return H, np.array([]), np.array([])

# ═══════════════════════════════════════════════════════════════════════════════
# TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

timing_stats = {}

def log_timing(section, elapsed_ms):
    if section not in timing_stats:
        timing_stats[section] = []
    timing_stats[section].append(elapsed_ms)
    if len(timing_stats[section]) > 30:
        timing_stats[section].pop(0)

def print_timing_summary():
    print(f"\n  ┌────────────────────── PERFORMANCE SUMMARY ──────────────────────┐")
    for section, values in timing_stats.items():
        if values:
            avg = sum(values) / len(values)
            mn = min(values)
            mx = max(values)
            print(f"  │ {section:<22} avg:{avg:>6.0f}ms min:{mn:>5.0f}ms max:{mx:>6.0f}ms │")
    total_avg = sum(timing_stats.get("total_loop", [1000])) / max(1, len(timing_stats.get("total_loop", [1])))
    print(f"  │ {'LOOP FREQUENCY':<22} ~{1000/max(1,total_avg):>5.1f} loops/sec{' '*25} │")
    print(f"  └─────────────────────────────────────────────────────────────────┘\n")

# ═══════════════════════════════════════════════════════════════════════════════
# FILE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_STATE_FILE = "trade_state.json"
TRADES_LOG_FILE = "trades.txt"
ANALYTICS_FILE = "analytics.json"

def clean_trades_file():
    try:
        with open(TRADES_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
        print(f"  [file] {TRADES_LOG_FILE} cleaned/created")
    except Exception as e:
        print(f"  [file] error cleaning {TRADES_LOG_FILE}: {e}")

_cleaned = []
for _f in [TRADE_STATE_FILE, ANALYTICS_FILE]:
    try:
        if os.path.exists(_f):
            os.remove(_f)
            _cleaned.append(_f)
    except Exception:
        pass
del _f
gc.collect()
clean_trades_file()

print("HFT KuCoin Bot (25x / STRICT 10/10 ALL MANDATORY) initialising...")
if _cleaned:
    print(f"  [cleanup] Wiped: {', '.join(_cleaned)}")
del _cleaned

# ═══════════════════════════════════════════════════════════════════════════════
# KUCOIN FUTURES REST CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

KUCOIN_FUTURES_BASE = "https://api-futures.kucoin.com"

_http_session = requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})

def _kucoin_sign(api_secret, timestamp, method, endpoint, body=""):
    msg = timestamp + method.upper() + endpoint + body
    return base64.b64encode(
        hmac.new(api_secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

def _sign_passphrase(api_secret, passphrase):
    return base64.b64encode(
        hmac.new(api_secret.encode("utf-8"), passphrase.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

class KuCoinFuturesClient:
    def __init__(self, api_key, api_secret, api_passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        self._signed_passphrase = _sign_passphrase(api_secret, api_passphrase)
        self.session = _http_session

    def _headers(self, method, signed_endpoint, body=""):
        ts = str(int(time.time() * 1000))
        sign = _kucoin_sign(self.api_secret, ts, method, signed_endpoint, body)
        return {
            "KC-API-KEY": self.api_key, "KC-API-SIGN": sign, "KC-API-TIMESTAMP": ts,
            "KC-API-PASSPHRASE": self._signed_passphrase, "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    def get(self, endpoint, params=None, timeout=5):
        if params:
            qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            signed_path = f"{endpoint}?{qs}"
        else:
            signed_path = endpoint
        url = KUCOIN_FUTURES_BASE + signed_path
        hdrs = self._headers("GET", signed_path)
        resp = self.session.get(url, headers=hdrs, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint, payload, timeout=5):
        body = json.dumps(payload, separators=(",", ":"))
        url = KUCOIN_FUTURES_BASE + endpoint
        hdrs = self._headers("POST", endpoint, body)
        resp = self.session.post(url, headers=hdrs, data=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_account_overview(self):
        return self.get("/api/v1/account-overview", {"currency": "USDT"})

    def get_klines(self, symbol, granularity, start_ms=None, end_ms=None):
        params = {"symbol": symbol, "granularity": granularity}
        if start_ms: params["from"] = start_ms
        if end_ms: params["to"] = end_ms
        return self.get("/api/v1/kline/query", params).get("data", [])

    def get_ticker(self, symbol):
        return self.get("/api/v1/ticker", {"symbol": symbol})

    def get_ticker_realtime(self, symbol):
        return self.get("/api/v1/ticker", {"symbol": symbol})

    def get_position(self, symbol):
        return self.get("/api/v1/position", {"symbol": symbol})

    def place_order(self, symbol, side, size, leverage):
        payload = {
            "clientOid": hashlib.md5(f"{time.time()}{side}".encode()).hexdigest(),
            "symbol": symbol, "side": side, "type": "market",
            "size": size, "leverage": str(leverage)
        }
        return self.post("/api/v1/orders", payload)

    def close_position(self, symbol):
        payload = {
            "clientOid": hashlib.md5(str(time.time()).encode()).hexdigest(),
            "symbol": symbol, "type": "market", "closeOrder": True
        }
        return self.post("/api/v1/orders", payload)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_SYMBOL = "XBTUSDTM"
LEVERAGE = 25
LOOP_SLEEP = 0.5
MIN_BALANCE_USDT = 5.0
TRADE_BALANCE_PCT = 0.10

ML_LOOKBACK = 100

ANALYSIS_WINDOW_5M = 1200
ANALYSIS_WINDOW_1M = 500

KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100
NET_PROFIT_ROE = 2.55
TAKE_PROFIT_ROE = NET_PROFIT_ROE + RT_FEE_ROE_PCT
STOP_LOSS_ROE = -99.0

TP_PRICE_PCT = TAKE_PROFIT_ROE / LEVERAGE / 100.0
SL_PRICE_PCT = abs(STOP_LOSS_ROE) / LEVERAGE / 100.0

FULL_REFRESH_INTERVAL = 3600

try:
    with open("credentials_kucoin.txt", "r") as _f:
        _lines = _f.readlines()
        _API_KEY, _API_SECRET, _API_PASSPHRASE = _lines[0].strip(), _lines[1].strip(), _lines[2].strip()
    client = KuCoinFuturesClient(_API_KEY, _API_SECRET, _API_PASSPHRASE)
    API_CONNECTED = True
except Exception as e:
    print(f"  [WARN] API credentials not found or invalid: {e}\n  [WARN] Running in SIMULATION-ONLY mode")
    client, API_CONNECTED = None, False

# ═══════════════════════════════════════════════════════════════════════════════
# CANDLE BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class CandleBuffer:
    def __init__(self, symbol, timeframe, max_candles=1200):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_candles = max_candles
        self.candles = []
        self.last_full_refresh = 0
        self.granularity = int(timeframe[0])
        self.duration_ms = self.granularity * 60 * 1000
        self.lock = Lock()

    def _parse_kline(self, k):
        try:
            return {
                "time": int(k[0]) / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            }
        except (IndexError, TypeError, ValueError):
            return None

    def initialize(self, client_obj):
        if not API_CONNECTED: return False
        with self.lock:
            print(f"  [Buffer:{self.timeframe}] Fetching {self.max_candles} candles (one-time)...")
            self.candles = []
            now_ms = int(time.time() * 1000)
            limit = 200
            num_requests = (self.max_candles + limit - 1) // limit

            for i in range(num_requests):
                end_ms = now_ms - (i * limit * self.duration_ms)
                start_ms = end_ms - (limit * self.duration_ms)
                try:
                    klines = client_obj.get_klines(self.symbol, self.granularity, start_ms=start_ms, end_ms=end_ms)
                    for k in reversed(klines):
                        candle = self._parse_kline(k)
                        if candle: self.candles.append(candle)
                except Exception as e:
                    print(f"  [Buffer:{self.timeframe}] Init error batch {i}: {e}")

            self._deduplicate()
            self.candles = self.candles[-self.max_candles:]
            self.last_full_refresh = time.time()
            print(f"  [Buffer:{self.timeframe}] Ready: {len(self.candles)} candles")
            return len(self.candles) >= 50

    def _deduplicate(self):
        seen_times = set()
        unique = []
        for c in reversed(self.candles):
            if c["time"] not in seen_times:
                seen_times.add(c["time"])
                unique.append(c)
        self.candles = list(reversed(unique))

    def update(self, client_obj, fetch_limit=3):
        if not API_CONNECTED or not self.candles: return 0
        with self.lock:
            now_ms = int(time.time() * 1000)
            end_ms = now_ms
            start_ms = end_ms - (fetch_limit * self.duration_ms)
            new_candles = []
            try:
                klines = client_obj.get_klines(self.symbol, self.granularity, start_ms=start_ms, end_ms=end_ms)
                for k in klines:
                    candle = self._parse_kline(k)
                    if candle: new_candles.append(candle)
            except Exception: return 0

            if self.candles:
                last_time = self.candles[-1]["time"]
                new_to_add = [c for c in new_candles if c["time"] > last_time]
            else:
                new_to_add = new_candles

            if new_to_add:
                self.candles.extend(new_to_add)
                self.candles = self.candles[-self.max_candles:]
            return len(new_to_add)

    def needs_full_refresh(self):
        if not self.last_full_refresh: return True
        return (time.time() - self.last_full_refresh) > FULL_REFRESH_INTERVAL

    def get_closes(self):
        with self.lock: return [c["close"] for c in self.candles]

    def get_candles(self):
        with self.lock: return self.candles.copy()

    def is_ready(self, min_candles=50):
        with self.lock: return len(self.candles) >= min_candles

    def __len__(self):
        with self.lock: return len(self.candles)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_account_balance():
    if not API_CONNECTED: return 0.0
    try: return float(client.get_account_overview()["data"]["availableBalance"])
    except Exception: return 0.0

def get_price(symbol):
    if not API_CONNECTED: return 0.0
    try: return float(client.get_ticker(symbol)["data"]["price"])
    except Exception: return 0.0

def get_realtime_price(symbol):
    if not API_CONNECTED: return 0.0
    try:
        data = client.get_ticker_realtime(symbol)["data"]
        ask = float(data.get("bestAskPrice") or 0)
        bid = float(data.get("bestBidPrice") or 0)
        if ask > 0 and bid > 0: return (ask + bid) / 2.0
        return float(data.get("price") or 0)
    except Exception: return 0.0

def get_position_info(symbol):
    empty = {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0, "mark_price": 0.0, "side": None, "size": 0}
    if not API_CONNECTED: return empty
    try:
        data = client.get_position(symbol).get("data", {})
        size = float(data.get("currentQty", 0) or 0)
        if size == 0: return empty
        upnl, margin = float(data.get("unrealisedPnl", 0) or 0), float(data.get("posMargin", 0) or 0)
        return {"is_open": True, "side": "long" if size > 0 else "short",
                "entry_price": float(data.get("avgEntryPrice", 0)), "mark_price": float(data.get("markPrice", 0)),
                "roe_pct": (upnl / margin * 100) if margin else 0.0, "size": size}
    except Exception: return empty

def get_buy_sell_volume_perc(candles):
    buy_vol = sell_vol = 0.0
    for c in candles:
        if c["close"] >= c["open"]: buy_vol += c["volume"]
        else: sell_vol += c["volume"]
    total_vol = buy_vol + sell_vol
    if total_vol == 0: return 50.0, 50.0
    return (buy_vol / total_vol) * 100.0, (sell_vol / total_vol) * 100.0

# ═══════════════════════════════════════════════════════════════════════════════
# CONCURRENT DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class ConcurrentDataFetcher:
    def __init__(self, buffer_5m, buffer_1m, symbol, max_workers=5):
        self.buffer_5m = buffer_5m
        self.buffer_1m = buffer_1m
        self.symbol = symbol
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def fetch_all_parallel(self, do_full_refresh=False):
        futures = {}
        results = {
            "price": 0.0, "balance": 0.0,
            "position": {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0,
                        "mark_price": 0.0, "side": None, "size": 0},
            "new_5m": 0, "new_1m": 0, "times": {}
        }
        t0 = time.perf_counter()

        if do_full_refresh or self.buffer_5m.needs_full_refresh():
            futures["5m"] = self.executor.submit(self._full_refresh_buffer, self.buffer_5m)
        else:
            futures["5m"] = self.executor.submit(self.buffer_5m.update, client, 3)

        if do_full_refresh or self.buffer_1m.needs_full_refresh():
            futures["1m"] = self.executor.submit(self._full_refresh_buffer, self.buffer_1m)
        else:
            futures["1m"] = self.executor.submit(self.buffer_1m.update, client, 3)

        futures["price"] = self.executor.submit(get_price, self.symbol)
        futures["balance"] = self.executor.submit(get_account_balance)
        futures["position"] = self.executor.submit(get_position_info, self.symbol)

        for name, future in futures.items():
            try:
                t_start = time.perf_counter()
                result = future.result(timeout=10)
                t_elapsed = (time.perf_counter() - t_start) * 1000
                results["times"][name] = t_elapsed
                if name == "5m": results["new_5m"] = result
                elif name == "1m": results["new_1m"] = result
                elif name == "price": results["price"] = result
                elif name == "balance": results["balance"] = result
                elif name == "position": results["position"] = result
            except Exception as e:
                results["times"][name] = -1
                print(f"  [Concurrent] Task '{name}' failed: {e}")

        results["parallel_total_ms"] = (time.perf_counter() - t0) * 1000
        return results

    def _full_refresh_buffer(self, buffer):
        buffer.initialize(client)
        return len(buffer)

    def shutdown(self):
        self.executor.shutdown(wait=False)

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def scale_to_sine(close_prices_5m, argmin_idx, argmax_idx):
    if len(close_prices_5m) < 32: return 50.0, 50.0
    sine_wave, _ = talib.HT_SINE(close_prices_5m)
    sine_wave = np.nan_to_num(-sine_wave)
    sine_window = sine_wave[-ANALYSIS_WINDOW_5M:] if len(sine_wave) >= ANALYSIS_WINDOW_5M else sine_wave
    cycle_min, cycle_max = sine_window[argmin_idx], sine_window[argmax_idx]
    rng = cycle_max - cycle_min if cycle_max != cycle_min else 1e-9
    current_sine = sine_wave[-1]
    dist_to_min = max(0, min(100, ((current_sine - cycle_min) / rng) * 100))
    dist_to_max = max(0, min(100, ((cycle_max - current_sine) / rng) * 100))
    return dist_to_min, dist_to_max

def analyze_fft_dominance_1m(close_prices_1m):
    if len(close_prices_1m) < 32: return False, False, 0.0, 0.0
    sine_wave_1m, lead_sine_1m = talib.HT_SINE(close_prices_1m)
    sine_wave_1m = np.nan_to_num(-sine_wave_1m)
    lead_sine_1m = np.nan_to_num(-lead_sine_1m)
    valid_mask = ~(np.isnan(sine_wave_1m) | np.isnan(lead_sine_1m))
    sine_wave_1m = sine_wave_1m[valid_mask]
    lead_sine_1m = lead_sine_1m[valid_mask]
    if len(sine_wave_1m) < 32: return False, False, 0.0, 0.0

    n = len(sine_wave_1m)
    complex_signal = sine_wave_1m + 1j * lead_sine_1m
    complex_signal = complex_signal * np.hanning(n) - np.mean(complex_signal)
    fft_result = fft(complex_signal)
    fft_power = np.abs(fft_result) ** 2
    freq_bins = fftfreq(n)
    powers, frequencies = fft_power[1:], freq_bins[1:]
    neg_powers, pos_powers = powers[frequencies < 0], powers[frequencies > 0]

    neg_power = np.sum(np.sort(neg_powers)[-12:]) if len(neg_powers) >= 12 else (np.sum(neg_powers) if len(neg_powers) > 0 else 0.0)
    pos_power = np.sum(np.sort(pos_powers)[-12:]) if len(pos_powers) >= 12 else (np.sum(pos_powers) if len(pos_powers) > 0 else 0.0)
    total_power = neg_power + pos_power
    if total_power == 0: return False, False, 0.0, 0.0

    neg_ratio = (neg_power / total_power) * 100
    pos_ratio = (pos_power / total_power) * 100
    DOMINANCE_THRESHOLD = 0.1
    return neg_ratio > (pos_ratio + DOMINANCE_THRESHOLD), pos_ratio > (neg_ratio + DOMINANCE_THRESHOLD), neg_ratio, pos_ratio

def calculate_momentum(close_arr, period=14):
    if len(close_arr) < period + 1: return np.nan
    return float(talib.MOM(close_arr, timeperiod=period)[-1])

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = np.nanmin(momentum)
    max_momentum = np.nanmax(momentum)
    min_percentage_custom = minimum_percentage / 100
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])
    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
    avg_mtf = np.nanmean(filtered_close)
    current_momentum = momentum[-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
        percent_to_max_momentum = ((current_momentum - min_momentum) / (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
    momentum_signal = percent_to_max_combined - percent_to_min_combined
    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

def price_regression(close):
    close_data = np.array(close)
    timestamps = np.arange(len(close_data))
    model = LinearRegression()
    model.fit(timestamps.reshape(-1, 1), close_data)
    num_targets = 1
    future_timestamps = np.arange(len(close_data), len(close_data) + num_targets)
    future_prices = model.predict(future_timestamps.reshape(-1, 1))
    return future_timestamps, future_prices

def generate_ml_forecast(candles, timeframe, forecast_periods=5):
    if len(candles) < ML_LOOKBACK: return 0.0, 0.0
    try:
        closes = clean_data([float(c['close']) for c in candles[-ML_LOOKBACK:]])
        highs = clean_data([float(c['high']) for c in candles[-ML_LOOKBACK:]])
        lows = clean_data([float(c['low']) for c in candles[-ML_LOOKBACK:]])
        volumes = clean_data([float(c['volume']) for c in candles[-ML_LOOKBACK:]])
        
        df = pd.DataFrame({'close': closes, 'high': highs, 'low': lows, 'volume': volumes})
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        for lag in range(1, 6): df[f'lagged_return_{lag}'] = df['returns'].shift(lag)
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
            
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        df['hurst'] = 0.5
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    H, _, _ = compute_Hc(window_data, kind='price', simplified=True)
                    df.at[df.index[i], 'hurst'] = H
                except: pass
        
        df['fft_real'] = 0.0; df['fft_imag'] = 0.0; df['fft_power'] = 0.0
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    fft_result = fft(window_data - np.mean(window_data))
                    power = np.abs(fft_result[1:]) ** 2
                    if len(power) > 0:
                        dominant_idx = np.argmax(power) + 1
                        df.at[df.index[i], 'fft_real'] = np.real(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_imag'] = np.imag(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_power'] = power[dominant_idx - 1]
                except: pass
                
        df['target'] = df['close'].shift(-forecast_periods) / df['close'] - 1
        df = df.dropna()
        if len(df) < 50: return 0.0, 0.0
        
        feature_columns = [col for col in df.columns if col not in ['close', 'high', 'low', 'volume', 'target']]
        X = df[feature_columns].values
        y = df['target'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        last_data = df.iloc[-1:][feature_columns].values
        last_data_scaled = scaler.transform(last_data)
        predicted_return = model.predict(last_data_scaled)[0]
        
        current_close = float(df['close'].iloc[-1])
        forecast_price = current_close * (1.0 + predicted_return)
        
        recent_returns = df['returns'].iloc[-5:].values
        if len(recent_returns) >= 3:
            if np.mean(recent_returns) > 0: forecast_price = max(forecast_price, current_close * 1.001)
            else: forecast_price = min(forecast_price, current_close * 0.999)
            
        recent_min = float(df['close'].iloc[-20:].min())
        recent_max = float(df['close'].iloc[-20:].max())
        forecast_price = max(forecast_price, recent_min * 0.95)
        forecast_price = min(forecast_price, recent_max * 1.05)
        
        return float(forecast_price), current_close
    except Exception as e:
        # print(f"  [ML Forecast] Error generating for {timeframe}: {e}")
        return 0.0, 0.0

def get_fft_market_mood(closes, n_components=5):
    if len(closes) < 50: return None, None
    closes_arr = np.array(closes, dtype=np.float64)
    current_close = float(closes_arr[-1])
    fft_vals = fftpack.rfft(closes_arr)
    idx = np.argsort(np.abs(fft_vals))[::-1][:n_components]
    filtered = np.zeros_like(fft_vals)
    filtered[idx] = fft_vals[idx]
    reconstructed = fftpack.irfft(filtered)
    slope_recon = (reconstructed[-1] - reconstructed[-4]) / 3.0
    offset = current_close - reconstructed[-1]
    forecast_price = reconstructed[-1] + slope_recon + offset
    mood = "Bullish" if slope_recon > 0 else "Bearish"
    return mood, float(forecast_price)


def compute_signals(buffer_5m, buffer_1m, live_price):
    candles_5m = buffer_5m.get_candles()
    candles_1m = buffer_1m.get_candles()
    closes_5m_raw = [c["close"] for c in candles_5m]
    closes_1m_raw = [c["close"] for c in candles_1m]

    if len(closes_5m_raw) < 50 or len(closes_1m_raw) < 50: return None

    close_arr_5m = np.array(closes_5m_raw, dtype=float)
    close_arr_1m = np.array(closes_1m_raw, dtype=float)

    # 1. Sine Scale (5m)
    last_1200_5m = close_arr_5m[-ANALYSIS_WINDOW_5M:]
    argmin_idx_5m = int(np.argmin(last_1200_5m))
    argmax_idx_5m = int(np.argmax(last_1200_5m))
    dist_to_min, dist_to_max = scale_to_sine(close_arr_5m, argmin_idx_5m, argmax_idx_5m)
    cond_sine_long = dist_to_min < dist_to_max
    cond_sine_short = dist_to_max < dist_to_min

    # 2. Extrema Cycle (1m) — STRICTLY LOWEST LOW VS HIGHEST HIGH REVERSALS
    candles_1m_window = candles_1m[-500:]
    
    # Strictly use 'low' array for absolute lowest and 'high' array for absolute highest
    lows_500 = np.array([c["low"] for c in candles_1m_window], dtype=np.float64)
    highs_500 = np.array([c["high"] for c in candles_1m_window], dtype=np.float64)
    closes_500 = np.array([c["close"] for c in candles_1m_window], dtype=np.float64)
    
    window_len_1m = len(candles_1m_window)
    
    # Find exact indices of the absolute extremes
    argmin_idx_1m = int(np.argmin(lows_500))
    argmax_idx_1m = int(np.argmax(highs_500))
    
    cycle_min_price = float(lows_500[argmin_idx_1m])
    cycle_max_price = float(highs_500[argmax_idx_1m])
    current_close_1m = float(close_arr_1m[-1])

    # Calculate exactly how many bars ago each occurred
    # Index 0 is oldest, Index (window_len_1m - 1) is current bar
    bars_ago_min = (window_len_1m - 1) - argmin_idx_1m
    bars_ago_max = (window_len_1m - 1) - argmax_idx_1m

    # Convert UTC timestamp to Local Datetime for printing
    try:
        ts_min = datetime.datetime.fromtimestamp(candles_1m_window[argmin_idx_1m]["time"]).strftime("%H:%M:%S")
        ts_max = datetime.datetime.fromtimestamp(candles_1m_window[argmax_idx_1m]["time"]).strftime("%H:%M:%S")
    except Exception:
        ts_min = ts_max = "N/A"

    # Calculate Thresholds for Momentum Context using close prices
    min_th, max_th, avg_mtf, momentum_sig, range_p = calculate_thresholds(
        closes_500, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
    )

    # Pure recency comparison (Lower bars_ago = More recent)
    if bars_ago_min > bars_ago_max:
        most_recent_extreme = "ARGMIN (LOW)"
        cond_cycle_long = True
        cond_cycle_short = False
    elif bars_ago_max > bars_ago_min:
        most_recent_extreme = "ARGMAX (HIGH)"
        cond_cycle_short = True
        cond_cycle_long = False
    else:
        most_recent_extreme = "TIE"
        cond_cycle_long = False
        cond_cycle_short = False

    # 3. Momentum (1m)
    mom_1m = calculate_momentum(close_arr_1m)
    if np.isnan(mom_1m): return None
    cond_mom_long = mom_1m > 0
    cond_mom_short = mom_1m < 0

    # 4. Volume (1m)
    bullish_perc, bearish_perc = get_buy_sell_volume_perc(candles_1m)
    cond_vol_long = bullish_perc > bearish_perc
    cond_vol_short = bearish_perc > bullish_perc

    # 5. FFT Dominance (1m)
    fft_long, fft_short, neg_ratio, pos_ratio = analyze_fft_dominance_1m(close_arr_1m)
    cond_fft_long = fft_long
    cond_fft_short = fft_short

    # 6. ML Forecast (1m) — MANDATORY (Random Forest)
    ml_forecast_price, ml_current_close = generate_ml_forecast(candles_1m, "1m")
    ml_forecast_valid = (ml_forecast_price > 0.0 and ml_current_close > 0.0)
    cond_ml_long  = ml_forecast_valid and (ml_forecast_price > ml_current_close)
    cond_ml_short = ml_forecast_valid and (ml_forecast_price < ml_current_close)

    # 7. FFT Market Mood (1m) — MANDATORY
    fft_mood, fft_mood_forecast = get_fft_market_mood(closes_1m_raw)
    cond_fft_mood_long  = (fft_mood == "Bullish")
    cond_fft_mood_short = (fft_mood == "Bearish")

    # 8. SMA Stack (1m) — MANDATORY (Rule: close < SMA9 -> LONG)
    sma9_1m  = float(np.mean(close_arr_1m[-9:]))
    cond_sma_long_1m  = (current_close_1m < sma9_1m)
    cond_sma_short_1m = (current_close_1m > sma9_1m)

    # 9. SMA Stack (5m) — MANDATORY (Rule: close < SMA9 -> LONG)
    current_close_5m = float(close_arr_5m[-1])
    sma9_5m  = float(np.mean(close_arr_5m[-9:]))
    cond_sma_long_5m  = (current_close_5m < sma9_5m)
    cond_sma_short_5m = (current_close_5m > sma9_5m)

    # 10. Price Regression (1m) — MANDATORY
    _, reg_future_prices = price_regression(close_arr_1m)
    reg_forecast = float(reg_future_prices[0])
    cond_reg_long = reg_forecast > current_close_1m
    cond_reg_short = reg_forecast < current_close_1m

    # ══════════════════════════════════════════════════════════════════
    # LOGIC: ALL 10 CONDITIONS MANDATORY
    # ══════════════════════════════════════════════════════════════════
    is_long  = (cond_sine_long  and cond_cycle_long  and cond_mom_long  and cond_vol_long
                and cond_fft_long  and cond_ml_long  and cond_fft_mood_long
                and cond_sma_long_1m  and cond_sma_long_5m and cond_reg_long)
    is_short = (cond_sine_short and cond_cycle_short and cond_mom_short and cond_vol_short
                and cond_fft_short and cond_ml_short and cond_fft_mood_short
                and cond_sma_short_1m and cond_sma_short_5m and cond_reg_short)

    return {
        "price": live_price, "is_long": is_long, "is_short": is_short,
        "cond_flags": {
            "sine_long": cond_sine_long, "sine_short": cond_sine_short,
            "cycle_long": cond_cycle_long, "cycle_short": cond_cycle_short,
            "mom_long": cond_mom_long, "mom_short": cond_mom_short,
            "vol_long": cond_vol_long, "vol_short": cond_vol_short,
            "fft_long": cond_fft_long, "fft_short": cond_fft_short,
            "ml_long": cond_ml_long, "ml_short": cond_ml_short,
            "fft_mood_long": cond_fft_mood_long, "fft_mood_short": cond_fft_mood_short,
            "sma_long_1m": cond_sma_long_1m, "sma_short_1m": cond_sma_short_1m,
            "sma_long_5m": cond_sma_long_5m, "sma_short_5m": cond_sma_short_5m,
            "reg_long": cond_reg_long, "reg_short": cond_reg_short,
        },
        "dist_to_min": dist_to_min, "dist_to_max": dist_to_max,
        "cycle_min_price": cycle_min_price, "cycle_max_price": cycle_max_price,
        "current_close_1m": current_close_1m, "current_close_5m": current_close_5m,
        "ts_min": ts_min, "ts_max": ts_max,
        "most_recent_extreme": most_recent_extreme,
        "momentum_signal": momentum_sig,
        "mom_1m": mom_1m, "bullish_perc": bullish_perc, "bearish_perc": bearish_perc,
        "neg_ratio": neg_ratio, "pos_ratio": pos_ratio,
        "ml_forecast_price": ml_forecast_price, "ml_current_close": ml_current_close,
        "fft_mood": fft_mood,
        "fft_mood_forecast": fft_mood_forecast if fft_mood_forecast is not None else 0.0,
        "sma9_1m": sma9_1m, "sma9_5m": sma9_5m,
        "reg_forecast": reg_forecast,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SL & TP CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(entry_price, side):
    tp_dist = entry_price * TP_PRICE_PCT
    sl_dist = entry_price * SL_PRICE_PCT
    if side == "long": sl_price, tp_price = entry_price - sl_dist, entry_price + tp_dist
    else: sl_price, tp_price = entry_price + sl_dist, entry_price - tp_dist
    print(f"  [Risk Calc] TP: {tp_price:.2f} ({TAKE_PROFIT_ROE}% ROE) | SL: {sl_price:.2f} ({STOP_LOSS_ROE}% ROE)")
    return float(sl_price), float(tp_price)

def check_sim_tp_sl(entry_price, current_price, side):
    if side == "long": price_change_pct = ((current_price - entry_price) / entry_price) * 100
    else: price_change_pct = ((entry_price - current_price) / entry_price) * 100
    roe_pct = price_change_pct * LEVERAGE - RT_FEE_ROE_PCT
    if roe_pct >= TAKE_PROFIT_ROE: return True, "TAKE PROFIT", roe_pct
    elif roe_pct <= STOP_LOSS_ROE: return True, "STOP LOSS", roe_pct
    return False, None, roe_pct

# ═══════════════════════════════════════════════════════════════════════════════
# STATE & JOURNALING
# ═══════════════════════════════════════════════════════════════════════════════

def save_trade_state(state):
    try:
        with open(TRADE_STATE_FILE, "w") as f: json.dump(state, f, indent=2)
    except Exception as e: print(f"  [state] save error: {e}")

def load_trade_state():
    if not os.path.exists(TRADE_STATE_FILE): return None
    try:
        with open(TRADE_STATE_FILE, "r") as f: return json.load(f)
    except Exception: return None

def clear_trade_state():
    try:
        if os.path.exists(TRADE_STATE_FILE): os.remove(TRADE_STATE_FILE)
    except Exception: pass

def write_trade_to_journal(trade_result):
    mode = trade_result.get("mode", "LIVE")
    line = (f"{'='*60}\nMODE: {mode}\nTYPE: {trade_result.get('type', 'N/A').upper()}\n"
            f"ENTRY TIME: {trade_result.get('entry_time', 'N/A')}\nEXIT TIME: {trade_result.get('exit_time', 'N/A')}\n"
            f"DURATION: {trade_result.get('duration', 'N/A')}\nENTRY PRICE: {trade_result.get('entry_price', 0):.2f}\n"
            f"EXIT PRICE: {trade_result.get('exit_price', 0):.2f}\nPRICE CHANGE: {trade_result.get('price_change_pct', 0):+.4f}%\n"
            f"GROSS ROE: {trade_result.get('gross_roe', 0):+.2f}%\nRT FEE DEDUCTED: {trade_result.get('rt_fee_pct', 0):.2f}%\n"
            f"NET ROE: {trade_result.get('roe', 0):+.2f}%\nREASON: {trade_result.get('reason', 'N/A')}\n"
            f"LEVERAGE: {LEVERAGE}x\nSTRATEGY: STRICT 10/10 ALL Mandatory | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}%\n{'='*60}\n\n")
    try:
        with open(TRADES_LOG_FILE, "a", encoding="utf-8") as f: f.write(line)
    except Exception as e: print(f"  [journal] write error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_contract_size(symbol):
    if not API_CONNECTED: return 0.001
    try: return float(client.get(f"/api/v1/contracts/{symbol}")["data"]["multiplier"])
    except Exception: return 0.001

def execute_entry(symbol, side, balance, price):
    try:
        trade_balance = balance * TRADE_BALANCE_PCT
        contracts = max(1, int((trade_balance * LEVERAGE) / (price * get_contract_size(symbol))))
        resp = client.place_order(symbol, side, contracts, LEVERAGE)
        if resp.get("data", {}).get("orderId"):
            print(f"  >>> {side.upper()} PLACED: {contracts} contracts @ ~{price:.2f}  (used {trade_balance:.2f} USDT / {TRADE_BALANCE_PCT*100:.0f}% of balance)")
            return True
        print(f"  [ORDER FAIL] {side.upper()} [{resp.get('code')}]: {resp.get('msg')}")
        return False
    except Exception as e:
        print(f"  [ORDER ERROR] {side.upper()}: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_duration(start_dt, end_dt):
    total_sec = int((end_dt - start_dt).total_seconds())
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_conditions(sig):
    f = sig["cond_flags"]
    print(f"  1. Sine (5m):      dMin:{sig['dist_to_min']:.1f}% dMax:{sig['dist_to_max']:.1f}% L:{f['sine_long']} S:{f['sine_short']} [MANDATORY]")
    print(f"  2. Cycle (1m):     Low:{sig['cycle_min_price']:.2f}@{sig['ts_min']} High:{sig['cycle_max_price']:.2f}@{sig['ts_max']} | Recency:{sig['most_recent_extreme']} | MomSig:{sig['momentum_signal']:.2f} | L:{f['cycle_long']} S:{f['cycle_short']} [MANDATORY]")
    print(f"  3. Mom (1m):       {sig['mom_1m']:.2f} L:{f['mom_long']} S:{f['mom_short']} [MANDATORY]")
    print(f"  4. Vol (1m):       Bull:{sig['bullish_perc']:.1f}% Bear:{sig['bearish_perc']:.1f}% L:{f['vol_long']} S:{f['vol_short']} [MANDATORY]")
    print(f"  5. FFT Dom (1m):   Neg:{sig['neg_ratio']:.2f}% Pos:{sig['pos_ratio']:.2f}% L:{f['fft_long']} S:{f['fft_short']} [MANDATORY]")
    
    ml_fc  = sig.get("ml_forecast_price", 0.0)
    ml_cur = sig.get("ml_current_close", 0.0)
    ml_diff = ((ml_fc - ml_cur) / ml_cur * 100) if ml_cur else 0.0
    print(f"  6. ML RF  (1m):    Cur:{ml_cur:.2f} Fcst:{ml_fc:.2f} ({ml_diff:+.4f}%) L:{f['ml_long']} S:{f['ml_short']} [MANDATORY]")
    
    fft_fc  = sig.get("fft_mood_forecast", 0.0)
    fft_cur = sig.get("current_close_1m", 0.0)
    fft_diff = ((fft_fc - fft_cur) / fft_cur * 100) if fft_cur else 0.0
    print(f"  7. FFT Mood (1m):  {sig.get('fft_mood','N/A'):<8} Cur:{fft_cur:.2f} FcstNext:{fft_fc:.2f} ({fft_diff:+.4f}%) L:{f['fft_mood_long']} S:{f['fft_mood_short']} [MANDATORY]")
    
    c1 = sig.get("current_close_1m", 0.0)
    s9_1 = sig.get("sma9_1m", 0.0)
    print(f"  8. SMA Stack (1m): Close:{c1:.2f} SMA9:{s9_1:.2f} L:{f['sma_long_1m']} S:{f['sma_short_1m']} [MANDATORY]")
    
    c5 = sig.get("current_close_5m", 0.0)
    s9_5 = sig.get("sma9_5m", 0.0)
    print(f"  9. SMA Stack (5m): Close:{c5:.2f} SMA9:{s9_5:.2f} L:{f['sma_long_5m']} S:{f['sma_short_5m']} [MANDATORY]")
    
    reg_fc = sig.get("reg_forecast", 0.0)
    reg_diff = ((reg_fc - c1) / c1 * 100) if c1 else 0.0
    print(f"  10.LinReg (1m):    Cur:{c1:.2f} Fcst:{reg_fc:.2f} ({reg_diff:+.4f}%) L:{f['reg_long']} S:{f['reg_short']} [MANDATORY]")

    # Explicit Overall Count Check
    long_all  = sum([f['sine_long'],  f['cycle_long'],  f['mom_long'],  f['vol_long'],  f['fft_long'],
                     f['ml_long'],  f['fft_mood_long'],  f['sma_long_1m'],  f['sma_long_5m'], f['reg_long']])
    short_all = sum([f['sine_short'], f['cycle_short'], f['mom_short'], f['vol_short'], f['fft_short'],
                     f['ml_short'], f['fft_mood_short'], f['sma_short_1m'], f['sma_short_5m'], f['reg_short']])
    print(f"  ═══ LONG:{long_all}/10 SHORT:{short_all}/10 (Rule: ALL 10 MANDATORY) -> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"HFT KuCoin Bot - STRICT 10/10 ALL CONDITIONS MANDATORY")
    print(f"{'='*60}")
    print(f"Symbol:              {TRADE_SYMBOL}")
    print(f"Leverage:            {LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_ROE}% ROE | SL: {STOP_LOSS_ROE}% ROE")
    print(f"Loop Sleep:          {LOOP_SLEEP}s")
    print(f"Entry Logic:         ALL 10 CONDITIONS MANDATORY")
    print(f"Extrema Logic:       Strictly Lowest Low vs Highest High (Last 500 1m bars)")
    print(f"SMA 1m Rule:         LONG: close < SMA9 | SHORT: close > SMA9")
    print(f"SMA 5m Rule:         LONG: close < SMA9 | SHORT: close > SMA9")
    print(f"Position Sizing:     {TRADE_BALANCE_PCT*100:.0f}% of balance per trade")
    print(f"API Connected:       {API_CONNECTED}")
    print(f"{'='*60}\n")

    buffer_5m = CandleBuffer(TRADE_SYMBOL, "5m", ANALYSIS_WINDOW_5M)
    buffer_1m = CandleBuffer(TRADE_SYMBOL, "1m", ANALYSIS_WINDOW_1M)

    if API_CONNECTED:
        print("  === INITIALIZING BUFFERS (one-time) ===\n")
        t0 = time.perf_counter()
        buffer_5m.initialize(client)
        buffer_1m.initialize(client)
        print(f"  Total init: {(time.perf_counter()-t0)*1000:.0f}ms\n")
        if not buffer_5m.is_ready() or not buffer_1m.is_ready():
            print("  [ERROR] Buffer init failed. Exiting.")
            return

    fetcher = ConcurrentDataFetcher(buffer_5m, buffer_1m, TRADE_SYMBOL, max_workers=5)

    saved_state = load_trade_state()
    if saved_state:
        print(f"  [recovery] State: {saved_state.get('mode')} {saved_state.get('side')} @ {saved_state.get('entry_price', 0):.2f}")

    loop_count = 0
    last_timing_print = time.time()
    last_buffer_log = 0

    print("  === STARTING MAIN LOOP ===\n")

    while True:
        try:
            loop_start = time.perf_counter()
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            loop_count += 1

            do_full_refresh = buffer_5m.needs_full_refresh() or buffer_1m.needs_full_refresh()
            data = fetcher.fetch_all_parallel(do_full_refresh=do_full_refresh)

            current_price = data["price"]
            balance = data["balance"]
            pos = data["position"]
            has_sufficient_balance = balance >= MIN_BALANCE_USDT

            parallel_ms = data["parallel_total_ms"]
            log_timing("parallel_total", parallel_ms)
            for k, v in data["times"].items():
                log_timing(f"task_{k}", v)

            if loop_count - last_buffer_log >= 100:
                print(f"  [Buffers] 5m:{len(buffer_5m)} 1m:{len(buffer_1m)} | Parallel: {parallel_ms:.0f}ms | Tasks: {data['times']}")
                last_buffer_log = loop_count

            t = time.perf_counter()
            sig = compute_signals(buffer_5m, buffer_1m, current_price)
            compute_ms = (time.perf_counter() - t) * 1000
            log_timing("compute_signals", compute_ms)

            # ── LIVE POSITION ACTIVE ──
            if pos["is_open"]:
                roe, side, entry_price, mark_price = pos["roe_pct"], pos["side"], pos["entry_price"], pos["mark_price"]
                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] === LIVE {side.upper()} | Entry:{entry_price:.2f} Mark:{mark_price:.2f} ROE:{roe:+.2f}% ===")
                    if sig: print_conditions(sig)
                reason = None
                if roe >= TAKE_PROFIT_ROE: reason = "TAKE PROFIT"
                elif roe <= STOP_LOSS_ROE: reason = "STOP LOSS"
                if reason:
                    print(f"  >>> [{reason}] {roe:+.2f}% ROE - Closing...")
                    client.close_position(TRADE_SYMBOL)
                    price_change_pct = ((mark_price - entry_price) / entry_price) * 100 if side == "long" else ((entry_price - mark_price) / entry_price) * 100
                    start_dt = datetime.datetime.strptime(saved_state["entry_time"], "%Y-%m-%d %H:%M:%S")
                    trade_result = {"mode": "LIVE", "entry_time": saved_state["entry_time"], "exit_time": now_str,
                                    "type": side, "entry_price": entry_price, "exit_price": mark_price,
                                    "price_change_pct": price_change_pct, "gross_roe": price_change_pct * LEVERAGE,
                                    "rt_fee_pct": RT_FEE_ROE_PCT, "roe": price_change_pct * LEVERAGE - RT_FEE_ROE_PCT,
                                    "duration": format_duration(start_dt, datetime.datetime.now()), "reason": reason}
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                loop_ms = (time.perf_counter() - loop_start) * 1000
                log_timing("total_loop", loop_ms)
                time.sleep(LOOP_SLEEP)
                continue

            # ── SIMULATION POSITION ACTIVE ──
            if saved_state and saved_state.get("mode") == "SIMULATION":
                sim_entry_price, sim_side, sim_entry_time = saved_state["entry_price"], saved_state["side"], saved_state["entry_time"]
                sim_now_price = get_realtime_price(TRADE_SYMBOL) or current_price
                hit, reason, sim_roe = check_sim_tp_sl(sim_entry_price, sim_now_price, sim_side)
                if loop_count % 10 == 0:
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    gross_roe = ((sim_now_price - sim_entry_price) / sim_entry_price * 100 * LEVERAGE) if sim_side == "long" else ((sim_entry_price - sim_now_price) / sim_entry_price * 100 * LEVERAGE)
                    print(f"\n[{now_str}] === SIM {sim_side.upper()} | Entry:{sim_entry_price:.2f} Mark:{sim_now_price:.2f} | Gross:{gross_roe:+.2f}% | Net (after {RT_FEE_ROE_PCT:.1f}% fee):{sim_roe:+.2f}% ===")
                    if sig: print_conditions(sig)
                if hit and reason:
                    print(f"  >>> [SIM {reason}] {sim_roe:+.2f}% ROE")
                    price_change_pct = ((sim_now_price - sim_entry_price) / sim_entry_price * 100 if sim_side == "long" else ((sim_entry_price - sim_now_price) / sim_entry_price * 100))
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    trade_result = {"mode": "SIMULATION", "entry_time": sim_entry_time, "exit_time": now_str,
                                    "type": sim_side, "entry_price": sim_entry_price, "exit_price": sim_now_price,
                                    "price_change_pct": price_change_pct, "gross_roe": price_change_pct * LEVERAGE,
                                    "rt_fee_pct": RT_FEE_ROE_PCT, "roe": sim_roe,
                                    "duration": format_duration(start_dt, now), "reason": reason}
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                loop_ms = (time.perf_counter() - loop_start) * 1000
                log_timing("total_loop", loop_ms)
                time.sleep(LOOP_SLEEP)
                continue

            # ── ORPHANED STATE CLEANUP ──
            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print("  [recovery] Orphaned live state - clearing.")
                clear_trade_state()
                saved_state = None

            # ── SCANNING FOR NEW ENTRY ──
            if not pos["is_open"] and not saved_state:
                if not sig:
                    loop_ms = (time.perf_counter() - loop_start) * 1000
                    log_timing("total_loop", loop_ms)
                    time.sleep(LOOP_SLEEP)
                    continue

                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] Scanning (FLAT) | Net: {parallel_ms:.0f}ms | Calc: {compute_ms:.0f}ms")
                    print_conditions(sig)
                    print(f"  Balance: {balance:.2f} USDT  (trade alloc: {balance*TRADE_BALANCE_PCT:.2f} USDT / {TRADE_BALANCE_PCT*100:.0f}%)")

                if sig["is_long"]:
                    print(f"\n  *** ALL 10/10 CONDITIONS MET -> LONG ***")
                    entry_now = get_realtime_price(TRADE_SYMBOL) or sig["price"]
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing LONG @ {entry_now:.2f}...")
                        if execute_entry(TRADE_SYMBOL, "buy", balance, entry_now):
                            sl_price, tp_price = calculate_sl_tp(entry_now, "long")
                            saved_state = {"active": True, "mode": "LIVE", "side": "long",
                                           "entry_price": entry_now, "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIM] LONG @ {entry_now:.2f} (low balance)")
                        sl_price, tp_price = calculate_sl_tp(entry_now, "long")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "long",
                                       "entry_price": entry_now, "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                        save_trade_state(saved_state)

                elif sig["is_short"]:
                    print(f"\n  *** ALL 10/10 CONDITIONS MET -> SHORT ***")
                    entry_now = get_realtime_price(TRADE_SYMBOL) or sig["price"]
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing SHORT @ {entry_now:.2f}...")
                        if execute_entry(TRADE_SYMBOL, "sell", balance, entry_now):
                            sl_price, tp_price = calculate_sl_tp(entry_now, "short")
                            saved_state = {"active": True, "mode": "LIVE", "side": "short",
                                           "entry_price": entry_now, "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIM] SHORT @ {entry_now:.2f} (low balance)")
                        sl_price, tp_price = calculate_sl_tp(entry_now, "short")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "short",
                                       "entry_price": entry_now, "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                        save_trade_state(saved_state)

            if time.time() - last_timing_print >= 60:
                print_timing_summary()
                last_timing_print = time.time()

            loop_ms = (time.perf_counter() - loop_start) * 1000
            log_timing("total_loop", loop_ms)
            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n  [INFO] Keyboard interrupt received. Shutting down...")
            fetcher.shutdown()
            print_timing_summary()
            break
        except Exception as e:
            print(f"  [ERROR] Loop exception: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()