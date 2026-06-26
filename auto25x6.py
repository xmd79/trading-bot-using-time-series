"""
HFT Auto Trading Bot — KuCoin Futures Edition v6c (STRICT BINARY 8-COND + ML + FFT)
==========================================================================
ENTRY LOGIC — ALL 8 CONDITIONS MUST BE TRUE SIMULTANEOUSLY:
  Every condition outputs EXACTLY one of {LONG=TRUE, SHORT=TRUE}.
  Never both true, never both false. Strict XOR per condition.
  1. Sine Scale (5m)           — directional bias from adaptive HT_SINE
  2. Extrema Cycle (1m)        — timing: most recent extreme matches direction
  3. Momentum (1m)             — MOM >= 0 long, < 0 short
  4. Volume (1m)               — bullish% >= bearish% long, else short
  5. FFT Dominance (1m)        — TOP 5 consistent freqs, 10pp threshold + fallback
  6. ML Forecast (1m)          — Random Forest multi-indicator regression forecast
  7. FFT Cycle Target (1m)     — WEIGHTED SCORING (phase 4, dir 3, reach 2, path 1)
  8. Regression Forecast (1m)  — weighted linear regression price forecast

POSITION SIZING: 5% of available balance per trade
LEVERAGE: 25x
TP: 2.55% NET ROE (5.55% gross after 3.0% RT fee)
SL: -50% ROE
"""

import time
import hmac
import json
import base64
import hashlib
import datetime
import os
import requests
import logging
import gc
import numpy as np
import pandas as pd
import talib
from decimal import Decimal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import scipy.fftpack as fftpack
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
            print(f"  │ {section:<24} avg:{avg:>6.0f}ms min:{mn:>5.0f}ms max:{mx:>6.0f}ms │")
    total_avg = sum(timing_stats.get("total_loop", [1000])) / max(1, len(timing_stats.get("total_loop", [1])))
    print(f"  │ {'LOOP FREQUENCY':<24} ~{1000/max(1,total_avg):>5.1f} loops/sec{' '*25} │")
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

print("HFT KuCoin Bot v6c (STRICT BINARY ALL-8 + RF ML + FFT) initialising...")
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
LOOP_SLEEP = 3.0
MIN_BALANCE_USDT = 5.0
TRADE_BALANCE_PCT = 0.05

ML_LOOKBACK = 100
REG_FORECAST_LOOKBACK = 60
ANALYSIS_WINDOW_5M = 1200
ANALYSIS_WINDOW_1M = 500

KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100
NET_PROFIT_ROE = 2.55
TAKE_PROFIT_ROE = NET_PROFIT_ROE + RT_FEE_ROE_PCT
STOP_LOSS_ROE = -50.0

TP_PRICE_PCT = TAKE_PROFIT_ROE / LEVERAGE / 100.0
SL_PRICE_PCT = abs(STOP_LOSS_ROE) / LEVERAGE / 100.0

FFT_TARGET_TOP_N = 3
FFT_TARGET_MIN_RATIO = 1.0
FFT_TARGET_WINDOW = 256
FFT_DOMINANCE_THRESHOLD = 10.0
FFT_TOP_N_FREQ = 5

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
        if not API_CONNECTED:
            return False
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
                        if candle:
                            self.candles.append(candle)
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
        if not API_CONNECTED or not self.candles:
            return 0
        with self.lock:
            now_ms = int(time.time() * 1000)
            end_ms = now_ms
            start_ms = end_ms - (fetch_limit * self.duration_ms)
            new_candles = []
            try:
                klines = client_obj.get_klines(self.symbol, self.granularity, start_ms=start_ms, end_ms=end_ms)
                for k in klines:
                    candle = self._parse_kline(k)
                    if candle:
                        new_candles.append(candle)
            except Exception:
                return 0
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
        if not self.last_full_refresh:
            return True
        return (time.time() - self.last_full_refresh) > FULL_REFRESH_INTERVAL

    def get_closes(self):
        with self.lock:
            return [c["close"] for c in self.candles]

    def get_candles(self):
        with self.lock:
            return self.candles.copy()

    def is_ready(self, min_candles=50):
        with self.lock:
            return len(self.candles) >= min_candles

    def __len__(self):
        with self.lock:
            return len(self.candles)

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
        if c["close"] >= c["open"]:
            buy_vol += c["volume"]
        else:
            sell_vol += c["volume"]
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
                if name == "5m":   results["new_5m"] = result
                elif name == "1m": results["new_1m"] = result
                elif name == "price":   results["price"] = result
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
# ML & FFT UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_data(arr):
    arr = np.array(arr, dtype=np.float64)
    for i in range(len(arr)):
        if not np.isfinite(arr[i]) or arr[i] <= 0:
            arr[i] = arr[i-1] if i > 0 else arr[0]
    return arr

def calculate_thresholds(closes):
    if len(closes) == 0: return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
    min_th = Decimal(str(np.min(closes)))
    q25 = Decimal(str(np.percentile(closes, 25)))
    q75 = Decimal(str(np.percentile(closes, 75)))
    max_th = Decimal(str(np.max(closes)))
    return min_th, q25, q75, max_th

def compute_Hc(data, kind='price', simplified=True):
    if simplified:
        N = len(data)
        if N < 20: return 0.5, [], []
        max_k = int(np.floor(N / 2))
        RS_list = []
        lags = range(2, max_k)
        for lag in lags:
            num_subseries = N // lag
            if num_subseries < 1: continue
            RS_vals = []
            for i in range(num_subseries):
                sub_series = data[i*lag : (i+1)*lag]
                if len(sub_series) < 2: continue
                mean_s = np.mean(sub_series)
                std_s = np.std(sub_series, ddof=1)
                if std_s == 0: continue
                cumdev = np.cumsum(sub_series - mean_s)
                R = np.max(cumdev) - np.min(cumdev)
                S = std_s
                RS_vals.append(R / S)
            if len(RS_vals) == 0: continue
            RS_list.append((lag, np.mean(RS_vals)))
        if len(RS_list) < 2: return 0.5, [], []
        lags_arr = np.array([x[0] for x in RS_list])
        rs_arr = np.array([x[1] for x in RS_list])
        try:
            poly = np.polyfit(np.log(lags_arr), np.log(rs_arr), 1)
            H = poly[0]
        except Exception:
            H = 0.5
        return H, lags_arr, rs_arr
    return 0.5, [], []

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS — CORE INDICATORS
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
    if len(close_prices_1m) < 32: return 50.0, 50.0
    close_arr = np.array(close_prices_1m, dtype=np.float64)
    sine_raw, lead_raw = talib.HT_SINE(close_arr)
    valid_mask = np.isfinite(sine_raw) & np.isfinite(lead_raw)
    sine_valid = sine_raw[valid_mask]
    lead_valid = lead_raw[valid_mask]
    if len(sine_valid) < 32: return 50.0, 50.0
    sine_valid = -sine_valid
    lead_valid = -lead_valid
    n = len(sine_valid)
    analytic = sine_valid + 1j * lead_valid
    analytic -= np.mean(analytic)
    window = np.hanning(n)
    analytic *= window
    spectrum = fft(analytic)
    power = np.abs(spectrum) ** 2
    freqs = fftfreq(n)
    min_freq = 1.0 / (n / 2)
    max_freq = 1.0 / 4.0
    band_mask = ((freqs < -min_freq) & (freqs >= -max_freq)) | \
                ((freqs > min_freq) & (freqs <= max_freq))
    valid_power = power[band_mask]
    valid_freqs = freqs[band_mask]
    if len(valid_power) == 0: return 50.0, 50.0
    top_n = min(FFT_TOP_N_FREQ, len(valid_power))
    top_indices = np.argsort(valid_power)[-top_n:]
    top_neg_power = 0.0
    top_pos_power = 0.0
    for idx in top_indices:
        if valid_freqs[idx] < 0: top_neg_power += valid_power[idx]
        else: top_pos_power += valid_power[idx]
    total = top_neg_power + top_pos_power
    if total < 1e-12: return 50.0, 50.0
    neg_ratio = (top_neg_power / total) * 100.0
    pos_ratio = (top_pos_power / total) * 100.0
    return neg_ratio, pos_ratio


def calculate_momentum(close_arr, period=14):
    if len(close_arr) < period + 1: return np.nan
    return float(talib.MOM(close_arr, timeperiod=period)[-1])

# ═══════════════════════════════════════════════════════════════════════════════
# ML FORECAST (RANDOM FOREST) — Condition #6
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ml_forecast(candles_1m):
    forecast_periods = 5
    if len(candles_1m) < ML_LOOKBACK:
        return 0.0, 0.0
    try:
        closes = np.array([float(c['close']) for c in candles_1m[-ML_LOOKBACK:]], dtype=np.float64)
        highs = np.array([float(c['high']) for c in candles_1m[-ML_LOOKBACK:]], dtype=np.float64)
        lows = np.array([float(c['low']) for c in candles_1m[-ML_LOOKBACK:]], dtype=np.float64)
        volumes = np.array([float(c['volume']) for c in candles_1m[-ML_LOOKBACK:]], dtype=np.float64)
        
        closes = clean_data(closes)
        highs = clean_data(highs)
        lows = clean_data(lows)
        volumes = clean_data(volumes)
        
        df = pd.DataFrame({'close': closes, 'high': highs, 'low': lows, 'volume': volumes})
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for lag in range(1, 6):
            df[f'lagged_return_{lag}'] = df['returns'].shift(lag)
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
                    
        df['fft_real'] = 0.0
        df['fft_imag'] = 0.0
        df['fft_power'] = 0.0
        for i in range(20, len(df)):
            window_data = df['close'].iloc[i-20:i].values
            if len(window_data) > 10:
                try:
                    fft_result = fft(window_data - np.mean(window_data))
                    freqs = np.fft.fftfreq(len(window_data))
                    power = np.abs(fft_result[1:]) ** 2
                    if len(power) > 0:
                        dominant_idx = np.argmax(power) + 1
                        df.at[df.index[i], 'fft_real'] = np.real(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_imag'] = np.imag(fft_result[dominant_idx])
                        df.at[df.index[i], 'fft_power'] = power[dominant_idx - 1]
                except: pass
                    
        df['target'] = df['close'].shift(-forecast_periods) / df['close'] - 1
        df_clean = df.dropna()
        
        if len(df_clean) < 50: return 0.0, 0.0
            
        feature_columns = [col for col in df.columns if col not in ['close', 'high', 'low', 'volume', 'target']]
        
        X_train = df_clean[feature_columns].values
        y_train = df_clean['target'].values
        
        scaler = StandardScaler()
        X_scaled_train = scaler.fit_transform(X_train)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X_scaled_train, y_train)
        
        # FIX: Predict on absolute latest row to avoid stale data
        last_data = df.iloc[-1:][feature_columns].values
        last_data_scaled = scaler.transform(last_data)
        predicted_return = model.predict(last_data_scaled)[0]
        
        current_close = Decimal(str(df['close'].iloc[-1]))
        forecast_price = current_close * (Decimal('1') + Decimal(str(predicted_return)))
        
        recent_returns = df['returns'].iloc[-5:].dropna().values
        if len(recent_returns) >= 3:
            if np.mean(recent_returns) > 0:
                forecast_price = max(forecast_price, current_close * Decimal('1.001'))
            else:
                forecast_price = min(forecast_price, current_close * Decimal('0.999'))
                
        recent_min = Decimal(str(df['close'].iloc[-20:].min()))
        recent_max = Decimal(str(df['close'].iloc[-20:].max()))
        forecast_price = max(forecast_price, recent_min * Decimal('0.95'))
        forecast_price = min(forecast_price, recent_max * Decimal('1.05'))
        
        return float(forecast_price), float(current_close)
    except Exception as e:
        print(f"  [ML Forecast] Error: {e}")
        return 0.0, 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# FFT FORECAST (DOMINANT FREQUENCY)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fft_forecast(candles_1m):
    if len(candles_1m) < 20: return 0.0, "N/A"
    try:
        closes = np.array([float(c['close']) for c in candles_1m], dtype=np.float64)
        closes = clean_data(closes)
        current_close = Decimal(str(closes[-1]))
        mean_close = np.mean(closes)
        
        fft_result = fft(closes - mean_close)
        freqs = np.fft.fftfreq(len(closes))
        magnitudes = np.abs(fft_result)
        
        pos_idx = np.where(freqs > 0)[0]
        neg_idx = np.where(freqs < 0)[0]
        pos_amp, neg_amp = 0, 0
        dominant_freq, dominant_amp, dominant_phase = 0, 0, 0
        cycle_direction = "N/A"
        
        if len(pos_idx) > 0:
            p_idx = pos_idx[np.argmax(magnitudes[pos_idx])]
            pos_amp = magnitudes[p_idx] / len(closes) * 2
            pos_phase = np.angle(fft_result[p_idx])
            pos_freq = freqs[p_idx]
        if len(neg_idx) > 0:
            n_idx = neg_idx[np.argmax(magnitudes[neg_idx])]
            neg_amp = magnitudes[n_idx] / len(closes) * 2
            neg_phase = np.angle(fft_result[n_idx])
            neg_freq = freqs[n_idx]
            
        if pos_amp > neg_amp:
            dominant_freq, dominant_amp, dominant_phase = pos_freq, pos_amp, pos_phase
            cycle_direction = "Down"
        else:
            dominant_freq, dominant_amp, dominant_phase = neg_freq, neg_amp, neg_phase
            cycle_direction = "Up"
            
        future_t = np.arange(len(closes), len(closes) + 5)
        forecast_val = 0
        if dominant_amp > 0:
            forecast_val = dominant_amp * np.cos(2 * np.pi * dominant_freq * future_t[-1] + dominant_phase)
            
        forecast_price = Decimal(str(forecast_val + mean_close))
        min_th, _, max_th, _ = calculate_thresholds(closes)
        amp_adjust = (max_th - min_th) / Decimal('2')
        mean_close_decimal = Decimal(str(mean_close))
        
        if cycle_direction == "Up":
            forecast_price = max(forecast_price, current_close)
            forecast_price = min(forecast_price, mean_close_decimal + amp_adjust)
        else:
            forecast_price = min(forecast_price, current_close)
            forecast_price = max(forecast_price, mean_close_decimal - amp_adjust)
            
        return float(forecast_price), cycle_direction
    except Exception as e:
        print(f"  [FFT Forecast] Error: {e}")
        return 0.0, "N/A"

# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE TARGET EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_target(closes, n_components=5, target_distance=56):
    if len(closes) < 20: return None, 0.0, 0.0, 0.0, "N/A"
    try:
        fft_res = fftpack.rfft(closes) 
        frequencies = fftpack.rfftfreq(len(closes))
        idx = np.argsort(np.abs(fft_res))[::-1][:n_components]
        filtered_fft = np.zeros_like(fft_res)
        filtered_fft[idx] = fft_res[idx]
        filtered_signal = fftpack.irfft(filtered_fft)
        
        current_close = closes[-1]
        target_price = filtered_signal[-1] + target_distance
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        diff = target_price - current_close
        market_mood = "Bullish" if diff > 0 else "Bearish"
        fastest_target = current_close + target_distance / 2
        entry_price = closes[-1]    
        stop_loss = entry_price - 3 * np.std(closes)
        return current_time, entry_price, stop_loss, fastest_target, market_mood
    except Exception as e:
        print(f"  [Get Target] Error: {e}")
        return None, 0.0, 0.0, 0.0, "N/A"

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS — FFT CYCLE TARGET (Condition #7)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fft_cycle_target_both(close_prices_1m, entry_price):
    EMPTY = {
        "phase_lock_long": False, "phase_lock_short": False,
        "init_dir_long": False, "init_dir_short": False,
        "reaches_tp_long": False, "reaches_tp_short": False,
        "path_safe_long": False, "path_safe_short": False,
        "target_price_long": 0.0, "target_price_short": 0.0,
        "proj_min": 0.0, "proj_max": 0.0,
        "dom_period": 0.0, "dom_freq": 0.0, "cycle_amp_pct": 0.0,
        "long_score": 0, "short_score": 0, "long_ok": False, "short_ok": False,
    }
    try:
        n_window = FFT_TARGET_WINDOW
        if len(close_prices_1m) < n_window: return EMPTY
        prices = np.array(close_prices_1m[-n_window:], dtype=np.float64)
        x = np.arange(n_window, dtype=np.float64)
        trend_coeffs = np.polyfit(x, prices, 1)
        trend = np.polyval(trend_coeffs, x)
        detrended = prices - trend
        window_func = np.hanning(n_window)
        windowed = detrended * window_func
        fft_coeffs = rfft(windowed)
        freqs = rfftfreq(n_window)
        magnitudes = np.abs(fft_coeffs)
        MIN_PERIOD_BARS = 4
        MAX_FREQ = 1.0 / MIN_PERIOD_BARS
        valid_mask = (freqs > 1e-9) & (freqs <= MAX_FREQ)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 1: return EMPTY
        valid_mags = magnitudes[valid_indices]
        valid_freqs = freqs[valid_indices]
        dom_idx_in_valid = int(np.argmax(valid_mags))
        dom_rfft_idx = valid_indices[dom_idx_in_valid]
        dom_freq = float(valid_freqs[dom_idx_in_valid])
        dom_period = 1.0 / dom_freq if dom_freq > 1e-9 else n_window
        dom_coeff = fft_coeffs[dom_rfft_idx]
        phase_at_end = 2.0 * np.pi * dom_freq * (n_window - 1) + np.angle(dom_coeff)
        sin_phase = np.sin(phase_at_end)
        phase_lock_long  = (sin_phase < 0)
        phase_lock_short = (sin_phase > 0)
        projection_steps = max(5, int(dom_period / 2))
        window_mean = np.mean(window_func)
        if window_mean < 1e-9: window_mean = 1.0
        window_correction = 1.0 / window_mean
        top_n_indices = valid_indices[np.argsort(valid_mags)[-FFT_TARGET_TOP_N:]]
        projected_prices = []
        for step in range(1, projection_steps + 1):
            t_future = float(n_window + step)
            projected_osc = 0.0
            for idx in top_n_indices:
                freq_i = float(freqs[idx])
                coeff_i = fft_coeffs[idx]
                phase_factor = np.exp(2j * np.pi * freq_i * t_future)
                contribution = np.real(coeff_i * phase_factor) * (2.0 / n_window)
                projected_osc += contribution
            trend_at_future = np.polyval(trend_coeffs, t_future)
            proj_price = float(trend_at_future + projected_osc * window_correction)
            projected_prices.append(proj_price)
        if not projected_prices: return EMPTY
        proj_max = max(projected_prices)
        proj_min = min(projected_prices)
        n_init = min(5, len(projected_prices))
        x_init = np.arange(n_init, dtype=np.float64)
        init_slope = float(np.polyfit(x_init, projected_prices[:n_init], 1)[0])
        init_dir_long  = (init_slope >= 0)
        init_dir_short = (init_slope < 0)
        tp_long_price  = entry_price * (1.0 + TP_PRICE_PCT)
        tp_short_price = entry_price * (1.0 - TP_PRICE_PCT)
        sl_long_price  = entry_price * (1.0 - SL_PRICE_PCT)
        sl_short_price = entry_price * (1.0 + SL_PRICE_PCT)
        required_long_tp = entry_price + (tp_long_price - entry_price) * FFT_TARGET_MIN_RATIO
        reaches_tp_long  = (proj_max >= required_long_tp)
        path_safe_long   = (proj_min >= sl_long_price)
        required_short_tp = entry_price - (entry_price - tp_short_price) * FFT_TARGET_MIN_RATIO
        reaches_tp_short  = (proj_min <= required_short_tp)
        path_safe_short   = (proj_max <= sl_short_price)
        target_price_long  = proj_max
        target_price_short = proj_min
        dom_amplitude_raw = float(valid_mags[dom_idx_in_valid]) * (2.0 / n_window) * window_correction
        cycle_amp_pct = (dom_amplitude_raw / (entry_price + 1e-12)) * 100.0
        weights = [4, 3, 2, 1]
        long_checks  = [phase_lock_long,  init_dir_long,  reaches_tp_long,  path_safe_long]
        short_checks = [phase_lock_short, init_dir_short, reaches_tp_short, path_safe_short]
        long_score  = sum(w * c for w, c in zip(weights, long_checks))
        short_score = sum(w * c for w, c in zip(weights, short_checks))
        long_ok  = (long_score >= short_score)
        short_ok = (short_score > long_score)
        return {
            "phase_lock_long": phase_lock_long, "phase_lock_short": phase_lock_short,
            "init_dir_long": init_dir_long, "init_dir_short": init_dir_short,
            "reaches_tp_long": reaches_tp_long, "reaches_tp_short": reaches_tp_short,
            "path_safe_long": path_safe_long, "path_safe_short": path_safe_short,
            "target_price_long": target_price_long, "target_price_short": target_price_short,
            "proj_min": proj_min, "proj_max": proj_max,
            "dom_period": dom_period, "dom_freq": dom_freq, "cycle_amp_pct": cycle_amp_pct,
            "long_score": long_score, "short_score": short_score,
            "long_ok": long_ok, "short_ok": short_ok,
        }
    except Exception as e:
        print(f"  [FFT Target] Error: {e}")
        return EMPTY

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS — REGRESSION FORECAST (Condition #8)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_regression_forecast(closes_1m, lookback=60):
    if len(closes_1m) < lookback: return 0.0, 0.0, 0.0
    window = np.array(closes_1m[-lookback:], dtype=np.float64)
    x = np.arange(lookback, dtype=np.float64)
    current_price = float(window[-1])
    decay = 0.05
    weights = np.exp(-decay * (lookback - 1 - x))
    weights /= weights.sum()
    W = np.diag(weights)
    X = np.column_stack([np.ones(lookback), x])
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ window
        coeffs = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        coeffs = np.polyfit(x, window, 1)[::-1]
    intercept, slope = float(coeffs[0]), float(coeffs[1])
    forecast_price = intercept + slope * lookback
    slope_normalized = slope / (current_price + 1e-12)
    return float(forecast_price), current_price, slope_normalized

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(buffer_5m, buffer_1m, live_price):
    candles_5m = buffer_5m.get_candles()
    candles_1m = buffer_1m.get_candles()
    closes_5m_raw = [c["close"] for c in candles_5m]
    closes_1m_raw = [c["close"] for c in candles_1m]
    if len(closes_5m_raw) < 50 or len(closes_1m_raw) < 50: return None

    close_arr_5m = np.array(closes_5m_raw, dtype=float)
    close_arr_1m = np.array(closes_1m_raw, dtype=float)

    # 1. Sine
    last_1200_5m = close_arr_5m[-ANALYSIS_WINDOW_5M:]
    argmin_idx_5m = int(np.argmin(last_1200_5m))
    argmax_idx_5m = int(np.argmax(last_1200_5m))
    dist_to_min, dist_to_max = scale_to_sine(close_arr_5m, argmin_idx_5m, argmax_idx_5m)
    cond_sine_long  = (dist_to_min < dist_to_max)
    cond_sine_short = (dist_to_max < dist_to_min)

    # 2. Cycle
    candles_1m_window = candles_1m[-500:]
    last_500_1m = close_arr_1m[-500:]
    window_len_1m = len(last_500_1m)
    argmin_idx_1m = int(np.argmin(last_500_1m))
    argmax_idx_1m = int(np.argmax(last_500_1m))
    cycle_min_price = float(last_500_1m[argmin_idx_1m])
    cycle_max_price = float(last_500_1m[argmax_idx_1m])
    current_close_1m = float(close_arr_1m[-1])
    bars_ago_min = (window_len_1m - 1) - argmin_idx_1m
    bars_ago_max = (window_len_1m - 1) - argmax_idx_1m
    try:
        ts_min = datetime.datetime.fromtimestamp(candles_1m_window[argmin_idx_1m]["time"], datetime.timezone.utc).strftime("%H:%M:%S")
        ts_max = datetime.datetime.fromtimestamp(candles_1m_window[argmax_idx_1m]["time"], datetime.timezone.utc).strftime("%H:%M:%S")
    except Exception: ts_min = ts_max = "N/A"

    if bars_ago_min <= bars_ago_max:
        most_recent_extreme = "ARGMIN (LOW)"
        cond_cycle_long, cond_cycle_short = True, False
    else:
        most_recent_extreme = "ARGMAX (HIGH)"
        cond_cycle_long, cond_cycle_short = False, True

    # 3. Mom
    mom_1m = calculate_momentum(close_arr_1m)
    if np.isnan(mom_1m): return None
    cond_mom_long  = (mom_1m >= 0)
    cond_mom_short = (mom_1m < 0)

    # 4. Vol
    bullish_perc, bearish_perc = get_buy_sell_volume_perc(candles_1m)
    cond_vol_long  = (bullish_perc >= bearish_perc)
    cond_vol_short = (bearish_perc > bullish_perc)

    # 5. FFT Dom
    neg_ratio, pos_ratio = analyze_fft_dominance_1m(close_arr_1m)
    if neg_ratio > pos_ratio + FFT_DOMINANCE_THRESHOLD:
        cond_fft_long, cond_fft_short = True, False
    elif pos_ratio > neg_ratio + FFT_DOMINANCE_THRESHOLD:
        cond_fft_long, cond_fft_short = False, True
    else:
        cond_fft_long  = (neg_ratio >= pos_ratio)
        cond_fft_short = (pos_ratio > neg_ratio)

    # 6. ML Forecast (Random Forest)
    ml_forecast_price, ml_current_close = generate_ml_forecast(candles_1m)
    if ml_forecast_price <= 0.0 or ml_current_close <= 0.0: return None
    cond_ml_long  = (ml_forecast_price >= ml_current_close)
    cond_ml_short = (ml_forecast_price < ml_current_close)

    # 7. FFT Target
    fft_target = compute_fft_cycle_target_both(closes_1m_raw, live_price)
    if fft_target["dom_period"] < 4.0: return None
    cond_fft_target_long  = fft_target["long_ok"]
    cond_fft_target_short = fft_target["short_ok"]

    # 8. Regr
    reg_forecast, reg_current, reg_slope = compute_regression_forecast(closes_1m_raw, REG_FORECAST_LOOKBACK)
    if reg_forecast <= 0.0 or reg_current <= 0.0: return None
    cond_reg_long  = (reg_forecast >= reg_current)
    cond_reg_short = (reg_forecast < reg_current)

    # Extra FFT & Target
    fft_forecast_price, fft_cycle_dir = generate_fft_forecast(candles_1m)
    target_time, target_entry, target_sl, fastest_target, market_mood = get_target(close_arr_1m)

    is_long  = (cond_sine_long and cond_cycle_long and cond_mom_long and cond_vol_long
                and cond_fft_long and cond_ml_long and cond_fft_target_long and cond_reg_long)
    is_short = (cond_sine_short and cond_cycle_short and cond_mom_short and cond_vol_short
                and cond_fft_short and cond_ml_short and cond_fft_target_short and cond_reg_short)

    long_true_count  = sum([cond_sine_long,  cond_cycle_long,  cond_mom_long,  cond_vol_long,
                            cond_fft_long,  cond_ml_long,  cond_fft_target_long,  cond_reg_long])
    short_true_count = sum([cond_sine_short, cond_cycle_short, cond_mom_short, cond_vol_short,
                            cond_fft_short, cond_ml_short, cond_fft_target_short, cond_reg_short])

    return {
        "price": live_price, "is_long": is_long, "is_short": is_short,
        "cond_flags": {
            "sine_long": cond_sine_long,      "sine_short": cond_sine_short,
            "cycle_long": cond_cycle_long,     "cycle_short": cond_cycle_short,
            "mom_long": cond_mom_long,         "mom_short": cond_mom_short,
            "vol_long": cond_vol_long,         "vol_short": cond_vol_short,
            "fft_long": cond_fft_long,         "fft_short": cond_fft_short,
            "ml_long": cond_ml_long,           "ml_short": cond_ml_short,
            "fft_target_long": cond_fft_target_long, "fft_target_short": cond_fft_target_short,
            "reg_long": cond_reg_long,         "reg_short": cond_reg_short,
        },
        "long_true_count": long_true_count, "short_true_count": short_true_count,
        "dist_to_min": dist_to_min, "dist_to_max": dist_to_max,
        "most_recent_extreme": most_recent_extreme,
        "mom_1m": mom_1m, "bullish_perc": bullish_perc, "bearish_perc": bearish_perc,
        "neg_ratio": neg_ratio, "pos_ratio": pos_ratio,
        "ml_forecast_price": ml_forecast_price, "ml_current_close": ml_current_close,
        "fft_target": fft_target,
        "reg_forecast": reg_forecast, "reg_current": reg_current, "reg_slope": reg_slope,
        "fft_forecast_price": fft_forecast_price, "fft_cycle_dir": fft_cycle_dir,
        "target_entry": target_entry, "fastest_target": fastest_target, 
        "target_sl": target_sl, "market_mood": market_mood
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SL & TP CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(entry_price, side):
    tp_dist = entry_price * TP_PRICE_PCT
    sl_dist = entry_price * SL_PRICE_PCT
    if side == "long": sl_price, tp_price = entry_price - sl_dist, entry_price + tp_dist
    else: sl_price, tp_price = entry_price + sl_dist, entry_price - tp_dist
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
    line = (f"{'='*65}\nMODE: {mode}\nTYPE: {trade_result.get('type', 'N/A').upper()}\n"
            f"ENTRY TIME: {trade_result.get('entry_time', 'N/A')}\nEXIT TIME:  {trade_result.get('exit_time', 'N/A')}\n"
            f"DURATION:   {trade_result.get('duration', 'N/A')}\nENTRY PRICE: {trade_result.get('entry_price', 0):.2f}\n"
            f"EXIT PRICE:  {trade_result.get('exit_price', 0):.2f}\nNET ROE:     {trade_result.get('roe', 0):+.2f}%\n"
            f"REASON:      {trade_result.get('reason', 'N/A')}\nREGR FCAST:  Price: {trade_result.get('reg_forecast', 0.0):.2f}\n"
            f"{'='*65}\n\n")
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
            print(f"  >>> {side.upper()} PLACED: {contracts} contracts @ ~{price:.2f}  "
                  f"(used {trade_balance:.2f} USDT / {TRADE_BALANCE_PCT*100:.0f}% of balance)")
            return True
        print(f"  [ORDER FAIL] {side.upper()} [{resp.get('code')}]: {resp.get('msg')}")
        return False
    except Exception as e:
        print(f"  [ORDER ERROR] {side.upper()}: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _ck(truth):
    return "\033[92m✓\033[0m" if truth else "\033[91m✗\033[0m"

def format_duration(start_dt, end_dt):
    total_sec = int((end_dt - start_dt).total_seconds())
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_conditions(sig):
    f = sig["cond_flags"]
    print(f"  ┌────────────────────── ALL 8 CONDITIONS ──────────────────────┐")
    print(f"  │ 1. Sine (5m)       dMin:{sig['dist_to_min']:>5.1f}% dMax:{sig['dist_to_max']:>5.1f}%"
          f"  L:{_ck(f['sine_long'])}  S:{_ck(f['sine_short'])}       │")
    print(f"  │ 2. Cycle (1m)      {sig['most_recent_extreme']:<14s}"
          f"  L:{_ck(f['cycle_long'])}  S:{_ck(f['cycle_short'])}       │")
    print(f"  │ 3. Mom (1m)        {sig['mom_1m']:>10.2f}"
          f"       L:{_ck(f['mom_long'])}  S:{_ck(f['mom_short'])}       │")
    print(f"  │ 4. Vol (1m)        Bull:{sig['bullish_perc']:>5.1f}% Bear:{sig['bearish_perc']:>5.1f}%"
          f"  L:{_ck(f['vol_long'])}  S:{_ck(f['vol_short'])} │")
    print(f"  │ 5. FFT Dom (1m)    Neg:{sig['neg_ratio']:>5.1f}% Pos:{sig['pos_ratio']:>5.1f}%"
          f"  L:{_ck(f['fft_long'])}  S:{_ck(f['fft_short'])} │")
    ml_fc  = sig.get("ml_forecast_price", 0.0)
    ml_cur = sig.get("ml_current_close", 0.0)
    ml_d   = ((ml_fc - ml_cur) / ml_cur * 100) if ml_cur > 0 else 0.0
    print(f"  │ 6. ML (1m)         Fcst:{ml_fc:>10.2f} ({ml_d:+.4f}%)"
          f"  L:{_ck(f['ml_long'])}  S:{_ck(f['ml_short'])} │")
    ft = sig.get("fft_target", {})
    ftp_l, ftp_s = ft.get("target_price_long", 0.0), ft.get("target_price_short", 0.0)
    fl_sc, fs_sc = ft.get("long_score", 0), ft.get("short_score", 0)
    print(f"  │ 7. FFT Target(v6c) L_Tgt:{ftp_l:>9.2f} S_Tgt:{ftp_s:>9.2f}"
          f"       │")
    print(f"  │    Score: L:{fl_sc}/10 S:{fs_sc}/10  Gate: L:{_ck(f['fft_target_long'])}  S:{_ck(f['fft_target_short'])}"
          f"       │")
    reg_fc, reg_cur = sig.get("reg_forecast", 0.0), sig.get("reg_current", 0.0)
    reg_d = ((reg_fc - reg_cur) / reg_cur * 100) if reg_cur > 0 else 0.0
    print(f"  │ 8. Regr Fcst (1m)  Fcst:{reg_fc:>10.2f} ({reg_d:+.4f}%)"
          f"  L:{_ck(f['reg_long'])}  S:{_ck(f['reg_short'])} │")
    
    # Extra Info
    fft_fc = sig.get("fft_forecast_price", 0.0)
    fft_cd = sig.get("fft_cycle_dir", "N/A")
    ft_fast = sig.get("fastest_target", 0.0)
    ft_mood = sig.get("market_mood", "N/A")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │ EXTRA: FFT Fcst:{fft_fc:.2f}({fft_cd}) | FastTgt:{ft_fast:.2f} | Mood:{ft_mood} │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    
    lc, sc = sig["long_true_count"], sig["short_true_count"]
    print(f"  │ SCORE: LONG {lc}/8  SHORT {sc}/8  (ALL 8 MUST BE TRUE)      │")
    print(f"  │ RESULT:  LONG:{_ck(sig['is_long'])}   SHORT:{_ck(sig['is_short'])}"
          f"{' '*22}│")
    print(f"  └──────────────────────────────────────────────────────────────┘")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print(f" HFT KuCoin Bot v6c — ALL 8 CONDITIONS (STRICT BINARY)")
    print(f"{'='*65}")
    print(f" Loop Sleep:          {LOOP_SLEEP}s | GC: Enabled")
    print(f" API Connected:       {API_CONNECTED}")
    print(f"{'='*65}\n")

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

    fetcher = ConcurrentDataFetcher(buffer_5m, buffer_1m, TRADE_SYMBOL)
    trade_state = load_trade_state()
    if trade_state and trade_state.get("is_open"):
        print(f"  [STATE] Resuming tracked trade: {trade_state.get('side','?').upper()} "
              f"@ {trade_state.get('entry_price',0):.2f}")

    loop_count = 0

    try:
        while True:
            gc.collect() # Enforce fresh memory and prevent cache staleness
            t_loop_start = time.perf_counter()

            t_net_start = time.perf_counter()
            do_full = (loop_count == 0)
            data = fetcher.fetch_all_parallel(do_full_refresh=do_full)
            net_ms = (time.perf_counter() - t_net_start) * 1000

            live_price = data["price"]
            balance = data["balance"]
            position = data["position"]
            trade_alloc = balance * TRADE_BALANCE_PCT

            if trade_state and trade_state.get("is_open"):
                entry_p = trade_state["entry_price"]
                side = trade_state["side"]
                hit, reason, roe = check_sim_tp_sl(entry_p, live_price, side)
                if hit:
                    exit_price = live_price
                    price_chg = ((exit_price - entry_p) / entry_p) * 100 if side == "long" else ((entry_p - exit_price) / entry_p) * 100
                    net_roe = (price_chg * LEVERAGE) - RT_FEE_ROE_PCT
                    entry_dt = datetime.datetime.fromisoformat(trade_state["entry_time"]).replace(tzinfo=datetime.timezone.utc)
                    exit_dt = datetime.datetime.now(datetime.timezone.utc)
                    dur = format_duration(entry_dt, exit_dt)
                    print(f"\n  {'*'*55}\n  *** {reason} *** {side.upper()} | ROE: {net_roe:+.2f}%\n  {'*'*55}\n")
                    if API_CONNECTED and position.get("is_open"):
                        try: client.close_position(TRADE_SYMBOL)
                        except Exception as e: print(f"  [EXCHANGE] Close error: {e}")
                    write_trade_to_journal({"mode": "LIVE" if API_CONNECTED else "SIM", "type": side, "entry_time": trade_state["entry_time"], "exit_time": exit_dt.isoformat(), "duration": dur, "entry_price": entry_p, "exit_price": exit_price, "roe": net_roe, "reason": reason, "reg_forecast": trade_state.get("reg_forecast", 0.0)})
                    clear_trade_state()
                    trade_state = None
                else:
                    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
                    print(f"  [{now_str}] IN {side.upper()} | ROE: {roe:+.2f}% | Price: {live_price:.2f} | Net:{net_ms:.0f}ms", end="\r")
                time.sleep(LOOP_SLEEP)
                loop_count += 1
                continue

            t_calc_start = time.perf_counter()
            sig = compute_signals(buffer_5m, buffer_1m, live_price)
            calc_ms = (time.perf_counter() - t_calc_start) * 1000

            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            if sig is None:
                print(f"  [{now_str}] Waiting for data... (5m:{len(buffer_5m)} 1m:{len(buffer_1m)})")
                time.sleep(LOOP_SLEEP)
                loop_count += 1
                continue

            pos_label = "FLAT"
            if sig["is_long"]:  pos_label = "LONG"
            elif sig["is_short"]: pos_label = "SHORT"

            print(f"\n[{now_str}] Scanning ({pos_label}) | Net:{net_ms:.0f}ms Calc:{calc_ms:.0f}ms")
            print_conditions(sig)
            print(f"  Balance: {balance:.2f} USDT  (trade alloc: {trade_alloc:.2f} USDT / {TRADE_BALANCE_PCT*100:.0f}%)")

            if sig["is_long"] or sig["is_short"]:
                side = "long" if sig["is_long"] else "short"
                if balance < MIN_BALANCE_USDT:
                    print(f"  [SKIP] Balance {balance:.2f} USDT < minimum {MIN_BALANCE_USDT} USDT")
                elif not API_CONNECTED:
                    print(f"  [SIM] Would enter {side.upper()} @ {live_price:.2f} (no API)")
                    trade_state = {
                        "is_open": True, "side": side, "entry_price": live_price,
                        "entry_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "fft_target_price": sig["fft_target"].get("target_price_long" if side == "long" else "target_price_short", 0.0),
                        "reg_forecast": sig.get("reg_forecast", 0.0),
                    }
                    save_trade_state(trade_state)
                else:
                    success = execute_entry(TRADE_SYMBOL, side, balance, live_price)
                    if success:
                        trade_state = {
                            "is_open": True, "side": side, "entry_price": live_price,
                            "entry_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "fft_target_price": sig["fft_target"].get("target_price_long" if side == "long" else "target_price_short", 0.0),
                            "reg_forecast": sig.get("reg_forecast", 0.0),
                        }
                        save_trade_state(trade_state)

            total_ms = (time.perf_counter() - t_loop_start) * 1000
            log_timing("total_loop", total_ms)
            loop_count += 1
            if loop_count % 200 == 0: print_timing_summary()
            time.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        print(f"\n\n  [EXIT] Bot stopped cleanly.")
        fetcher.shutdown()
    except Exception as e:
        print(f"\n  [FATAL] {e}")
        import traceback; traceback.print_exc()
        fetcher.shutdown()

if __name__ == "__main__":
    main()
