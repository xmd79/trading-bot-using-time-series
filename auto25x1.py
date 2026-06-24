"""
HFT Auto Trading Bot — KuCoin Futures Edition (OPTIMIZED)
=========================================================
CHANGES FROM ORIGINAL:
- CandleBuffer class: Downloads 1200 candles ONCE at startup
- Each iteration: Only fetches 1-2 NEW candles per timeframe (2 requests vs 12)
- Added timing/cronometers to identify bottlenecks
- Reduced from ~15 requests/iteration to ~5 requests/iteration
- Added periodic full refresh (every 60 minutes) to prevent drift

Strictly 5 Conditions:
  1. Sine Scale (5m TF): Uses argmin/argmax of last 1200 5m values as cycle boundaries
  2. Extrema Cycle (1m TF): argmin/argmax of last 1200 1m closes
  3. Momentum (1m TF): MOM > 0 (LONG) | MOM < 0 (SHORT)
  4. Volume (1m TF): Bullish % > Bearish % (LONG) | Bearish % > Bullish % (SHORT)
  5. FFT Dominance (1m TF): Top 12 Negative Freq Power > Top 12 Positive Freq Power (LONG)

25x Leverage
TP: 2.55% NET Profit (5.55% Gross ROE after 3.0% RT fee deduction)
SL: -25% ROE
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

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
from scipy.fft import fft, fftfreq
from threading import Lock

# ═══════════════════════════════════════════════════════════════════════════════
# TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class Timer:
    """Simple cronometer for profiling code sections."""
    def __init__(self, name=""):
        self.name = name
        self.start = 0
        self.elapsed = 0
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000  # ms
        if self.elapsed > 100:  # Only log if > 100ms
            print(f"  [TIMER] {self.name}: {self.elapsed:.1f}ms")

# Global timing stats
timing_stats = {
    "candle_update": [],
    "ticker": [],
    "balance": [],
    "position": [],
    "compute_signals": [],
    "total_loop": [],
}

def log_timing(section, elapsed_ms):
    """Log timing and keep last 20 values."""
    timing_stats[section].append(elapsed_ms)
    if len(timing_stats[section]) > 20:
        timing_stats[section].pop(0)

def print_timing_summary():
    """Print average timing for each section."""
    print(f"\n  ┌───────────────────── PERFORMANCE SUMMARY ─────────────────────┐")
    for section, values in timing_stats.items():
        if values:
            avg = sum(values) / len(values)
            last = values[-1]
            print(f"  │ {section:<20} avg: {avg:>7.1f}ms  last: {last:>7.1f}ms  ({len(values)} samples) │")
    total_avg = sum(timing_stats.get("total_loop", [0])) / max(1, len(timing_stats.get("total_loop", [1])))
    print(f"  │ {'LOOP FREQUENCY':<20} ~{1000/max(1,total_avg):>5.1f} loops/sec{' '*23} │")
    print(f"  └───────────────────────────────────────────────────────────────┘\n")

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

print("HFT KuCoin Bot (25x / OPTIMIZED) initialising...")
if _cleaned:
    print(f"  [cleanup] Wiped: {', '.join(_cleaned)}")
del _cleaned

# ═══════════════════════════════════════════════════════════════════════════════
# KUCOIN FUTURES REST CLIENT (WITH SESSION FOR CONNECTION POOLING)
# ═══════════════════════════════════════════════════════════════════════════════

KUCOIN_FUTURES_BASE = "https://api-futures.kucoin.com"

# Reusable session for connection pooling (faster repeated requests)
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
LOOP_SLEEP = 1  # Reduced from 3 since we're much faster now
MIN_BALANCE_USDT = 5.0

ANALYSIS_WINDOW_5M = 1200 
ANALYSIS_WINDOW_1M = 1200 

KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100  # 3.0%
NET_PROFIT_ROE = 2.55                                  
TAKE_PROFIT_ROE = NET_PROFIT_ROE + RT_FEE_ROE_PCT      # 5.55%
STOP_LOSS_ROE = -25.0                                  

TP_PRICE_PCT = TAKE_PROFIT_ROE / LEVERAGE / 100.0  # 0.222%
SL_PRICE_PCT = abs(STOP_LOSS_ROE) / LEVERAGE / 100.0  # 1.0%

# Buffer refresh interval (seconds) - full refresh to prevent drift
FULL_REFRESH_INTERVAL = 3600  # 1 hour

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
# CANDLE BUFFER - OPTIMIZED DATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class CandleBuffer:
    """
    Maintains a local buffer of candles.
    
    OPTIMIZATION:
    - initialize(): Full fetch of 1200 candles (ONE-TIME, ~12 requests)
    - update(): Fetches only last 2-3 candles (1 request per call)
    - Reduces per-iteration requests from 12 to 1 per timeframe
    """
    
    def __init__(self, symbol, timeframe, max_candles=1200):
        self.symbol = symbol
        self.timeframe = timeframe  # "1m" or "5m"
        self.max_candles = max_candles
        self.candles = []
        self.last_full_refresh = 0
        self.granularity = int(timeframe[0])  # 1 or 5
        self.duration_ms = self.granularity * 60 * 1000
        self.lock = Lock()
        
    def _parse_kline(self, k):
        """Parse a single kline from API response."""
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
        """
        Initial full fetch of historical candles.
        Makes multiple requests to get max_candles worth of data.
        This is called ONCE at startup.
        """
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
            
            # Keep only the most recent max_candles and remove duplicates by time
            self._deduplicate()
            self.candles = self.candles[-self.max_candles:]
            self.last_full_refresh = time.time()
            print(f"  [Buffer:{self.timeframe}] ✓ Ready: {len(self.candles)} candles")
            return len(self.candles) >= 50
    
    def _deduplicate(self):
        """Remove duplicate candles (same timestamp)."""
        seen_times = set()
        unique = []
        for c in reversed(self.candles):  # Process newest first
            if c["time"] not in seen_times:
                seen_times.add(c["time"])
                unique.append(c)
        self.candles = list(reversed(unique))  # Back to oldest-first order
    
    def update(self, client_obj, fetch_limit=3):
        """
        Fetch only the latest candles and append if new.
        Returns number of new candles added.
        ONLY 1 API REQUEST per call!
        """
        if not API_CONNECTED or not self.candles:
            return 0
            
        with self.lock:
            now_ms = int(time.time() * 1000)
            
            # Fetch the last few candles
            end_ms = now_ms
            start_ms = end_ms - (fetch_limit * self.duration_ms)
            
            new_candles = []
            try:
                klines = client_obj.get_klines(self.symbol, self.granularity, start_ms=start_ms, end_ms=end_ms)
                for k in klines:
                    candle = self._parse_kline(k)
                    if candle:
                        new_candles.append(candle)
            except Exception as e:
                print(f"  [Buffer:{self.timeframe}] Update error: {e}")
                return 0
            
            # Find which candles are new (not already in buffer)
            if self.candles:
                last_time = self.candles[-1]["time"]
                new_to_add = [c for c in new_candles if c["time"] > last_time]
            else:
                new_to_add = new_candles
            
            if new_to_add:
                self.candles.extend(new_to_add)
                # Trim to max size
                self.candles = self.candles[-self.max_candles:]
            
            return len(new_to_add)
    
    def needs_full_refresh(self):
        """Check if buffer needs a full refresh to prevent drift."""
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
# DATA HELPERS (SIMPLIFIED)
# ═══════════════════════════════════════════════════════════════════════════════

def get_account_balance():
    if not API_CONNECTED: return 0.0
    try: return float(client.get_account_overview()["data"]["availableBalance"])
    except Exception: return 0.0

def get_price(symbol):
    if not API_CONNECTED: return 0.0
    try: return float(client.get_ticker(symbol)["data"]["price"])
    except Exception: return 0.0

def get_buy_sell_volume_perc(candles):
    """Calculate bullish/bearish volume percentages."""
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
# TECHNICAL ANALYSIS (UNCHANGED LOGIC)
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
    """
    FFT Dominance using HT_SINE quadrature components.
    - Negative freq power > Positive freq power → LONG
    - Positive freq power > Negative freq power → SHORT
    """
    if len(close_prices_1m) < 32: 
        return False, False, 0.0, 0.0
    
    sine_wave_1m, lead_sine_1m = talib.HT_SINE(close_prices_1m)
    sine_wave_1m = np.nan_to_num(-sine_wave_1m)
    lead_sine_1m = np.nan_to_num(-lead_sine_1m)
    
    valid_mask = ~(np.isnan(sine_wave_1m) | np.isnan(lead_sine_1m))
    sine_wave_1m = sine_wave_1m[valid_mask]
    lead_sine_1m = lead_sine_1m[valid_mask]
    
    if len(sine_wave_1m) < 32:
        return False, False, 0.0, 0.0
    
    n = len(sine_wave_1m)
    complex_signal = sine_wave_1m + 1j * lead_sine_1m
    window = np.hanning(n)
    complex_signal = complex_signal * window
    complex_signal = complex_signal - np.mean(complex_signal)
    
    fft_result = fft(complex_signal)
    fft_power = np.abs(fft_result) ** 2
    freq_bins = fftfreq(n)
    
    powers = fft_power[1:]
    frequencies = freq_bins[1:]
    
    neg_mask = frequencies < 0
    pos_mask = frequencies > 0
    
    neg_powers = powers[neg_mask]
    pos_powers = powers[pos_mask]
    
    if len(neg_powers) >= 12:
        top_12_neg = np.sort(neg_powers)[-12:]
        neg_power = np.sum(top_12_neg)
    elif len(neg_powers) > 0:
        neg_power = np.sum(neg_powers)
    else:
        neg_power = 0.0
        
    if len(pos_powers) >= 12:
        top_12_pos = np.sort(pos_powers)[-12:]
        pos_power = np.sum(top_12_pos)
    elif len(pos_powers) > 0:
        pos_power = np.sum(pos_powers)
    else:
        pos_power = 0.0
    
    total_power = neg_power + pos_power
    if total_power == 0: 
        return False, False, 0.0, 0.0
    
    neg_ratio = (neg_power / total_power) * 100
    pos_ratio = (pos_power / total_power) * 100
    
    DOMINANCE_THRESHOLD = 0.1
    fft_long_signal = neg_ratio > (pos_ratio + DOMINANCE_THRESHOLD)
    fft_short_signal = pos_ratio > (neg_ratio + DOMINANCE_THRESHOLD)
    
    return fft_long_signal, fft_short_signal, neg_ratio, pos_ratio


def calculate_momentum(close_arr, period=14):
    if len(close_arr) < period + 1: return np.nan
    return float(talib.MOM(close_arr, timeperiod=period)[-1])


def compute_signals(buffer_5m, buffer_1m, live_price):
    """
    Compute trading signals using pre-fetched candle buffers.
    NO API CALLS in this function!
    """
    candles_5m = buffer_5m.get_candles()
    candles_1m = buffer_1m.get_candles()
    
    closes_5m_raw = [c["close"] for c in candles_5m]
    closes_1m_raw = [c["close"] for c in candles_1m]
    
    if len(closes_5m_raw) < 50 or len(closes_1m_raw) < 50: 
        return None
    
    close_arr_5m = np.array(closes_5m_raw, dtype=float)
    close_arr_1m = np.array(closes_1m_raw, dtype=float)
    
    # 1. Sine Scale (5m)
    last_1200_5m = close_arr_5m[-ANALYSIS_WINDOW_5M:]
    argmin_idx_5m = int(np.argmin(last_1200_5m))
    argmax_idx_5m = int(np.argmax(last_1200_5m))
    dist_to_min, dist_to_max = scale_to_sine(close_arr_5m, argmin_idx_5m, argmax_idx_5m)
    cond_sine_long = dist_to_min < dist_to_max
    cond_sine_short = dist_to_max < dist_to_min
    
    # 2. Extrema Cycle (1m)
    candles_1m_window = candles_1m[-500:]
    last_1200_1m = close_arr_1m[-500:]
    window_len_1m = len(last_1200_1m)
    argmin_idx_1m = int(np.argmin(last_1200_1m))
    argmax_idx_1m = int(np.argmax(last_1200_1m))
    cycle_min_price = float(last_1200_1m[argmin_idx_1m])
    cycle_max_price = float(last_1200_1m[argmax_idx_1m])
    current_close_1m = float(close_arr_1m[-1])
    bars_ago_min = (window_len_1m - 1) - argmin_idx_1m
    bars_ago_max = (window_len_1m - 1) - argmax_idx_1m
    try:
        ts_min = datetime.datetime.fromtimestamp(candles_1m_window[argmin_idx_1m]["time"], datetime.timezone.utc).strftime("%H:%M:%S UTC")
        ts_max = datetime.datetime.fromtimestamp(candles_1m_window[argmax_idx_1m]["time"], datetime.timezone.utc).strftime("%H:%M:%S UTC")
    except Exception:
        ts_min = ts_max = "N/A"
    
    if bars_ago_min < bars_ago_max:
        most_recent_extreme = "ARGMIN (LOW)"
    elif bars_ago_max < bars_ago_min:
        most_recent_extreme = "ARGMAX (HIGH)"
    else:
        most_recent_extreme = "TIE"
    
    cond_cycle_long = (argmin_idx_1m > argmax_idx_1m) and (current_close_1m > cycle_min_price)
    cond_cycle_short = (argmax_idx_1m > argmin_idx_1m) and (current_close_1m < cycle_max_price)
    
    # 3. Momentum (1m)
    mom_1m = calculate_momentum(close_arr_1m)
    if np.isnan(mom_1m): return None
    cond_mom_long = mom_1m > 0
    cond_mom_short = mom_1m < 0
    
    # 4. Volume (1m Percentages)
    bullish_perc, bearish_perc = get_buy_sell_volume_perc(candles_1m)
    cond_vol_long = bullish_perc > bearish_perc
    cond_vol_short = bearish_perc > bullish_perc
    
    # 5. FFT Dominance (1m Top 12 Freqs)
    fft_long, fft_short, neg_ratio, pos_ratio = analyze_fft_dominance_1m(close_arr_1m)
    cond_fft_long = fft_long
    cond_fft_short = fft_short
    
    is_long = cond_sine_long and cond_cycle_long and cond_mom_long and cond_vol_long and cond_fft_long
    is_short = cond_sine_short and cond_cycle_short and cond_mom_short and cond_vol_short and cond_fft_short
    
    return {
        "price": live_price, "is_long": is_long, "is_short": is_short,
        "cond_flags": {
            "sine_long": cond_sine_long, "sine_short": cond_sine_short,
            "cycle_long": cond_cycle_long, "cycle_short": cond_cycle_short,
            "mom_long": cond_mom_long, "mom_short": cond_mom_short,
            "vol_long": cond_vol_long, "vol_short": cond_vol_short,
            "fft_long": cond_fft_long, "fft_short": cond_fft_short,
        },
        "dist_to_min": dist_to_min, "dist_to_max": dist_to_max,
        "argmin_idx_1m": argmin_idx_1m, "argmax_idx_1m": argmax_idx_1m,
        "cycle_min_price": cycle_min_price, "cycle_max_price": cycle_max_price,
        "current_close_1m": current_close_1m,
        "bars_ago_min": bars_ago_min, "bars_ago_max": bars_ago_max,
        "ts_min": ts_min, "ts_max": ts_max,
        "most_recent_extreme": most_recent_extreme,
        "window_len_1m": window_len_1m,
        "mom_1m": mom_1m, "bullish_perc": bullish_perc, "bearish_perc": bearish_perc,
        "neg_ratio": neg_ratio, "pos_ratio": pos_ratio,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SL & TP CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(entry_price, side):
    tp_dist = entry_price * TP_PRICE_PCT
    sl_dist = entry_price * SL_PRICE_PCT
    if side == "long": sl_price, tp_price = entry_price - sl_dist, entry_price + tp_dist
    else: sl_price, tp_price = entry_price + sl_dist, entry_price - tp_dist
    print(f"  [Risk Calc] TP Dist: {tp_dist:.2f} ({TAKE_PROFIT_ROE}% ROE gross / {NET_PROFIT_ROE}% net)")
    print(f"  [Risk Calc] SL Dist: {sl_dist:.2f} ({STOP_LOSS_ROE}% ROE)")
    print(f"  [Risk Calc] TP Price: {tp_price:.2f} | SL Price: {sl_price:.2f}")
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
            f"LEVERAGE: {LEVERAGE}x\nSTRATEGY: 5m Sine + 1m Extrema/Mom/Vol/FFT\n{'='*60}\n\n")
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
        contracts = max(1, int((balance * LEVERAGE) / (price * get_contract_size(symbol))))
        resp = client.place_order(symbol, side, contracts, LEVERAGE)
        if resp.get("data", {}).get("orderId"):
            print(f"  >>> {side.upper()} PLACED: {contracts} contracts @ ~{price:.2f}")
            return True
        print(f"  [ORDER FAIL] {side.upper()} [{resp.get('code')}]: {resp.get('msg')}")
        return False
    except Exception as e:
        print(f"  [ORDER ERROR] {side.upper()}: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

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

def format_duration(start_dt, end_dt):
    total_sec = int((end_dt - start_dt).total_seconds())
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_conditions(sig):
    """Print detailed values for each condition."""
    f = sig["cond_flags"]
    
    print(f"  1. Sine Scale (5m)")
    print(f"     Dist to Min: {sig['dist_to_min']:.2f}% | Dist to Max: {sig['dist_to_max']:.2f}%")
    print(f"     LONG: {str(f['sine_long']):<5} | SHORT: {str(f['sine_short']):<5}")
    
    print(f"  2. Extrema Cycle (1m)  [window: {sig['window_len_1m']} bars]")
    print(f"     LOWEST  LOW  → price: {sig['cycle_min_price']:.2f}  | idx: {sig['argmin_idx_1m']:4d} | bars ago: {sig['bars_ago_min']:4d} | time: {sig['ts_min']}")
    print(f"     HIGHEST HIGH → price: {sig['cycle_max_price']:.2f}  | idx: {sig['argmax_idx_1m']:4d} | bars ago: {sig['bars_ago_max']:4d} | time: {sig['ts_max']}")
    print(f"     Current Close: {sig['current_close_1m']:.2f}")
    print(f"     Most recent extreme: {sig['most_recent_extreme']}")
    print(f"     LONG: {str(f['cycle_long']):<5} | SHORT: {str(f['cycle_short']):<5}")
    
    print(f"  3. Momentum (1m)")
    print(f"     MOM Value: {sig['mom_1m']:.2f}")
    print(f"     LONG: {str(f['mom_long']):<5} | SHORT: {str(f['mom_short']):<5}")
    
    print(f"  4. Volume (1m)")
    print(f"     Bullish %: {sig['bullish_perc']:.2f}% | Bearish %: {sig['bearish_perc']:.2f}%")
    print(f"     LONG: {str(f['vol_long']):<5} | SHORT: {str(f['vol_short']):<5}")
    
    print(f"  5. FFT Dominance (1m Top 12 Freqs)")
    print(f"     Neg Freq Power: {sig['neg_ratio']:.2f}% | Pos Freq Power: {sig['pos_ratio']:.2f}%")
    print(f"     Dominance Diff: {abs(sig['neg_ratio'] - sig['pos_ratio']):.2f}%")
    print(f"     LONG: {str(f['fft_long']):<5} | SHORT: {str(f['fft_short']):<5}")
    
    print(f"  ──────────────────────────────────────────────────")
    long_true = sum([f['sine_long'], f['cycle_long'], f['mom_long'], f['vol_long'], f['fft_long']])
    short_true = sum([f['sine_short'], f['cycle_short'], f['mom_short'], f['vol_short'], f['fft_short']])
    print(f"  True Count:         LONG: {long_true}/5      | SHORT: {short_true}/5")
    print(f"  Overall LONG:  {sig['is_long']}")
    print(f"  Overall SHORT: {sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"HFT KuCoin Bot Configuration (OPTIMIZED):")
    print(f"{'='*60}")
    print(f"Symbol:              {TRADE_SYMBOL}")
    print(f"Leverage:            {LEVERAGE}x")
    print(f"RT Fee ROE:          {RT_FEE_ROE_PCT}%")
    print(f"Target TP:           {TAKE_PROFIT_ROE}% Gross ROE ({NET_PROFIT_ROE}% Net ROE)")
    print(f"Stop Loss:           {STOP_LOSS_ROE}% ROE")
    print(f"TP Price Move:       {TP_PRICE_PCT*100:.3f}%")
    print(f"SL Price Move:       {SL_PRICE_PCT*100:.3f}%")
    print(f"Data Vectors:        1200 x 5m (~4.1 days) + 1200 x 1m (20 hours)")
    print(f"Loop Sleep:          {LOOP_SLEEP}s (reduced from 3s)")
    print(f"Buffer Refresh:      Every {FULL_REFRESH_INTERVAL}s (1 hour)")
    print(f"API Connected:       {API_CONNECTED}")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZE CANDLE BUFFERS (ONE-TIME FULL FETCH)
    # ══════════════════════════════════════════════════════════════════════════
    
    buffer_5m = CandleBuffer(TRADE_SYMBOL, "5m", ANALYSIS_WINDOW_5M)
    buffer_1m = CandleBuffer(TRADE_SYMBOL, "1m", ANALYSIS_WINDOW_1M)
    
    if API_CONNECTED:
        print("\n  ═══ INITIALIZING CANDLE BUFFERS (one-time fetch) ═══\n")
        
        t0 = time.perf_counter()
        buffer_5m.initialize(client)
        t1 = time.perf_counter()
        print(f"  5m buffer init: {(t1-t0)*1000:.0f}ms")
        
        buffer_1m.initialize(client)
        t2 = time.perf_counter()
        print(f"  1m buffer init: {(t2-t1)*1000:.0f}ms")
        print(f"  Total init time: {(t2-t0)*1000:.0f}ms")
        
        if not buffer_5m.is_ready() or not buffer_1m.is_ready():
            print("\n  [ERROR] Failed to initialize candle buffers. Exiting.")
            return
        
        print(f"\n  ═══ BUFFERS READY - Starting main loop ═══\n")
    else:
        print("\n  [SIMULATION MODE] No API connection - using empty buffers")
    
    saved_state = load_trade_state()
    if saved_state:
        print("  [recovery] Found leftover state — verifying...")
        print(f"    State: {saved_state.get('mode', 'unknown')} {saved_state.get('side', 'unknown')}")
        print(f"    Entry: {saved_state.get('entry_price', 0):.2f}")
    
    loop_count = 0
    last_timing_print = time.time()

    while True:
        try:
            loop_start = time.perf_counter()
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            loop_count += 1

            # ══════════════════════════════════════════════════════════════
            # UPDATE CANDLE BUFFERS (ONLY 2 REQUESTS!)
            # ══════════════════════════════════════════════════════════════
            t = time.perf_counter()
            
            # Check if full refresh needed (prevent drift)
            if buffer_5m.needs_full_refresh() or buffer_1m.needs_full_refresh():
                print(f"  [Buffer] Performing scheduled full refresh...")
                buffer_5m.initialize(client)
                buffer_1m.initialize(client)
            else:
                # Quick update - only fetch new candles
                new_5m = buffer_5m.update(client, fetch_limit=3)
                new_1m = buffer_1m.update(client, fetch_limit=3)
                if loop_count % 50 == 0:
                    print(f"  [Buffer] 5m: +{new_5m} candles (total: {len(buffer_5m)}) | 1m: +{new_1m} candles (total: {len(buffer_1m)})")
            
            candle_update_ms = (time.perf_counter() - t) * 1000
            log_timing("candle_update", candle_update_ms)

            # ══════════════════════════════════════════════════════════════
            # FETCH TICKER, BALANCE, POSITION (3 REQUESTS)
            # ══════════════════════════════════════════════════════════════
            t = time.perf_counter()
            current_price = get_price(TRADE_SYMBOL)
            ticker_ms = (time.perf_counter() - t) * 1000
            log_timing("ticker", ticker_ms)
            
            t = time.perf_counter()
            balance = get_account_balance()
            balance_ms = (time.perf_counter() - t) * 1000
            log_timing("balance", balance_ms)
            
            t = time.perf_counter()
            pos = get_position_info(TRADE_SYMBOL)
            position_ms = (time.perf_counter() - t) * 1000
            log_timing("position", position_ms)
            
            has_sufficient_balance = balance >= MIN_BALANCE_USDT
            
            # ══════════════════════════════════════════════════════════════
            # COMPUTE SIGNALS (NO API CALLS - PURE CALCULATION)
            # ══════════════════════════════════════════════════════════════
            t = time.perf_counter()
            sig = compute_signals(buffer_5m, buffer_1m, current_price)
            compute_ms = (time.perf_counter() - t) * 1000
            log_timing("compute_signals", compute_ms)
            
            # ══════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT
            # ══════════════════════════════════════════════════════════════
            
            if pos["is_open"]:
                roe, side, entry_price, mark_price = pos["roe_pct"], pos["side"], pos["entry_price"], pos["mark_price"]
                
                if loop_count % 5 == 0:
                    print(f"\n[{now_str}] ╔══════ LIVE POSITION ══════╗")
                    print(f"  Side:       {side.upper()}\n  Entry:      {entry_price:.2f}\n  Mark:       {mark_price:.2f}\n  ROE:        {roe:+.2f}%")
                    print(f"  TP Target:  {TAKE_PROFIT_ROE}% ROE\n  SL Target:  {STOP_LOSS_ROE}% ROE")
                    if saved_state and saved_state.get("tp") and saved_state.get("sl"):
                        print(f"  TP Price:   {saved_state['tp']:.2f}\n  SL Price:   {saved_state['sl']:.2f}")
                    print(f"  ─────────────────────────────")
                    if sig: print_conditions(sig)
                    print(f"  ═════════════════════════════")
                
                reason = None
                if roe >= TAKE_PROFIT_ROE: reason = "TAKE PROFIT"
                elif roe <= STOP_LOSS_ROE: reason = "STOP LOSS"
                    
                if reason:
                    print(f"  >>> [{reason}] Triggered at {roe:+.2f}% ROE. Closing position...")
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
                    print(f"  [TRADE CLOSED] Written to {TRADES_LOG_FILE}")
                
                # Log timing and sleep
                loop_ms = (time.perf_counter() - loop_start) * 1000
                log_timing("total_loop", loop_ms)
                if time.time() - last_timing_print > 300:  # Print every 5 minutes
                    print_timing_summary()
                    last_timing_print = time.time()
                time.sleep(LOOP_SLEEP)
                continue

            # ══════════════════════════════════════════════════════════════
            # SIMULATION POSITION MANAGEMENT
            # ══════════════════════════════════════════════════════════════
            
            if saved_state and saved_state.get("mode") == "SIMULATION":
                sim_entry_price, sim_side, sim_entry_time = saved_state["entry_price"], saved_state["side"], saved_state["entry_time"]
                hit, reason, sim_roe = check_sim_tp_sl(sim_entry_price, current_price, sim_side)
                
                if loop_count % 5 == 0:
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    print(f"\n[{now_str}] ╔══════ SIMULATION POSITION ══════╗")
                    print(f"  Side:       {sim_side.upper()}\n  Entry:      {sim_entry_price:.2f}\n  Current:    {current_price:.2f}\n  Sim ROE:    {sim_roe:+.2f}% (Net)\n  Elapsed:    {format_duration(start_dt, now)}")
                    print(f"  TP Target:  {TAKE_PROFIT_ROE}% ROE\n  SL Target:  {STOP_LOSS_ROE}% ROE")
                    if saved_state.get("tp") and saved_state.get("sl"):
                        print(f"  TP Price:   {saved_state['tp']:.2f}\n  SL Price:   {saved_state['sl']:.2f}")
                    print(f"  ─────────────────────────────")
                    if sig: print_conditions(sig)
                    print(f"  ═══════════════════════════════")
                
                if hit and reason:
                    print(f"  >>> [SIM {reason}] Triggered at {sim_roe:+.2f}% ROE")
                    price_change_pct = ((current_price - sim_entry_price) / sim_entry_price) * 100 if sim_side == "long" else ((sim_entry_price - current_price) / sim_entry_price) * 100
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    trade_result = {"mode": "SIMULATION", "entry_time": sim_entry_time, "exit_time": now_str,
                                    "type": sim_side, "entry_price": sim_entry_price, "exit_price": current_price,
                                    "price_change_pct": price_change_pct, "gross_roe": price_change_pct * LEVERAGE,
                                    "rt_fee_pct": RT_FEE_ROE_PCT, "roe": sim_roe,
                                    "duration": format_duration(start_dt, now), "reason": reason}
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                    print(f"  [SIM TRADE CLOSED] Written to {TRADES_LOG_FILE}")
                
                loop_ms = (time.perf_counter() - loop_start) * 1000
                log_timing("total_loop", loop_ms)
                if time.time() - last_timing_print > 300:
                    print_timing_summary()
                    last_timing_print = time.time()
                time.sleep(LOOP_SLEEP)
                continue

            # ══════════════════════════════════════════════════════════════
            # RECOVERY: Clear orphaned state
            # ══════════════════════════════════════════════════════════════
            
            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print("  [recovery] Live state active but no position on exchange. Clearing state.")
                clear_trade_state()
                saved_state = None

            # ══════════════════════════════════════════════════════════════
            # SIGNAL SCANNING (FLAT STATE)
            # ══════════════════════════════════════════════════════════════
            
            if not pos["is_open"] and not saved_state:
                if not sig:
                    if loop_count % 20 == 0: 
                        print(f"[{now_str}] Gathering data... (5m: {len(buffer_5m)}, 1m: {len(buffer_1m)})")
                    loop_ms = (time.perf_counter() - loop_start) * 1000
                    log_timing("total_loop", loop_ms)
                    time.sleep(LOOP_SLEEP)
                    continue

                if loop_count % 5 == 0:
                    print(f"\n[{now_str}] Scanning (FLAT) [Requests/iter: ~5 vs ~15 before]")
                    print_conditions(sig)
                    print(f"  ──────────────────────────────────────────────────")
                    print(f"  Balance: {balance:.2f} USDT | Sufficient: {has_sufficient_balance}")
                    print(f"  Buffers: 5m={len(buffer_5m)} | 1m={len(buffer_1m)}")

                if sig["is_long"]:
                    print(f"\n  *** ALL 5 CONDITIONS MET FOR LONG ***")
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing LONG entry...")
                        if execute_entry(TRADE_SYMBOL, "buy", balance, sig["price"]):
                            sl_price, tp_price = calculate_sl_tp(sig["price"], "long")
                            saved_state = {"active": True, "mode": "LIVE", "side": "long", 
                                           "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIMULATION] Insufficient balance. Starting SIMULATED LONG...")
                        sl_price, tp_price = calculate_sl_tp(sig["price"], "long")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "long", 
                                       "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                        save_trade_state(saved_state)

                elif sig["is_short"]:
                    print(f"\n  *** ALL 5 CONDITIONS MET FOR SHORT ***")
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing SHORT entry...")
                        if execute_entry(TRADE_SYMBOL, "sell", balance, sig["price"]):
                            sl_price, tp_price = calculate_sl_tp(sig["price"], "short")
                            saved_state = {"active": True, "mode": "LIVE", "side": "short", 
                                           "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIMULATION] Insufficient balance. Starting SIMULATED SHORT...")
                        sl_price, tp_price = calculate_sl_tp(sig["price"], "short")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "short", 
                                       "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                        save_trade_state(saved_state)

            # ══════════════════════════════════════════════════════════════
            # LOOP TIMING & SLEEP
            # ══════════════════════════════════════════════════════════════
            
            loop_ms = (time.perf_counter() - loop_start) * 1000
            log_timing("total_loop", loop_ms)
            
            # Print performance summary every 5 minutes
            if time.time() - last_timing_print > 300:
                print_timing_summary()
                last_timing_print = time.time()
            
            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n[Bot] Shutting down safely.")
            print("\nFinal Performance Summary:")
            print_timing_summary()
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()