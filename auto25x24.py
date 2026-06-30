"""
HFT Auto Trading Bot — KuCoin Futures Edition (CONCURRENT OPTIMIZED)
====================================================================
SIGNAL LOGIC:
  - Mandatory: Momentum (1m), Volume (1m), AND ML Forecast MUST be true for the direction.
  - ML Forecast (1m): forecast_price > current_close → LONG allowed;
                      forecast_price < current_close → SHORT allowed.
  - Flexible: At least 4 out of 5 total conditions must be true for the direction.
  - (Since Mom + Vol = 2 mandatory true, you only need 1 more from Sine/Cycle).

POSITION SIZING:
  - Only 5% of available balance is used per trade.
  - Remaining 95% stays untouched in the account.

25x Leverage
TP: 2.55% NET Profit (5.55% Gross ROE after 3.0% RT fee deduction)
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
import numpy as np
import talib
import gc
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            last = values[-1]
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

print("HFT KuCoin Bot (25x / 4-ALL-AGREE HARMONIC LOGIC) initialising...")
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
LOOP_SLEEP = 0.5
MIN_BALANCE_USDT = 5.0
TRADE_BALANCE_PCT = 0.05          # Use only 5% of balance per trade

ML_LOOKBACK = 100                  # Bars used for ML forecast feature window

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
            "price": 0.0,
            "balance": 0.0,
            "position": {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0, 
                        "mark_price": 0.0, "side": None, "size": 0},
            "new_5m": 0,
            "new_1m": 0,
            "times": {}
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
                
                if name == "5m":
                    results["new_5m"] = result
                elif name == "1m":
                    results["new_1m"] = result
                elif name == "price":
                    results["price"] = result
                elif name == "balance":
                    results["balance"] = result
                elif name == "position":
                    results["position"] = result
                    
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


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC OSCILLATOR — BIDIRECTIONAL DUAL CIRCUIT (Condition 4)
# ═══════════════════════════════════════════════════════════════════════════════
#
# NOTE: base phase/cycle data now derives from the 5m timeframe (close_arr_5m /
# candles_5m) instead of 1m — 1m HT_SINE was too noisy / whipped phase too fast.
# Volume VROC nudge still reads 1m tick-level buy/sell dominance for fine nudging.
#
# Two independent oscillators run simultaneously, each STRICTLY divided into
# 4 quadrants of exactly 25% of the 360° cycle each (90° per quadrant):
#
#  UP-CYCLE  (θ, driven by +sine):
#    Q1u [0°–90°)   : Reversal-Dip     — EM exhaust LOW, dip confirmed → LONG entry
#    Q2u [90°–180°) : Accumulation     — price rising off bottom        → LONG cont.
#    Q3u [180°–270°): Pump / Mark-up   — ascending toward peak          → LONG cont.
#    Q4u [270°–360°): Reversal-Top     — EM exhaust HIGH, top confirmed → cycle end
#
#  DOWN-CYCLE (φ, driven by –sine, i.e. inverted):
#    Q4d [0°–90°)   : Reversal-Top     — EM exhaust HIGH, top confirmed → SHORT entry
#    Q3d [90°–180°) : Distribution     — price falling off top          → SHORT cont.
#    Q2d [180°–270°): Falling          — descending toward floor        → SHORT cont.
#    Q1d [270°–360°): Reversal-Dip     — EM exhaust LOW, dip confirmed  → cycle end
#
#  Active circuit selection (ARGMIN/ARGMAX pole, on the 5m window):
#    recent extreme = ARGMIN (low) → UP-CYCLE  active (price bouncing up)
#    recent extreme = ARGMAX (high)→ DOWN-CYCLE active (price falling down)
#
#  STAGE-ORDER ENFORCEMENT (quadrature):
#    Each call may only advance the active circuit's stage by ONE quadrant
#    relative to the previous call (or hold the same stage) — never skip a
#    stage (e.g. straight from Reversal-Dip to Pump). A raw phase reading
#    that jumps ahead more than one quadrant is clamped to "previous + 1".
#    A full circuit flip (handled separately by the ARGMIN/ARGMAX pole
#    switch) resets the stage tracker for the new circuit to its raw reading.
#
#  Volume VROC nudges phase ±15° toward the active cycle's entry quadrant.
#  Forecast projects θ or φ forward HARMONIC_FORECAST_BARS along the active sine.
#  Output: strictly binary — LONG xor SHORT, never both, never neither.
# ═══════════════════════════════════════════════════════════════════════════════

HARMONIC_FORECAST_BARS = 8      # bars ahead for TP-direction forecast (5m bars)
HARMONIC_VROC_PERIOD   = 10     # bars for Volume Rate-of-Change (1m tick bars)

UP_QUAD_LABELS   = ["Q1u-REVERSAL-DIP", "Q2u-ACCUMULATION", "Q3u-PUMP", "Q4u-REVERSAL-TOP"]
DOWN_QUAD_LABELS = ["Q4d-REVERSAL-TOP", "Q3d-DISTRIBUTION", "Q2d-FALLING", "Q1d-REVERSAL-DIP"]

# ── UNIFIED 6-STAGE MARKET QUADRATURE ─────────────────────────────────────
# Q4u (UP top-reversal) and Q4d (DOWN top-reversal) are the SAME physical
# point, and Q1d (DOWN cycle-end dip) and Q1u (UP cycle-start dip) are also
# the SAME physical point. Treating them as 4 independent UP slots + 4
# independent DOWN slots caused "Incoming" to wrongly wrap back to Q1u right
# after Q4u, skipping the real next stage (Q3d-Distribution). Collapsing the
# two duplicated boundary stages gives ONE true circular sequence of 6:
#
#   0: Q1u-REVERSAL-DIP   1: Q2u-ACCUMULATION   2: Q3u-PUMP
#   3: Q4u-REVERSAL-TOP   4: Q3d-DISTRIBUTION   5: Q2d-FALLING   -> wraps to 0
#
# Local (circuit, 0-3) indices map onto this ring as follows:
#   UP   local 0,1,2,3       -> unified 0,1,2,3
#   DOWN local 0(Q4d),1(Q3d),2(Q2d),3(Q1d) -> unified 3,4,5,0
UNIFIED_STAGE_LABELS = [
    "Q1u-REVERSAL-DIP",
    "Q2u-ACCUMULATION",
    "Q3u-PUMP",
    "Q4u-REVERSAL-TOP",
    "Q3d-DISTRIBUTION",
    "Q2d-FALLING",
]
N_UNIFIED_STAGES = len(UNIFIED_STAGE_LABELS)
_DOWN_LOCAL_TO_UNIFIED = {0: 3, 1: 4, 2: 5, 3: 0}

# Persistent unified-ring stage tracker — single continuous index 0-5 that
# only ever holds or advances by exactly +1 (mod 6) per call. No more
# per-circuit reset, so the boundary transition (Q4u -> Q3d, and Q2d -> Q1u)
# is enforced correctly instead of jumping/wrapping incorrectly.
_HO_STAGE_STATE = {"unified_idx": None}


def _to_unified_idx(circuit_name, local_idx):
    if circuit_name == "UP":
        return local_idx
    return _DOWN_LOCAL_TO_UNIFIED[local_idx]


def _physical_side_for_unified(unified_idx):
    """Physical circuit side a unified-ring stage actually belongs to —
    independent of the inverted trend used for ho_long/ho_short. Stages
    0-3 (Q1u/Q2u/Q3u/Q4u) are the UP-circuit's own stages; stages 4-5
    (Q3d/Q2d) are the DOWN-circuit's own stages. This is what should be
    printed next to each quadrant NAME so "Distribution"/"Falling" always
    show as DN, never UP, regardless of the inverted trend reading."""
    return "UP" if unified_idx <= 3 else "DN"


def _tag_for_unified(unified_idx):
    """Corrected display tag for a unified-ring stage, matching the same
    inversion rule used for ho_long/ho_short/trend_str below: stages 0-2
    (Q1u/Q2u/Q3u) read as DOWN, stages 3-5 (Q4u/Q3d/Q2d) read as UP.
    Used ONLY for the active/current stage so it agrees with the header's
    "Active: X-CYCLE  trend_arrow" line."""
    return "UP" if unified_idx >= 3 else "DN"


def _sine_to_phase(sine_val, prev_sine_val):
    """Map a sine value [-1,+1] + direction to a 0–360° phase angle."""
    raw = float(np.degrees(np.arcsin(np.clip(sine_val, -1.0, 1.0))))
    if sine_val >= prev_sine_val:
        return raw + 90.0          # ascending half: 0° (bottom) … 180° (top)
    else:
        return 270.0 - raw         # descending half: 180° (top) … 360° (bottom)


def _enforce_sequential_stage(circuit_name, raw_local_idx):
    """
    Quadrature guard on the UNIFIED 6-stage ring: only allow the confirmed
    unified stage index (0-5) to hold or advance by exactly +1 (mod 6) per
    call relative to the previous confirmed unified stage. This guarantees
    the true physical order Q1u->Q2u->Q3u->Q4u->Q3d->Q2d->Q1u(wrap) is
    always respected, with no skipped stages and no premature wrap.
    Returns (confirmed_unified_idx, prev_unified_idx).
    """
    state = _HO_STAGE_STATE
    raw_unified = _to_unified_idx(circuit_name, raw_local_idx)
    prev_unified = state["unified_idx"]

    if prev_unified is None:
        state["unified_idx"] = raw_unified
        return raw_unified, raw_unified

    next_allowed = (prev_unified + 1) % N_UNIFIED_STAGES
    if raw_unified == prev_unified or raw_unified == next_allowed:
        confirmed = raw_unified
    else:
        confirmed = next_allowed   # force single sequential step forward

    state["unified_idx"] = confirmed
    return confirmed, prev_unified


def compute_harmonic_oscillator(candles_5m, close_arr_5m, live_price,
                                 argmin_idx_5m, argmax_idx_5m,
                                 bars_ago_min, bars_ago_max,
                                 bullish_perc, bearish_perc,
                                 candles_1m=None):
    """
    Base data is the 5m timeframe (slower, far less noisy than 1m).
    `candles_1m` (optional) is only used for the fine-grained volume VROC nudge.

    Returns:
        ho_long          (bool)  : strictly binary
        ho_short         (bool)  : strictly binary, always opposite of ho_long
        theta_deg        (float) : active oscillator phase 0–360
        forecast_price   (float) : projected price HARMONIC_FORECAST_BARS (5m) ahead
        trend_str        (str)   : 'UP' or 'DOWN'
        quad_label       (str)   : active quadrant label with cycle indicator
        extra            (dict)  : quad_progress_pct, cycle_progress_pct,
                                    last_quad_label, next_quad_label, cycle_period
    """
    n = len(close_arr_5m)
    if n < max(32, HARMONIC_VROC_PERIOD + 5):
        return (True, False, 90.0, live_price, "UP", "UP:Q2u-ACCUMULATION",
                {"quad_progress_pct": 0.0, "cycle_progress_pct": 25.0,
                 "last_quad_label": "UP:Q1u-REVERSAL-DIP",
                 "next_quad_label": "UP:Q3u-PUMP", "cycle_period": 20.0})

    # ── 1. COMPUTE BASE SINE ARRAYS (5m) ─────────────────────────────────────
    try:
        sine_raw, _ = talib.HT_SINE(close_arr_5m)
        up_sine   = np.nan_to_num(-sine_raw)   # +sine: bottom=–1, top=+1
        down_sine = np.nan_to_num(sine_raw)    # –sine: top=–1,    bottom=+1
    except Exception:
        up_sine = down_sine = np.zeros(n)

    # ── 2. VOLUME VROC (fine nudge from 1m ticks if available, else neutral) ─
    if candles_1m and len(candles_1m) >= HARMONIC_VROC_PERIOD + 1:
        vols = np.array([c["volume"] for c in candles_1m[-HARMONIC_VROC_PERIOD-1:]], dtype=np.float64)
        vroc = (float(vols[-1]) - float(vols[0])) / (float(vols[0]) + 1e-9)
    else:
        vroc = 0.0
    energy_dir = 1.0 if bullish_perc > bearish_perc else -1.0
    vol_nudge  = 15.0 * np.clip(abs(vroc), 0.0, 1.0)  # magnitude only; sign per cycle

    # ── 3. CYCLE PERIOD (shared, 5m bars) ─────────────────────────────────────
    try:
        period_arr   = talib.HT_DCPERIOD(close_arr_5m)
        valid_p      = period_arr[np.isfinite(period_arr)]
        cycle_period = float(np.median(valid_p[-20:])) if len(valid_p) >= 5 else 20.0
    except Exception:
        cycle_period = 20.0
    cycle_period = max(4.0, min(cycle_period, 200.0))

    window_p  = close_arr_5m[-int(cycle_period):]
    amplitude = (float(np.max(window_p)) - float(np.min(window_p))) / 2.0
    midline   = float(np.mean(window_p))

    # ── 4. UP-CYCLE OSCILLATOR (θ) ───────────────────────────────────────────
    up_cur  = float(up_sine[-1])
    up_prev = float(up_sine[-2]) if n >= 2 else up_cur
    theta_sine = _sine_to_phase(up_cur, up_prev)
    up_vol_shift = -vol_nudge * energy_dir   # negative = toward 0° (Q1u entry)
    theta_deg = (theta_sine + up_vol_shift) % 360.0
    theta_advance = (HARMONIC_FORECAST_BARS / cycle_period) * 360.0
    theta_future  = (theta_deg + theta_advance) % 360.0
    up_forecast   = midline + amplitude * np.sin(np.radians(theta_future - 90.0))

    # ── 5. DOWN-CYCLE OSCILLATOR (φ) ─────────────────────────────────────────
    dn_cur  = float(down_sine[-1])
    dn_prev = float(down_sine[-2]) if n >= 2 else dn_cur
    phi_sine = _sine_to_phase(dn_cur, dn_prev)
    dn_vol_shift = vol_nudge * energy_dir    # positive sell energy → toward 0° (Q4d entry)
    phi_deg = (phi_sine + dn_vol_shift) % 360.0
    phi_advance  = (HARMONIC_FORECAST_BARS / cycle_period) * 360.0
    phi_future   = (phi_deg + phi_advance) % 360.0
    dn_forecast  = midline - amplitude * np.sin(np.radians(phi_future - 90.0))

    # ── 6. ACTIVE CIRCUIT SELECTION (ARGMIN/ARGMAX pole, 5m window) ──────────
    up_cycle_active = (bars_ago_min < bars_ago_max)   # True = up-cycle

    if up_cycle_active:
        circuit_name  = "UP"
        active_phase  = theta_deg
        raw_stage_idx = int(theta_deg // 90.0) % 4
        labels        = UP_QUAD_LABELS
        forecast_price = float(np.clip(up_forecast, live_price * 0.95, live_price * 1.05))
    else:
        circuit_name  = "DOWN"
        active_phase  = phi_deg
        raw_stage_idx = int(phi_deg // 90.0) % 4
        labels        = DOWN_QUAD_LABELS
        forecast_price = float(np.clip(dn_forecast, live_price * 0.95, live_price * 1.05))

    # ── 7. QUADRATURE STAGE GUARD — sequential progression on unified ring ──
    confirmed_unified, prev_unified = _enforce_sequential_stage(circuit_name, raw_stage_idx)
    next_unified = (confirmed_unified + 1) % N_UNIFIED_STAGES

    # Confirmed-quadrant phase boundaries (each EXACTLY 25% / 90° of its own
    # circuit's cycle). raw_stage_idx is still the LOCAL (0-3) index used for
    # the phase-percentage math below; the unified index above is only for
    # display/sequencing across the UP<->DOWN boundary.
    quad_lo = raw_stage_idx * 90.0
    quad_progress_pct  = float(np.clip(((active_phase - quad_lo) / 90.0) * 100.0, 0.0, 100.0))
    cycle_progress_pct = float(np.clip((active_phase / 360.0) * 100.0, 0.0, 100.0))

    # Direction from the CONFIRMED unified stage, not raw phase, so binary
    # output always matches the displayed/enforced stage.
    # NOTE — INVERTED ON PURPOSE: live observation showed the raw UP/DOWN
    # circuit label moves opposite to actual price action (when this engine
    # reads "UP" the market is actually falling, and vice versa). Quadrant
    # NAMES (Reversal-Dip/Accumulation/Pump/...) are left untouched — only
    # the final ho_long/ho_short/trend_str/displayed-tag are flipped here.
    # Unified stages 0,1,2 (Q1u/Q2u/Q3u) -> inverted to DOWN
    # Unified stages 3,4,5 (Q4u/Q3d/Q2d) -> inverted to UP
    if confirmed_unified < 3:
        ho_long, ho_short, trend_str = False, True, "DOWN"
    else:
        ho_long, ho_short, trend_str = True, False, "UP"

    # Active/current tag uses the trend-correlated rule (agrees with the
    # header's "Active: X-CYCLE  trend_arrow" line). Last/Incoming use each
    # stage's own NATURAL physical side instead — Distribution/Falling are
    # always tagged DN, never flipped to UP just because the currently
    # active stage's inverted trend happens to read UP.
    active_label = f"{_tag_for_unified(confirmed_unified)}:{UNIFIED_STAGE_LABELS[confirmed_unified]}"
    last_full    = f"{_physical_side_for_unified(prev_unified)}:{UNIFIED_STAGE_LABELS[prev_unified]}"
    next_full    = f"{_physical_side_for_unified(next_unified)}:{UNIFIED_STAGE_LABELS[next_unified]}"

    extra = {
        "quad_progress_pct":  quad_progress_pct,
        "cycle_progress_pct": cycle_progress_pct,
        "last_quad_label":    last_full,
        "next_quad_label":    next_full,
        "cycle_period":       cycle_period,
    }

    return ho_long, ho_short, active_phase, forecast_price, trend_str, active_label, extra

def calculate_momentum_cyclicity(close_arr, period=14, window=100):
    """
    Momentum condition based on MOM extrema recency, not sign.
    Computes MOM series over `window` bars, finds the index of the
    most negative value (argmin) and most positive value (argmax).
    If argmin is more recent (closer to now) → momentum is LONG (cycle bottom).
    If argmax is more recent                 → momentum is SHORT (cycle top).
    Returns: (mom_last, bars_ago_min, bars_ago_max, is_long, is_short)
    Never both-True, never both-False.
    """
    needed = period + window + 1
    if len(close_arr) < needed:
        return np.nan, 0, 0, True, False   # fallback LONG

    mom_full = talib.MOM(close_arr, timeperiod=period)
    # take the last `window` valid MOM values
    mom_window = mom_full[-window:]
    mom_window = np.nan_to_num(mom_window, nan=0.0)

    argmin_idx = int(np.argmin(mom_window))   # index of most negative (trough)
    argmax_idx = int(np.argmax(mom_window))   # index of most positive (peak)

    w = len(mom_window)
    bars_ago_min = (w - 1) - argmin_idx   # 0 = current bar
    bars_ago_max = (w - 1) - argmax_idx

    mom_last = float(mom_full[-1]) if np.isfinite(mom_full[-1]) else 0.0

    # strictly binary: most recent extreme determines direction
    # argmin (most negative MOM) more recent → momentum trough just passed → LONG reversal incoming
    # argmax (most positive MOM) more recent → momentum peak just passed   → SHORT reversal incoming
    # tie-break: use sign of current MOM value
    if bars_ago_min < bars_ago_max:
        is_long, is_short = True, False   # trough more recent → troughed after → LONG
    elif bars_ago_max < bars_ago_min:
        is_long, is_short = False, True   # peak more recent → peaked after → SHORT
    else:
        # exact tie — fall back to inverse sign
        is_long  = mom_last <= 0.0
        is_short = not is_long

    return mom_last, bars_ago_min, bars_ago_max, is_long, is_short

def generate_ml_forecast(candles_1m):
    """
    Lightweight ML-style forecast using talib indicators + linear regression slope.
    Uses only numpy and talib (no sklearn/pandas dependency).

    Returns:
        forecast_price (float): predicted price one step ahead
        current_close  (float): last close used as baseline
    """
    if len(candles_1m) < ML_LOOKBACK:
        return 0.0, 0.0

    window = candles_1m[-ML_LOOKBACK:]
    closes  = np.array([c["close"]  for c in window], dtype=np.float64)
    highs   = np.array([c["high"]   for c in window], dtype=np.float64)
    lows    = np.array([c["low"]    for c in window], dtype=np.float64)
    volumes = np.array([c["volume"] for c in window], dtype=np.float64)

    # Replace any NaN / zero with previous value
    for arr in (closes, highs, lows, volumes):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]) or arr[i] == 0:
                arr[i] = arr[i-1] if i > 0 else arr[0]

    current_close = float(closes[-1])

    try:
        # ── Feature 1: RSI momentum bias ─────────────────────────────────────
        rsi = talib.RSI(closes, timeperiod=14)
        rsi_last = float(rsi[-1]) if np.isfinite(rsi[-1]) else 50.0
        rsi_bias = (rsi_last - 50.0) / 50.0           # −1..+1

        # ── Feature 2: MACD histogram sign & magnitude ────────────────────────
        macd, macd_sig, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        hist_last = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else 0.0
        hist_norm = hist_last / (current_close + 1e-12)

        # ── Feature 3: Bollinger Band position ───────────────────────────────
        bb_up, bb_mid, bb_lo = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        bb_range = float(bb_up[-1] - bb_lo[-1]) if np.isfinite(bb_up[-1]) and np.isfinite(bb_lo[-1]) else 1e-9
        if bb_range < 1e-12:
            bb_range = 1e-9
        bb_pos = float((closes[-1] - bb_lo[-1]) / bb_range) if np.isfinite(bb_lo[-1]) else 0.5   # 0..1

        # ── Feature 4: Short-term linear regression slope ────────────────────
        n_lr = 20
        lr_closes = closes[-n_lr:]
        x = np.arange(n_lr, dtype=np.float64)
        slope = float(np.polyfit(x, lr_closes, 1)[0])
        slope_norm = slope / (current_close + 1e-12)   # normalised slope

        # ── Feature 5: Volume-weighted price change ───────────────────────────
        recent_ret = (closes[-1] - closes[-6]) / (closes[-6] + 1e-12)
        vol_ratio = float(np.mean(volumes[-3:])) / (float(np.mean(volumes[-20:])) + 1e-12)
        vol_weighted_ret = recent_ret * vol_ratio

        # ── Combine features into a single directional score ─────────────────
        # Weights chosen to reflect relative signal reliability
        score = (
            0.30 * rsi_bias
          + 0.25 * np.sign(hist_norm) * min(abs(hist_norm) * 1e4, 1.0)
          + 0.20 * (bb_pos - 0.5) * 2          # map 0..1 → −1..+1
          + 0.15 * np.sign(slope_norm) * min(abs(slope_norm) * 1e3, 1.0)
          + 0.10 * np.sign(vol_weighted_ret) * min(abs(vol_weighted_ret) * 10, 1.0)
        )

        # ── Translate score into a 1-bar ahead price forecast ─────────────────
        # Max expected single-bar move ~ 0.3% at score magnitude of 1.0
        MAX_MOVE_FRAC = 0.003
        forecast_price = current_close * (1.0 + score * MAX_MOVE_FRAC)

        return float(forecast_price), current_close

    except Exception as e:
        print(f"  [ML Forecast] Error: {e}")
        return 0.0, current_close


def compute_signals(buffer_5m, buffer_1m, live_price):
    candles_5m = buffer_5m.get_candles()
    candles_1m  = buffer_1m.get_candles()

    closes_5m_raw = [c["close"] for c in candles_5m]
    closes_1m_raw = [c["close"] for c in candles_1m]

    if len(closes_5m_raw) < 50 or len(closes_1m_raw) < 50:
        return None

    close_arr_5m = np.array(closes_5m_raw, dtype=float)
    close_arr_1m = np.array(closes_1m_raw, dtype=float)

    # ── Shared pre-computation (needed by Harmonic block) ────────────────────
    candles_1m_window = candles_1m[-500:]
    last_500_1m       = close_arr_1m[-500:]
    window_len_1m     = len(last_500_1m)
    argmin_idx_1m     = int(np.argmin(last_500_1m))
    argmax_idx_1m     = int(np.argmax(last_500_1m))
    cycle_min_price   = float(last_500_1m[argmin_idx_1m])
    cycle_max_price   = float(last_500_1m[argmax_idx_1m])
    current_close_1m  = float(close_arr_1m[-1])
    bars_ago_min      = (window_len_1m - 1) - argmin_idx_1m
    bars_ago_max      = (window_len_1m - 1) - argmax_idx_1m
    most_recent_extreme = "ARGMIN (LOW)" if bars_ago_min > bars_ago_max else (
                          "ARGMAX (HIGH)" if bars_ago_max > bars_ago_min else "TIE")

    try:
        ts_min = datetime.datetime.fromtimestamp(candles_1m_window[argmin_idx_1m]["time"]).strftime("%H:%M:%S")
        ts_max = datetime.datetime.fromtimestamp(candles_1m_window[argmax_idx_1m]["time"]).strftime("%H:%M:%S")
    except Exception:
        ts_min = ts_max = "N/A"

    # ── 1. Momentum (1m) — MANDATORY ─────────────────────────────────────────
    # Cyclicity: most recent MOM extrema determines direction.
    # argmin (most negative MOM) more recent → dip/reversal → LONG
    # argmax (most positive MOM) more recent → peak/reversal → SHORT
    mom_1m, mom_bars_ago_min, mom_bars_ago_max, cond_mom_long, cond_mom_short =         calculate_momentum_cyclicity(close_arr_1m)
    if np.isnan(mom_1m):
        return None

    # ── 2. 3m Midpoint Gate ───────────────────────────────────────────────────
    # Re-derive from 5m candles (proxy: use last 3 candles of 1m as 3m substitute)
    # Full 3m candle data not separately buffered — compute from 1m closes
    last_3m_closes = closes_1m_raw[-180:] if len(closes_1m_raw) >= 180 else closes_1m_raw
    mid_3m_min  = float(np.min(last_3m_closes))
    mid_3m_max  = float(np.max(last_3m_closes))
    mid_3m      = (mid_3m_min + mid_3m_max) / 2.0
    cur_price   = live_price
    cond_mid_long  = cur_price < mid_3m          # below midpoint → upside room
    cond_mid_short = not cond_mid_long

    # ── 3. 45° Angle (Gann) ───────────────────────────────────────────────────
    # Keep exactly as original: expected price along 45° line from cycle low
    # Reference point: cycle min price at bars_ago_min bars ago
    # 1 bar of time = 1 unit of price movement scaled to range/window
    price_range   = cycle_max_price - cycle_min_price
    time_units    = float(window_len_1m)
    slope_per_bar = price_range / time_units if time_units > 0 else 0.0
    expected_45   = cycle_min_price + slope_per_bar * float(bars_ago_min)
    # Above 45° line = bearish (price moved too fast), below = bullish
    above_45      = (live_price > expected_45)
    pct_vs_45     = ((live_price - expected_45) / (expected_45 + 1e-9)) * 100.0
    cond_45_long  = not above_45
    cond_45_short = above_45

    # ── 4. Harmonic Oscillator (Sine + Cycle + Volume fused) — MANDATORY ──────
    # Base phase/cycle data now comes from the 5m timeframe (much less noisy
    # than 1m); 1m candles are still passed in for the fine volume-VROC nudge.
    last_5m_window   = close_arr_5m[-ANALYSIS_WINDOW_5M:] if len(close_arr_5m) >= ANALYSIS_WINDOW_5M else close_arr_5m
    window_len_5m    = len(last_5m_window)
    argmin_idx_5m    = int(np.argmin(last_5m_window))
    argmax_idx_5m    = int(np.argmax(last_5m_window))
    bars_ago_min_5m  = (window_len_5m - 1) - argmin_idx_5m
    bars_ago_max_5m  = (window_len_5m - 1) - argmax_idx_5m

    bullish_perc, bearish_perc = get_buy_sell_volume_perc(candles_1m)
    ho_long, ho_short, theta_deg, forecast_price, trend_str, quad_label, ho_extra = compute_harmonic_oscillator(
        candles_5m, last_5m_window, live_price,
        argmin_idx_5m, argmax_idx_5m,
        bars_ago_min_5m, bars_ago_max_5m,
        bullish_perc, bearish_perc,
        candles_1m=candles_1m
    )
    cond_ho_long  = ho_long
    cond_ho_short = ho_short

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY LOGIC:
    #   MANDATORY: Mom (1) AND Harmonic (4) must agree on direction
    #   FLEXIBLE:  Need all 4/4 total (strict — all must agree)
    #              This is maximum-accuracy mode: every condition must be True
    #   Result is strictly binary — is_long XOR is_short always
    # ══════════════════════════════════════════════════════════════════════════
    long_votes  = [cond_mom_long,  cond_mid_long,  cond_45_long,  cond_ho_long]
    short_votes = [cond_mom_short, cond_mid_short, cond_45_short, cond_ho_short]

    long_true_count  = sum(long_votes)
    short_true_count = sum(short_votes)

    # Mandatory: Mom + HO must both agree
    mom_ho_agree_long  = cond_mom_long  and cond_ho_long
    mom_ho_agree_short = cond_mom_short and cond_ho_short

    # Signal requires mandatory pair + all 4 agree (highest accuracy gate)
    is_long  = mom_ho_agree_long  and (long_true_count  == 4)
    is_short = mom_ho_agree_short and (short_true_count == 4)

    # Strict mutual exclusivity fallback (can't be both)
    if is_long and is_short:
        is_long, is_short = False, False

    return {
        "price":      live_price,
        "is_long":    is_long,
        "is_short":   is_short,
        "cond_flags": {
            "mom_long":  cond_mom_long,  "mom_short":  cond_mom_short,
            "mid_long":  cond_mid_long,  "mid_short":  cond_mid_short,
            "ang_long":  cond_45_long,   "ang_short":  cond_45_short,
            "ho_long":   cond_ho_long,   "ho_short":   cond_ho_short,
        },
        "long_true_count":  long_true_count,
        "short_true_count": short_true_count,
        # Momentum
        "mom_1m": mom_1m,
        "mom_bars_ago_min": mom_bars_ago_min,
        "mom_bars_ago_max": mom_bars_ago_max,
        # 3m Midpoint
        "mid_3m_min": mid_3m_min, "mid_3m_max": mid_3m_max, "mid_3m": mid_3m,
        # 45° Angle
        "expected_45": expected_45, "pct_vs_45": pct_vs_45, "above_45": above_45,
        # Harmonic Oscillator outputs
        "theta_deg":      theta_deg,
        "forecast_price": forecast_price,
        "trend_str":      trend_str,
        "quad_label":     quad_label,
        "quad_progress_pct":  ho_extra["quad_progress_pct"],
        "cycle_progress_pct": ho_extra["cycle_progress_pct"],
        "last_quad_label":    ho_extra["last_quad_label"],
        "next_quad_label":    ho_extra["next_quad_label"],
        "ho_cycle_period":    ho_extra["cycle_period"],
        "bullish_perc":   bullish_perc,
        "bearish_perc":   bearish_perc,
        # Cycle data (kept for reference / journaling)
        "argmin_idx_1m": argmin_idx_1m, "argmax_idx_1m": argmax_idx_1m,
        "cycle_min_price": cycle_min_price, "cycle_max_price": cycle_max_price,
        "current_close_1m": current_close_1m,
        "bars_ago_min": bars_ago_min, "bars_ago_max": bars_ago_max,
        "ts_min": ts_min, "ts_max": ts_max,
        "most_recent_extreme": most_recent_extreme,
        "window_len_1m": window_len_1m,
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
    trade_type = trade_result.get("type", "N/A").upper()
    entry_time = trade_result.get("entry_time", "N/A")
    exit_time = trade_result.get("exit_time", "N/A")
    duration = trade_result.get("duration", "N/A")
    entry_price = trade_result.get("entry_price", 0)
    exit_price = trade_result.get("exit_price", 0)
    price_change_pct = trade_result.get("price_change_pct", 0)
    gross_roe = trade_result.get("gross_roe", 0)
    rt_fee_pct = trade_result.get("rt_fee_pct", 0)
    net_roe = trade_result.get("roe", 0)
    reason = trade_result.get("reason", "N/A")
    strategy = trade_result.get("strategy", f"4/4 Cond (Mom+HO Mandatory, ALL must agree) | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}%")
    
    separator = "=" * 60
    line = (
        f"{separator}\n"
        f"MODE: {mode}\n"
        f"TYPE: {trade_type}\n"
        f"ENTRY TIME: {entry_time}\n"
        f"EXIT TIME: {exit_time}\n"
        f"DURATION: {duration}\n"
        f"ENTRY PRICE: {entry_price:.2f}\n"
        f"EXIT PRICE: {exit_price:.2f}\n"
        f"PRICE CHANGE: {price_change_pct:+.4f}%\n"
        f"GROSS ROE: {gross_roe:+.2f}%\n"
        f"RT FEE DEDUCTED: {rt_fee_pct:.2f}%\n"
        f"NET ROE: {net_roe:+.2f}%\n"
        f"REASON: {reason}\n"
        f"LEVERAGE: {LEVERAGE}x\n"
        f"STRATEGY: {strategy}\n"
        f"{separator}\n\n"
    )
    try:
        with open(TRADES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"  [journal] Trade written to {TRADES_LOG_FILE}")
    except Exception as e:
        print(f"  [journal] write error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_contract_size(symbol):
    if not API_CONNECTED: return 0.001
    try: return float(client.get(f"/api/v1/contracts/{symbol}")["data"]["multiplier"])
    except Exception: return 0.001

def execute_entry(symbol, side, balance, price):
    try:
        trade_balance = balance * TRADE_BALANCE_PCT   # use only 5% of available balance
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
    f        = sig["cond_flags"]
    fc       = sig["forecast_price"]
    lp       = sig["price"]
    fc_pct   = ((fc - lp) / lp * 100) if lp else 0.0
    trend_arrow = "▲ UP" if sig["trend_str"] == "UP" else "▼ DOWN"
    ql       = sig["quad_label"]          # e.g. "UP:Q2u-ACCUMULATION" or "DN:Q3d-DISTRIBUTION"
    cycle_tag = ql.split(":")[0]          # "UP" or "DN"
    quad_tag  = ql.split(":")[-1] if ":" in ql else ql
    last_ql   = sig.get("last_quad_label", ql)
    next_ql   = sig.get("next_quad_label", ql)
    qpp       = sig.get("quad_progress_pct", 0.0)
    cpp       = sig.get("cycle_progress_pct", 0.0)

    print(f"  ┌─────────────────── DUAL-CIRCUIT HARMONIC STATE (5m) ────────────┐")
    print(f"  │ Active: {cycle_tag}-CYCLE  {trend_arrow:<8}  Phase: {sig['theta_deg']:>6.1f}°            │")
    print(f"  │ Last:    {last_ql:<28}                          │")
    print(f"  │ Current: {quad_tag:<28} (each quad = 25% of cycle) │")
    print(f"  │ Incoming:{next_ql:<28}                          │")
    print(f"  │ Quad progress: {qpp:5.1f}%   Cycle progress: {cpp:5.1f}%               │")
    print(f"  │ Forecast→{fc:.2f} ({fc_pct:+.3f}%)  CyclePeriod:{sig.get('ho_cycle_period',0):.1f}bars(5m){' '*10}│")
    print(f"  │ Vol: Bull:{sig['bullish_perc']:.1f}% Bear:{sig['bearish_perc']:.1f}%{" "*36}│")
    print(f"  └─────────────────────────────────────────────────────────────────┘")
    mom_recent = "argmax(HIGH)" if sig["mom_bars_ago_max"] < sig["mom_bars_ago_min"] else "argmin(LOW)"
    print(f"  1. Mom (1m):    {sig['mom_1m']:+.4f}  MinAgo:{sig['mom_bars_ago_min']}bars MaxAgo:{sig['mom_bars_ago_max']}bars Recent:{mom_recent}  L:{'✓' if f['mom_long'] else '✗'} S:{'✓' if f['mom_short'] else '✗'} [MANDATORY]")
    print(f"  2. 3mMid:       Min:{sig['mid_3m_min']:.2f} Max:{sig['mid_3m_max']:.2f} Mid:{sig['mid_3m']:.2f} | Cur:{lp:.2f} ({'above' if lp>=sig['mid_3m'] else 'below'})  L:{'✓' if f['mid_long'] else '✗'} S:{'✓' if f['mid_short'] else '✗'}")
    print(f"  3. 45°Angle:    Expected:{sig['expected_45']:.2f} | Cur:{lp:.2f} ({sig['pct_vs_45']:+.3f}%) {'▲Above' if sig['above_45'] else '▼Below'}  L:{'✓' if f['ang_long'] else '✗'} S:{'✓' if f['ang_short'] else '✗'}")
    print(f"  4. Harmonic:    [{cycle_tag}:{quad_tag}] θ={sig['theta_deg']:.1f}° Trend:{sig['trend_str']} Fcst:{fc:.2f}  L:{'✓' if f['ho_long'] else '✗'} S:{'✓' if f['ho_short'] else '✗'} [MANDATORY]")
    print(f"  ═══ LONG:{sig['long_true_count']}/4 SHORT:{sig['short_true_count']}/4 (Mom+HO Mandatory + ALL 4 Agree) -> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"HFT KuCoin Bot - 4/4 CONDITIONS (MOM+HARMONIC MANDATORY)")
    print(f"{'='*60}")
    print(f"Symbol:              {TRADE_SYMBOL}")
    print(f"Leverage:            {LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_ROE}% ROE | SL: {STOP_LOSS_ROE}% ROE")
    print(f"Loop Sleep:          {LOOP_SLEEP}s")
    print(f"Entry Logic:         Mom & HarmonicOscillator MANDATORY + All 4 Must Agree")
    print(f"Position Sizing:     {TRADE_BALANCE_PCT*100:.0f}% of balance per trade")
    print(f"API Connected:       {API_CONNECTED}")
    print(f"Trades Log File:     {TRADES_LOG_FILE}")
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

            # ══════════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT — LIVE POSITION EXISTS
            # ══════════════════════════════════════════════════════════════════
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
                    gross_roe = price_change_pct * LEVERAGE
                    net_roe = gross_roe - RT_FEE_ROE_PCT
                    trade_result = {
                        "mode": "LIVE",
                        "entry_time": saved_state["entry_time"],
                        "exit_time": now_str,
                        "type": side,
                        "entry_price": entry_price,
                        "exit_price": mark_price,
                        "price_change_pct": price_change_pct,
                        "gross_roe": gross_roe,
                        "rt_fee_pct": RT_FEE_ROE_PCT,
                        "roe": net_roe,
                        "duration": format_duration(start_dt, datetime.datetime.now()),
                        "reason": reason,
                        "strategy": f"4/4 Cond (Mom+HO Mandatory, ALL must agree) | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}%"
                    }
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                
                loop_ms = (time.perf_counter() - loop_start) * 1000
                log_timing("total_loop", loop_ms)
                time.sleep(LOOP_SLEEP)
                continue

            # ══════════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT — SIMULATION POSITION EXISTS
            # ══════════════════════════════════════════════════════════════════
            if saved_state and saved_state.get("mode") == "SIMULATION":
                sim_entry_price, sim_side, sim_entry_time = saved_state["entry_price"], saved_state["side"], saved_state["entry_time"]
                hit, reason, sim_roe = check_sim_tp_sl(sim_entry_price, current_price, sim_side)
                
                if loop_count % 10 == 0:
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    print(f"\n[{now_str}] === SIM {sim_side.upper()} | Entry:{sim_entry_price:.2f} Now:{current_price:.2f} ROE:{sim_roe:+.2f}% ===")
                    if sig: print_conditions(sig)
                
                if hit and reason:
                    print(f"  >>> [SIM {reason}] {sim_roe:+.2f}% ROE")
                    price_change_pct = ((current_price - sim_entry_price) / sim_entry_price) * 100 if sim_side == "long" else ((sim_entry_price - current_price) / sim_entry_price) * 100
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    gross_roe = price_change_pct * LEVERAGE
                    net_roe = gross_roe - RT_FEE_ROE_PCT
                    trade_result = {
                        "mode": "SIMULATION",
                        "entry_time": sim_entry_time,
                        "exit_time": now_str,
                        "type": sim_side,
                        "entry_price": sim_entry_price,
                        "exit_price": current_price,
                        "price_change_pct": price_change_pct,
                        "gross_roe": gross_roe,
                        "rt_fee_pct": RT_FEE_ROE_PCT,
                        "roe": net_roe,
                        "duration": format_duration(start_dt, datetime.datetime.now()),
                        "reason": reason,
                        "strategy": f"4/4 Cond (Mom+HO Mandatory, ALL must agree) | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}%"
                    }
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                
                loop_ms = (time.perf_counter() - loop_start) * 1000
                log_timing("total_loop", loop_ms)
                time.sleep(LOOP_SLEEP)
                continue

            # ══════════════════════════════════════════════════════════════════
            # SIGNAL CHECKING & NEW TRADE ENTRY
            # ══════════════════════════════════════════════════════════════════
            if sig:
                if loop_count % 20 == 0:
                    print(f"\n[{now_str}] Price:{current_price:.2f} | Bal:{balance:.2f} USDT | No Position")
                    print_conditions(sig)

                if sig["is_long"] or sig["is_short"]:
                    side = "buy" if sig["is_long"] else "sell"
                    side_label = "long" if sig["is_long"] else "short"

                    if API_CONNECTED and has_sufficient_balance:
                        print(f"\n  >>> SIGNAL: {side_label.upper()} @ {current_price:.2f} — Executing LIVE order...")
                        success = execute_entry(TRADE_SYMBOL, side, balance, current_price)
                        if success:
                            save_trade_state({
                                "mode": "LIVE",
                                "side": side_label,
                                "entry_price": current_price,
                                "entry_time": now_str
                            })
                            saved_state = {
                                "mode": "LIVE",
                                "side": side_label,
                                "entry_price": current_price,
                                "entry_time": now_str
                            }
                            sl_price, tp_price = calculate_sl_tp(current_price, side_label)
                            print(f"  [LIVE ENTRY] {side_label.upper()} @ {current_price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")
                    else:
                        if not API_CONNECTED:
                            print(f"\n  >>> SIGNAL: {side_label.upper()} @ {current_price:.2f} — Opening SIMULATION position...")
                        elif not has_sufficient_balance:
                            print(f"\n  >>> SIGNAL: {side_label.upper()} @ {current_price:.2f} — Balance too low ({balance:.2f} < {MIN_BALANCE_USDT}), opening SIMULATION position...")
                        
                        save_trade_state({
                            "mode": "SIMULATION",
                            "side": side_label,
                            "entry_price": current_price,
                            "entry_time": now_str
                        })
                        saved_state = {
                            "mode": "SIMULATION",
                            "side": side_label,
                            "entry_price": current_price,
                            "entry_time": now_str
                        }
                        sl_price, tp_price = calculate_sl_tp(current_price, side_label)
                        print(f"  [SIM ENTRY] {side_label.upper()} @ {current_price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")

            # ══════════════════════════════════════════════════════════════════
            # TIMING & LOOP MAINTENANCE
            # ══════════════════════════════════════════════════════════════════
            loop_ms = (time.perf_counter() - loop_start) * 1000
            log_timing("total_loop", loop_ms)
            
            if time.time() - last_timing_print >= 300:
                print_timing_summary()
                last_timing_print = time.time()
            
            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n\n  [SHUTDOWN] Keyboard interrupt received.")
            print_timing_summary()
            fetcher.shutdown()
            break
        except Exception as e:
            print(f"  [LOOP ERROR] {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()