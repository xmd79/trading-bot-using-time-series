"""
HFT Auto Trading Bot — KuCoin Futures Edition (CONCURRENT OPTIMIZED)
====================================================================
SIGNAL LOGIC:
  - Mandatory (5): Sine (1m), Volume (1m), ML Forecast, QuadTrend, QuadRev MUST all be True.
  - Flexible: At least 4 out of 9 total conditions must be True for the direction.
  - Conditions: 1.Sine* 2.Cycle 3.Mid 4.Mom 5.Vol* 6.FFT 7.ML* 8.QuadTrend* 9.QuadRev*
                (* = mandatory)

CYCLIC ENERGY CIRCUIT (Unit Circle Quadrant Engine):
  Price position within [argmin_price .. argmax_price] 500-bar range:
    Q1 (0–25%)  = most negative pole  — argmin / dip
    Q2 (25–50%) = lower ascending
    Q3 (50–75%) = upper ascending
    Q4 (75–100%)= most positive pole  — argmax / top

  COND 8 — QuadTrend (always exactly one True):
    LONG  = direction is "up"   (cycle Q1→Q2→Q3→Q4 in progress)
    SHORT = direction is "down" (cycle Q4→Q3→Q2→Q1 in progress)

  COND 9 — QuadRev (always exactly one True):
    LONG  = Q1 reversal confirmed (argmin most recent AND price in Q1 zone)
    SHORT = Q4 reversal confirmed (argmax most recent AND price in Q4 zone)
    Between reversals: inherits last confirmed reversal direction.

  Per-iteration print: [last_quad] → [current_quad] → [next_quad]
  Reversal alert printed when Q1 or Q4 reversal fires this bar.

POSITION SIZING:
  - Only 5% of available balance is used per trade.
  - Remaining 95% stays untouched in the account.

25x Leverage
TP: 2.55% NET Profit (5.55% Gross ROE after 3.0% RT fee deduction)
SL: -99% ROE

CHANGES FROM v11:
  - Quad condition split into TWO separate mandatory conditions (8 & 9)
  - Cond 8 QuadTrend: pure cycle direction (up/down), always one T/one F
  - Cond 9 QuadRev:   reversal trigger (Q1=long/Q4=short), always one T/one F
  - Both are mandatory — 9 conditions total, 5 mandatory, threshold 4/9
"""

import time
import hmac
import json
import base64
import hashlib
import datetime
import os
import logging
import requests
import numpy as np
import talib
import gc
from scipy.fft import fft, fftfreq
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging for the new FFT function
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

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

print("HFT KuCoin Bot (25x / 3-OUT-OF-5 LOGIC) initialising...")
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
MOM_LOOKBACK = 200                 # Lookback for Momentum recency analysis

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
# CYCLIC ENERGY CIRCUIT — UNIT CIRCLE QUADRANT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Two SEPARATE conditions produced — each always exactly one True / one False:
#
#  CONDITION 8 — QUAD TREND (direction of the current cycle in progress)
#    cond_quad_trend_long  = True  when direction is UP  (cycle started at Q1, traveling Q1→Q2→Q3→Q4)
#    cond_quad_trend_short = True  when direction is DOWN (cycle started at Q4, traveling Q4→Q3→Q2→Q1)
#    Tiebreak (no prior reversal yet): argmin recency wins; if still tied → long.
#
#  CONDITION 9 — QUAD REVERSAL (is the current bar a confirmed reversal trigger)
#    cond_quad_rev_long  = True  when Q1 reversal confirmed (argmin most recent AND price in Q1)
#    cond_quad_rev_short = True  when Q4 reversal confirmed (argmax most recent AND price in Q4)
#    Between reversals: inherits last confirmed direction. Guaranteed mutual exclusivity.
#
#  Unit circle mapping of [argmin_price .. argmax_price]:
#    Q1 : 0%–25%   — most negative pole (dip / argmin LOW)
#    Q2 : 25%–50%  — lower ascending
#    Q3 : 50%–75%  — upper ascending
#    Q4 : 75%–100% — most positive pole (top / argmax HIGH)
# ═══════════════════════════════════════════════════════════════════════════════

_cycle_state = {
    "direction": None,          # "up" | "down"  — persists between reversals
    "last_quad":    None,
    "current_quad": None,
    "reversal_type": None,      # "Q1_REVERSAL" | "Q4_REVERSAL"
    "reversal_count": 0,
}

_UP_SEQUENCE   = ["Q1", "Q2", "Q3", "Q4"]
_DOWN_SEQUENCE = ["Q4", "Q3", "Q2", "Q1"]

def _price_to_quadrant(price, low, high):
    """Map price within [low, high] to Q1/Q2/Q3/Q4. Clamp at boundaries."""
    if high <= low:
        return "Q1"
    pct = max(0.0, min(1.0, (price - low) / (high - low)))
    if pct < 0.25: return "Q1"
    if pct < 0.50: return "Q2"
    if pct < 0.75: return "Q3"
    return "Q4"

def _next_quad_in_direction(current_quad, direction):
    seq = _UP_SEQUENCE if direction == "up" else (_DOWN_SEQUENCE if direction == "down" else None)
    if seq is None:
        return "?"
    try:
        idx = seq.index(current_quad)
        return seq[idx + 1] if idx + 1 < len(seq) else seq[-1]
    except ValueError:
        return "?"

def compute_energy_quadrant(current_price, cycle_min_price, cycle_max_price,
                             bars_ago_min, bars_ago_max):
    """
    Compute both quadrant conditions for this iteration.

    Returns dict with keys used by compute_signals:
        current_quad        : Q1/Q2/Q3/Q4
        last_quad           : previous iteration's quad
        next_quad           : projected next quad
        direction           : "up" / "down"
        pct_in_range        : 0–100% position in cycle range
        reversal_type       : "Q1_REVERSAL" / "Q4_REVERSAL" (last confirmed)
        reversal_count      : total reversals seen

        -- COND 8: QUAD TREND --
        cond_quad_trend_long   (bool) : direction is "up"   — always opposite of trend_short
        cond_quad_trend_short  (bool) : direction is "down"  — always opposite of trend_long

        -- COND 9: QUAD REVERSAL --
        cond_quad_rev_long     (bool) : Q1 reversal active  — always opposite of rev_short
        cond_quad_rev_short    (bool) : Q4 reversal active  — always opposite of rev_long

    Invariants:
        cond_quad_trend_long  XOR cond_quad_trend_short  == True  (always)
        cond_quad_rev_long    XOR cond_quad_rev_short    == True  (always)
    """
    global _cycle_state

    current_quad = _price_to_quadrant(current_price, cycle_min_price, cycle_max_price)
    cycle_range  = cycle_max_price - cycle_min_price if cycle_max_price > cycle_min_price else 1e-9
    pct_in_range = max(0.0, min(100.0, (current_price - cycle_min_price) / cycle_range * 100))

    # ── Reversal detection ──────────────────────────────────────────────────
    # Q1 confirmed: argmin is most recent extreme AND price sits in Q1 zone
    q1_rev = (bars_ago_min < bars_ago_max) and (current_quad == "Q1")
    # Q4 confirmed: argmax is most recent extreme AND price sits in Q4 zone
    q4_rev = (bars_ago_max < bars_ago_min) and (current_quad == "Q4")

    # Update persistent direction on confirmed reversal
    if q1_rev:
        _cycle_state["direction"]     = "up"
        _cycle_state["reversal_type"] = "Q1_REVERSAL"
        _cycle_state["reversal_count"] += 1
    elif q4_rev:
        _cycle_state["direction"]     = "down"
        _cycle_state["reversal_type"] = "Q4_REVERSAL"
        _cycle_state["reversal_count"] += 1

    # ── CONDITION 8: QUAD TREND — always exactly one True ──────────────────
    # Determine direction; if still None (bot just started), use recency tiebreak
    if _cycle_state["direction"] == "up":
        cond_quad_trend_long  = True
        cond_quad_trend_short = False
    elif _cycle_state["direction"] == "down":
        cond_quad_trend_long  = False
        cond_quad_trend_short = True
    else:
        # Bootstrap: no reversal seen yet — use argmin recency as tiebreak
        if bars_ago_min <= bars_ago_max:   # argmin more recent or tied → up
            cond_quad_trend_long  = True
            cond_quad_trend_short = False
            _cycle_state["direction"] = "up"
        else:
            cond_quad_trend_long  = False
            cond_quad_trend_short = True
            _cycle_state["direction"] = "down"

    # ── CONDITION 9: QUAD REVERSAL — always exactly one True ───────────────
    # Active reversal: confirmed this bar takes precedence.
    # Between reversals: inherits last confirmed direction.
    if q1_rev:
        # Q1 reversal confirmed this bar → LONG reversal active
        cond_quad_rev_long  = True
        cond_quad_rev_short = False
    elif q4_rev:
        # Q4 reversal confirmed this bar → SHORT reversal active
        cond_quad_rev_long  = False
        cond_quad_rev_short = True
    else:
        # No new reversal — inherit from last confirmed direction
        # (same as trend: up→long active reversal side, down→short active)
        if _cycle_state["direction"] == "up":
            cond_quad_rev_long  = True
            cond_quad_rev_short = False
        else:
            cond_quad_rev_long  = False
            cond_quad_rev_short = True

    # ── Quad history ────────────────────────────────────────────────────────
    last_quad = _cycle_state["current_quad"]
    _cycle_state["last_quad"]    = last_quad
    _cycle_state["current_quad"] = current_quad

    direction = _cycle_state["direction"]
    next_quad = _next_quad_in_direction(current_quad, direction)

    return {
        "current_quad":  current_quad,
        "last_quad":     last_quad if last_quad else "—",
        "next_quad":     next_quad,
        "direction":     direction,
        "pct_in_range":  pct_in_range,
        "reversal_type": _cycle_state["reversal_type"] if _cycle_state["reversal_type"] else "—",
        "reversal_count": _cycle_state["reversal_count"],
        "q1_rev_this_bar": q1_rev,
        "q4_rev_this_bar": q4_rev,
        # COND 8
        "cond_quad_trend_long":  cond_quad_trend_long,
        "cond_quad_trend_short": cond_quad_trend_short,
        # COND 9
        "cond_quad_rev_long":    cond_quad_rev_long,
        "cond_quad_rev_short":   cond_quad_rev_short,
    }

def print_quadrant_state(qstate):
    """Print cyclic energy circuit status — both conditions — at each iteration."""
    arr  = "▲" if qstate["direction"] == "up" else "▼"
    seq_str = "Q1→Q2→Q3→Q4" if qstate["direction"] == "up" else "Q4→Q3→Q2→Q1"

    # Reversal alert tags
    rev_tag = ""
    if qstate["q1_rev_this_bar"]:
        rev_tag = "  ◄◄ Q1 REVERSAL CONFIRMED — LONG TRIGGER"
    elif qstate["q4_rev_this_bar"]:
        rev_tag = "  ◄◄ Q4 REVERSAL CONFIRMED — SHORT TRIGGER"

    print(f"  8. QuadTREND:  [{qstate['last_quad']}]→[{qstate['current_quad']}]→[{qstate['next_quad']}]"
          f"  Cycle:{arr}{seq_str}  Pos:{qstate['pct_in_range']:.1f}%"
          f"  L:{qstate['cond_quad_trend_long']} S:{qstate['cond_quad_trend_short']}  [MANDATORY]")
    print(f"  9. QuadREV:    Rev:{qstate['reversal_type']} #{qstate['reversal_count']}"
          f"  L:{qstate['cond_quad_rev_long']} S:{qstate['cond_quad_rev_short']}  [MANDATORY]{rev_tag}")

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

def clean_data(arr):
    """Replace NaN/Inf with linearly interpolated valid values."""
    arr = np.array(arr, dtype=np.float64)
    nans = np.isnan(arr) | np.isinf(arr)
    if np.any(nans):
        valid = ~nans
        if np.any(valid):
            arr[nans] = np.interp(np.where(nans)[0], np.where(valid)[0], arr[valid])
        else:
            arr[:] = 0.0
    return arr

def scale_to_sine(close_prices, argmin_idx, argmax_idx):
    """
    Scales HT_SINE output to 0-100% distance to min/max cycle extremes.
    Now accepts any timeframe's close prices (changed from 5m-only to support 1m).
    """
    if len(close_prices) < 32: return 50.0, 50.0
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(-sine_wave)
    
    sine_window = sine_wave
    if len(sine_window) < 2: return 50.0, 50.0
    
    cycle_min = sine_window[argmin_idx]
    cycle_max = sine_window[argmax_idx]
    rng = cycle_max - cycle_min if cycle_max != cycle_min else 1e-9
    current_sine = sine_wave[-1]
    dist_to_min = max(0, min(100, ((current_sine - cycle_min) / rng) * 100))
    dist_to_max = max(0, min(100, ((cycle_max - current_sine) / rng) * 100))
    return dist_to_min, dist_to_max

def fft_forecast_1m(candles, n_components=5):
    """
    True FFT forecast for 1m closes.

    1. Detrend closes (remove linear trend so DC/drift don't pollute spectrum).
    2. FFT → pick top-N dominant positive-frequency components by amplitude.
    3. Reconstruct signal at bar N (one bar ahead of the last observed bar).
    4. Add trend back → fft_forecast_price.
    5. dominant_freq = cycles-per-bar of the single highest-amplitude component.
    6. period_bars   = 1 / dominant_freq  (bars per cycle).
    7. Direction: fft_forecast_price > current_close → LONG, else SHORT (never tie).

    Returns:
        fft_forecast_price  (float)  – predicted close one bar ahead
        dominant_freq       (float)  – cycles per bar of dominant component
        period_bars         (float)  – bars per dominant cycle
        fft_long            (bool)
        fft_short           (bool)
    """
    closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
    closes = clean_data(closes)
    n = len(closes)

    if n < 32:
        cur = float(closes[-1]) if n > 0 else 0.0
        return cur, 0.0, 0.0, False, False

    # --- detrend ---
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, closes, 1)
    trend = slope * x + intercept
    detrended = closes - trend

    # --- FFT on positive frequencies only (indices 1 .. n//2) ---
    F = fft(detrended)
    freqs = np.fft.fftfreq(n)            # cycles per bar, range [-0.5, 0.5]
    pos_mask = freqs > 0
    pos_idx  = np.where(pos_mask)[0]

    if len(pos_idx) == 0:
        cur = float(closes[-1])
        return cur, 0.0, 0.0, False, False

    amplitudes = np.abs(F[pos_idx])
    # pick top-N components
    top_n = min(n_components, len(pos_idx))
    top_local = np.argsort(amplitudes)[-top_n:][::-1]
    top_global = pos_idx[top_local]

    # dominant = highest amplitude component
    dom_global = top_global[0]
    dominant_freq = float(freqs[dom_global])
    period_bars   = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')

    # reconstruct at bar index n  (one step ahead)
    next_bar = float(n)
    recon = 0.0
    for gi in top_global:
        amp   = np.abs(F[gi]) * 2.0 / n      # one-sided amplitude
        phase = np.angle(F[gi])
        freq  = float(freqs[gi])
        recon += amp * np.cos(2.0 * np.pi * freq * next_bar + phase)

    # add trend back at next bar
    trend_next = slope * next_bar + intercept
    fft_forecast_price = float(trend_next + recon)

    current_close = float(closes[-1])

    # guaranteed mutual exclusivity: strictly > vs <=
    fft_long  = fft_forecast_price > current_close
    fft_short = not fft_long

    return fft_forecast_price, dominant_freq, period_bars, fft_long, fft_short

def calculate_momentum_recency(close_arr, lookback=200, period=14):
    """
    [MANDATORY LOGIC - FIXED]
    Calculate talib.MOM for the last `lookback` values.
    
    LONG  = True  if the index of the MOST NEGATIVE (lowest) MOM value
                    was MORE RECENT (lower bars ago) than the MOST POSITIVE (highest) MOM value.
                    (Meaning: Momentum bottomed out and is recovering -> LONG)
                    
    SHORT = True  if the index of the MOST POSITIVE (highest) MOM value
                    was MORE RECENT (lower bars ago) than the MOST NEGATIVE (lowest) MOM value.
                    (Meaning: Momentum peaked and is rolling over -> SHORT)
    
    Returns:
        cond_mom_long (bool)
        cond_mom_short (bool)
        mom_current (float): last MOM value
        mom_max_val (float): highest positive MOM in window
        mom_min_val (float): lowest negative MOM in window
        mom_max_idx (int): index of highest MOM (0=oldest, lookback-1=newest)
        mom_min_idx (int): index of lowest MOM (0=oldest, lookback-1=newest)
        bars_ago_max (int): bars ago when highest MOM occurred
        bars_ago_min (int): bars ago when lowest MOM occurred
    """
    if len(close_arr) < lookback + period:
        mom_val = float(talib.MOM(close_arr, timeperiod=period)[-1]) if len(close_arr) >= period + 1 else 0.0
        if np.isnan(mom_val):
            mom_val = 0.0
        return mom_val > 0, mom_val < 0, mom_val, mom_val, mom_val, 0, 0, 0, 0
    
    window_close = close_arr[-(lookback + period):]
    mom_series = talib.MOM(window_close, timeperiod=period)
    mom_series = np.nan_to_num(mom_series)
    mom_200 = mom_series[-lookback:]
    
    if len(mom_200) < 2:
        mom_current = float(mom_200[-1]) if len(mom_200) > 0 else 0.0
        return mom_current > 0, mom_current < 0, mom_current, mom_current, mom_current, 0, 0, 0, 0
    
    mom_max_idx = int(np.argmax(mom_200))
    mom_min_idx = int(np.argmin(mom_200))
    
    mom_max_val = float(mom_200[mom_max_idx])
    mom_min_val = float(mom_200[mom_min_idx])
    mom_current = float(mom_200[-1])
    
    # Convert to "bars ago" (0 = current bar, higher = older)
    bars_ago_max = (len(mom_200) - 1) - mom_max_idx
    bars_ago_min = (len(mom_200) - 1) - mom_min_idx
    
    # MANDATORY LOGIC (FIXED):
    # Most recent = LOWEST number of bars ago
    # If HiMOM is most recent (low bars_ago_max) -> peaked -> SHORT
    # If LoMOM is most recent (low bars_ago_min) -> bottomed -> LONG
    cond_mom_long = bars_ago_min < bars_ago_max   # LoMOM more recent -> LONG
    cond_mom_short = bars_ago_max < bars_ago_min  # HiMOM more recent -> SHORT
    
    if mom_max_idx == mom_min_idx:
        cond_mom_long = False
        cond_mom_short = False
    
    return (cond_mom_long, cond_mom_short, mom_current, 
            mom_max_val, mom_min_val, mom_max_idx, mom_min_idx,
            bars_ago_max, bars_ago_min)

def generate_ml_forecast(candles_1m):
    """
    Lightweight ML-style forecast using talib indicators + linear regression slope.
    """
    if len(candles_1m) < ML_LOOKBACK:
        return 0.0, 0.0

    window = candles_1m[-ML_LOOKBACK:]
    closes  = np.array([c["close"]  for c in window], dtype=np.float64)
    highs   = np.array([c["high"]   for c in window], dtype=np.float64)
    lows    = np.array([c["low"]    for c in window], dtype=np.float64)
    volumes = np.array([c["volume"] for c in window], dtype=np.float64)

    for arr in (closes, highs, lows, volumes):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]) or arr[i] == 0:
                arr[i] = arr[i-1] if i > 0 else arr[0]

    current_close = float(closes[-1])

    try:
        rsi = talib.RSI(closes, timeperiod=14)
        rsi_last = float(rsi[-1]) if np.isfinite(rsi[-1]) else 50.0
        rsi_bias = (rsi_last - 50.0) / 50.0

        macd, macd_sig, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        hist_last = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else 0.0
        hist_norm = hist_last / (current_close + 1e-12)

        bb_up, bb_mid, bb_lo = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        bb_range = float(bb_up[-1] - bb_lo[-1]) if np.isfinite(bb_up[-1]) and np.isfinite(bb_lo[-1]) else 1e-9
        if bb_range < 1e-12:
            bb_range = 1e-9
        bb_pos = float((closes[-1] - bb_lo[-1]) / bb_range) if np.isfinite(bb_lo[-1]) else 0.5

        n_lr = 20
        lr_closes = closes[-n_lr:]
        x = np.arange(n_lr, dtype=np.float64)
        slope = float(np.polyfit(x, lr_closes, 1)[0])
        slope_norm = slope / (current_close + 1e-12)

        recent_ret = (closes[-1] - closes[-6]) / (closes[-6] + 1e-12)
        vol_ratio = float(np.mean(volumes[-3:])) / (float(np.mean(volumes[-20:])) + 1e-12)
        vol_weighted_ret = recent_ret * vol_ratio

        score = (
            0.30 * rsi_bias
          + 0.25 * np.sign(hist_norm) * min(abs(hist_norm) * 1e4, 1.0)
          + 0.20 * (bb_pos - 0.5) * 2
          + 0.15 * np.sign(slope_norm) * min(abs(slope_norm) * 1e3, 1.0)
          + 0.10 * np.sign(vol_weighted_ret) * min(abs(vol_weighted_ret) * 10, 1.0)
        )

        MAX_MOVE_FRAC = 0.003
        forecast_price = current_close * (1.0 + score * MAX_MOVE_FRAC)

        return float(forecast_price), current_close

    except Exception as e:
        print(f"  [ML Forecast] Error: {e}")
        return 0.0, current_close


def compute_signals(buffer_5m, buffer_1m, live_price):
    candles_5m = buffer_5m.get_candles()
    candles_1m = buffer_1m.get_candles()
    
    closes_5m_raw = [c["close"] for c in candles_5m]
    closes_1m_raw = [c["close"] for c in candles_1m]
    
    if len(closes_5m_raw) < 50 or len(closes_1m_raw) < 50: 
        return None
    
    close_arr_5m = np.array(closes_5m_raw, dtype=float)
    close_arr_1m = np.array(closes_1m_raw, dtype=float)
    
    # 1. Sine Scale (1m)
    last_500_1m_for_sine = close_arr_1m[-ANALYSIS_WINDOW_1M:]
    argmin_idx_sine = int(np.argmin(last_500_1m_for_sine))
    argmax_idx_sine = int(np.argmax(last_500_1m_for_sine))
    dist_to_min, dist_to_max = scale_to_sine(close_arr_1m, argmin_idx_sine, argmax_idx_sine)
    cond_sine_long = dist_to_min < dist_to_max
    cond_sine_short = dist_to_max < dist_to_min
    
    # 2. Extrema Cycle (1m) - STRICT MUTUAL EXCLUSIVITY
    # Uses live_price as current close for real-time accuracy
    candles_1m_window = candles_1m[-500:]
    last_500_1m = close_arr_1m[-500:]
    window_len_1m = len(last_500_1m)
    argmin_idx_1m = int(np.argmin(last_500_1m))
    argmax_idx_1m = int(np.argmax(last_500_1m))
    cycle_min_price = float(last_500_1m[argmin_idx_1m])   # lowest low in window
    cycle_max_price = float(last_500_1m[argmax_idx_1m])   # highest high in window
    current_close_1m = float(live_price)                   # REAL-TIME price
    cycle_middle_price = (cycle_min_price + cycle_max_price) / 2.0

    bars_ago_min = (window_len_1m - 1) - argmin_idx_1m
    bars_ago_max = (window_len_1m - 1) - argmax_idx_1m

    try:
        ts_min = datetime.datetime.fromtimestamp(candles_1m_window[argmin_idx_1m]["time"]).strftime("%Y-%m-%d %H:%M:%S")
        ts_max = datetime.datetime.fromtimestamp(candles_1m_window[argmax_idx_1m]["time"]).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts_min = ts_max = "N/A"

    # Recency rule: most recently formed extreme determines bias
    #   ARGMIN (lowest low) most recent -> bounced from bottom -> LONG
    #   ARGMAX (highest high) most recent -> peaked -> SHORT
    if bars_ago_min < bars_ago_max:
        most_recent_extreme = "ARGMIN (LOW)"
        cond_cycle_long  = True
        cond_cycle_short = False
        cycle_status_for_fft = "Up"
    elif bars_ago_max < bars_ago_min:
        most_recent_extreme = "ARGMAX (HIGH)"
        cond_cycle_long  = False
        cond_cycle_short = True
        cycle_status_for_fft = "Down"
    else:
        most_recent_extreme = "TIE"
        cond_cycle_long  = False
        cond_cycle_short = False
        cycle_status_for_fft = "Up"

    # 2b. Middle-line condition: price vs midpoint of cycle range
    #     Below middle -> bullish position in range -> long bias
    #     Above middle -> bearish position in range -> short bias
    cond_mid_long  = current_close_1m < cycle_middle_price
    cond_mid_short = current_close_1m > cycle_middle_price

    # 2c. Cyclic Energy Circuit — TWO SEPARATE MANDATORY CONDITIONS (8 & 9)
    qstate = compute_energy_quadrant(
        current_price=current_close_1m,
        cycle_min_price=cycle_min_price,
        cycle_max_price=cycle_max_price,
        bars_ago_min=bars_ago_min,
        bars_ago_max=bars_ago_max,
    )
    # COND 8 — Quad TREND: are we in an up-cycle or down-cycle right now?
    #   always exactly one True / one False
    cond_quad_trend_long  = qstate["cond_quad_trend_long"]
    cond_quad_trend_short = qstate["cond_quad_trend_short"]

    # COND 9 — Quad REVERSAL: is the current bar a confirmed Q1 (long) or Q4 (short) reversal?
    #   always exactly one True / one False
    cond_quad_rev_long  = qstate["cond_quad_rev_long"]
    cond_quad_rev_short = qstate["cond_quad_rev_short"]

    # 3. Momentum (1m) — MANDATORY FIXED LOGIC
    (cond_mom_long, cond_mom_short, mom_1m,
     mom_max_val, mom_min_val, mom_max_idx, mom_min_idx,
     mom_bars_ago_max, mom_bars_ago_min) = calculate_momentum_recency(
        close_arr_1m, lookback=MOM_LOOKBACK, period=14
    )
    
    # 4. Volume (1m)
    bullish_perc, bearish_perc = get_buy_sell_volume_perc(candles_1m)
    cond_vol_long = bullish_perc > bearish_perc
    cond_vol_short = bearish_perc > bullish_perc
    
    # 5. FFT Forecast (1m) — dominant frequency + one-bar-ahead price forecast
    fft_forecast_price = 0.0
    fft_dominant_freq  = 0.0
    fft_period_bars    = 0.0
    cond_fft_long      = False
    cond_fft_short     = False
    try:
        fft_forecast_price, fft_dominant_freq, fft_period_bars, cond_fft_long, cond_fft_short = \
            fft_forecast_1m(candles_1m, n_components=5)
    except Exception as e:
        logging.error(f"FFT Forecast Error: {e}")
        # fallback: guaranteed direction from cycle extrema
        cond_fft_long  = (cycle_status_for_fft == "Up")
        cond_fft_short = not cond_fft_long

    # 6. ML Forecast (1m) — MANDATORY
    ml_forecast_price, ml_current_close = generate_ml_forecast(candles_1m)
    ml_forecast_valid = (ml_forecast_price > 0.0 and ml_current_close > 0.0)
    cond_ml_long  = ml_forecast_valid and (ml_forecast_price > ml_current_close)
    cond_ml_short = ml_forecast_valid and (ml_forecast_price < ml_current_close)

    # ══════════════════════════════════════════════════════════════════
    # LOGIC: ALL 5 MANDATORY + MAJORITY of 9 on same side
    # Conditions: 1.Sine* 2.Cycle 3.Mid 4.Mom 5.Vol* 6.FFT 7.ML* 8.QuadTrend* 9.QuadRev*
    # ══════════════════════════════════════════════════════════════════
    long_true_count  = sum([cond_sine_long,  cond_cycle_long,  cond_mid_long,
                            cond_mom_long,   cond_vol_long,    cond_fft_long,
                            cond_ml_long,    cond_quad_trend_long,  cond_quad_rev_long])
    short_true_count = sum([cond_sine_short, cond_cycle_short, cond_mid_short,
                            cond_mom_short,  cond_vol_short,   cond_fft_short,
                            cond_ml_short,   cond_quad_trend_short, cond_quad_rev_short])

    is_long  = (cond_sine_long  and cond_vol_long  and cond_ml_long
                and cond_quad_trend_long  and cond_quad_rev_long
                and long_true_count > short_true_count)
    is_short = (cond_sine_short and cond_vol_short and cond_ml_short
                and cond_quad_trend_short and cond_quad_rev_short
                and short_true_count > long_true_count)
    
    return {
        "price": live_price, "is_long": is_long, "is_short": is_short,
        "cond_flags": {
            "sine_long": cond_sine_long, "sine_short": cond_sine_short,
            "cycle_long": cond_cycle_long, "cycle_short": cond_cycle_short,
            "mid_long": cond_mid_long, "mid_short": cond_mid_short,
            "mom_long": cond_mom_long, "mom_short": cond_mom_short,
            "vol_long": cond_vol_long, "vol_short": cond_vol_short,
            "fft_long": cond_fft_long, "fft_short": cond_fft_short,
            "ml_long": cond_ml_long, "ml_short": cond_ml_short,
            "quad_trend_long": cond_quad_trend_long, "quad_trend_short": cond_quad_trend_short,
            "quad_rev_long":   cond_quad_rev_long,   "quad_rev_short":   cond_quad_rev_short,
        },
        "long_true_count": long_true_count,
        "short_true_count": short_true_count,
        "qstate": qstate,
        "dist_to_min": dist_to_min, "dist_to_max": dist_to_max,
        "argmin_idx_1m": argmin_idx_1m, "argmax_idx_1m": argmax_idx_1m,
        "cycle_min_price": cycle_min_price, "cycle_max_price": cycle_max_price,
        "cycle_middle_price": cycle_middle_price,
        "current_close_1m": current_close_1m,
        "bars_ago_min": bars_ago_min, "bars_ago_max": bars_ago_max,
        "ts_min": ts_min, "ts_max": ts_max,
        "most_recent_extreme": most_recent_extreme,
        "window_len_1m": window_len_1m,
        "mom_1m": mom_1m,
        "mom_max_val": mom_max_val, "mom_min_val": mom_min_val,
        "mom_max_idx": mom_max_idx, "mom_min_idx": mom_min_idx,
        "mom_bars_ago_max": mom_bars_ago_max, "mom_bars_ago_min": mom_bars_ago_min,
        "bullish_perc": bullish_perc, "bearish_perc": bearish_perc,
        "fft_forecast_price": fft_forecast_price,
        "fft_dominant_freq": fft_dominant_freq,
        "fft_period_bars": fft_period_bars,
        "ml_forecast_price": ml_forecast_price, "ml_current_close": ml_current_close,
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
            f"LEVERAGE: {LEVERAGE}x\nSTRATEGY: 5 Mandatory (Sine+Vol+ML+QuadTrend+QuadRev) + Majority | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}%\n{'='*60}\n\n")
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
    cur = sig["current_close_1m"]
    mid = sig["cycle_middle_price"]
    pos_str = "BELOW" if cur < mid else ("ABOVE" if cur > mid else "AT")

    print(f"  1. Sine (1m):  dMin:{sig['dist_to_min']:.1f}% dMax:{sig['dist_to_max']:.1f}% L:{f['sine_long']} S:{f['sine_short']} [MANDATORY]")
    print(f"  2. Cycle (1m): ArgMin(LOW) {sig['cycle_min_price']:.2f}@{sig['ts_min']}({sig['bars_ago_min']}barsago) "
          f"ArgMax(HIGH) {sig['cycle_max_price']:.2f}@{sig['ts_max']}({sig['bars_ago_max']}barsago) "
          f"| MostRecent:{sig['most_recent_extreme']} | L:{f['cycle_long']} S:{f['cycle_short']}")
    print(f"  3. Mid  (1m):  Low:{sig['cycle_min_price']:.2f} Mid:{mid:.2f} High:{sig['cycle_max_price']:.2f} | "
          f"Now:{cur:.2f} is {pos_str} Mid | L:{f['mid_long']} S:{f['mid_short']}")
    print(f"  4. Mom  (1m):  {sig['mom_1m']:.2f} | HiMOM:{sig['mom_max_val']:.2f}@{sig['mom_bars_ago_max']}barsago "
          f"LoMOM:{sig['mom_min_val']:.2f}@{sig['mom_bars_ago_min']}barsago | L:{f['mom_long']} S:{f['mom_short']}")
    vol_state = "[NEUTRAL - waiting for dominant side]" if sig['bullish_perc'] == sig['bearish_perc'] else "[MANDATORY]"
    print(f"  5. Vol  (1m):  Bull:{sig['bullish_perc']:.1f}% Bear:{sig['bearish_perc']:.1f}% L:{f['vol_long']} S:{f['vol_short']} {vol_state}")
    print(f"  6. FFT  (1m):  DomFreq:{sig['fft_dominant_freq']:.6f}c/bar Period:{sig['fft_period_bars']:.1f}bars "
          f"Fcst:{sig['fft_forecast_price']:.2f} L:{f['fft_long']} S:{f['fft_short']}")
    ml_fc  = sig.get("ml_forecast_price", 0.0)
    ml_cur = sig.get("ml_current_close", 0.0)
    ml_diff = ((ml_fc - ml_cur) / ml_cur * 100) if ml_cur else 0.0
    print(f"  7. ML   (1m):  Cur:{ml_cur:.2f} Fcst:{ml_fc:.2f} ({ml_diff:+.4f}%) L:{f['ml_long']} S:{f['ml_short']} [MANDATORY]")
    if "qstate" in sig:
        print_quadrant_state(sig["qstate"])

    long_true  = sig["long_true_count"]
    short_true = sig["short_true_count"]

    # Mandatory gate summary (5 mandatory: Sine, Vol, ML, QuadTrend, QuadRev)
    mand_long  = sum([f['sine_long'],  f['vol_long'],  f['ml_long'],
                      f['quad_trend_long'],  f['quad_rev_long']])
    mand_short = sum([f['sine_short'], f['vol_short'], f['ml_short'],
                      f['quad_trend_short'], f['quad_rev_short']])

    print(f"  ═══ LONG:{long_true}/9 SHORT:{short_true}/9 "
          f"(Rule: 5 Mandatory + Majority) "
          f"| Mand L:{mand_long}/5 S:{mand_short}/5 "
          f"-> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"HFT KuCoin Bot - 5 MANDATORY + MAJORITY (9 CONDITIONS)")
    print(f"{'='*60}")
    print(f"Symbol:              {TRADE_SYMBOL}")
    print(f"Leverage:            {LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_ROE}% ROE | SL: {STOP_LOSS_ROE}% ROE")
    print(f"Loop Sleep:          {LOOP_SLEEP}s")
    print(f"Entry Logic:         5 Mandatory (Sine+Vol+ML+QuadTrend+QuadRev) + Majority of 9 on same side")
    print(f"Conditions:          1.Sine* 2.Cycle 3.Mid 4.Mom 5.Vol* 6.FFT 7.ML* 8.QuadTrend* 9.QuadRev*")
    print(f"Position Sizing:     {TRADE_BALANCE_PCT*100:.0f}% of balance per trade")
    print(f"Sine Timeframe:      1m")
    print(f"Mom Logic:           Recency-based (LoMOM recent=LONG, HiMOM recent=SHORT)")
    print(f"FFT Logic:           Detrended spectral 1-bar-ahead forecast")
    print(f"QuadTrend:           Up-cycle Q1→Q4 = LONG | Down-cycle Q4→Q1 = SHORT  [always 1T/1F]")
    print(f"QuadRev:             Q1 confirmed = LONG | Q4 confirmed = SHORT         [always 1T/1F]")
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
                    trade_result = {"mode": "SIMULATION", "entry_time": sim_entry_time, "exit_time": now_str,
                                    "type": sim_side, "entry_price": sim_entry_price, "exit_price": current_price,
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

            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print("  [recovery] Orphaned live state - clearing.")
                clear_trade_state()
                saved_state = None

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
                    print(f"\n  *** 5 MANDATORY + MAJORITY -> LONG ***")
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing LONG...")
                        if execute_entry(TRADE_SYMBOL, "buy", balance, sig["price"]):
                            sl_price, tp_price = calculate_sl_tp(sig["price"], "long")
                            saved_state = {"active": True, "mode": "LIVE", "side": "long", 
                                           "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIM] LONG (low balance)")
                        sl_price, tp_price = calculate_sl_tp(sig["price"], "long")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "long", 
                                       "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                        save_trade_state(saved_state)

                elif sig["is_short"]:
                    print(f"\n  *** 5 MANDATORY + MAJORITY -> SHORT ***")
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing SHORT...")
                        if execute_entry(TRADE_SYMBOL, "sell", balance, sig["price"]):
                            sl_price, tp_price = calculate_sl_tp(sig["price"], "short")
                            saved_state = {"active": True, "mode": "LIVE", "side": "short", 
                                           "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIM] SHORT (low balance)")
                        sl_price, tp_price = calculate_sl_tp(sig["price"], "short")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "short", 
                                       "entry_price": sig["price"], "sl": sl_price, "tp": tp_price, "entry_time": now_str}
                        save_trade_state(saved_state)

            loop_ms = (time.perf_counter() - loop_start) * 1000
            log_timing("total_loop", loop_ms)
            
            if time.time() - last_timing_print > 300:
                print_timing_summary()
                last_timing_print = time.time()
            
            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n[Bot] Shutting down safely.")
            print_timing_summary()
            fetcher.shutdown()
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()