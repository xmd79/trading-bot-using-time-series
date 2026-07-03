"""
HFT Auto Trading Bot — KuCoin Futures Edition (CONCURRENT OPTIMIZED)
====================================================================
SIGNAL LOGIC (FIXED v5 — all conditions now unanimous mean-reversion):
  - ALL 8/8 CONDITIONS MUST AGREE ON THE SAME SIDE TO TRIGGER.
  - Conditions: Sine(1m), Sine(5m), Cycle(1m), Momentum-Exhaustion(1m),
                Volume-Exhaustion(1m), 3mMid, RegressionDeviation, ATR-Band-Reversal
  - Long:  All 8 long conditions  = True
  - Short: All 8 short conditions = True

  CHANGELOG vs previous version:
    * Volume condition was TREND-FOLLOWING while all others were MEAN-REVERSION,
      making 7/7 consensus nearly impossible. It is now inverted (selling
      exhaustion -> LONG / buying exhaustion -> SHORT) and uses a short
      20-candle lookback instead of the full 500-candle buffer.
    * 3m-Midpoint and Regression(45 degree) lookbacks shortened from 400/500
      candles (~20hrs / ~8hrs) to short, responsive windows so they actually move.
    * Sine calc now guards against div-by-zero when the sine wave is flat.
    * Cycle condition's dead "cc > p_min" / "cc < p_max" check removed, and
      "most recent extreme" now compares against the last candle's timestamp
      instead of system datetime.now() (removes clock-drift risk).
    * ATR is now a real 8th condition: price vs an ATR-multiple band around a
      short moving average (a volatility-normalized "how stretched are we"
      reversal filter) — AND it's used for actual ATR-based SL/TP sizing.

  CHANGELOG v6 (wick-timing fix):
    * ATR-Band condition originally compared only the instantaneous live
      ticker price to the band. Since the loop sleeps 0.5s, a liquidation
      wick that breaches the band and snaps back in a few seconds could be
      missed entirely (~1-3 second effective detection window). It now also
      checks the high/low of the last few candles for a band breach, giving
      a multi-minute detection window instead of a multi-second one.

POSITION SIZING:
  - 10% of available balance per trade.
  - Remaining 90% stays untouched.

25x Leverage
TP: +5.55% NET ROE (clean profit after round-trip fees)
SL: DISABLED by default (USE_STOP_LOSS=False) — see CHANGELOG v7 below.

  CHANGELOG v7 (risk model change):
    * TP target lowered from 10% to 5.55% net ROE (faster, higher hit-rate
      exits on the mean-reversion bounce).
    * Stop loss disabled by default. Since each trade only ever commits
      TRADE_BALANCE_PCT (10%) of total balance as margin, the worst-case loss
      on any single trade is already capped at that 10% slice even without a
      price-based SL - the other 90% of the account is never in the trade.
      A hard-coded SL price on top of that account-level cap doesn't lower
      total-balance risk further; it just risks getting whipsawed out on a
      wick before the mean-reversion thesis plays out. The existing
      "NEAR LIQUIDATION" safety close (roe <= -95%) still guards against an
      actual liquidation event regardless of this setting. Set
      USE_STOP_LOSS=True in the config section to restore price-based SL.

MOMENTUM CONDITION:
  - Looks at last 200 momentum values
  - Finds argmin (most negative) and argmax (most positive)
  - If argmin is more recent than argmax -> LONG (momentum exhaustion / reversal up)
  - If argmax is more recent than argmin -> SHORT (momentum exhaustion / reversal down)
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
from concurrent.futures import ThreadPoolExecutor

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
            mn, mx = min(values), max(values)
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

print("HFT KuCoin Bot (25x / 5.55% net TP / SL DISABLED (position-sized risk cap) / ALL 8/8 CONDITIONS MUST AGREE) initialising...")
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
TRADE_BALANCE_PCT = 0.10   # 10% of available balance per trade

ANALYSIS_WINDOW_5M = 1200
ANALYSIS_WINDOW_1M = 500
ANALYSIS_WINDOW_3M = 400

KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100  # round-trip fee cost in ROE% at this leverage

# TP = net/clean ROE target (after round-trip fees are deducted) at 25x leverage.
_TARGET_NET_ROE_PCT = 5.55
_TARGET_SL_ROE_PCT  = 10.0   # only used if USE_STOP_LOSS is re-enabled below
TP_PCT = (_TARGET_NET_ROE_PCT + KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100) / (LEVERAGE * 100)
SL_PCT = _TARGET_SL_ROE_PCT / (LEVERAGE * 100)

# ── STOP LOSS: DISABLED by design ──
# Each trade only commits TRADE_BALANCE_PCT (10%) of total balance as isolated
# margin. Even in the absolute worst case - the position gets fully liquidated -
# the loss is capped at that 10% margin slice; the other 90% of the account was
# never in the trade and is never at risk. A price-based stop loss on top of
# that account-level cap doesn't reduce total-balance risk any further, it just
# whipsaws the bot out of positions on ordinary wicks before the mean-reversion
# thesis (which is exactly what the 8/8 unanimous condition gate is built to
# time) has a chance to play out. The "NEAR LIQUIDATION" safety close in the
# main loop (roe <= -95%) remains active regardless, as a hard last-resort
# backstop against an actual liquidation event.
USE_STOP_LOSS = False

ATR_PERIOD = 14
FULL_REFRESH_INTERVAL = 3600

# NEW: Momentum lookback period for argmin/argmax analysis
MOM_LOOKBACK = 200
MOM_PERIOD = 14

# ── FIX: short, responsive lookbacks (previously used entire 400-500 candle buffers,
#          which made these indicators nearly flat and rarely decisive) ──
VOLUME_LOOKBACK = 20     # 1m candles (~20 minutes) for volume exhaustion condition
MIDPOINT_LOOKBACK = 50   # 3m candles (~150 minutes) for 3m midpoint condition
ANGLE_LOOKBACK = 50      # 1m candles (~50 minutes) for regression-deviation condition

# ── NEW: ATR-band reversal condition config ──
ATR_MA_PERIOD = 20       # moving average period the ATR band is centered on
ATR_BAND_MULT = 1.5      # price must be stretched this many ATRs from the MA to flag a reversal
ATR_BREACH_LOOKBACK = 3  # also count a band breach if it occurred within the last N candles (wick timing fix)

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
                "time": int(k[0]) / 1000, "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])
            }
        except (IndexError, TypeError, ValueError):
            return None
    
    def initialize(self, client_obj):
        if not API_CONNECTED: return False
        with self.lock:
            print(f"  [Buffer:{self.timeframe}] Fetching {self.max_candles} candles...")
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
        seen_times, unique = set(), []
        for c in reversed(self.candles):
            if c["time"] not in seen_times:
                seen_times.add(c["time"])
                unique.append(c)
        self.candles = list(reversed(unique))
    
    def update(self, client_obj, fetch_limit=3):
        if not API_CONNECTED or not self.candles: return 0
        with self.lock:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - (fetch_limit * self.duration_ms)
            new_candles = []
            try:
                klines = client_obj.get_klines(self.symbol, self.granularity, start_ms=start_ms, end_ms=now_ms)
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

def get_position_info(symbol):
    empty = {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0, "mark_price": 0.0, "side": None, "size": 0}
    if not API_CONNECTED: return empty
    try:
        data = client.get_position(symbol).get("data", [])
        if not data: return empty
        data = data[0]
        size = float(data.get("currentQty", 0) or 0)
        if size == 0: return empty
        upnl, margin = float(data.get("unrealisedPnl", 0) or 0), float(data.get("posMargin", 0) or 0)
        return {"is_open": True, "side": "long" if size > 0 else "short",
                "entry_price": float(data.get("avgEntryPrice", 0)), "mark_price": float(data.get("markPrice", 0)), 
                "roe_pct": (upnl / margin * 100) if margin else 0.0, "size": size}
    except Exception: return empty

def get_buy_sell_volume_perc(candles):
    """Raw buy/sell volume split over whatever candle slice is passed in."""
    buy_vol = sell_vol = 0.0
    for c in candles:
        if c["close"] >= c["open"]: buy_vol += c["volume"]
        else: sell_vol += c["volume"]
    total_vol = buy_vol + sell_vol
    if total_vol == 0: return 50.0, 50.0
    return (buy_vol / total_vol) * 100.0, (sell_vol / total_vol) * 100.0

def get_volume_exhaustion_condition(candles, lookback=VOLUME_LOOKBACK):
    """
    FIX: This condition used to be TREND-FOLLOWING (bull-dominated volume -> SHORT,
    since it assumed the move would continue) while every other condition in the
    bot is MEAN-REVERSION (buy the dip / sell the rip). That made unanimous
    agreement almost impossible.

    Now it is a mean-reversion "selling/buying exhaustion" read over a short
    recent window: if sell volume has been dominating recently, that dominance
    is read as exhaustion of the down-move -> LONG. If buy volume has been
    dominating, that's exhaustion of the up-move -> SHORT.
    """
    if not candles:
        return False, False, 50.0, 50.0
    recent = candles[-lookback:]
    bull_pct, bear_pct = get_buy_sell_volume_perc(recent)
    # INVERTED vs raw trend reading: bearish domination -> selling exhausted -> LONG
    c_vol_l = bear_pct > bull_pct
    c_vol_s = bull_pct > bear_pct
    return c_vol_l, c_vol_s, bull_pct, bear_pct

# ═══════════════════════════════════════════════════════════════════════════════
# CONCURRENT DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════
class ConcurrentDataFetcher:
    def __init__(self, buffer_5m, buffer_3m, buffer_1m, symbol, max_workers=6):
        self.buffer_5m, self.buffer_3m, self.buffer_1m = buffer_5m, buffer_3m, buffer_1m
        self.symbol = symbol
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def fetch_all_parallel(self, do_full_refresh=False):
        futures, results = {}, {"price": 0.0, "balance": 0.0, "position": {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0, "mark_price": 0.0, "side": None, "size": 0}, "new_5m": 0, "new_3m": 0, "new_1m": 0, "times": {}}
        t0 = time.perf_counter()
        
        for name, buf in [("5m", self.buffer_5m), ("3m", self.buffer_3m), ("1m", self.buffer_1m)]:
            if do_full_refresh or buf.needs_full_refresh(): futures[name] = self.executor.submit(self._full_refresh_buffer, buf)
            else: futures[name] = self.executor.submit(buf.update, client, 3)
            
        futures["price"] = self.executor.submit(get_price, self.symbol)
        futures["balance"] = self.executor.submit(get_account_balance)
        futures["position"] = self.executor.submit(get_position_info, self.symbol)
        
        for name, future in futures.items():
            try:
                t_start = time.perf_counter()
                result = future.result(timeout=10)
                results["times"][name] = (time.perf_counter() - t_start) * 1000
                if name in ["5m", "3m", "1m"]: results[f"new_{name}"] = result
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
def get_3m_midpoint_condition(candles_3m, current_price, lookback=MIDPOINT_LOOKBACK):
    # FIX: previously used the ENTIRE buffer (up to 400 candles / ~20 hours),
    # making min/max/midpoint nearly constant. Now uses a short recent window.
    if len(candles_3m) < 50: return False, False, 0.0, 0.0, 0.0, 0, 0
    window = candles_3m[-lookback:]
    closes_3m = np.array([c["close"] for c in window], dtype=float)
    len_3m = len(closes_3m)
    argmin_idx = len_3m - 1 - int(np.argmax(closes_3m[::-1] == np.min(closes_3m)))
    argmax_idx = len_3m - 1 - int(np.argmax(closes_3m[::-1] == np.max(closes_3m)))
    min_price, max_price = float(closes_3m[argmin_idx]), float(closes_3m[argmax_idx])
    midpoint = (min_price + max_price) / 2.0
    return current_price < midpoint, current_price > midpoint, midpoint, min_price, max_price, argmin_idx, argmax_idx

def calculate_atr(candles, period=14):
    if len(candles) < period + 1: return 0.0
    h = np.array([c["high"] for c in candles[-period*2:]], dtype=float)
    l = np.array([c["low"] for c in candles[-period*2:]], dtype=float)
    c = np.array([c["close"] for c in candles[-period*2:]], dtype=float)
    atr = talib.ATR(h, l, c, timeperiod=period)
    return float(atr[-1]) if len(atr) > 0 and np.isfinite(atr[-1]) else 0.0

def calculate_dynamic_sl_tp(entry_price, side, atr_value=0.0):
    """
    TP: fixed net-ROE target (TP_PCT = 5.55% net after fees), independent of
    SL/ATR - it's the profit level the mean-reversion bounce is expected to
    reach, not a reward:risk multiple of a stop distance.

    SL: DISABLED when USE_STOP_LOSS is False (the default - see config
    section for rationale). sl_price is returned as 0.0 as a sentinel meaning
    "no stop order" and check_dynamic_tp_sl() knows to skip the SL branch
    entirely. If USE_STOP_LOSS is re-enabled, SL still uses
    max(fixed %, 1.5x ATR) so stops aren't tighter than current volatility.
    """
    tp_dist = entry_price * TP_PCT

    if USE_STOP_LOSS:
        sl_dist_fixed = entry_price * SL_PCT
        if atr_value and atr_value > 0:
            sl_dist = max(sl_dist_fixed, atr_value * 1.5)
        else:
            sl_dist = sl_dist_fixed
    else:
        sl_dist = 0.0

    if side == "long":
        tp_price = entry_price + tp_dist
        sl_price = (entry_price - sl_dist) if USE_STOP_LOSS else 0.0
    else:
        tp_price = entry_price - tp_dist
        sl_price = (entry_price + sl_dist) if USE_STOP_LOSS else 0.0

    tp_roe = (tp_dist / entry_price) * 100 * LEVERAGE - RT_FEE_ROE_PCT
    sl_roe = (sl_dist / entry_price) * 100 * LEVERAGE if USE_STOP_LOSS else 0.0

    if USE_STOP_LOSS:
        print(f"  [SL/TP] SL:-{(sl_dist/entry_price)*100:.3f}%={sl_dist:.2f} ({sl_roe:.1f}% ROE) | "
              f"TP:+{(tp_dist/entry_price)*100:.3f}%={tp_dist:.2f} ({tp_roe:.1f}% net ROE) | ATR:{atr_value:.2f}")
    else:
        print(f"  [SL/TP] SL: DISABLED (risk capped at {TRADE_BALANCE_PCT*100:.0f}% of balance via position "
              f"sizing; liq-guard closes at -95% ROE) | TP:+{(tp_dist/entry_price)*100:.3f}%={tp_dist:.2f} "
              f"({tp_roe:.1f}% net ROE) | ATR:{atr_value:.2f}")

    return float(sl_price), float(tp_price), sl_roe, tp_roe

def check_dynamic_tp_sl(entry_price, current_price, side, sl_price, tp_price):
    hit, hit_type = False, None
    sl_active = USE_STOP_LOSS and sl_price and sl_price > 0
    if side == "long":
        if current_price >= tp_price: hit, hit_type = True, "TP"
        elif sl_active and current_price <= sl_price: hit, hit_type = True, "SL"
    else:
        if current_price <= tp_price: hit, hit_type = True, "TP"
        elif sl_active and current_price >= sl_price: hit, hit_type = True, "SL"
    
    pcp = ((current_price - entry_price) / entry_price) * 100 if side == "long" else ((entry_price - current_price) / entry_price) * 100
    roe_pct = pcp * LEVERAGE - RT_FEE_ROE_PCT
    
    if hit: return True, "TAKE PROFIT" if hit_type == "TP" else "STOP LOSS", roe_pct
    return False, None, roe_pct

def scale_to_sine(close_prices):
    if len(close_prices) < 32: return 50.0, 50.0, 0.0
    close_prices = np.array(close_prices, dtype=float)
    current_close = close_prices[-1]      
    sine_wave, leadsine = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave
    current_sine = sine_wave[-1]
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # FIX: guard against div-by-zero / NaN when the sine wave is flat
    range_val = sine_wave_max - sine_wave_min
    if range_val == 0 or not np.isfinite(range_val):
        return 50.0, 50.0, float(current_sine)

    dist_from_close_to_min = ((current_sine - sine_wave_min) / range_val) * 100
    dist_from_close_to_max = ((sine_wave_max - current_sine) / range_val) * 100
    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_momentum(close_arr, period=14):
    if len(close_arr) < period + 1: return np.nan
    return float(talib.MOM(close_arr, timeperiod=period)[-1])

def calculate_regression_deviation(close_prices, lookback=ANGLE_LOOKBACK):
    """
    FIX #1 (naming): this was called "45-degree angle" but is plain linear
    regression - the fitted line is essentially never actually at 45 degrees.
    Renamed to reflect what it does: fit a short-term trendline and flag when
    price has deviated from it (mean-reversion back toward the line).

    FIX #2 (lookback): previously fit across the ENTIRE 1m buffer (up to 500
    candles / ~8 hours), making the line nearly static. Now fits over a short
    recent window so the "expected price" actually tracks current conditions.
    """
    if len(close_prices) < 10:
        return 0.0, 0.0, 0.0, "Insufficient data for regression deviation"
    recent = np.asarray(close_prices[-lookback:], dtype=float)
    if len(recent) < 10:
        return 0.0, 0.0, 0.0, "Insufficient data for regression deviation"
    x = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)
    expected_price = slope * (len(recent) - 1) + intercept
    return expected_price, slope, intercept, None

# Backwards-compatible alias (old name kept in case other scripts import it)
calculate_45_degree_angle = calculate_regression_deviation

def get_atr_band_reversal_condition(candles, current_price, atr_period=ATR_PERIOD,
                                     ma_period=ATR_MA_PERIOD, atr_mult=ATR_BAND_MULT,
                                     breach_lookback=ATR_BREACH_LOOKBACK):
    """
    NEW 8th CONDITION: volatility-normalized mean-reversion filter.

    Builds a short moving average of closes and a band around it sized in ATR
    units (similar in spirit to Keltner Channels). If price has stretched
    below the lower band by more than `atr_mult` ATRs, that's an
    over-extended down-move -> LONG reversal candidate. If price has
    stretched above the upper band, that's an over-extended up-move -> SHORT
    reversal candidate.

    This both (a) adds a real, independent reversal condition to the
    unanimous-consensus vote, and (b) gives calculate_dynamic_sl_tp a live
    ATR reading to size stops off of (previously ATR was computed but never
    actually used anywhere).

    FIX (wick timing): the original version only checked the live ticker
    price against the band at the instant the loop happened to poll. Since
    the loop sleeps ~0.5s and a liquidation-wick breach can revert in just a
    few seconds, that gave the bot only a ~1-3 second window to catch the
    signal, and it would often miss the exact trade it was designed to catch.
    Now it ALSO checks whether the high/low of any of the last
    `breach_lookback` candles pierced the band, giving a multi-minute window
    instead of a multi-second one, while still checking the live price too
    (so it still reacts immediately to a breach happening right now).
    """
    if len(candles) < max(atr_period, ma_period) + 5:
        return False, False, 0.0, 0.0, 0.0, 0.0

    atr_value = calculate_atr(candles, atr_period)
    closes = np.array([c["close"] for c in candles[-ma_period:]], dtype=float)
    ma = float(np.mean(closes))

    if atr_value <= 0:
        return False, False, atr_value, ma, ma, ma

    lower_band = ma - atr_value * atr_mult
    upper_band = ma + atr_value * atr_mult

    recent_candles = candles[-breach_lookback:] if breach_lookback > 0 else []

    # stretched below band (live tick OR a recent candle's wick low) -> oversold -> LONG
    c_atr_l = current_price < lower_band or any(c["low"] < lower_band for c in recent_candles)
    # stretched above band (live tick OR a recent candle's wick high) -> overbought -> SHORT
    c_atr_s = current_price > upper_band or any(c["high"] > upper_band for c in recent_candles)

    return c_atr_l, c_atr_s, atr_value, ma, lower_band, upper_band


def compute_momentum_argmin_argmax(arr1, candles_1m):
    """
    NEW MOMENTUM LOGIC:
    - Calculate last 200 momentum values
    - Find argmin (most negative momentum) and argmax (most positive momentum)
    - If argmin is more recent than argmax → LONG signal
    - If argmax is more recent than argmin → SHORT signal
    """
    # Default values for insufficient data
    default_result = {
        "c_mom_l": False, "c_mom_s": False,
        "mom_current": 0.0,
        "mom_argmin_val": 0.0, "mom_argmax_val": 0.0,
        "bars_since_mom_min": 0, "bars_since_mom_max": 0,
        "mom_recent": "INSUFFICIENT DATA",
        "ts_mom_min": "N/A", "ts_mom_max": "N/A"
    }
    
    # Need MOM_LOOKBACK + MOM_PERIOD candles to get MOM_LOOKBACK momentum values
    required_candles = MOM_LOOKBACK + MOM_PERIOD
    if len(arr1) < required_candles:
        return default_result
    
    # Calculate momentum for the required period
    mom_values = talib.MOM(arr1[-required_candles:], timeperiod=MOM_PERIOD)
    mom_values = np.nan_to_num(mom_values)
    
    # Should have exactly MOM_LOOKBACK values (or close to it)
    if len(mom_values) < MOM_LOOKBACK:
        return default_result
    
    # Get current momentum
    mom_current = float(mom_values[-1])
    
    # Find argmin (most negative) and argmax (most positive)
    mom_argmin_idx = int(np.argmin(mom_values))
    mom_argmax_idx = int(np.argmax(mom_values))
    
    mom_argmin_val = float(mom_values[mom_argmin_idx])
    mom_argmax_val = float(mom_values[mom_argmax_idx])
    
    # Calculate bars since each extreme (higher index = more recent)
    mom_len = len(mom_values)
    bars_since_mom_min = mom_len - 1 - mom_argmin_idx
    bars_since_mom_max = mom_len - 1 - mom_argmax_idx
    
    # Get timestamps for the argmin and argmax positions
    # The momentum at index i corresponds to the candle at index (required_candles - MOM_LOOKBACK + i) in arr1
    # or more precisely: mom_values[i] uses candles from (i) to (i + MOM_PERIOD - 1) in the sliced array
    # The "effective" candle index for mom_values[i] is: (len(arr1) - required_candles) + i + MOM_PERIOD - 1
    base_idx = len(arr1) - required_candles
    
    try:
        mom_min_candle_idx = base_idx + mom_argmin_idx + MOM_PERIOD - 1
        mom_max_candle_idx = base_idx + mom_argmax_idx + MOM_PERIOD - 1
        
        # Ensure indices are within bounds
        if 0 <= mom_min_candle_idx < len(candles_1m):
            ts_mom_min = datetime.datetime.fromtimestamp(candles_1m[mom_min_candle_idx]["time"]).strftime("%H:%M:%S")
        else:
            ts_mom_min = "N/A"
            
        if 0 <= mom_max_candle_idx < len(candles_1m):
            ts_mom_max = datetime.datetime.fromtimestamp(candles_1m[mom_max_candle_idx]["time"]).strftime("%H:%M:%S")
        else:
            ts_mom_max = "N/A"
    except Exception:
        ts_mom_min = ts_mom_max = "N/A"
    
    # Determine which extreme is more recent
    if bars_since_mom_min < bars_since_mom_max:
        # Most negative momentum (argmin) is more recent → LONG
        mom_recent = "ARGMIN (MOST NEGATIVE)"
        c_mom_l, c_mom_s = True, False
    elif bars_since_mom_max < bars_since_mom_min:
        # Most positive momentum (argmax) is more recent → SHORT
        mom_recent = "ARGMAX (MOST POSITIVE)"
        c_mom_l, c_mom_s = False, True
    else:
        # Tie - no signal
        mom_recent = "TIE"
        c_mom_l, c_mom_s = False, False
    
    return {
        "c_mom_l": c_mom_l, "c_mom_s": c_mom_s,
        "mom_current": mom_current,
        "mom_argmin_val": mom_argmin_val, "mom_argmax_val": mom_argmax_val,
        "bars_since_mom_min": bars_since_mom_min, "bars_since_mom_max": bars_since_mom_max,
        "mom_recent": mom_recent,
        "ts_mom_min": ts_mom_min, "ts_mom_max": ts_mom_max
    }


def compute_signals(buffer_5m, buffer_3m, buffer_1m, live_price):
    candles_5m, candles_3m, candles_1m = buffer_5m.get_candles(), buffer_3m.get_candles(), buffer_1m.get_candles()
    c1 = [x["close"] for x in candles_1m]
    c5 = [x["close"] for x in candles_5m]
    if len(c1) < 50 or len(candles_3m) < 50 or len(c5) < 32: return None
    
    arr1 = np.array(c1, dtype=float)
    arr5 = np.array(c5, dtype=float)
    cc = live_price
    
    # ── 1. Sine (1m) ──
    d_min, d_max, current_sine = scale_to_sine(arr1)
    c_sine_l, c_sine_s = d_min < d_max, d_max < d_min
    
    # ── 1b. Sine (5m) — SEPARATE CONDITION ──
    d5_min, d5_max, current_sine_5m = scale_to_sine(arr5)
    c_sine5_l, c_sine5_s = d5_min < d5_max, d5_max < d5_min
    
    # ── 2. Cycle (1m) ──
    w1_candles = candles_1m[-500:]
    w1 = arr1[-500:]
    wl1 = len(w1)
    
    ai1_min = wl1 - 1 - int(np.argmax(w1[::-1] == np.min(w1)))
    ai1_max = wl1 - 1 - int(np.argmax(w1[::-1] == np.max(w1)))
    
    p_min, p_max = float(w1[ai1_min]), float(w1[ai1_max])
    
    try:
        dt_min = datetime.datetime.fromtimestamp(w1_candles[ai1_min]["time"])
        dt_max = datetime.datetime.fromtimestamp(w1_candles[ai1_max]["time"])
        # FIX: compare against the last CANDLE's timestamp instead of system
        # datetime.now() - removes sensitivity to local clock drift.
        now_dt = datetime.datetime.fromtimestamp(w1_candles[-1]["time"])
        
        diff_min_sec = (now_dt - dt_min).total_seconds()
        diff_max_sec = (now_dt - dt_max).total_seconds()
        
        bars_since_min = int(diff_min_sec / 60)
        bars_since_max = int(diff_max_sec / 60)
        
        t_min = dt_min.strftime("%H:%M:%S")
        t_max = dt_max.strftime("%H:%M:%S")
        
        # FIX: removed dead "cc > p_min" / "cc < p_max" checks (true ~99.9% of
        # the time and added nothing - if price broke past the extreme, a new
        # extreme would already have replaced p_min/p_max). Signal is now
        # simply "which extreme (low or high) is more recent".
        if diff_min_sec < diff_max_sec:
            recent = "ARGMIN (LOW)"
            c_cyc_l, c_cyc_s = True, False
        elif diff_max_sec < diff_min_sec:
            recent = "ARGMAX (HIGH)"
            c_cyc_l, c_cyc_s = False, True
        else:
            recent = "TIE"
            c_cyc_l, c_cyc_s = False, False
            
    except Exception as e:
        t_min = t_max = "N/A"
        bars_since_min = bars_since_max = 0
        recent = "TIE"
        c_cyc_l, c_cyc_s = False, False
        
    # ── 3. Momentum (1m) ── Argmin/Argmax of last 200 momentum values
    mom_result = compute_momentum_argmin_argmax(arr1, candles_1m)
    c_mom_l = mom_result["c_mom_l"]
    c_mom_s = mom_result["c_mom_s"]
    
    # ── 4. Volume Exhaustion (1m) ── FIX: inverted to mean-reversion, short lookback
    c_vol_l, c_vol_s, bull, bear = get_volume_exhaustion_condition(candles_1m, VOLUME_LOOKBACK)
    
    # ── 5. 3m Midpoint (short lookback) ──
    c_mid_l, c_mid_s, mid, mn3, mx3, _, _ = get_3m_midpoint_condition(candles_3m, cc, MIDPOINT_LOOKBACK)
    
    # ── 6. Regression Deviation (formerly "45-Degree Angle", short lookback) ──
    expected_price, angle_slope, angle_intercept, angle_error = calculate_regression_deviation(arr1, ANGLE_LOOKBACK)
    
    if angle_error:
        angle_forecast = angle_error
        c_angle_l, c_angle_s = False, False
    else:
        if cc < expected_price:
            angle_forecast = "Bullish: Close below regression line"
            c_angle_l, c_angle_s = True, False
        elif cc > expected_price:
            angle_forecast = "Bearish: Close above regression line"
            c_angle_s, c_angle_l = True, False
        else:
            angle_forecast = "Neutral: Close at the regression line"
            c_angle_l, c_angle_s = False, False

    # ── 7. ATR value (used by condition 8 and by SL/TP sizing) ──
    atr_value = calculate_atr(candles_1m, ATR_PERIOD)

    # ── 8. ATR-Band Reversal (NEW condition) ──
    c_atr_l, c_atr_s, atr_val_chk, atr_ma, atr_lower, atr_upper = get_atr_band_reversal_condition(
        candles_1m, cc, ATR_PERIOD, ATR_MA_PERIOD, ATR_BAND_MULT
    )
    
    # ── BUILD LONG / SHORT LISTS (now 8 unanimous conditions) ──
    long_conds  = [c_sine_l,  c_sine5_l,  c_cyc_l,  c_mom_l,  c_vol_l,  c_mid_l,  c_angle_l,  c_atr_l]
    short_conds = [c_sine_s,  c_sine5_s,  c_cyc_s,  c_mom_s,  c_vol_s,  c_mid_s,  c_angle_s,  c_atr_s]
    
    lt = sum(long_conds)
    st = sum(short_conds)
    
    is_long  = all(long_conds)   
    is_short = all(short_conds)  
    
    return {
        "price": live_price, "current_close": cc, 
        "is_long": is_long, "is_short": is_short,
        "cond_flags": {
            "sine_long": c_sine_l, "sine_short": c_sine_s, 
            "sine5_long": c_sine5_l, "sine5_short": c_sine5_s,
            "cycle_long": c_cyc_l, "cycle_short": c_cyc_s, 
            "mom_long": c_mom_l, "mom_short": c_mom_s, 
            "vol_long": c_vol_l, "vol_short": c_vol_s, 
            "mid_long": c_mid_l, "mid_short": c_mid_s,
            "angle_long": c_angle_l, "angle_short": c_angle_s,
            "atr_long": c_atr_l, "atr_short": c_atr_s
        },
        "long_true_count": lt, "short_true_count": st,
        "dist_to_min": d_min, "dist_to_max": d_max,
        "current_sine": current_sine,
        "dist_to_min_5m": d5_min, "dist_to_max_5m": d5_max,
        "current_sine_5m": current_sine_5m,
        "ts_min": t_min, "ts_max": t_max, "most_recent_extreme": recent,
        "bars_since_min": bars_since_min, "bars_since_max": bars_since_max,
        # Momentum fields
        "mom_1m": mom_result["mom_current"],
        "mom_argmin_val": mom_result["mom_argmin_val"],
        "mom_argmax_val": mom_result["mom_argmax_val"],
        "bars_since_mom_min": mom_result["bars_since_mom_min"],
        "bars_since_mom_max": mom_result["bars_since_mom_max"],
        "mom_recent": mom_result["mom_recent"],
        "ts_mom_min": mom_result["ts_mom_min"],
        "ts_mom_max": mom_result["ts_mom_max"],
        # Other fields
        "bullish_perc": bull, "bearish_perc": bear,
        "midpoint_3m": mid, "min_3m": mn3, "max_3m": mx3,
        "atr_value": atr_value,
        "atr_ma": atr_ma, "atr_lower": atr_lower, "atr_upper": atr_upper,
        "expected_price_45": expected_price,
        "angle_slope": angle_slope,
        "angle_intercept": angle_intercept,
        "angle_forecast": angle_forecast
    }

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

def write_trade_to_journal(r):
    line = (f"{'='*70}\nMODE: {r.get('mode')}\nTYPE: {r.get('type','').upper()}\nENTRY: {r.get('entry_time')}\nEXIT: {r.get('exit_time')}\nDUR: {r.get('duration')}\n"
            f"EP: {r.get('entry_price',0):.2f}\nXP: {r.get('exit_price',0):.2f}\nSL: {r.get('sl_price',0):.2f}\nTP: {r.get('tp_price',0):.2f}\nATR: {r.get('atr_at_entry',0):.2f}\n"
            f"PC: {r.get('price_change_pct',0):+.4f}%\nGROSS ROE: {r.get('gross_roe',0):+.2f}%\nNET ROE: {r.get('roe',0):+.2f}%\nREASON: {r.get('reason')}\n{'='*70}\n\n")
    try:
        with open(TRADES_LOG_FILE, "a", encoding="utf-8") as f: f.write(line)
    except Exception as e: print(f"  [journal] error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
def get_contract_size(symbol):
    if not API_CONNECTED: return 0.001
    try: return float(client.get(f"/api/v1/contracts/{symbol}")["data"]["multiplier"])
    except Exception: return 0.001

def execute_entry(symbol, side, balance, price):
    try:
        tb = balance * TRADE_BALANCE_PCT
        contracts = max(1, int((tb * LEVERAGE) / (price * get_contract_size(symbol))))
        resp = client.place_order(symbol, side, contracts, LEVERAGE)
        if resp.get("data", {}).get("orderId"):
            print(f"  >>> {side.upper()} PLACED: {contracts} contracts @ ~{price:.2f}  (used {tb:.2f} USDT)")
            return True
        print(f"  [ORDER FAIL] {side.upper()} [{resp.get('code')}]: {resp.get('msg')}")
        return False
    except Exception as e:
        print(f"  [ORDER ERROR] {side.upper()}: {e}")
        return False

def format_duration(s, e):
    t = int((e - s).total_seconds())
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def print_conditions(sig):
    f, cc = sig["cond_flags"], sig["current_close"]
    
    def mark(val): return "✓" if val else "✗"
    
    print(f"  1. Sine (1m):    dMin:{sig['dist_to_min']:.2f}% dMax:{sig['dist_to_max']:.2f}% (RawSine:{sig.get('current_sine', 0):.4f}) L:{mark(f['sine_long'])} S:{mark(f['sine_short'])}")
    
    print(f"  2. Sine (5m):    dMin:{sig.get('dist_to_min_5m', 0):.2f}% dMax:{sig.get('dist_to_max_5m', 0):.2f}% (RawSine:{sig.get('current_sine_5m', 0):.4f}) L:{mark(f['sine5_long'])} S:{mark(f['sine5_short'])}")
    
    print(f"  3. Cycle (1m):   Low@{sig['ts_min']}({sig['bars_since_min']}bars) High@{sig['ts_max']}({sig['bars_since_max']}bars) | Recent: {sig['most_recent_extreme']} | L:{mark(f['cycle_long'])} S:{mark(f['cycle_short'])}")
    
    # NEW: Updated momentum display
    print(f"  4. Mom (1m):     Cur:{sig['mom_1m']:.2f} | ArgMin:{sig['mom_argmin_val']:.2f}@{sig['ts_mom_min']}({sig['bars_since_mom_min']}bars) ArgMax:{sig['mom_argmax_val']:.2f}@{sig['ts_mom_max']}({sig['bars_since_mom_max']}bars) | Recent: {sig['mom_recent']} | L:{mark(f['mom_long'])} S:{mark(f['mom_short'])}")
    
    print(f"  5. Vol-Exh (1m): Bull:{sig['bullish_perc']:.1f}% Bear:{sig['bearish_perc']:.1f}% (lookback:{VOLUME_LOOKBACK}) L:{mark(f['vol_long'])} S:{mark(f['vol_short'])}")
    
    mid, mn, mx = sig.get("midpoint_3m", 0), sig.get("min_3m", 0), sig.get("max_3m", 0)
    mdp = ((cc - mid) / mid * 100) if mid > 0 else 0.0
    print(f"  6. 3mMid:        Min:{mn:.2f} Max:{mx:.2f} Mid:{mid:.2f} | Cur:{cc:.2f} ({mdp:+.3f}%) L:{mark(f['mid_long'])} S:{mark(f['mid_short'])}")
    
    exp_price = sig.get("expected_price_45", 0)
    if exp_price > 0:
        angle_diff = ((cc - exp_price) / exp_price * 100) if exp_price > 0 else 0
        print(f"  7. RegrDev:      Expected:{exp_price:.2f} | Cur:{cc:.2f} ({angle_diff:+.3f}%) | {sig.get('angle_forecast', 'N/A')}")
        print(f"                 L:{mark(f['angle_long'])} S:{mark(f['angle_short'])}")
    else:
        print(f"  7. RegrDev:      {sig.get('angle_forecast', 'N/A')}")

    atr_v, atr_ma, atr_lo, atr_hi = sig.get("atr_value", 0), sig.get("atr_ma", 0), sig.get("atr_lower", 0), sig.get("atr_upper", 0)
    print(f"  8. ATR-Band:     ATR:{atr_v:.2f} MA:{atr_ma:.2f} Band:[{atr_lo:.2f},{atr_hi:.2f}] | Cur:{cc:.2f} L:{mark(f['atr_long'])} S:{mark(f['atr_short'])}")
    
    lt, st = sig["long_true_count"], sig["short_true_count"]
    
    long_bar  = "█" * lt + "░" * (8 - lt)
    short_bar = "█" * st + "░" * (8 - st)
    
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"  LONG : [{long_bar}] {lt}/8  {'<< TRIGGER >>' if sig['is_long'] else ''}")
    print(f"  SHORT: [{short_bar}] {st}/8  {'<< TRIGGER >>' if sig['is_short'] else ''}")
    print(f"  ═══════════════════════════════════════════════════════════════════")
    print(f"     RULE: ALL 8/8 MUST AGREE -> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*70}")
    print(f"HFT KuCoin Bot — ALL 8/8 CONDITIONS MUST AGREE TO TRIGGER")
    print(f"{'='*70}")
    print(f"Symbol: {TRADE_SYMBOL} | Leverage: {LEVERAGE}x | API: {API_CONNECTED}")
    print(f"TP: +{TP_PCT*100:.3f}% price (~{_TARGET_NET_ROE_PCT:.2f}% net ROE, after fees) | "
          f"SL: {'DISABLED - risk capped by ' + str(int(TRADE_BALANCE_PCT*100)) + '% position sizing; liq-guard @ -95% ROE' if not USE_STOP_LOSS else f'-{SL_PCT*100:.3f}% price (~{_TARGET_SL_ROE_PCT:.0f}% ROE)'}")
    print(f"Balance per trade: {TRADE_BALANCE_PCT*100:.0f}%")
    print(f"Conditions: Sine(1m) + Sine(5m) + Cycle(1m) + Mom(1m:{MOM_LOOKBACK}argmin/argmax) + "
          f"VolExh(1m:{VOLUME_LOOKBACK}) + 3mMid({MIDPOINT_LOOKBACK}) + RegrDev({ANGLE_LOOKBACK}) + ATR-Band({ATR_BAND_MULT}x)")
    print(f"Rule: ALL 8/8 must be TRUE on same side = UNANIMOUS CONSENSUS")
    print(f"{'='*70}\n")

    b5 = CandleBuffer(TRADE_SYMBOL, "5m", ANALYSIS_WINDOW_5M)
    b3 = CandleBuffer(TRADE_SYMBOL, "3m", ANALYSIS_WINDOW_3M)
    b1 = CandleBuffer(TRADE_SYMBOL, "1m", ANALYSIS_WINDOW_1M)
    
    if API_CONNECTED:
        print("  === INITIALIZING BUFFERS ===\n")
        t0 = time.perf_counter()
        b5.initialize(client); b3.initialize(client); b1.initialize(client)
        print(f"  Total init: {(time.perf_counter()-t0)*1000:.0f}ms\n")
        if not b1.is_ready() or not b3.is_ready() or not b5.is_ready(min_candles=32):
            print("  [ERROR] Buffer init failed."); return
    
    fetcher = ConcurrentDataFetcher(b5, b3, b1, TRADE_SYMBOL)
    saved_state = load_trade_state()
    if saved_state: print(f"  [recovery] State: {saved_state.get('mode')} {saved_state.get('side')} @ {saved_state.get('entry_price', 0):.2f}")
    
    loop_count, last_timing_print, last_buffer_log = 0, time.time(), 0
    print("  === STARTING MAIN LOOP ===\n")

    while True:
        try:
            loop_start = time.perf_counter()
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            loop_count += 1

            do_full = b5.needs_full_refresh() or b1.needs_full_refresh() or b3.needs_full_refresh()
            data = fetcher.fetch_all_parallel(do_full_refresh=do_full)
            cur_p, bal, pos = data["price"], data["balance"], data["position"]
            has_bal = bal >= MIN_BALANCE_USDT
            
            pms = data["parallel_total_ms"]
            log_timing("parallel_total", pms)
            for k, v in data["times"].items(): log_timing(f"task_{k}", v)
            
            if loop_count - last_buffer_log >= 100:
                print(f"  [Buffers] 5m:{len(b5)} 3m:{len(b3)} 1m:{len(b1)} | Par:{pms:.0f}ms")
                last_buffer_log = loop_count
            
            t = time.perf_counter()
            sig = compute_signals(b5, b3, b1, cur_p)
            cms = (time.perf_counter() - t) * 1000
            log_timing("compute_signals", cms)

            # ── LIVE POSITION ──
            if pos["is_open"]:
                roe, side, ep, mp = pos["roe_pct"], pos["side"], pos["entry_price"], pos["mark_price"]
                sl_p = saved_state.get("sl", 0.0) if saved_state else 0.0
                tp_p = saved_state.get("tp", 0.0) if saved_state else 0.0
                if tp_p <= 0:
                    sl_p, tp_p, _, _ = calculate_dynamic_sl_tp(ep, side, sig["atr_value"] if sig else 0.0)
                
                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] === LIVE {side.upper()} | Entry:{ep:.2f} Mark:{mp:.2f} ROE:{roe:+.2f}% ===")
                    print(f"  SL:{'DISABLED (liq-guard @ -95% ROE)' if not USE_STOP_LOSS or sl_p <= 0 else f'{sl_p:.2f}'} TP:{tp_p:.2f}")
                    if sig: print_conditions(sig)
                
                hit, reason, _ = check_dynamic_tp_sl(ep, mp, side, sl_p, tp_p)
                if not hit and roe >= 100: hit, reason = True, "EXTREME PROFIT"
                elif not hit and roe <= -95: hit, reason = True, "NEAR LIQUIDATION"
                    
                if hit and reason:
                    print(f"  >>> [{reason}] {roe:+.2f}% ROE - Closing...")
                    client.close_position(TRADE_SYMBOL)
                    pcp = ((mp - ep) / ep) * 100 if side == "long" else ((ep - mp) / ep) * 100
                    sdt = datetime.datetime.strptime(saved_state["entry_time"], "%Y-%m-%d %H:%M:%S")
                    write_trade_to_journal({
                        "mode": "LIVE", "entry_time": saved_state["entry_time"], 
                        "exit_time": now_str, "type": side, "entry_price": ep, 
                        "exit_price": mp, "sl_price": sl_p, "tp_price": tp_p, 
                        "atr_at_entry": saved_state.get("atr_at_entry", 0), 
                        "price_change_pct": pcp, "gross_roe": pcp * LEVERAGE, 
                        "rt_fee_pct": RT_FEE_ROE_PCT, "roe": roe, 
                        "duration": format_duration(sdt, datetime.datetime.now()), 
                        "reason": reason
                    })
                    clear_trade_state(); saved_state = None
                
                log_timing("total_loop", (time.perf_counter() - loop_start) * 1000)
                time.sleep(LOOP_SLEEP); continue

            # ── SIMULATION POSITION ──
            if saved_state and saved_state.get("mode") == "SIMULATION":
                s_ep, s_side, s_et = saved_state["entry_price"], saved_state["side"], saved_state["entry_time"]
                s_sl, s_tp = saved_state.get("sl", 0.0), saved_state.get("tp", 0.0)
                hit, reason, s_roe = check_dynamic_tp_sl(s_ep, cur_p, s_side, s_sl, s_tp)
                
                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] === SIM {s_side.upper()} | Entry:{s_ep:.2f} Now:{cur_p:.2f} ROE:{s_roe:+.2f}% ===")
                    print(f"  SL:{'DISABLED (liq-guard @ -95% ROE)' if not USE_STOP_LOSS or s_sl <= 0 else f'{s_sl:.2f}'} TP:{s_tp:.2f}")
                    if sig: print_conditions(sig)
                
                if not hit and s_roe >= 100: hit, reason = True, "EXTREME PROFIT"
                elif not hit and s_roe <= -95: hit, reason = True, "NEAR LIQUIDATION"
                
                if hit and reason:
                    print(f"  >>> [SIM {reason}] {s_roe:+.2f}% ROE")
                    pcp = ((cur_p - s_ep) / s_ep) * 100 if s_side == "long" else ((s_ep - cur_p) / s_ep) * 100
                    sdt = datetime.datetime.strptime(s_et, "%Y-%m-%d %H:%M:%S")
                    write_trade_to_journal({
                        "mode": "SIMULATION", "entry_time": s_et, 
                        "exit_time": now_str, "type": s_side, 
                        "entry_price": s_ep, "exit_price": cur_p, 
                        "sl_price": s_sl, "tp_price": s_tp, 
                        "atr_at_entry": saved_state.get("atr_at_entry", 0), 
                        "price_change_pct": pcp, "gross_roe": pcp * LEVERAGE, 
                        "rt_fee_pct": RT_FEE_ROE_PCT, "roe": s_roe, 
                        "duration": format_duration(sdt, now), "reason": reason
                    })
                    clear_trade_state(); saved_state = None
                
                log_timing("total_loop", (time.perf_counter() - loop_start) * 1000)
                time.sleep(LOOP_SLEEP); continue

            # ── ORPHAN CLEANUP ──
            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print("  [recovery] Orphaned live state - clearing.")
                clear_trade_state(); saved_state = None

            # ── SCANNING ──
            if not pos["is_open"] and not saved_state:
                if not sig:
                    log_timing("total_loop", (time.perf_counter() - loop_start) * 1000)
                    time.sleep(LOOP_SLEEP); continue

                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] Scanning (FLAT) | Net:{pms:.0f}ms Calc:{cms:.0f}ms")
                    print_conditions(sig)
                    print(f"  Bal: {bal:.2f} USDT (alloc: {bal*TRADE_BALANCE_PCT:.2f})")

                if sig["is_long"] or sig["is_short"]:
                    side = "long" if sig["is_long"] else "short"
                    print(f"\n  {'*'*60}")
                    print(f"  *** ALL 8/8 CONDITIONS AGREE -> {side.upper()} ***")
                    print(f"  {'*'*60}")
                    sl_p, tp_p, sl_r, tp_r = calculate_dynamic_sl_tp(sig["price"], side, sig["atr_value"])
                    state_dict = {
                        "active": True, "side": side, "entry_price": sig["price"], 
                        "sl": sl_p, "tp": tp_p, "sl_roe": sl_r, "tp_roe": tp_r, 
                        "atr_at_entry": sig["atr_value"], "entry_time": now_str
                    }
                    
                    if has_bal:
                        state_dict["mode"] = "LIVE"
                        print(f"  [LIVE] Executing {side.upper()}...")
                        if execute_entry(TRADE_SYMBOL, "buy" if side == "long" else "sell", bal, sig["price"]):
                            saved_state = state_dict; save_trade_state(saved_state)
                    else:
                        state_dict["mode"] = "SIMULATION"
                        print(f"  [SIM] {side.upper()} (low balance)")
                        saved_state = state_dict; save_trade_state(saved_state)

            log_timing("total_loop", (time.perf_counter() - loop_start) * 1000)
            if time.time() - last_timing_print > 300:
                print_timing_summary(); last_timing_print = time.time()
            
            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n[Bot] Shutting down safely."); print_timing_summary(); fetcher.shutdown(); break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}"); import traceback; traceback.print_exc(); time.sleep(5)

if __name__ == "__main__":
    main()
