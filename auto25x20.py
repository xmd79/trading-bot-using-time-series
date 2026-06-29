"""
HFT Auto Trading Bot — KuCoin Futures Edition (CONCURRENT OPTIMIZED)
====================================================================
SIGNAL LOGIC:
  - Mandatory: Momentum (1m), Volume (1m), AND 3m Midpoint MUST be true.
  - Flexible: At least 1 out of 2 flexible conditions (Sine, Cycle) must be true.
  - Total: All 3 mandatory + ≥1 flexible = 4/5 Total.

POSITION SIZING:
  - Only 5% of available balance per trade.
  - Remaining 95% stays untouched.

25x Leverage
Fixed TP: +2.5% price move | Fixed SL: -2.5% price move
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

print("HFT KuCoin Bot (25x / 5% net TP / 99% SL / Mom+Vol MANDATORY) initialising...")
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
TRADE_BALANCE_PCT = 0.05

ANALYSIS_WINDOW_5M = 1200
ANALYSIS_WINDOW_1M = 500
ANALYSIS_WINDOW_3M = 400

KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100

# TP: gross ROE needed = target_net + fees = 5% + 3% = 8% -> price move = 8/(25*100)
# SL: 99% ROE loss -> price move = 99/(25*100)
_TARGET_NET_ROE_PCT = 2.0
_TARGET_SL_ROE_PCT  = 99.0
TP_PCT = (_TARGET_NET_ROE_PCT + KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100) / (LEVERAGE * 100)
SL_PCT = _TARGET_SL_ROE_PCT / (LEVERAGE * 100)

ATR_PERIOD = 14  # kept for signal display only (not used for TP/SL)

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
def get_3m_midpoint_condition(candles_3m, current_price):
    if len(candles_3m) < 50: return False, False, 0.0, 0.0, 0.0, 0, 0
    closes_3m = np.array([c["close"] for c in candles_3m], dtype=float)
    len_3m = len(closes_3m)
    argmin_idx = len_3m - 1 - int(np.argmin(closes_3m[::-1]))
    argmax_idx = len_3m - 1 - int(np.argmax(closes_3m[::-1]))
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
    sl_dist = entry_price * SL_PCT
    tp_dist = entry_price * TP_PCT

    if side == "long":
        sl_price, tp_price = entry_price - sl_dist, entry_price + tp_dist
    else:
        sl_price, tp_price = entry_price + sl_dist, entry_price - tp_dist

    sl_roe = SL_PCT * 100 * LEVERAGE
    tp_roe = TP_PCT * 100 * LEVERAGE - RT_FEE_ROE_PCT
    print(f"  [Fixed PCT] SL:-{SL_PCT*100:.1f}%={sl_dist:.2f} ({sl_roe:.1f}% ROE) | TP:+{TP_PCT*100:.1f}%={tp_dist:.2f} ({tp_roe:.1f}% net ROE)")
    return float(sl_price), float(tp_price), sl_roe, tp_roe

def check_dynamic_tp_sl(entry_price, current_price, side, sl_price, tp_price):
    hit, hit_type = False, None
    if side == "long":
        if current_price >= tp_price: hit, hit_type = True, "TP"
        elif current_price <= sl_price: hit, hit_type = True, "SL"
    else:
        if current_price <= tp_price: hit, hit_type = True, "TP"
        elif current_price >= sl_price: hit, hit_type = True, "SL"
    
    pcp = ((current_price - entry_price) / entry_price) * 100 if side == "long" else ((entry_price - current_price) / entry_price) * 100
    roe_pct = pcp * LEVERAGE - RT_FEE_ROE_PCT
    
    if hit: return True, "TAKE PROFIT" if hit_type == "TP" else "STOP LOSS", roe_pct
    return False, None, roe_pct

def scale_to_sine(close_prices_5m, argmin_idx, argmax_idx):
    if len(close_prices_5m) < 32: return 50.0, 50.0
    sine_wave, _ = talib.HT_SINE(close_prices_5m)
    sine_wave = np.nan_to_num(-sine_wave)
    sine_window = sine_wave[-ANALYSIS_WINDOW_5M:] if len(sine_wave) >= ANALYSIS_WINDOW_5M else sine_wave
    c_min, c_max = sine_window[argmin_idx], sine_window[argmax_idx]
    rng = c_max - c_min if c_max != c_min else 1e-9
    cur = sine_wave[-1]
    return max(0, min(100, ((cur - c_min) / rng) * 100)), max(0, min(100, ((c_max - cur) / rng) * 100))

def calculate_momentum(close_arr, period=14):
    if len(close_arr) < period + 1: return np.nan
    return float(talib.MOM(close_arr, timeperiod=period)[-1])

def compute_signals(buffer_5m, buffer_3m, buffer_1m, live_price):
    candles_5m, candles_3m, candles_1m = buffer_5m.get_candles(), buffer_3m.get_candles(), buffer_1m.get_candles()
    c5, c1 = [x["close"] for x in candles_5m], [x["close"] for x in candles_1m]
    if len(c5) < 50 or len(c1) < 50 or len(candles_3m) < 50: return None
    
    arr5, arr1 = np.array(c5, dtype=float), np.array(c1, dtype=float)
    cc = live_price
    
    # 1. Sine (5m)
    w5 = arr5[-ANALYSIS_WINDOW_5M:]
    l5 = len(w5)
    ai5_min = l5 - 1 - int(np.argmin(w5[::-1]))
    ai5_max = l5 - 1 - int(np.argmax(w5[::-1]))
    d_min, d_max = scale_to_sine(arr5, ai5_min, ai5_max)
    c_sine_l, c_sine_s = d_min < d_max, d_max < d_min
    
    # 2. Cycle (1m)
    w1_candles = candles_1m[-500:]
    w1 = arr1[-500:]
    wl1 = len(w1)
    ai1_min = wl1 - 1 - int(np.argmin(w1[::-1]))
    ai1_max = wl1 - 1 - int(np.argmax(w1[::-1]))
    p_min, p_max = float(w1[ai1_min]), float(w1[ai1_max])
    bmin, bmax = (wl1 - 1) - ai1_min, (wl1 - 1) - ai1_max
    
    try:
        t_min = datetime.datetime.fromtimestamp(w1_candles[ai1_min]["time"]).strftime("%H:%M:%S")
        t_max = datetime.datetime.fromtimestamp(w1_candles[ai1_max]["time"]).strftime("%H:%M:%S")
    except Exception: t_min = t_max = "N/A"
    
    if bmin > bmax:
        recent = "ARGMIN (LOW)"; c_cyc_l, c_cyc_s = cc > p_min, False
    elif bmax > bmin:
        recent = "ARGMAX (HIGH)"; c_cyc_s, c_cyc_l = cc < p_max, False
    else:
        recent = "TIE"; c_cyc_l, c_cyc_s = False, False
        
    # 3. Mom (1m)
    mom = calculate_momentum(arr1)
    if np.isnan(mom): return None
    c_mom_l, c_mom_s = mom > 0, mom < 0
    
    # 4. Vol (1m)
    bull, bear = get_buy_sell_volume_perc(candles_1m)
    c_vol_l, c_vol_s = bull > bear, bear > bull
    
    # 5. 3m Mid
    c_mid_l, c_mid_s, mid, mn3, mx3, _, _ = get_3m_midpoint_condition(candles_3m, cc)
    
    lon = [c_sine_l, c_cyc_l, c_mom_l, c_vol_l, c_mid_l]
    sho = [c_sine_s, c_cyc_s, c_mom_s, c_vol_s, c_mid_s]
    lt, st = sum(lon), sum(sho)
    # Mandatory: Mom + Vol only
    lm, sm = c_mom_l and c_vol_l, c_mom_s and c_vol_s
    # Flexible: Sine + Cycle + 3mMid (need ≥2/3)
    lf = sum([c_sine_l, c_cyc_l, c_mid_l])
    sf = sum([c_sine_s, c_cyc_s, c_mid_s])
    
    return {
        "price": live_price, "current_close": cc, "is_long": lm and lf >= 3, "is_short": sm and sf >= 3,
        "cond_flags": {"sine_long": c_sine_l, "sine_short": c_sine_s, "cycle_long": c_cyc_l, "cycle_short": c_cyc_s, "mom_long": c_mom_l, "mom_short": c_mom_s, "vol_long": c_vol_l, "vol_short": c_vol_s, "mid_long": c_mid_l, "mid_short": c_mid_s},
        "long_true_count": lt, "short_true_count": st, "long_flex": lf, "short_flex": sf,
        "long_mandatory_met": lm, "short_mandatory_met": sm,
        "dist_to_min": d_min, "dist_to_max": d_max,
        "ts_min": t_min, "ts_max": t_max, "most_recent_extreme": recent,
        "mom_1m": mom, "bullish_perc": bull, "bearish_perc": bear,
        "midpoint_3m": mid, "min_3m": mn3, "max_3m": mx3,
        "atr_value": calculate_atr(candles_1m, ATR_PERIOD)
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
    print(f"  1. Sine (5m):    dMin:{sig['dist_to_min']:.1f}% dMax:{sig['dist_to_max']:.1f}% L:{f['sine_long']} S:{f['sine_short']} [flex]")
    print(f"  2. Cycle (1m):   Low@{sig['ts_min']} High@{sig['ts_max']} | Recent: {sig['most_recent_extreme']} | L:{f['cycle_long']} S:{f['cycle_short']} [flex]")
    print(f"  3. Mom (1m):     {sig['mom_1m']:.2f} L:{f['mom_long']} S:{f['mom_short']} [MANDATORY]")
    print(f"  4. Vol (1m):     Bull:{sig['bullish_perc']:.1f}% Bear:{sig['bearish_perc']:.1f}% L:{f['vol_long']} S:{f['vol_short']} [MANDATORY]")
    mid, mn, mx = sig.get("midpoint_3m",0), sig.get("min_3m",0), sig.get("max_3m",0)
    mdp = ((cc - mid) / mid * 100) if mid > 0 else 0.0
    print(f"  5. 3mMid:        Min:{mn:.2f} Max:{mx:.2f} Mid:{mid:.2f} | Cur:{cc:.2f} ({mdp:+.3f}%) L:{f['mid_long']} S:{f['mid_short']} [flex]")
    lt, st = sig["long_true_count"], sig["short_true_count"]
    lm, sm = sig["long_mandatory_met"], sig["short_mandatory_met"]
    lf, sf = sig["long_flex"], sig["short_flex"]
    print(f"  ═══ LONG:{lt}/5 (Mand:{'✓' if lm else '✗'} Flex:{lf}/3) SHORT:{st}/5 (Mand:{'✓' if sm else '✗'} Flex:{sf}/3)")
    print(f"     Rule: ALL 5 CONDITIONS MUST BE TRUE -> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*70}\nHFT KuCoin Bot - Mom+Vol MANDATORY + ≥2/3 Flex (Sine/Cycle/3mMid)\n{'='*70}")
    print(f"Symbol: {TRADE_SYMBOL} | Leverage: {LEVERAGE}x | API: {API_CONNECTED}")
    print(f"TP: +{TP_PCT*100:.3f}% price (~{_TARGET_NET_ROE_PCT:.0f}% net ROE) | SL: -{SL_PCT*100:.3f}% price (~{_TARGET_SL_ROE_PCT:.0f}% ROE)\n{'='*70}\n")

    b5 = CandleBuffer(TRADE_SYMBOL, "5m", ANALYSIS_WINDOW_5M)
    b3 = CandleBuffer(TRADE_SYMBOL, "3m", ANALYSIS_WINDOW_3M)
    b1 = CandleBuffer(TRADE_SYMBOL, "1m", ANALYSIS_WINDOW_1M)
    
    if API_CONNECTED:
        print("  === INITIALIZING BUFFERS ===\n")
        t0 = time.perf_counter()
        b5.initialize(client); b3.initialize(client); b1.initialize(client)
        print(f"  Total init: {(time.perf_counter()-t0)*1000:.0f}ms\n")
        if not b5.is_ready() or not b1.is_ready() or not b3.is_ready():
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

            # --- LIVE POSITION ---
            if pos["is_open"]:
                roe, side, ep, mp = pos["roe_pct"], pos["side"], pos["entry_price"], pos["mark_price"]
                sl_p = saved_state.get("sl", 0.0) if saved_state else 0.0
                tp_p = saved_state.get("tp", 0.0) if saved_state else 0.0
                if sl_p <= 0 or tp_p <= 0:
                    sl_p, tp_p, _, _ = calculate_dynamic_sl_tp(ep, side, sig["atr_value"] if sig else 0.0)
                
                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] === LIVE {side.upper()} | Entry:{ep:.2f} Mark:{mp:.2f} ROE:{roe:+.2f}% ===")
                    print(f"  SL:{sl_p:.2f} TP:{tp_p:.2f}")
                    if sig: print_conditions(sig)
                
                hit, reason, _ = check_dynamic_tp_sl(ep, mp, side, sl_p, tp_p)
                if not hit and roe >= 100: hit, reason = True, "EXTREME PROFIT"
                elif not hit and roe <= -95: hit, reason = True, "NEAR LIQUIDATION"
                    
                if hit and reason:
                    print(f"  >>> [{reason}] {roe:+.2f}% ROE - Closing...")
                    client.close_position(TRADE_SYMBOL)
                    pcp = ((mp - ep) / ep) * 100 if side == "long" else ((ep - mp) / ep) * 100
                    sdt = datetime.datetime.strptime(saved_state["entry_time"], "%Y-%m-%d %H:%M:%S")
                    write_trade_to_journal({"mode": "LIVE", "entry_time": saved_state["entry_time"], "exit_time": now_str, "type": side, "entry_price": ep, "exit_price": mp, "sl_price": sl_p, "tp_price": tp_p, "atr_at_entry": saved_state.get("atr_at_entry", 0), "price_change_pct": pcp, "gross_roe": pcp * LEVERAGE, "rt_fee_pct": RT_FEE_ROE_PCT, "roe": roe, "duration": format_duration(sdt, datetime.datetime.now()), "reason": reason})
                    clear_trade_state(); saved_state = None
                
                log_timing("total_loop", (time.perf_counter() - loop_start) * 1000)
                time.sleep(LOOP_SLEEP); continue

            # --- SIMULATION POSITION ---
            if saved_state and saved_state.get("mode") == "SIMULATION":
                s_ep, s_side, s_et = saved_state["entry_price"], saved_state["side"], saved_state["entry_time"]
                s_sl, s_tp = saved_state.get("sl", 0.0), saved_state.get("tp", 0.0)
                hit, reason, s_roe = check_dynamic_tp_sl(s_ep, cur_p, s_side, s_sl, s_tp)
                
                if loop_count % 10 == 0:
                    print(f"\n[{now_str}] === SIM {s_side.upper()} | Entry:{s_ep:.2f} Now:{cur_p:.2f} ROE:{s_roe:+.2f}% ===")
                    print(f"  SL:{s_sl:.2f} TP:{s_tp:.2f}")
                    if sig: print_conditions(sig)
                
                if not hit and s_roe >= 100: hit, reason = True, "EXTREME PROFIT"
                elif not hit and s_roe <= -95: hit, reason = True, "NEAR LIQUIDATION"
                
                if hit and reason:
                    print(f"  >>> [SIM {reason}] {s_roe:+.2f}% ROE")
                    pcp = ((cur_p - s_ep) / s_ep) * 100 if s_side == "long" else ((s_ep - cur_p) / s_ep) * 100
                    sdt = datetime.datetime.strptime(s_et, "%Y-%m-%d %H:%M:%S")
                    write_trade_to_journal({"mode": "SIMULATION", "entry_time": s_et, "exit_time": now_str, "type": s_side, "entry_price": s_ep, "exit_price": cur_p, "sl_price": s_sl, "tp_price": s_tp, "atr_at_entry": saved_state.get("atr_at_entry", 0), "price_change_pct": pcp, "gross_roe": pcp * LEVERAGE, "rt_fee_pct": RT_FEE_ROE_PCT, "roe": s_roe, "duration": format_duration(sdt, now), "reason": reason})
                    clear_trade_state(); saved_state = None
                
                log_timing("total_loop", (time.perf_counter() - loop_start) * 1000)
                time.sleep(LOOP_SLEEP); continue

            # --- ORPHAN CLEANUP ---
            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print("  [recovery] Orphaned live state - clearing.")
                clear_trade_state(); saved_state = None

            # --- SCANNING ---
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
                    print(f"\n  *** 4/5 CONDITIONS MET -> {side.upper()} ***")
                    sl_p, tp_p, sl_r, tp_r = calculate_dynamic_sl_tp(sig["price"], side, sig["atr_value"])
                    state_dict = {"active": True, "side": side, "entry_price": sig["price"], "sl": sl_p, "tp": tp_p, "sl_roe": sl_r, "tp_roe": tp_r, "atr_at_entry": sig["atr_value"], "entry_time": now_str}
                    
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
