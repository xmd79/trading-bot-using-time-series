"""
HFT Auto Trading Bot — KuCoin Futures Edition (CONCURRENT OPTIMIZED)
====================================================================
SIGNAL LOGIC:
  - Mandatory: Momentum (1m), Volume (1m), ML Forecast, AND TP Reachability MUST be true.
  - ML Forecast (1m): forecast_price > current_close → LONG allowed;
                      forecast_price < current_close → SHORT allowed.
  - TP Reachability: Consensus Forecast Price MUST transit (cross) the TP target.
  - Flexible: At least 3 out of 5 total conditions must be true for the direction.
  - (Since Mom + Vol = 2 mandatory true, you only need 1 more from Sine/Cycle/FFT).

POSITION SIZING:
  - Only 5% of available balance is used per trade.
  - Remaining 95% stays untouched in the account.

25x Leverage
TP: 2.55% NET Profit (5.55% Gross ROE after 3.0% RT fee deduction)
SL: -90% ROE
"""

import sys
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

print("HFT KuCoin Bot (25x / 3-OUT-OF-5 LOGIC / TP-REACHABILITY ML) initialising...")
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
ML_LOOKBACK = 100
ANALYSIS_WINDOW_5M = 1200 
ANALYSIS_WINDOW_1M = 500
KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100
NET_PROFIT_ROE = 2.55                                  
TAKE_PROFIT_ROE = NET_PROFIT_ROE + RT_FEE_ROE_PCT      
STOP_LOSS_ROE = -90.0                                  
TP_PRICE_PCT = TAKE_PROFIT_ROE / LEVERAGE / 100.0
SL_PRICE_PCT = abs(STOP_LOSS_ROE) / LEVERAGE / 100.0
FULL_REFRESH_INTERVAL = 3600
TP_REACH_SIMULATIONS = 300
TP_REACH_FORECAST_STEPS = 30

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
# TRADE HISTORY TRACKER (Clean Single Line)
# ═══════════════════════════════════════════════════════════════════════════════

class TradeHistoryTracker:
    def __init__(self):
        self.trades = []
        self.lock = Lock()
        self.total_profit_roe = 0.0
        self.total_loss_roe = 0.0
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.long_trades = 0
        self.short_trades = 0
        self.long_wins = 0
        self.short_wins = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.current_streak = 0
        self.best_trade_roe = -999.0
        self.worst_trade_roe = 999.0
        self.session_start = datetime.datetime.now()
        self.last_trade_time = None
        
    def add_trade(self, trade_result):
        with self.lock:
            roe = trade_result.get("roe", 0.0)
            side = trade_result.get("type", "long")
            
            self.trades.append(trade_result)
            self.total_trades += 1
            self.last_trade_time = datetime.datetime.now()
            
            if side == "long": self.long_trades += 1
            else: self.short_trades += 1
            
            if roe >= 0:
                self.wins += 1
                self.total_profit_roe += roe
                if side == "long": self.long_wins += 1
                self.current_streak = self.current_streak + 1 if self.current_streak >= 0 else 1
                self.max_consecutive_wins = max(self.max_consecutive_wins, self.current_streak)
            else:
                self.losses += 1
                self.total_loss_roe += abs(roe)
                self.current_streak = self.current_streak - 1 if self.current_streak <= 0 else -1
                self.max_consecutive_losses = max(self.max_consecutive_losses, abs(self.current_streak))
            
            self.best_trade_roe = max(self.best_trade_roe, roe)
            self.worst_trade_roe = min(self.worst_trade_roe, roe)
    
    def get_stats(self):
        with self.lock:
            net_roe = self.total_profit_roe - self.total_loss_roe
            win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0
            time_since_last = "N/A"
            if self.last_trade_time:
                delta = datetime.datetime.now() - self.last_trade_time
                total_sec = int(delta.total_seconds())
                if total_sec < 60: time_since_last = f"{total_sec}s ago"
                elif total_sec < 3600: time_since_last = f"{total_sec//60}m {total_sec%60}s ago"
                else: time_since_last = f"{total_sec // 3600}h {(total_sec % 3600) // 60}m ago"
            return {
                "total_trades": self.total_trades, "wins": self.wins, "losses": self.losses, "win_rate": win_rate,
                "total_profit_roe": self.total_profit_roe, "total_loss_roe": self.total_loss_roe, "net_roe": net_roe,
                "best_trade_roe": self.best_trade_roe, "worst_trade_roe": self.worst_trade_roe,
                "time_since_last_trade": time_since_last
            }
    
    def print_stats_line(self, position_status="FLAT", current_roe=0.0):
        s = self.get_stats()
        
        if s['net_roe'] > 0: net_str = f"+{s['net_roe']:.2f}%"
        elif s['net_roe'] < 0: net_str = f"{s['net_roe']:.2f}%"
        else: net_str = "0.00%"
            
        wr_str = f"{s['win_rate']:.1f}%"
        pos_str = "FLAT" if position_status == "FLAT" else f"{position_status} ({current_roe:+.2f}%)"
        
        line = f"  [P&L] Trades: {s['total_trades']} W:{s['wins']} L:{s['losses']} WR:{wr_str} | Profit: +{s['total_profit_roe']:.2f}% Loss: -{s['total_loss_roe']:.2f}% | ═══ NET: {net_str} | Pos: {pos_str} | Last: {s['time_since_last_trade']}   "
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()
        
    def print_trade_alert(self, trade_result):
        roe = trade_result.get("roe", 0.0)
        side = trade_result.get("type", "N/A").upper()
        reason = trade_result.get("reason", "N/A")
        mode = trade_result.get("mode", "N/A")
        entry = trade_result.get("entry_price", 0)
        exit_p = trade_result.get("exit_price", 0)
        duration = trade_result.get("duration", "N/A")
        
        icon = "🟢" if roe >= 0 else "🔴"
        result_str = f"PROFIT +{roe:.2f}%" if roe >= 0 else f"LOSS {roe:.2f}%"
        
        print() 
        print(f"  {icon}════════════════════════════════════════{icon}")
        print(f"  │ TRADE CLOSED: {mode} {side}")
        print(f"  │ {reason}")
        print(f"  │ Entry: {entry:.2f} → Exit: {exit_p:.2f} | Duration: {duration}")
        print(f"  │ Result: {result_str}")
        
        s = self.get_stats()
        if s["net_roe"] >= 0: net_str = f"+{s['net_roe']:.2f}%"
        else: net_str = f"{s['net_roe']:.2f}%"
        print(f"  │ Running Total: {s['wins']}W/{s['losses']}L | Net: {net_str}")
        print(f"  {icon}════════════════════════════════════════{icon}")

trade_tracker = TradeHistoryTracker()

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
            return {"time": int(k[0]) / 1000, "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
        except (IndexError, TypeError, ValueError): return None
    
    def initialize(self, client_obj):
        """Initialize buffer with full historical data - takes only client_obj as argument"""
        if not API_CONNECTED: return False
        with self.lock:
            print(f"  [Buffer:{self.timeframe}] Fetching {self.max_candles} candles...")
            self.candles = []
            now_ms = int(time.time() * 1000)
            limit = 200
            for i in range((self.max_candles + limit - 1) // limit):
                end_ms = now_ms - (i * limit * self.duration_ms)
                start_ms = end_ms - (limit * self.duration_ms)
                try:
                    for k in reversed(client_obj.get_klines(self.symbol, self.granularity, start_ms=start_ms, end_ms=end_ms)):
                        candle = self._parse_kline(k)
                        if candle: self.candles.append(candle)
                except Exception as e: print(f"  [Buffer:{self.timeframe}] Init error batch {i}: {e}")
            self._deduplicate()
            self.candles = self.candles[-self.max_candles:]
            self.last_full_refresh = time.time()
            print(f"  [Buffer:{self.timeframe}] Ready: {len(self.candles)} candles")
            return len(self.candles) >= 50
    
    def _deduplicate(self):
        seen, unique = set(), []
        for c in reversed(self.candles):
            if c["time"] not in seen: seen.add(c["time"]); unique.append(c)
        self.candles = list(reversed(unique))
    
    def update(self, client_obj, fetch_limit=3):
        """Update buffer with recent candles - takes client_obj and optional fetch_limit"""
        if not API_CONNECTED or not self.candles: return 0
        with self.lock:
            try:
                klines = client_obj.get_klines(self.symbol, self.granularity, start_ms=int(time.time()*1000) - (fetch_limit*self.duration_ms), end_ms=int(time.time()*1000))
                new = [self._parse_kline(k) for k in klines if self._parse_kline(k)]
                if self.candles: new = [c for c in new if c["time"] > self.candles[-1]["time"]]
                if new: self.candles.extend(new); self.candles = self.candles[-self.max_candles:]
                return len(new)
            except Exception: return 0
            
    def needs_full_refresh(self):
        return (time.time() - self.last_full_refresh) > FULL_REFRESH_INTERVAL if self.last_full_refresh else True
    
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
        data = client.get_position(symbol).get("data", {})
        size = float(data.get("currentQty", 0) or 0)
        if size == 0: return empty
        upnl, margin = float(data.get("unrealisedPnl", 0) or 0), float(data.get("posMargin", 0) or 0)
        return {"is_open": True, "side": "long" if size > 0 else "short", "entry_price": float(data.get("avgEntryPrice", 0)), "mark_price": float(data.get("markPrice", 0)), "roe_pct": (upnl / margin * 100) if margin else 0.0, "size": size}
    except Exception: return empty

def get_buy_sell_volume_perc(candles):
    buy_vol = sell_vol = 0.0
    for c in candles:
        if c["close"] >= c["open"]: buy_vol += c["volume"]
        else: sell_vol += c["volume"]
    total = buy_vol + sell_vol
    return (buy_vol / total) * 100.0, (sell_vol / total) * 100.0 if total else (50.0, 50.0)

# ═══════════════════════════════════════════════════════════════════════════════
# CONCURRENT DATA FETCHER (FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

class ConcurrentDataFetcher:
    def __init__(self, buffer_5m, buffer_1m, symbol, max_workers=5):
        self.buffer_5m, self.buffer_1m, self.symbol = buffer_5m, buffer_1m, symbol
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def fetch_all_parallel(self, do_full_refresh=False):
        futures, results = {}, {"price": 0.0, "balance": 0.0, "position": {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0, "mark_price": 0.0, "side": None, "size": 0}, "new_5m": 0, "new_1m": 0, "times": {}}
        t0 = time.perf_counter()
        
        # FIX: initialize() takes only client, update() takes client and fetch_limit
        # Must call them separately with correct arguments
        need_full_5m = do_full_refresh or self.buffer_5m.needs_full_refresh()
        need_full_1m = do_full_refresh or self.buffer_1m.needs_full_refresh()
        
        if need_full_5m:
            futures["5m"] = self.executor.submit(self.buffer_5m.initialize, client)
        else:
            futures["5m"] = self.executor.submit(self.buffer_5m.update, client, 3)
            
        if need_full_1m:
            futures["1m"] = self.executor.submit(self.buffer_1m.initialize, client)
        else:
            futures["1m"] = self.executor.submit(self.buffer_1m.update, client, 3)
        
        futures["price"] = self.executor.submit(get_price, self.symbol)
        futures["balance"] = self.executor.submit(get_account_balance)
        futures["position"] = self.executor.submit(get_position_info, self.symbol)
        
        for name, future in futures.items():
            try:
                t_s = time.perf_counter()
                result = future.result(timeout=10)
                results["times"][name] = (time.perf_counter() - t_s) * 1000
                if name == "5m": results["new_5m"] = len(self.buffer_5m) if need_full_5m else result
                elif name == "1m": results["new_1m"] = len(self.buffer_1m) if need_full_1m else result
                elif name == "price": results["price"] = result
                elif name == "balance": results["balance"] = result
                elif name == "position": results["position"] = result
            except Exception as e: results["times"][name] = -1; print(f"  [Concurrent] Task '{name}' failed: {e}")
        results["parallel_total_ms"] = (time.perf_counter() - t0) * 1000
        return results
    
    def shutdown(self): self.executor.shutdown(wait=False)

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def scale_to_sine(close_prices_5m, argmin_idx, argmax_idx):
    if len(close_prices_5m) < 32: return 50.0, 50.0
    sine_wave = np.nan_to_num(-talib.HT_SINE(close_prices_5m)[0])
    sine_window = sine_wave[-ANALYSIS_WINDOW_5M:] if len(sine_wave) >= ANALYSIS_WINDOW_5M else sine_wave
    c_min, c_max = sine_window[argmin_idx], sine_window[argmax_idx]
    rng = c_max - c_min if c_max != c_min else 1e-9
    cur = sine_wave[-1]
    return max(0, min(100, ((cur - c_min) / rng) * 100)), max(0, min(100, ((c_max - cur) / rng) * 100))

def analyze_fft_dominance_1m(close_prices_1m):
    if len(close_prices_1m) < 32: return False, False, 0.0, 0.0
    sw, lsw = np.nan_to_num(-talib.HT_SINE(close_prices_1m)[0]), np.nan_to_num(-talib.HT_SINE(close_prices_1m)[1])
    mask = ~(np.isnan(sw) | np.isnan(lsw))
    sw, lsw, n = sw[mask], lsw[mask], len(sw[mask])
    if n < 32: return False, False, 0.0, 0.0
    cs = (sw + 1j * lsw) * np.hanning(n); cs -= np.mean(cs)
    fp = np.abs(fft(cs))**2; freqs = fftfreq(n); p, f = fp[1:], freqs[1:]
    np_, pp = np.sum(p[f<0]), np.sum(p[f>0])
    t = np_ + pp
    if t == 0: return False, False, 0.0, 0.0
    nr, pr = (np_/t)*100, (pp/t)*100
    return nr > pr + 0.1, pr > nr + 0.1, nr, pr

def calculate_momentum(close_arr, period=14):
    return float(talib.MOM(close_arr, timeperiod=period)[-1]) if len(close_arr) >= period + 1 else np.nan

def generate_ml_forecast(candles_1m):
    if len(candles_1m) < ML_LOOKBACK: return 0.0, 0.0
    w = candles_1m[-ML_LOOKBACK:]
    c, h, l, v = np.array([x["close"] for x in w], dtype=np.float64), np.array([x["high"] for x in w], dtype=np.float64), np.array([x["low"] for x in w], dtype=np.float64), np.array([x["volume"] for x in w], dtype=np.float64)
    for arr in (c, h, l, v):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]) or arr[i] == 0: arr[i] = arr[i-1] if i > 0 else arr[0]
    cur = float(c[-1])
    try:
        rsi_b = (float(talib.RSI(c, 14)[-1]) - 50.0) / 50.0 if np.isfinite(talib.RSI(c, 14)[-1]) else 0.0
        hn = float(talib.MACD(c, 12, 26, 9)[2][-1]) / (cur + 1e-12) if np.isfinite(talib.MACD(c, 12, 26, 9)[2][-1]) else 0.0
        bb = talib.BBANDS(c, 20, 2, 2, 0); bbr = float(bb[0][-1] - bb[2][-1])
        bp = float((c[-1] - bb[2][-1]) / bbr) if np.isfinite(bb[2][-1]) and bbr > 1e-12 else 0.5
        sl = float(np.polyfit(np.arange(20), c[-20:], 1)[0]) / (cur + 1e-12)
        vr = float(np.mean(v[-3:])) / (float(np.mean(v[-20:])) + 1e-12)
        vw = ((c[-1] - c[-6]) / (c[-6] + 1e-12)) * vr
        score = 0.30*rsi_b + 0.25*np.sign(hn)*min(abs(hn)*1e4, 1.0) + 0.20*(bp-0.5)*2 + 0.15*np.sign(sl)*min(abs(sl)*1e3, 1.0) + 0.10*np.sign(vw)*min(abs(vw)*10, 1.0)
        return cur * (1.0 + score * 0.003), cur
    except Exception: return 0.0, cur

# ═══════════════════════════════════════════════════════════════════════════════
# TP REACHABILITY ML ENSEMBLE (REALISTIC BOUND FORECASTING)
# ═══════════════════════════════════════════════════════════════════════════════

def _fft_price_forecast(closes, steps):
    try:
        w = closes[-200:]; n = len(w)
        if n < 20: return closes[-1]
        
        # 1. Extract linear trend
        x = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(x, w, 1)
        
        # 2. Detrend and center
        detrended = w - (slope * x + intercept)
        
        # 3. FFT to find dominant cycle safely
        windowed = detrended * np.hanning(n)
        f = fft(windowed)
        freqs = fftfreq(n)
        mags = np.abs(f)
        
        # Ignore DC and Nyquist to find actual cycles
        mags[0] = 0
        if n % 2 == 0: mags[n//2] = 0
        
        dom_idx = np.argmax(mags)
        dom_freq = freqs[dom_idx]
        dom_phase = np.angle(f[dom_idx])
        dom_amp = mags[dom_idx] / n  # Proper scaling
        
        # 4. Extrapolate trend + single dominant cycle
        t_target = n + steps - 1
        forecast = (slope * t_target + intercept) + (dom_amp * np.sin(2 * np.pi * dom_freq * t_target + dom_phase))
        
        return forecast
    except Exception:
        return closes[-1]

def _rw_price_forecast(closes, steps, n_sims):
    try:
        rets = np.diff(closes[-100:]) / closes[-101:-1]
        mu, sigma = np.mean(rets[-20:]), np.std(rets)
        finals = []
        for _ in range(n_sims):
            p = closes[-1]
            for _ in range(steps): p *= (1 + np.random.normal(mu, sigma))
            finals.append(p)
        return np.median(finals)
    except Exception: return closes[-1]

def _forest_price_forecast(closes, steps):
    try:
        s10 = np.polyfit(np.arange(10), closes[-10:], 1)[0]
        s20 = np.polyfit(np.arange(20), closes[-20:], 1)[0]
        s50 = np.polyfit(np.arange(min(50, len(closes))), closes[-min(50, len(closes)):], 1)[0]
        blend = 0.5*s10 + 0.3*s20 + 0.2*s50
        return closes[-1] + (blend * steps)
    except Exception: return closes[-1]

def _lstm_price_forecast(closes, volumes):
    try:
        n = len(closes)
        if n < 60: return closes[-1]
        cn = (closes[-60:] - np.mean(closes[-60:])) / (np.std(closes[-60:]) + 1e-12)
        vn = (volumes[-60:] - np.mean(volumes[-60:])) / (np.std(volumes[-60:]) + 1e-12)
        rets = np.diff(closes[-60:]) / closes[-61:-1]
        comb = np.stack([cn, vn, np.concatenate([[0], rets])], axis=-1)
        W_i = np.array([[0.3, 0.1, 0.2, -0.1],[0.2, 0.4, 0.1, 0.2],[0.1, 0.1, 0.3, 0.1]])
        W_hi = np.array([[0.2,-0.1, 0.3, 0.1],[0.1, 0.3,-0.2, 0.2],[0.2, 0.1, 0.1,-0.1],[-0.1,0.2, 0.2, 0.3]])
        b_i = np.array([0.1,-0.1, 0.1, 0.0])
        W_c = np.array([[0.2, 0.3, 0.1, 0.2],[0.3, 0.1, 0.2, 0.1],[0.1, 0.2, 0.3, 0.2]])
        W_hc = np.array([[0.1, 0.2, 0.3, 0.1],[0.2, 0.1, 0.1, 0.3],[0.3, 0.2, 0.1, 0.2],[0.1, 0.3, 0.2, 0.1]])
        b_c = np.array([0.0, 0.0, 0.0, 0.0])
        h, c_state = np.zeros(4), np.zeros(4)
        sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10)))
        for t in range(len(comb)-20, len(comb)):
            x = comb[t]
            i = sig(np.dot(W_i.T, x) + np.dot(W_hi.T, h) + b_i)
            ct = np.tanh(np.dot(W_c.T, x) + np.dot(W_hc.T, h) + b_c)
            c_state = 0.5 * c_state + i * ct
            h = i * np.tanh(c_state)
        trend_score = h[0]*0.6 + h[1]*0.4
        expected_ret = trend_score * 0.005 * 30
        return closes[-1] * (1 + expected_ret)
    except Exception: return closes[-1]

def tp_reachability_forecast(candles_1m, current_price, tp_long_price, sl_long_price, tp_short_price, sl_short_price):
    if len(candles_1m) < 100 or current_price <= 0:
        return False, False, current_price, {"fft": current_price, "random_walk": current_price, "forest": current_price, "lstm": current_price}
    
    closes = np.array([c["close"] for c in candles_1m], dtype=np.float64)
    volumes = np.array([c["volume"] for c in candles_1m], dtype=np.float64)
    
    # REALISTIC BOUNDS: Maximum 1% move per step (30% total over 30 steps)
    # This entirely prevents mathematical artifacts like -2344763.33
    max_move = current_price * 0.01 * TP_REACH_FORECAST_STEPS
    min_bound = current_price - max_move
    max_bound = current_price + max_move
    
    p_fft = np.clip(_fft_price_forecast(closes, TP_REACH_FORECAST_STEPS), min_bound, max_bound)
    p_rw = np.clip(_rw_price_forecast(closes, TP_REACH_FORECAST_STEPS, TP_REACH_SIMULATIONS), min_bound, max_bound)
    p_for = np.clip(_forest_price_forecast(closes, TP_REACH_FORECAST_STEPS), min_bound, max_bound)
    p_lstm = np.clip(_lstm_price_forecast(closes, volumes), min_bound, max_bound)
    
    method_prices = {"fft": p_fft, "random_walk": p_rw, "forest": p_for, "lstm": p_lstm}
    consensus_price = 0.20*p_fft + 0.25*p_rw + 0.30*p_for + 0.25*p_lstm
    
    reach_long = consensus_price >= tp_long_price
    reach_short = consensus_price <= tp_short_price
    
    return reach_long, reach_short, consensus_price, method_prices

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(buffer_5m, buffer_1m, live_price):
    c5, c1 = buffer_5m.get_candles(), buffer_1m.get_candles()
    cl5, cl1 = [x["close"] for x in c5], [x["close"] for x in c1]
    if len(cl5) < 50 or len(cl1) < 50: return None
    a5, a1 = np.array(cl5, dtype=float), np.array(cl1, dtype=float)
    
    l1200 = a5[-ANALYSIS_WINDOW_5M:]
    dmin, dmax = scale_to_sine(a5, int(np.argmin(l1200)), int(np.argmax(l1200)))
    c_sin_l, c_sin_s = dmin < dmax, dmax < dmin
    
    w1, l500 = c1[-500:], a1[-500:]; wl = len(l500)
    ai1, ax1 = int(np.argmin(l500)), int(np.argmax(l500))
    cpmin, cpmax, cur1 = float(l500[ai1]), float(l500[ax1]), float(a1[-1])
    bmin, bmax = (wl-1)-ai1, (wl-1)-ax1
    try: tsmin = datetime.datetime.fromtimestamp(w1[ai1]["time"], datetime.timezone.utc).strftime("%H:%M:%S UTC")
    except: tsmin = "N/A"
    try: tsmax = datetime.datetime.fromtimestamp(w1[ax1]["time"], datetime.timezone.utc).strftime("%H:%M:%S UTC")
    except: tsmax = "N/A"
    
    if bmin < bmax: mre, c_cyc_l, c_cyc_s = "ARGMIN (LOW)", cur1 > cpmin, False
    elif bmax < bmin: mre, c_cyc_s, c_cyc_l = "ARGMAX (HIGH)", cur1 < cpmax, False
    else: mre, c_cyc_l, c_cyc_s = "TIE", False, False
    
    mom = calculate_momentum(a1)
    if np.isnan(mom): return None
    c_mom_l, c_mom_s = mom > 0, mom < 0
    
    bp, sp = get_buy_sell_volume_perc(c1)
    c_vol_l, c_vol_s = bp > sp, sp > bp
    
    f_l, f_s, nr, pr = analyze_fft_dominance_1m(a1)
    
    mlp, mlc = generate_ml_forecast(c1)
    mlv = mlp > 0.0 and mlc > 0.0
    c_ml_l, c_ml_s = mlv and mlp > mlc, mlv and mlp < mlc
    
    tp_d, sl_d = live_price * TP_PRICE_PCT, live_price * SL_PRICE_PCT
    tpl, sll, tps, sls = live_price + tp_d, live_price - sl_d, live_price - tp_d, live_price + sl_d
    
    r_l, r_s, cons_p, meth_p = tp_reachability_forecast(c1, live_price, tpl, sll, tps, sls)
    
    lt = sum([c_sin_l, c_cyc_l, c_mom_l, c_vol_l, f_l])
    st = sum([c_sin_s, c_cyc_s, c_mom_s, c_vol_s, f_s])
    
    is_l = c_mom_l and c_vol_l and c_ml_l and r_l and lt >= 3
    is_s = c_mom_s and c_vol_s and c_ml_s and r_s and st >= 3
    
    return {
        "price": live_price, "is_long": is_l, "is_short": is_s,
        "cond_flags": {"sine_long": c_sin_l, "sine_short": c_sin_s, "cycle_long": c_cyc_l, "cycle_short": c_cyc_s,
                       "mom_long": c_mom_l, "mom_short": c_mom_s, "vol_long": c_vol_l, "vol_short": c_vol_s,
                       "fft_long": f_l, "fft_short": f_s, "ml_long": c_ml_l, "ml_short": c_ml_s,
                       "tp_reach_long": r_l, "tp_reach_short": r_s},
        "long_true_count": lt, "short_true_count": st,
        "dist_to_min": dmin, "dist_to_max": dmax, "argmin_idx_1m": ai1, "argmax_idx_1m": ax1,
        "cycle_min_price": cpmin, "cycle_max_price": cpmax, "current_close_1m": cur1,
        "bars_ago_min": bmin, "bars_ago_max": bmax, "ts_min": tsmin, "ts_max": tsmax, "most_recent_extreme": mre, "window_len_1m": wl,
        "mom_1m": mom, "bullish_perc": bp, "bearish_perc": sp, "neg_ratio": nr, "pos_ratio": pr,
        "ml_forecast_price": mlp, "ml_current_close": mlc,
        "tp_reach_long": r_l, "tp_reach_short": r_s, "consensus_price": cons_p, "method_prices": meth_p,
        "tp_long_price": tpl, "sl_long_price": sll, "tp_short_price": tps, "sl_short_price": sls
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SL & TP CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(ep, side):
    td, sd = ep * TP_PRICE_PCT, ep * SL_PRICE_PCT
    sl, tp = (ep - sd, ep + td) if side == "long" else (ep + sd, ep - td)
    print(f"  [Risk Calc] TP: {tp:.2f} ({TAKE_PROFIT_ROE}% ROE) | SL: {sl:.2f} ({STOP_LOSS_ROE}% ROE)")
    return float(sl), float(tp)

def check_sim_tp_sl(ep, cp, side):
    pcp = ((cp - ep) / ep) * 100 if side == "long" else ((ep - cp) / ep) * 100
    roe = pcp * LEVERAGE - RT_FEE_ROE_PCT
    if roe >= TAKE_PROFIT_ROE: return True, "TAKE PROFIT", roe
    elif roe <= STOP_LOSS_ROE: return True, "STOP LOSS", roe
    return False, None, roe

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

def write_trade_to_journal(tr):
    line = (f"{'='*60}\nMODE: {tr.get('mode', 'LIVE')}\nTYPE: {tr.get('type', 'N/A').upper()}\n"
            f"ENTRY TIME: {tr.get('entry_time', 'N/A')}\nEXIT TIME: {tr.get('exit_time', 'N/A')}\n"
            f"DURATION: {tr.get('duration', 'N/A')}\nENTRY PRICE: {tr.get('entry_price', 0):.2f}\n"
            f"EXIT PRICE: {tr.get('exit_price', 0):.2f}\nPRICE CHANGE: {tr.get('price_change_pct', 0):+.4f}%\n"
            f"GROSS ROE: {tr.get('gross_roe', 0):+.2f}%\nRT FEE DEDUCTED: {tr.get('rt_fee_pct', 0):.2f}%\n"
            f"NET ROE: {tr.get('roe', 0):+.2f}%\nREASON: {tr.get('reason', 'N/A')}\n"
            f"LEVERAGE: {LEVERAGE}x\nSTRATEGY: 3/5 Cond (Mom+Vol+ML+TPReach Mandatory) | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}%\n{'='*60}\n\n")
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
        tb = balance * TRADE_BALANCE_PCT
        contracts = max(1, int((tb * LEVERAGE) / (price * get_contract_size(symbol))))
        resp = client.place_order(symbol, side, contracts, LEVERAGE)
        if resp.get("data", {}).get("orderId"):
            print(f"  >>> {side.upper()} PLACED: {contracts} contracts @ ~{price:.2f}  (used {tb:.2f} USDT / {TRADE_BALANCE_PCT*100:.0f}%)")
            return True
        print(f"  [ORDER FAIL] {side.upper()} [{resp.get('code')}]: {resp.get('msg')}")
        return False
    except Exception as e:
        print(f"  [ORDER ERROR] {side.upper()}: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_duration(s, e):
    ts = int((e - s).total_seconds())
    h, rem = divmod(ts, 3600)
    return f"{h:02d}:{divmod(rem, 60)[0]:02d}:{divmod(rem, 60)[1]:02d}"

def print_conditions(sig):
    f = sig["cond_flags"]
    print(f"  1. Sine (5m):  dMin:{sig['dist_to_min']:.1f}% dMax:{sig['dist_to_max']:.1f}% L:{f['sine_long']} S:{f['sine_short']}")
    print(f"  2. Cycle (1m): Low@{sig['ts_min']} High@{sig['ts_max']} | Most Recent: {sig['most_recent_extreme']} | L:{f['cycle_long']} S:{f['cycle_short']}")
    print(f"  3. Mom (1m):   {sig['mom_1m']:.2f} L:{f['mom_long']} S:{f['mom_short']} [MANDATORY]")
    print(f"  4. Vol (1m):   Bull:{sig['bullish_perc']:.1f}% Bear:{sig['bearish_perc']:.1f}% L:{f['vol_long']} S:{f['vol_short']} [MANDATORY]")
    print(f"  5. FFT (1m):   Neg:{sig['neg_ratio']:.2f}% Pos:{sig['pos_ratio']:.2f}% L:{f['fft_long']} S:{f['fft_short']}")
    ml_fc, ml_cur = sig.get("ml_forecast_price", 0.0), sig.get("ml_current_close", 0.0)
    ml_diff = ((ml_fc - ml_cur) / ml_cur * 100) if ml_cur else 0.0
    print(f"  6. ML (1m):    Cur:{ml_cur:.2f} Fcst:{ml_fc:.2f} ({ml_diff:+.4f}%) L:{f['ml_long']} S:{f['ml_short']} [MANDATORY]")
    
    mp = sig.get("method_prices", {})
    cp = sig.get("consensus_price", 0.0)
    tpl = sig.get("tp_long_price", 0.0)
    tps = sig.get("tp_short_price", 0.0)
    
    print(f"  7. TP-Reach:   Tgt-L:{tpl:.2f} Tgt-S:{tps:.2f}")
    print(f"     Forecast-> FFT:{mp.get('fft',0):.2f} RW:{mp.get('random_walk',0):.2f} Forest:{mp.get('forest',0):.2f} LSTM:{mp.get('lstm',0):.2f}")
    
    if cp >= tpl: cons_dir = "🟢 LONG CONSENSUS (Transits TP)"
    elif cp <= tps: cons_dir = "🔴 SHORT CONSENSUS (Transits TP)"
    else: cons_dir = "⚪ NEUTRAL (Fails to Transit TP)"
        
    print(f"     ═══ CONSENSUS PRICE: {cp:.2f} -> {cons_dir}")
    print(f"     L:{f['tp_reach_long']} S:{f['tp_reach_short']} [MANDATORY]")
    
    print(f"  ═══ LONG:{sig['long_true_count']}/5 SHORT:{sig['short_true_count']}/5 (Rule: Mom+Vol+ML+TPReach Mandatory + >=3/5) -> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"HFT KuCoin Bot - 3/5 CONDITIONS (MOM+VOL+ML+TPREACH MANDATORY)")
    print(f"{'='*60}")
    print(f"Symbol:              {TRADE_SYMBOL}")
    print(f"Leverage:            {LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_ROE}% ROE | SL: {STOP_LOSS_ROE}% ROE")
    print(f"Loop Sleep:          {LOOP_SLEEP}s")
    print(f"Entry Logic:         Mom & Vol & ML & TP-Reach MANDATORY + At least 3/5 Total")
    print(f"Position Sizing:     {TRADE_BALANCE_PCT*100:.0f}% of balance per trade")
    print(f"TP Reach Rule:       Consensus Forecast MUST transit TP Target Price")
    print(f"API Connected:       {API_CONNECTED}")
    print(f"{'='*60}\n")

    buffer_5m = CandleBuffer(TRADE_SYMBOL, "5m", ANALYSIS_WINDOW_5M)
    buffer_1m = CandleBuffer(TRADE_SYMBOL, "1m", ANALYSIS_WINDOW_1M)
    
    if API_CONNECTED:
        print("  === INITIALIZING BUFFERS ===\n")
        t0 = time.perf_counter()
        buffer_5m.initialize(client)
        buffer_1m.initialize(client)
        print(f"  Total init: {(time.perf_counter()-t0)*1000:.0f}ms\n")
        if not buffer_5m.is_ready() or not buffer_1m.is_ready():
            print("  [ERROR] Buffer init failed. Exiting."); return
    
    fetcher = ConcurrentDataFetcher(buffer_5m, buffer_1m, TRADE_SYMBOL, max_workers=5)
    saved_state = load_trade_state()
    if saved_state: print(f"  [recovery] State: {saved_state.get('mode')} {saved_state.get('side')} @ {saved_state.get('entry_price', 0):.2f}")
    
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

            do_full = buffer_5m.needs_full_refresh() or buffer_1m.needs_full_refresh()
            data = fetcher.fetch_all_parallel(do_full_refresh=do_full)
            
            cp, bal, pos = data["price"], data["balance"], data["position"]
            has_bal = bal >= MIN_BALANCE_USDT
            p_ms = data["parallel_total_ms"]
            log_timing("parallel_total", p_ms)
            for k, v in data["times"].items(): log_timing(f"task_{k}", v)
            
            if loop_count - last_buffer_log >= 100:
                print()
                print(f"  [Buffers] 5m:{len(buffer_5m)} 1m:{len(buffer_1m)} | Parallel: {p_ms:.0f}ms")
                last_buffer_log = loop_count
            
            t = time.perf_counter()
            sig = compute_signals(buffer_5m, buffer_1m, cp)
            c_ms = (time.perf_counter() - t) * 1000
            log_timing("compute_signals", c_ms)

            if pos["is_open"]:
                p_status = f"LIVE {pos['side'].upper()}"
                c_roe = pos["roe_pct"]
            elif saved_state and saved_state.get("mode") == "SIMULATION":
                p_status = f"SIM {saved_state['side'].upper()}"
                _, _, c_roe = check_sim_tp_sl(saved_state["entry_price"], cp, saved_state["side"])
            else:
                p_status = "FLAT"
                c_roe = 0.0

            trade_tracker.print_stats_line(position_status=p_status, current_roe=c_roe)

            if pos["is_open"]:
                roe, side, ep, mp = pos["roe_pct"], pos["side"], pos["entry_price"], pos["mark_price"]
                
                if loop_count % 10 == 0:
                    print()
                    print(f"[{now_str}] === LIVE {side.upper()} | Entry:{ep:.2f} Mark:{mp:.2f} ROE:{roe:+.2f}% ===")
                    if sig: print_conditions(sig)
                
                reason = None
                if roe >= TAKE_PROFIT_ROE: reason = "TAKE PROFIT"
                elif roe <= STOP_LOSS_ROE: reason = "STOP LOSS"
                    
                if reason:
                    print()
                    print(f"  >>> [{reason}] {roe:+.2f}% ROE - Closing...")
                    client.close_position(TRADE_SYMBOL)
                    pcp = ((mp - ep) / ep) * 100 if side == "long" else ((ep - mp) / ep) * 100
                    sdt = datetime.datetime.strptime(saved_state["entry_time"], "%Y-%m-%d %H:%M:%S")
                    tr = {"mode": "LIVE", "entry_time": saved_state["entry_time"], "exit_time": now_str, "type": side, "entry_price": ep, "exit_price": mp, "price_change_pct": pcp, "gross_roe": pcp * LEVERAGE, "rt_fee_pct": RT_FEE_ROE_PCT, "roe": pcp * LEVERAGE - RT_FEE_ROE_PCT, "duration": format_duration(sdt, datetime.datetime.now()), "reason": reason}
                    write_trade_to_journal(tr)
                    trade_tracker.add_trade(tr)
                    trade_tracker.print_trade_alert(tr)
                    clear_trade_state(); saved_state = None
                
                time.sleep(LOOP_SLEEP); continue

            if saved_state and saved_state.get("mode") == "SIMULATION":
                sep, ss, set_ = saved_state["entry_price"], saved_state["side"], saved_state["entry_time"]
                hit, reason, sroe = check_sim_tp_sl(sep, cp, ss)
                
                if loop_count % 10 == 0:
                    print()
                    print(f"[{now_str}] === SIM {ss.upper()} | Entry:{sep:.2f} Now:{cp:.2f} ROE:{sroe:+.2f}% ===")
                    if sig: print_conditions(sig)
                
                if hit and reason:
                    print()
                    print(f"  >>> [SIM {reason}] {sroe:+.2f}% ROE")
                    pcp = ((cp - sep) / sep) * 100 if ss == "long" else ((sep - cp) / sep) * 100
                    sdt = datetime.datetime.strptime(set_, "%Y-%m-%d %H:%M:%S")
                    tr = {"mode": "SIMULATION", "entry_time": set_, "exit_time": now_str, "type": ss, "entry_price": sep, "exit_price": cp, "price_change_pct": pcp, "gross_roe": pcp * LEVERAGE, "rt_fee_pct": RT_FEE_ROE_PCT, "roe": sroe, "duration": format_duration(sdt, now), "reason": reason}
                    write_trade_to_journal(tr)
                    trade_tracker.add_trade(tr)
                    trade_tracker.print_trade_alert(tr)
                    clear_trade_state(); saved_state = None
                
                time.sleep(LOOP_SLEEP); continue

            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print()
                print("  [recovery] Orphaned live state - clearing."); clear_trade_state(); saved_state = None

            if not pos["is_open"] and not saved_state:
                if not sig: time.sleep(LOOP_SLEEP); continue

                if loop_count % 10 == 0:
                    print()
                    print(f"[{now_str}] Scanning (FLAT) | Calc: {c_ms:.0f}ms")
                    print_conditions(sig)

                if sig["is_long"]:
                    print()
                    print(f"\n  *** 4 MANDATORY MET + 3/5 Total -> LONG ***")
                    if has_bal:
                        print(f"  [LIVE] Executing LONG...")
                        if execute_entry(TRADE_SYMBOL, "buy", bal, sig["price"]):
                            sl, tp = calculate_sl_tp(sig["price"], "long")
                            saved_state = {"active": True, "mode": "LIVE", "side": "long", "entry_price": sig["price"], "sl": sl, "tp": tp, "entry_time": now_str, "consensus_price": sig["consensus_price"]}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIM] LONG (low balance)")
                        sl, tp = calculate_sl_tp(sig["price"], "long")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "long", "entry_price": sig["price"], "sl": sl, "tp": tp, "entry_time": now_str, "consensus_price": sig["consensus_price"]}
                        save_trade_state(saved_state)

                elif sig["is_short"]:
                    print()
                    print(f"\n  *** 4 MANDATORY MET + 3/5 Total -> SHORT ***")
                    if has_bal:
                        print(f"  [LIVE] Executing SHORT...")
                        if execute_entry(TRADE_SYMBOL, "sell", bal, sig["price"]):
                            sl, tp = calculate_sl_tp(sig["price"], "short")
                            saved_state = {"active": True, "mode": "LIVE", "side": "short", "entry_price": sig["price"], "sl": sl, "tp": tp, "entry_time": now_str, "consensus_price": sig["consensus_price"]}
                            save_trade_state(saved_state)
                    else:
                        print(f"  [SIM] SHORT (low balance)")
                        sl, tp = calculate_sl_tp(sig["price"], "short")
                        saved_state = {"active": True, "mode": "SIMULATION", "side": "short", "entry_price": sig["price"], "sl": sl, "tp": tp, "entry_time": now_str, "consensus_price": sig["consensus_price"]}
                        save_trade_state(saved_state)

            loop_ms = (time.perf_counter() - loop_start) * 1000
            log_timing("total_loop", loop_ms)
            
            if time.time() - last_timing_print > 300:
                print()
                print_timing_summary()
                last_timing_print = time.time()
            
            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n\n[Bot] Shutting down safely.")
            print_timing_summary()
            s = trade_tracker.get_stats()
            net_str = f"+{s['net_roe']:.2f}%" if s['net_roe'] > 0 else (f"{s['net_roe']:.2f}%" if s['net_roe'] < 0 else "0.00%")
            print(f"  [Final P&L] Trades: {s['total_trades']} W:{s['wins']} L:{s['losses']} WR:{s['win_rate']:.1f}% | Net ROE: {net_str}")
            fetcher.shutdown()
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()
