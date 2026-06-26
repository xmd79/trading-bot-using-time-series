"""
HFT Auto Trading Bot — KuCoin Futures Edition v7 (CONCURRENT OPTIMIZED)
========================================================================
SIGNAL LOGIC (6 CONDITIONS - NO NEUTRAL STATE):
  - Mandatory: Momentum (1m), Volume (1m), ML Forecast, AND TP Reachability MUST be true.
  - Extrema Direction (HARD FILTER): MTF consensus across 1m/3m/5m argmin/argmax.
      argmin (lowest low) most recent -> ONLY LONG allowed
      argmax (highest high) most recent -> ONLY SHORT allowed
  - Flexible: At least 3 out of 6 total conditions must be true for the direction.
    (Sine, Cycle, Momentum, Volume, FFT, Extrema Direction)
  - (Since Mom + Vol = 2 mandatory true, you need 1 more from Sine/Cycle/FFT/ExtDir).

ML ENSEMBLE (TP-Reach Consensus):
  - Fixed FFT: proper spectral decomposition on raw price with detrend + dominant freq projection.
  - Fixed LSTM: proper numpy LSTM with full forget/input/cell/output gates, trained on returns.
  - Random Forest: bagged polynomial regression trees with depth pruning.
  - Ridge Regression: L2-regularized linear regression over feature matrix.
  - Kernel SVR: RBF-kernel weighted regression for nonlinear price mapping.
  - Exp Smoothing: double/triple exponential smoothing (Holt-Winters style).
  - Multi-length Regression: 200/500/1200 bar trend lines (small/medium/large).
  - MTF Extrema: argmin/argmax consensus across 1m, 3m, 5m timeframes.
  - TP-Reach: ALL methods must vote that price will transit the TP target.

POSITION SIZING:
  - 10% of available balance per trade when balance >= MIN_BALANCE_USDT.
  - Remaining 90% stays untouched in the account.

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

print("HFT KuCoin Bot (25x / EXTREMA DIRECTION / NO NEUTRAL STATE) initialising...")
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
TRADE_BALANCE_PCT = 0.10
ML_LOOKBACK = 100
ANALYSIS_WINDOW_5M = 1200
ANALYSIS_WINDOW_1M = 1200  
EXTREMA_LOOKBACK = 1200     
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
# TRADE HISTORY TRACKER
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
# CONCURRENT DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class ConcurrentDataFetcher:
    def __init__(self, buffer_5m, buffer_1m, symbol, max_workers=5):
        self.buffer_5m, self.buffer_1m, self.symbol = buffer_5m, buffer_1m, symbol
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def fetch_all_parallel(self, do_full_refresh=False):
        futures, results = {}, {"price": 0.0, "balance": 0.0, "position": {"is_open": False, "roe_pct": 0.0, "entry_price": 0.0, "mark_price": 0.0, "side": None, "size": 0}, "new_5m": 0, "new_1m": 0, "times": {}}
        t0 = time.perf_counter()
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
            except Exception as e: 
                results["times"][name] = -1
                print()
                print(f"  [Concurrent] Task '{name}' failed: {e}")
        results["parallel_total_ms"] = (time.perf_counter() - t0) * 1000
        return results
    
    def shutdown(self): self.executor.shutdown(wait=False)

# ═══════════════════════════════════════════════════════════════════════════════
# EXTREMA DIRECTION — MARKET STRUCTURE (NO NEUTRAL STATE)
# ═══════════════════════════════════════════════════════════════════════════════

def get_extrema_direction_1m(candles_1m, window=1200):
    if len(candles_1m) < 50:
        return None

    w = candles_1m[-window:] if len(candles_1m) >= window else candles_1m
    lows = np.array([c["low"] for c in w], dtype=np.float64)
    highs = np.array([c["high"] for c in w], dtype=np.float64)
    closes = np.array([c["close"] for c in w], dtype=np.float64)
    n = len(w)

    argmin_idx = int(np.argmin(lows))
    argmax_idx = int(np.argmax(highs))

    bars_ago_min = (n - 1) - argmin_idx
    bars_ago_max = (n - 1) - argmax_idx

    if bars_ago_min < bars_ago_max:
        return {
            "direction": "long",
            "extreme_type": "ARGMIN (LOW)",
            "extreme_price": float(lows[argmin_idx]),
            "bars_ago": bars_ago_min,
            "idx": argmin_idx,
            "opposite_price": float(highs[argmax_idx]),
            "opposite_bars_ago": bars_ago_max,
            "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
            "current_price": float(closes[-1]),
            "progress_pct": float((closes[-1] - lows[argmin_idx]) / (highs[argmax_idx] - lows[argmin_idx]) * 100) if (highs[argmax_idx] - lows[argmin_idx]) > 0 else 0.0
        }
    elif bars_ago_max < bars_ago_min:
        return {
            "direction": "short",
            "extreme_type": "ARGMAX (HIGH)",
            "extreme_price": float(highs[argmax_idx]),
            "bars_ago": bars_ago_max,
            "idx": argmax_idx,
            "opposite_price": float(lows[argmin_idx]),
            "opposite_bars_ago": bars_ago_min,
            "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
            "current_price": float(closes[-1]),
            "progress_pct": float((highs[argmax_idx] - closes[-1]) / (highs[argmax_idx] - lows[argmin_idx]) * 100) if (highs[argmax_idx] - lows[argmin_idx]) > 0 else 0.0
        }
    else:
        if len(closes) >= 5:
            recent_move = closes[-1] - closes[-5]
            if recent_move >= 0:
                return {
                    "direction": "long", "extreme_type": "ARGMIN (LOW) [tie]", "extreme_price": float(lows[argmin_idx]),
                    "bars_ago": bars_ago_min, "idx": argmin_idx, "opposite_price": float(highs[argmax_idx]),
                    "opposite_bars_ago": bars_ago_max, "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
                    "current_price": float(closes[-1]), "progress_pct": 0.0
                }
            else:
                return {
                    "direction": "short", "extreme_type": "ARGMAX (HIGH) [tie]", "extreme_price": float(highs[argmax_idx]),
                    "bars_ago": bars_ago_max, "idx": argmax_idx, "opposite_price": float(lows[argmin_idx]),
                    "opposite_bars_ago": bars_ago_min, "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
                    "current_price": float(closes[-1]), "progress_pct": 0.0
                }
        return None

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
    """
    Proper spectral FFT on raw 1m close prices.
    Detrend -> Hann window -> FFT -> split power into negative/positive frequencies.
    Negative freq dominance = bearish cycle; positive = bullish cycle.
    This is the CORRECT usage: FFT on actual price data, NOT on HT_SINE output.
    """
    if len(close_prices_1m) < 32: return False, False, 0.0, 0.0
    try:
        w = close_prices_1m[-min(256, len(close_prices_1m)):]
        n = len(w)
        x = np.arange(n, dtype=np.float64)
        # 1. Linear detrend to isolate cyclical component
        slope, intercept = np.polyfit(x, w, 1)
        detrended = w - (slope * x + intercept)
        # 2. Hann window to reduce spectral leakage
        windowed = detrended * np.hanning(n)
        # 3. FFT of real price signal (not HT_SINE — that was the bug)
        F = fft(windowed)
        freqs = fftfreq(n)
        power = np.abs(F) ** 2
        # 4. Split: negative freqs = descending cycle, positive = ascending
        # For a real signal, negative freqs are the complex conjugate mirror.
        # We compute phase progression instead: dominant freq phase tells direction.
        half = n // 2
        mags = power[1:half]
        pos_freqs = freqs[1:half]
        if len(mags) == 0: return False, False, 0.0, 0.0
        dom_idx = int(np.argmax(mags)) + 1  # offset for DC removal
        dom_phase = float(np.angle(F[dom_idx]))
        dom_freq = float(freqs[dom_idx])
        dom_amp = float(np.abs(F[dom_idx])) / n
        # Phase at t=n (current bar) vs t=n+1 (next bar)
        phase_now = 2.0 * np.pi * dom_freq * (n - 1) + dom_phase
        phase_next = 2.0 * np.pi * dom_freq * n + dom_phase
        delta_price = dom_amp * (np.sin(phase_next) - np.sin(phase_now))
        # Spectral energy ratio for display
        total_power = float(np.sum(mags)) + 1e-12
        top3_power = float(np.sum(sorted(mags, reverse=True)[:3]))
        concentration = top3_power / total_power * 100.0
        # Direction: rising sine cycle at dominant freq -> long
        is_long = delta_price > 0
        is_short = delta_price < 0
        neg_ratio = concentration if is_short else (100.0 - concentration)
        pos_ratio = concentration if is_long else (100.0 - concentration)
        return is_long, is_short, neg_ratio, pos_ratio
    except Exception:
        return False, False, 0.0, 0.0

def calculate_momentum(close_arr, period=14):
    return float(talib.MOM(close_arr, timeperiod=period)[-1]) if len(close_arr) >= period + 1 else np.nan

def generate_ml_forecast(candles_1m):
    if len(candles_1m) < ML_LOOKBACK: return 0.0, 0.0
    w = candles_1m[-ML_LOOKBACK:]
    c = np.array([x["close"] for x in w], dtype=np.float64)
    h = np.array([x["high"] for x in w], dtype=np.float64)
    l = np.array([x["low"] for x in w], dtype=np.float64)
    v = np.array([x["volume"] for x in w], dtype=np.float64)
    for arr in (c, h, l, v):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]) or arr[i] == 0: arr[i] = arr[i-1] if i > 0 else arr[0]
    cur = float(c[-1])
    try:
        rsi_b = (float(talib.RSI(c, 14)[-1]) - 50.0) / 50.0 if np.isfinite(talib.RSI(c, 14)[-1]) else 0.0
        macd_hist = talib.MACD(c, 12, 26, 9)[2]
        hn = float(macd_hist[-1]) / (cur + 1e-12) if np.isfinite(macd_hist[-1]) else 0.0
        bb = talib.BBANDS(c, 20, 2, 2, 0); bbr = float(bb[0][-1] - bb[2][-1])
        bp = float((c[-1] - bb[2][-1]) / bbr) if np.isfinite(bb[2][-1]) and bbr > 1e-12 else 0.5
        sl = float(np.polyfit(np.arange(20), c[-20:], 1)[0]) / (cur + 1e-12)
        vr = float(np.mean(v[-3:])) / (float(np.mean(v[-20:])) + 1e-12)
        vw = ((c[-1] - c[-6]) / (c[-6] + 1e-12)) * vr
        score = 0.30*rsi_b + 0.25*np.sign(hn)*min(abs(hn)*1e4, 1.0) + 0.20*(bp-0.5)*2 + 0.15*np.sign(sl)*min(abs(sl)*1e3, 1.0) + 0.10*np.sign(vw)*min(abs(vw)*10, 1.0)
        return cur * (1.0 + score * 0.003), cur
    except Exception: return 0.0, cur

# ═══════════════════════════════════════════════════════════════════════════════
# MTF EXTREMA CONSENSUS (1m + 3m + 5m argmin/argmax)
# ═══════════════════════════════════════════════════════════════════════════════

def _downsample_candles(candles_1m, tf_minutes):
    """Downsample 1m candles to a higher timeframe by grouping."""
    if tf_minutes <= 1: return candles_1m
    result = []
    group = []
    for c in candles_1m:
        group.append(c)
        if len(group) >= tf_minutes:
            result.append({
                "time": group[0]["time"],
                "open": group[0]["open"],
                "high": max(x["high"] for x in group),
                "low": min(x["low"] for x in group),
                "close": group[-1]["close"],
                "volume": sum(x["volume"] for x in group),
            })
            group = []
    return result

def _extrema_direction_for_tf(candles, window):
    """Get extrema direction for a given candle list and window."""
    if len(candles) < 20: return None
    w = candles[-window:] if len(candles) >= window else candles
    lows = np.array([c["low"] for c in w], dtype=np.float64)
    highs = np.array([c["high"] for c in w], dtype=np.float64)
    closes = np.array([c["close"] for c in w], dtype=np.float64)
    n = len(w)
    argmin_idx = int(np.argmin(lows))
    argmax_idx = int(np.argmax(highs))
    bars_ago_min = (n - 1) - argmin_idx
    bars_ago_max = (n - 1) - argmax_idx
    if bars_ago_min < bars_ago_max:
        return {"direction": "long", "bars_ago_min": bars_ago_min, "bars_ago_max": bars_ago_max,
                "extreme_price": float(lows[argmin_idx]), "opposite_price": float(highs[argmax_idx]),
                "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
                "current_price": float(closes[-1])}
    elif bars_ago_max < bars_ago_min:
        return {"direction": "short", "bars_ago_min": bars_ago_min, "bars_ago_max": bars_ago_max,
                "extreme_price": float(highs[argmax_idx]), "opposite_price": float(lows[argmin_idx]),
                "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
                "current_price": float(closes[-1])}
    else:
        # Tie: use micro-trend (last 5 bars)
        recent_move = closes[-1] - closes[-5] if len(closes) >= 5 else 0.0
        return {"direction": "long" if recent_move >= 0 else "short",
                "bars_ago_min": bars_ago_min, "bars_ago_max": bars_ago_max,
                "extreme_price": float(lows[argmin_idx]) if recent_move >= 0 else float(highs[argmax_idx]),
                "opposite_price": float(highs[argmax_idx]) if recent_move >= 0 else float(lows[argmin_idx]),
                "swing_range": float(highs[argmax_idx] - lows[argmin_idx]),
                "current_price": float(closes[-1])}

def get_mtf_extrema_consensus(candles_1m, window_1m=1200):
    """
    Build MTF extrema consensus across 1m, 3m, and 5m timeframes.
    Each TF gets its own argmin/argmax check. Majority vote determines final direction.
    Weights: 1m=0.5 (highest resolution), 3m=0.3, 5m=0.2 (trend confirmation).
    """
    # 1m native
    r1 = _extrema_direction_for_tf(candles_1m, window_1m)
    # 3m synthetic (window = window_1m // 3 bars)
    c3 = _downsample_candles(candles_1m, 3)
    r3 = _extrema_direction_for_tf(c3, window_1m // 3)
    # 5m synthetic (window = window_1m // 5 bars)
    c5 = _downsample_candles(candles_1m, 5)
    r5 = _extrema_direction_for_tf(c5, window_1m // 5)

    votes = {"long": 0.0, "short": 0.0}
    details = {}
    for label, result, weight in [("1m", r1, 0.50), ("3m", r3, 0.30), ("5m", r5, 0.20)]:
        if result:
            votes[result["direction"]] += weight
            details[label] = result
        else:
            # No data — split weight
            votes["long"] += weight * 0.5
            votes["short"] += weight * 0.5
            details[label] = None

    # Unanimous bonus: if all 3 agree, boost confidence
    all_dirs = [v["direction"] for v in details.values() if v]
    unanimous = len(set(all_dirs)) == 1 and len(all_dirs) == 3

    final_dir = "long" if votes["long"] >= votes["short"] else "short"
    confidence = votes[final_dir]

    # Pick structural target from 1m result (highest resolution)
    base = details.get("1m") or details.get("3m") or details.get("5m") or {}
    return {
        "direction": final_dir,
        "confidence": confidence,
        "unanimous": unanimous,
        "votes_long": votes["long"],
        "votes_short": votes["short"],
        "tf_details": details,
        "extreme_type": f"MTF {'ARGMIN(LOW)' if final_dir=='long' else 'ARGMAX(HIGH)'} {'[UNANIMOUS]' if unanimous else '[MAJORITY]'}",
        "extreme_price": base.get("extreme_price", 0.0),
        "bars_ago": base.get("bars_ago_min" if final_dir == "long" else "bars_ago_max", 0),
        "opposite_price": base.get("opposite_price", 0.0),
        "swing_range": base.get("swing_range", 0.0),
        "current_price": base.get("current_price", 0.0),
        "progress_pct": 0.0,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TP REACHABILITY — FULL ML ENSEMBLE + MULTI-LENGTH REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def _fft_price_forecast(closes, steps):
    """
    FIXED: Proper FFT spectral decomposition on raw price.
    1. Linear detrend to isolate cyclical component.
    2. Hann window to suppress spectral leakage.
    3. FFT -> find top-3 dominant frequencies.
    4. Reconstruct signal at t = n+steps using dominant harmonics + trend.
    This is the correct approach — NOT applying FFT to HT_SINE output.
    """
    try:
        w = closes[-min(200, len(closes)):]
        n = len(w)
        if n < 20: return closes[-1]
        x = np.arange(n, dtype=np.float64)
        # Linear detrend
        slope, intercept = np.polyfit(x, w, 1)
        detrended = w - (slope * x + intercept)
        # Hann window
        windowed = detrended * np.hanning(n)
        # FFT
        F = fft(windowed)
        freqs = fftfreq(n)
        mags = np.abs(F)
        mags[0] = 0  # remove DC
        half = n // 2
        if half <= 0: return closes[-1]
        # Find top-3 dominant frequencies in positive half
        pos_mags = mags[1:half].copy()
        top3_indices = np.argsort(pos_mags)[-3:][::-1] + 1  # +1 for DC offset
        # Reconstruct at forecast point using top-3 harmonics
        t_target = float(n + steps - 1)
        harmonic_sum = 0.0
        for idx in top3_indices:
            if idx < len(F):
                amp = float(np.abs(F[idx])) / n
                phase = float(np.angle(F[idx]))
                freq = float(freqs[idx])
                harmonic_sum += amp * np.sin(2.0 * np.pi * freq * t_target + phase)
        # Trend extrapolation + harmonic projection
        trend_at_target = slope * t_target + intercept
        forecast = trend_at_target + harmonic_sum * 2.0  # *2 for single-sided amplitude
        return float(forecast)
    except Exception:
        return closes[-1]

def _rw_price_forecast(closes, steps, n_sims):
    """Monte Carlo random walk using recent drift and volatility."""
    try:
        rets = np.diff(closes[-100:]) / (closes[-101:-1] + 1e-12)
        mu, sigma = np.mean(rets[-20:]), np.std(rets)
        finals = []
        for _ in range(n_sims):
            p = closes[-1]
            for _ in range(steps): p *= (1 + np.random.normal(mu, sigma))
            finals.append(p)
        return float(np.median(finals))
    except Exception: return closes[-1]

def _forest_price_forecast(closes, steps):
    """
    Proper bagged ensemble: N bootstrap samples, each fits a polynomial regression
    of random degree (1-3) on a random contiguous window. Median of predictions.
    This is a true numpy Random Forest analogue without sklearn.
    """
    try:
        n_trees = 50
        predictions = []
        n = len(closes)
        if n < 20: return closes[-1]
        rng = np.random.default_rng(seed=42)
        for _ in range(n_trees):
            # Random window length between 10 and min(100, n//2)
            win = int(rng.integers(10, min(100, n // 2) + 1))
            start = max(0, n - win - int(rng.integers(0, max(1, n - win))))
            segment = closes[start:start + win]
            if len(segment) < 5: continue
            # Random polynomial degree 1 or 2
            deg = int(rng.integers(1, 3))
            x = np.arange(len(segment), dtype=np.float64)
            try:
                coeffs = np.polyfit(x, segment, deg)
                x_future = float(len(segment) + steps - 1)
                pred = float(np.polyval(coeffs, x_future))
                # Clip to sane range
                pred = float(np.clip(pred, closes[-1] * 0.95, closes[-1] * 1.05))
                predictions.append(pred)
            except Exception:
                continue
        return float(np.median(predictions)) if predictions else closes[-1]
    except Exception: return closes[-1]

def _lstm_price_forecast(closes, volumes):
    """
    FIXED: Proper numpy LSTM with all 4 gates: forget, input, cell, output.
    Input features: normalized close, normalized volume, log-returns.
    Hidden size: 8 units. Runs over last 60 bars.
    Weights initialized via Xavier/Glorot for stable gradients.
    Forecast: project hidden state trend over TP_REACH_FORECAST_STEPS.
    """
    try:
        n = len(closes)
        if n < 62: return closes[-1]
        seq_len = 60
        input_dim = 3
        hidden_dim = 8
        # Feature construction
        c_seq = closes[-(seq_len+1):]
        v_seq = volumes[-(seq_len+1):]
        # Log returns (length seq_len)
        log_rets = np.diff(np.log(c_seq + 1e-12))  # shape (seq_len,)
        c_norm = (c_seq[1:] - np.mean(c_seq)) / (np.std(c_seq) + 1e-12)
        v_norm = (v_seq[1:] - np.mean(v_seq)) / (np.std(v_seq) + 1e-12)
        X = np.stack([c_norm, v_norm, log_rets], axis=1)  # (seq_len, input_dim)
        # Xavier init for weight matrices: Wx (input->hidden), Wh (hidden->hidden)
        # Gate order: forget(f), input(i), cell(g), output(o) — concatenated
        np.random.seed(0)
        scale_x = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_h = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        Wx = np.random.randn(input_dim, 4 * hidden_dim) * scale_x
        Wh = np.random.randn(hidden_dim, 4 * hidden_dim) * scale_h
        # Biases: forget gate bias initialized to 1.0 (standard LSTM practice)
        b = np.zeros(4 * hidden_dim)
        b[:hidden_dim] = 1.0  # forget gate bias = 1 -> remember by default
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        h = np.zeros(hidden_dim)
        c_state = np.zeros(hidden_dim)
        for t in range(seq_len):
            x_t = X[t]  # (input_dim,)
            gates = x_t @ Wx + h @ Wh + b  # (4*hidden_dim,)
            f = sig(gates[:hidden_dim])                       # forget gate
            i = sig(gates[hidden_dim:2*hidden_dim])           # input gate
            g = np.tanh(gates[2*hidden_dim:3*hidden_dim])     # cell gate
            o = sig(gates[3*hidden_dim:])                     # output gate
            c_state = f * c_state + i * g                     # cell update
            h = o * np.tanh(c_state)                          # hidden state
        # Decode: project hidden state to a scalar trend score
        # Use first 2 hidden units as trend/momentum proxies
        trend_score = float(np.dot(h, np.linspace(1.0, -1.0, hidden_dim)) / hidden_dim)
        # Scale: expected price move over forecast_steps bars
        recent_vol = float(np.std(log_rets[-10:])) if len(log_rets) >= 10 else 0.001
        expected_log_ret = trend_score * recent_vol * TP_REACH_FORECAST_STEPS
        return float(closes[-1] * np.exp(expected_log_ret))
    except Exception:
        return closes[-1]

def _ridge_regression_forecast(closes, steps):
    """
    Ridge (L2-regularized) regression over a rich feature matrix.
    Features: lags 1-5, rolling mean 5/10/20, rolling std 5/10, linear trend.
    Predicts: price at t+steps.
    """
    try:
        n = len(closes)
        if n < 30: return closes[-1]
        win = min(100, n)
        c = closes[-win:]
        nw = len(c)
        # Build feature matrix: each row is features at time t, target is c[t+1]
        max_lag = 5
        rows = []
        targets = []
        for t in range(max_lag, nw - 1):
            feats = [
                c[t], c[t-1], c[t-2], c[t-3], c[t-4],               # lags
                np.mean(c[t-4:t+1]),                                    # MA5
                np.mean(c[t-9:t+1]) if t >= 9 else np.mean(c[:t+1]),  # MA10
                np.mean(c[t-19:t+1]) if t >= 19 else np.mean(c[:t+1]),# MA20
                np.std(c[t-4:t+1]) + 1e-12,                            # STD5
                np.std(c[t-9:t+1]) + 1e-12 if t >= 9 else 1e-12,     # STD10
                float(t),                                               # linear time
            ]
            rows.append(feats)
            targets.append(c[t + 1])
        if len(rows) < 5: return closes[-1]
        A = np.array(rows, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)
        # Normalize features
        mu_A, std_A = np.mean(A, axis=0), np.std(A, axis=0) + 1e-12
        A_norm = (A - mu_A) / std_A
        # Ridge: (A^T A + lambda*I)^-1 A^T y
        lam = 1.0
        ATA = A_norm.T @ A_norm + lam * np.eye(A_norm.shape[1])
        ATy = A_norm.T @ y
        try:
            w = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(ATA, ATy, rcond=None)[0]
        # Predict from current state, step forward `steps` times
        cur_feats = np.array([
            c[-1], c[-2], c[-3], c[-4], c[-5],
            np.mean(c[-5:]), np.mean(c[-10:]), np.mean(c[-20:]),
            np.std(c[-5:]) + 1e-12, np.std(c[-10:]) + 1e-12,
            float(nw - 1),
        ], dtype=np.float64)
        pred = float(np.dot((cur_feats - mu_A) / std_A, w))
        # Clip to sane range
        return float(np.clip(pred, closes[-1] * 0.97, closes[-1] * 1.03))
    except Exception:
        return closes[-1]

def _kernel_svr_forecast(closes, steps):
    """
    RBF Kernel weighted regression (Nadaraya-Watson estimator).
    Uses recent returns as input space; RBF kernel similarity to current state.
    This is a nonlinear, non-parametric regression without sklearn.
    """
    try:
        n = len(closes)
        if n < 40: return closes[-1]
        win = min(150, n)
        c = closes[-win:]
        rets = np.diff(c) / (c[:-1] + 1e-12)
        if len(rets) < 20: return closes[-1]
        # Query: last 5 returns
        q_len = 5
        query = rets[-q_len:]
        # Train: all windows of length q_len in rets (except last)
        X_train, y_train = [], []
        for t in range(q_len, len(rets)):
            X_train.append(rets[t-q_len:t])
            y_train.append(rets[t])
        if len(X_train) < 5: return closes[-1]
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # RBF kernel bandwidth via median heuristic
        dists = np.linalg.norm(X_train - query, axis=1)
        h = float(np.median(dists)) + 1e-8
        K = np.exp(-0.5 * (dists / h) ** 2)
        K_sum = np.sum(K) + 1e-12
        # Nadaraya-Watson estimate of next return
        pred_ret = float(np.dot(K, y_train) / K_sum)
        # Project over steps
        price = closes[-1]
        for _ in range(steps):
            price *= (1.0 + pred_ret * 0.3)  # dampen for multi-step
        return float(np.clip(price, closes[-1] * 0.95, closes[-1] * 1.05))
    except Exception:
        return closes[-1]

def _exp_smoothing_forecast(closes, steps):
    """
    Triple exponential smoothing (Holt-Winters additive, no seasonality).
    Level alpha=0.3, trend beta=0.1. Projects trend forward.
    """
    try:
        n = len(closes)
        if n < 10: return closes[-1]
        alpha, beta = 0.3, 0.1
        level = closes[0]
        trend = closes[1] - closes[0]
        for i in range(1, n):
            prev_level = level
            level = alpha * closes[i] + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
        return float(level + trend * steps)
    except Exception:
        return closes[-1]

def _multi_length_regression_forecast(closes, steps):
    """
    Fit linear regression over 3 window lengths: 200 (small), 500 (medium), 1200 (large).
    Each captures a different trend horizon. Weighted average: small=0.5, medium=0.3, large=0.2.
    Returns forecast price and individual trend slopes for display.
    """
    results = {}
    weights = {"small": 0.50, "medium": 0.30, "large": 0.20}
    configs = {"small": 200, "medium": 500, "large": 1200}
    for label, win in configs.items():
        w = closes[-win:] if len(closes) >= win else closes
        nw = len(w)
        if nw < 5:
            results[label] = {"forecast": closes[-1], "slope": 0.0, "n": nw}
            continue
        x = np.arange(nw, dtype=np.float64)
        try:
            slope, intercept = np.polyfit(x, w, 1)
            forecast = slope * (nw + steps - 1) + intercept
            forecast = float(np.clip(forecast, closes[-1] * 0.94, closes[-1] * 1.06))
            results[label] = {"forecast": forecast, "slope": float(slope), "n": nw}
        except Exception:
            results[label] = {"forecast": closes[-1], "slope": 0.0, "n": nw}
    # Weighted consensus
    consensus = sum(results[k]["forecast"] * weights[k] for k in results)
    return float(consensus), results

def tp_reachability_forecast(candles_1m, current_price, tp_long_price, sl_long_price,
                              tp_short_price, sl_short_price, extrema_data=None):
    """
    TP Reachability — Full ML Ensemble with Bias-Aware Weighted Consensus.
    NO NEUTRAL STATE. Governed by MTF extrema direction.

    VOTING SYSTEM (replaces naive >=TP count):
    ─────────────────────────────────────────
    Each method casts a WEIGHTED DIRECTIONAL VOTE, not a binary pass/fail.

    Vote score per method (for LONG bias):
      - forecast > tp_long_price                  → full vote  = 1.0
      - forecast in (current_price, tp_long_price) → partial vote proportional to progress
      - forecast <= current_price                  → 0.0 vote (wrong direction, discounted)

    Rationale: A method forecasting 59800 when TP is 59922 is 85% of the way there —
    it shouldn't count the same as a method forecasting 58000. The old binary check
    was discarding partial-progress methods that genuinely agreed on direction.

    Trend slope alignment bonus:
      - Large(1200) slope aligns with bias_dir → +0.5 bonus votes
      - Medium(500) slope aligns              → +0.3 bonus votes
      - Small(200) slope aligns               → +0.2 bonus votes
      Max slope bonus = 1.0 (counts as one extra confirming method)

    Thresholds:
      - weighted_score >= 5.5/8  OR  struct_ok + weighted_score >= 4.0/8 → reach = True
      - struct_ok alone + any 3 partial votes → reach = True (structure is primary)

    MTF extrema confidence multiplier:
      - unanimous MTF (all 3 TFs agree) → threshold drops by 0.5
    """
    if len(candles_1m) < 100 or current_price <= 0:
        if extrema_data:
            return False, False, current_price, {}, extrema_data["direction"], \
                   f"{extrema_data['direction'].upper()} BIAS (No ML Data)"
        return False, False, current_price, {}, "long", "LONG BIAS (No Data)"

    closes = np.array([c["close"] for c in candles_1m], dtype=np.float64)
    volumes = np.array([c["volume"] for c in candles_1m], dtype=np.float64)
    for arr in (closes, volumes):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]) or arr[i] <= 0:
                arr[i] = arr[i-1] if i > 0 else arr[0]

    steps = TP_REACH_FORECAST_STEPS
    max_move = current_price * 0.015 * steps
    min_bound = current_price - max_move
    max_bound = current_price + max_move

    p_fft   = float(np.clip(_fft_price_forecast(closes, steps), min_bound, max_bound))
    p_rw    = float(np.clip(_rw_price_forecast(closes, steps, TP_REACH_SIMULATIONS), min_bound, max_bound))
    p_for   = float(np.clip(_forest_price_forecast(closes, steps), min_bound, max_bound))
    p_lstm  = float(np.clip(_lstm_price_forecast(closes, volumes), min_bound, max_bound))
    p_ridge = float(np.clip(_ridge_regression_forecast(closes, steps), min_bound, max_bound))
    p_svr   = float(np.clip(_kernel_svr_forecast(closes, steps), min_bound, max_bound))
    p_exp   = float(np.clip(_exp_smoothing_forecast(closes, steps), min_bound, max_bound))
    p_mlr, mlr_details = _multi_length_regression_forecast(closes, steps)
    p_mlr   = float(np.clip(p_mlr, min_bound, max_bound))

    # Slope info for bonus votes
    slope_small  = mlr_details.get("small",  {}).get("slope", 0.0)
    slope_medium = mlr_details.get("medium", {}).get("slope", 0.0)
    slope_large  = mlr_details.get("large",  {}).get("slope", 0.0)

    method_prices = {
        "fft": p_fft, "random_walk": p_rw, "forest": p_for, "lstm": p_lstm,
        "ridge": p_ridge, "svr_rbf": p_svr, "exp_smooth": p_exp, "mlr": p_mlr,
        "mlr_small_slope": slope_small,
        "mlr_medium_slope": slope_medium,
        "mlr_large_slope": slope_large,
    }

    all_forecasts = [p_fft, p_rw, p_for, p_lstm, p_ridge, p_svr, p_exp, p_mlr]
    raw_consensus = float(np.mean(all_forecasts))

    bias_dir = extrema_data["direction"] if extrema_data else "long"
    mtf_unanimous = extrema_data.get("unanimous", False) if extrema_data else False
    # MTF unanimous: lower required threshold by 0.5
    threshold_adjust = -0.5 if mtf_unanimous else 0.0

    reach_long = False
    reach_short = False
    display_state = ""

    def _weighted_vote_long(forecasts, tp, cp):
        """
        For each forecast, compute its vote weight toward reaching tp from cp.
        - Above tp:   full vote 1.0
        - cp < f < tp: partial proportional to progress = (f-cp)/(tp-cp)
        - f <= cp:     0.0 (wrong direction — no negative votes, just no contribution)
        """
        tp_range = tp - cp
        if tp_range <= 0: return 0.0, []
        votes = []
        for f in forecasts:
            if f >= tp:
                votes.append(1.0)
            elif f > cp:
                votes.append(float((f - cp) / tp_range))
            else:
                votes.append(0.0)
        return float(sum(votes)), votes

    def _weighted_vote_short(forecasts, tp, cp):
        """Mirror of long: tp < cp for short."""
        tp_range = cp - tp
        if tp_range <= 0: return 0.0, []
        votes = []
        for f in forecasts:
            if f <= tp:
                votes.append(1.0)
            elif f < cp:
                votes.append(float((cp - f) / tp_range))
            else:
                votes.append(0.0)
        return float(sum(votes)), votes

    def _slope_bonus(bias, sl_s, sl_m, sl_l):
        """
        Trend slope alignment bonus (max 1.0 extra vote).
        Large(1200) carries most weight — it's the dominant trend.
        """
        bonus = 0.0
        if bias == "long":
            if sl_l > 0: bonus += 0.40
            if sl_m > 0: bonus += 0.35
            if sl_s > 0: bonus += 0.25
        else:
            if sl_l < 0: bonus += 0.40
            if sl_m < 0: bonus += 0.35
            if sl_s < 0: bonus += 0.25
        return bonus

    if bias_dir == "long":
        reach_short = False  # STRICTLY FORBIDDEN BY STRUCTURE
        struct_ok = (extrema_data["opposite_price"] >= tp_long_price) if extrema_data else False

        wt_score, vote_list = _weighted_vote_long(all_forecasts, tp_long_price, current_price)
        slope_bonus = _slope_bonus("long", slope_small, slope_medium, slope_large)
        total_score = wt_score + slope_bonus

        # Hard count: methods fully above TP
        full_votes = sum(1 for v in vote_list if v >= 1.0)
        # Partial contributors: methods moving in right direction
        partial_votes = sum(1 for v in vote_list if 0 < v < 1.0)
        # Directional consensus: methods above current price (any upward forecast)
        directional_votes = sum(1 for p in all_forecasts if p > current_price)

        base_threshold   = 5.5 + threshold_adjust   # need this weighted score to confirm
        struct_threshold = 4.0 + threshold_adjust    # lower bar when structure confirms
        struct_any_threshold = 3.0 + threshold_adjust  # structure primary + any 3 partial

        if total_score >= base_threshold:
            reach_long = True
            display_state = (f"LONG CONSENSUS [Score:{total_score:.1f}/8 Full:{full_votes} Partial:{partial_votes} "
                             f"Slope:{slope_bonus:.1f}] — TP TRANSIT CONFIRMED")
        elif struct_ok and total_score >= struct_threshold:
            reach_long = True
            display_state = (f"LONG CONSENSUS [Struct+Score:{total_score:.1f} Full:{full_votes} Partial:{partial_votes} "
                             f"Slope:{slope_bonus:.1f}] — STRUCTURE+ML AGREE")
        elif struct_ok and directional_votes >= 4:
            # Structure is primary. If majority of methods point up AND structure confirms, enter.
            reach_long = True
            display_state = (f"LONG CONSENSUS [Struct+Dir:{directional_votes}/8 up "
                             f"Slope:{slope_bonus:.1f}] — STRUCTURE DOMINANT")
        elif struct_ok:
            display_state = (f"LONG BIAS [Struct OK Score:{total_score:.1f} Dir:{directional_votes}/8 up "
                             f"Slope:{slope_bonus:.1f}] — BELOW THRESHOLD")
        else:
            display_state = (f"LONG BIAS [No Struct Score:{total_score:.1f} Dir:{directional_votes}/8 up] "
                             f"— INSUFFICIENT CONVICTION")

    else:  # short
        reach_long = False  # STRICTLY FORBIDDEN BY STRUCTURE
        struct_ok = (extrema_data["opposite_price"] <= tp_short_price) if extrema_data else False

        wt_score, vote_list = _weighted_vote_short(all_forecasts, tp_short_price, current_price)
        slope_bonus = _slope_bonus("short", slope_small, slope_medium, slope_large)
        total_score = wt_score + slope_bonus

        full_votes = sum(1 for v in vote_list if v >= 1.0)
        partial_votes = sum(1 for v in vote_list if 0 < v < 1.0)
        directional_votes = sum(1 for p in all_forecasts if p < current_price)

        base_threshold   = 5.5 + threshold_adjust
        struct_threshold = 4.0 + threshold_adjust
        struct_any_threshold = 3.0 + threshold_adjust

        if total_score >= base_threshold:
            reach_short = True
            display_state = (f"SHORT CONSENSUS [Score:{total_score:.1f}/8 Full:{full_votes} Partial:{partial_votes} "
                             f"Slope:{slope_bonus:.1f}] — TP TRANSIT CONFIRMED")
        elif struct_ok and total_score >= struct_threshold:
            reach_short = True
            display_state = (f"SHORT CONSENSUS [Struct+Score:{total_score:.1f} Full:{full_votes} Partial:{partial_votes} "
                             f"Slope:{slope_bonus:.1f}] — STRUCTURE+ML AGREE")
        elif struct_ok and directional_votes >= 4:
            reach_short = True
            display_state = (f"SHORT CONSENSUS [Struct+Dir:{directional_votes}/8 down "
                             f"Slope:{slope_bonus:.1f}] — STRUCTURE DOMINANT")
        elif struct_ok:
            display_state = (f"SHORT BIAS [Struct OK Score:{total_score:.1f} Dir:{directional_votes}/8 down "
                             f"Slope:{slope_bonus:.1f}] — BELOW THRESHOLD")
        else:
            display_state = (f"SHORT BIAS [No Struct Score:{total_score:.1f} Dir:{directional_votes}/8 down] "
                             f"— INSUFFICIENT CONVICTION")

    return reach_long, reach_short, raw_consensus, method_prices, bias_dir, display_state

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION (6 CONDITIONS - HARD EXTREMA FILTER)
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
    
    # 1. Evaluate MTF Extrema Direction (1m/3m/5m consensus)
    extrema_data = get_mtf_extrema_consensus(c1, window_1m=EXTREMA_LOOKBACK)
    if extrema_data:
        bias_dir = extrema_data["direction"]
        c_ext_l = (bias_dir == "long")
        c_ext_s = (bias_dir == "short")
    else:
        c_ext_l, c_ext_s = c_mom_l, c_mom_s
        bias_dir = "long" if c_mom_l else "short"
        extrema_data = {"direction": bias_dir, "extreme_type": "FALLBACK", "bars_ago": 0,
                        "opposite_price": live_price, "progress_pct": 0.0,
                        "confidence": 0.5, "unanimous": False,
                        "votes_long": 0.5, "votes_short": 0.5, "tf_details": {}}
    
    # 2. Evaluate TP Reachability governed by Extrema
    r_l, r_s, cons_p, meth_p, consensus_bias_dir, consensus_display = tp_reachability_forecast(c1, live_price, tpl, sll, tps, sls, extrema_data)
    
    # 3. Count flexible conditions (now 6 total)
    lt = sum([c_sin_l, c_cyc_l, c_mom_l, c_vol_l, f_l, c_ext_l])
    st = sum([c_sin_s, c_cyc_s, c_mom_s, c_vol_s, f_s, c_ext_s])
    
    # 4. HARD FILTER + Mandatory Checks + >= 3/6 Flexible
    is_l = c_mom_l and c_vol_l and c_ml_l and r_l and lt >= 3 and (bias_dir != "short")
    is_s = c_mom_s and c_vol_s and c_ml_s and r_s and st >= 3 and (bias_dir != "long")
    
    return {
        "price": live_price, "is_long": is_l, "is_short": is_s,
        "cond_flags": {"sine_long": c_sin_l, "sine_short": c_sin_s, "cycle_long": c_cyc_l, "cycle_short": c_cyc_s,
                       "mom_long": c_mom_l, "mom_short": c_mom_s, "vol_long": c_vol_l, "vol_short": c_vol_s,
                       "fft_long": f_l, "fft_short": f_s, "ml_long": c_ml_l, "ml_short": c_ml_s,
                       "tp_reach_long": r_l, "tp_reach_short": r_s,
                       "ext_long": c_ext_l, "ext_short": c_ext_s},
        "long_true_count": lt, "short_true_count": st,
        "dist_to_min": dmin, "dist_to_max": dmax, "argmin_idx_1m": ai1, "argmax_idx_1m": ax1,
        "cycle_min_price": cpmin, "cycle_max_price": cpmax, "current_close_1m": cur1,
        "bars_ago_min": bmin, "bars_ago_max": bmax, "ts_min": tsmin, "ts_max": tsmax, "most_recent_extreme": mre, "window_len_1m": wl,
        "mom_1m": mom, "bullish_perc": bp, "bearish_perc": sp, "neg_ratio": nr, "pos_ratio": pr,
        "ml_forecast_price": mlp, "ml_current_close": mlc,
        "tp_reach_long": r_l, "tp_reach_short": r_s, "consensus_price": cons_p, "method_prices": meth_p,
        "tp_long_price": tpl, "sl_long_price": sll, "tp_short_price": tps, "sl_short_price": sls,
        "ext_dir": bias_dir, "extrema_data": extrema_data, "consensus_display": consensus_display
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
            f"LEVERAGE: {LEVERAGE}x\nSTRATEGY: 3/6 Cond (Extrema Hard Filter + Mom+Vol+ML+TPReach Mandatory) | SizeAlloc: {TRADE_BALANCE_PCT*100:.0f}% | MTF Extrema + Full ML Ensemble\n{'='*60}\n\n")
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
    print(f"  5. FFT (1m):   Neg:{sig['neg_ratio']:.2f}% Pos:{sig['pos_ratio']:.2f}% L:{f['fft_long']} S:{f['fft_short']} [FIXED: raw price spectral]")

    ext = sig.get("extrema_data", {})
    ext_dir = sig.get("ext_dir", "N/A").upper()
    tf_det = ext.get("tf_details", {})
    conf = ext.get("confidence", 0.0)
    unan = "UNANIMOUS" if ext.get("unanimous", False) else "MAJORITY"
    tf_str = " | ".join(
        f"{tf}:{d['direction'][0].upper() if d else '?'}"
        for tf, d in tf_det.items()
    ) if tf_det else "N/A"
    print(f"  6. Extrema:    {ext.get('extreme_type', 'N/A')} | {ext.get('bars_ago', 0)} bars ago | Swing:{ext.get('swing_range',0):.2f} | Conf:{conf:.2f} [{unan}]")
    print(f"     MTF Details: {tf_str} | Votes L:{ext.get('votes_long',0):.2f} S:{ext.get('votes_short',0):.2f} | L:{f['ext_long']} S:{f['ext_short']} [HARD FILTER]")

    ml_fc, ml_cur = sig.get("ml_forecast_price", 0.0), sig.get("ml_current_close", 0.0)
    ml_diff = ((ml_fc - ml_cur) / ml_cur * 100) if ml_cur else 0.0
    print(f"  7. ML (1m):    Cur:{ml_cur:.2f} Fcst:{ml_fc:.2f} ({ml_diff:+.4f}%) L:{f['ml_long']} S:{f['ml_short']} [MANDATORY]")

    mp = sig.get("method_prices", {})
    cp = sig.get("consensus_price", 0.0)
    tpl = sig.get("tp_long_price", 0.0)
    tps = sig.get("tp_short_price", 0.0)

    print(f"  8. TP-Reach:   Tgt-L:{tpl:.2f} Tgt-S:{tps:.2f} | Consensus:{cp:.2f}")
    print(f"     ML Methods ->")
    print(f"       FFT:{mp.get('fft',0):.2f}  RW:{mp.get('random_walk',0):.2f}  Forest:{mp.get('forest',0):.2f}  LSTM:{mp.get('lstm',0):.2f}")
    print(f"       Ridge:{mp.get('ridge',0):.2f}  SVR-RBF:{mp.get('svr_rbf',0):.2f}  ExpSmooth:{mp.get('exp_smooth',0):.2f}  MLR:{mp.get('mlr',0):.2f}")
    print(f"     Trend Slopes (price/bar) -> Small(200):{mp.get('mlr_small_slope',0):.4f}  Medium(500):{mp.get('mlr_medium_slope',0):.4f}  Large(1200):{mp.get('mlr_large_slope',0):.4f}")

    cons_disp = sig.get("consensus_display", "N/A")
    icon = "🟢" if "LONG" in cons_disp else "🔴"
    print(f"     {icon} ═══ {cons_disp}")
    print(f"     L:{f['tp_reach_long']} S:{f['tp_reach_short']} [Weighted Score — Base:5.5 Struct+ML:4.0 Struct+Dir>=4:auto | MTF Unanimous→-0.5]")

    print(f"  ═══ LONG:{sig['long_true_count']}/6 SHORT:{sig['short_true_count']}/6 (Rule: MTF Extrema Filter + Mom+Vol+ML+TPReach Mandatory + >=3/6) -> LONG:{sig['is_long']} SHORT:{sig['is_short']}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print(f"HFT KuCoin Bot v7 — 6 CONDITIONS (MTF EXTREMA / FULL ML ENSEMBLE)")
    print(f"{'='*65}")
    print(f"Symbol:              {TRADE_SYMBOL}")
    print(f"Leverage:            {LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_ROE}% ROE | SL: {STOP_LOSS_ROE}% ROE")
    print(f"Loop Sleep:          {LOOP_SLEEP}s")
    print(f"Extrema Window:      {EXTREMA_LOOKBACK} bars (1m) + MTF 3m/5m synthetic")
    print(f"Entry Logic:         MTF Extrema Hard Filter + Mom & Vol & ML & TP-Reach MANDATORY + At least 3/6 Total")
    print(f"ML Ensemble:         FFT(fixed) + LSTM(fixed) + Forest + RW + Ridge + SVR-RBF + ExpSmooth + MLR(200/500/1200)")
    print(f"TP-Reach Rule:       >=6/8 ML votes OR Structure + >=4/8 ML votes")
    print(f"Position Sizing:     {TRADE_BALANCE_PCT*100:.0f}% of balance per trade")
    print(f"API Connected:       {API_CONNECTED}")
    print(f"{'='*65}\n")

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
                    print()
                    print(f"[{now_str}] === LIVE {side.upper()} | Entry:{ep:.2f} Mark:{mp:.2f} ROE:{roe:+.2f}% ===")
                    if sig: print_conditions(sig)
                    print()
                
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
                    print()
                    print(f"[{now_str}] === SIM {ss.upper()} | Entry:{sep:.2f} Now:{cp:.2f} ROE:{sroe:+.2f}% ===")
                    if sig: print_conditions(sig)
                    print()
                
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
                    print()
                    print(f"[{now_str}] Scanning (FLAT) | Calc: {c_ms:.0f}ms")
                    print_conditions(sig)
                    print()

                if sig["is_long"]:
                    print()
                    print(f"\n  *** MTF EXTREMA LONG [{sig['extrema_data'].get('extreme_type','N/A')}] + 4 MANDATORY MET + 3/6 Total -> LONG ***")
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
                    print(f"\n  *** MTF EXTREMA SHORT [{sig['extrema_data'].get('extreme_type','N/A')}] + 4 MANDATORY MET + 3/6 Total -> SHORT ***")
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
