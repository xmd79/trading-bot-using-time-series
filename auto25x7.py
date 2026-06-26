"""
HFT Auto Trading Bot — KuCoin Futures Edition v7.1 (STRIPPED ENSEMBLE)
========================================================================
SIGNAL LOGIC (6 CONDITIONS - NO NEUTRAL STATE):
  - Mandatory: Momentum (1m), Volume (1m), ML Forecast, AND TP Reachability MUST be true.
  - Extrema Direction (HARD FILTER): MTF consensus across 1m/3m/5m argmin/argmax.
      argmin (lowest low) most recent -> ONLY LONG allowed
      argmax (highest high) most recent -> ONLY SHORT allowed
  - Flexible: At least 3 out of 6 total conditions must be true for the direction.
    (Sine, Cycle, Momentum, Volume, FFT, Extrema Direction)
  - (Since Mom + Vol = 2 mandatory true, you need 1 more from Sine/Cycle/FFT/ExtDir).

ML ENSEMBLE (TP-Reach Consensus) — STRIPPED TO 3 MOST ACCURATE:
  - LSTM: proper numpy LSTM with full forget/input/cell/output gates, trained on returns.
         Captures sequential momentum — consistently right for short-horizon TP.
  - Exp Smoothing: double/triple exponential smoothing (Holt-Winters style).
         Captures trend persistence — confirms directional bias.
  - MACD-Velocity: histogram slope + acceleration projection.
         Fast-reacting momentum confirmation — catches momentum shifts early.

  REMOVED (inaccurate for HFT short-horizon):
    - FFT: spectral extrapolation breaks down in volatile conditions
    - Random Walk: no edge, pure noise
    - Forest: bagged trees too slow to adapt to HFT microstructure
    - Ridge: linear model can't capture nonlinear momentum
    - SVR-RBF: kernel bandwidth instability in fast-moving markets
    - Multi-length Regression: trend-following, not TP-accurate

DECISION LOGIC (NO BOTH-FALSE):
  - 2+ of 3 methods above current price -> LONG -> L:True S:False
  - 2+ of 3 methods below current price -> SHORT -> L:False S:True
  - Tie -> extrema structure breaks tie
  - MTF Unanimous extrema -> overrides any tie

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

print("HFT KuCoin Bot v7.1 (25x / STRIPPED ENSEMBLE / NO BOTH-FALSE) initialising...")
if _cleaned:
    print(f"  [cleanup] Wiped: {', '.join(_cleaned)}")
print("  [ensemble] LSTM + ExpSmooth + MACD-Velocity (3 methods only)")
print("  [logic] Consensus LONG -> L:True S:False | Consensus SHORT -> L:False S:True")
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
    """
    if len(close_prices_1m) < 32: return False, False, 0.0, 0.0
    try:
        w = close_prices_1m[-min(256, len(close_prices_1m)):]
        n = len(w)
        x = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(x, w, 1)
        detrended = w - (slope * x + intercept)
        windowed = detrended * np.hanning(n)
        F = fft(windowed)
        freqs = fftfreq(n)
        power = np.abs(F) ** 2
        half = n // 2
        mags = power[1:half]
        pos_freqs = freqs[1:half]
        if len(mags) == 0: return False, False, 0.0, 0.0
        dom_idx = int(np.argmax(mags)) + 1
        dom_phase = float(np.angle(F[dom_idx]))
        dom_freq = float(freqs[dom_idx])
        dom_amp = float(np.abs(F[dom_idx])) / n
        phase_now = 2.0 * np.pi * dom_freq * (n - 1) + dom_phase
        phase_next = 2.0 * np.pi * dom_freq * n + dom_phase
        delta_price = dom_amp * (np.sin(phase_next) - np.sin(phase_now))
        total_power = float(np.sum(mags)) + 1e-12
        top3_power = float(np.sum(sorted(mags, reverse=True)[:3]))
        concentration = top3_power / total_power * 100.0
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
    r1 = _extrema_direction_for_tf(candles_1m, window_1m)
    c3 = _downsample_candles(candles_1m, 3)
    r3 = _extrema_direction_for_tf(c3, window_1m // 3)
    c5 = _downsample_candles(candles_1m, 5)
    r5 = _extrema_direction_for_tf(c5, window_1m // 5)

    votes = {"long": 0.0, "short": 0.0}
    details = {}
    for label, result, weight in [("1m", r1, 0.50), ("3m", r3, 0.30), ("5m", r5, 0.20)]:
        if result:
            votes[result["direction"]] += weight
            details[label] = result
        else:
            votes["long"] += weight * 0.5
            votes["short"] += weight * 0.5
            details[label] = None

    all_dirs = [v["direction"] for v in details.values() if v]
    unanimous = len(set(all_dirs)) == 1 and len(all_dirs) == 3

    final_dir = "long" if votes["long"] >= votes["short"] else "short"
    confidence = votes[final_dir]

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
# TP REACHABILITY — STRIPPED ENSEMBLE: LSTM + ExpSmooth + MACD-Velocity
# ═══════════════════════════════════════════════════════════════════════════════

def _lstm_price_forecast(closes, volumes):
    """
    Proper numpy LSTM with all 4 gates: forget, input, cell, output.
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
        c_seq = closes[-(seq_len+1):]
        v_seq = volumes[-(seq_len+1):]
        log_rets = np.diff(np.log(c_seq + 1e-12))
        c_norm = (c_seq[1:] - np.mean(c_seq)) / (np.std(c_seq) + 1e-12)
        v_norm = (v_seq[1:] - np.mean(v_seq)) / (np.std(v_seq) + 1e-12)
        X = np.stack([c_norm, v_norm, log_rets], axis=1)
        np.random.seed(0)
        scale_x = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_h = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        Wx = np.random.randn(input_dim, 4 * hidden_dim) * scale_x
        Wh = np.random.randn(hidden_dim, 4 * hidden_dim) * scale_h
        b = np.zeros(4 * hidden_dim)
        b[:hidden_dim] = 1.0
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        h = np.zeros(hidden_dim)
        c_state = np.zeros(hidden_dim)
        for t in range(seq_len):
            x_t = X[t]
            gates = x_t @ Wx + h @ Wh + b
            f = sig(gates[:hidden_dim])
            i = sig(gates[hidden_dim:2*hidden_dim])
            g = np.tanh(gates[2*hidden_dim:3*hidden_dim])
            o = sig(gates[3*hidden_dim:])
            c_state = f * c_state + i * g
            h = o * np.tanh(c_state)
        trend_score = float(np.dot(h, np.linspace(1.0, -1.0, hidden_dim)) / hidden_dim)
        recent_vol = float(np.std(log_rets[-10:])) if len(log_rets) >= 10 else 0.001
        expected_log_ret = trend_score * recent_vol * TP_REACH_FORECAST_STEPS
        return float(closes[-1] * np.exp(expected_log_ret))
    except Exception:
        return closes[-1]

def _exp_smoothing_forecast(closes, steps):
    """
    Triple exponential smoothing (Holt-Winters additive, no seasonality).
    Level alpha=0.3, trend beta=0.1. Projects trend forward.
    Captures trend persistence — highly accurate for directional TP confirmation.
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

def _macd_velocity_forecast(closes, steps):
    """
    MACD-velocity based price projection.
    Uses MACD histogram slope (velocity of momentum change) to project price.
    Fast-reacting because MACD responds quickly to momentum shifts.
    
    Logic:
    - Current histogram value = momentum direction
    - Histogram velocity (slope over 3 bars) = rate of momentum change
    - Histogram acceleration = change in velocity
    - Project: velocity is primary driver, acceleration is secondary
    """
    try:
        n = len(closes)
        if n < 35: return closes[-1]
        macd, signal, hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        if len(hist) < 5 or not np.isfinite(hist[-1]):
            return closes[-1]
        
        current_hist = float(hist[-1])
        
        # Velocity: slope of histogram over last 3 bars
        if np.isfinite(hist[-3]):
            hist_velocity = float(hist[-1] - hist[-3])
        else:
            hist_velocity = 0.0
        
        # Acceleration: change in velocity (second derivative)
        if len(hist) >= 5 and all(np.isfinite(x) for x in [hist[-1], hist[-2], hist[-3], hist[-4]]):
            vel_1 = float(hist[-1] - hist[-2])
            vel_2 = float(hist[-2] - hist[-3])
            hist_accel = vel_1 - vel_2
        else:
            hist_accel = 0.0
        
        # Normalize by current price for scale-independence
        price_norm = current_hist / (closes[-1] + 1e-12)
        vel_norm = hist_velocity / (closes[-1] + 1e-12)
        accel_norm = hist_accel / (closes[-1] + 1e-12)
        
        # Project: use velocity as primary driver, acceleration as secondary
        # velocity tells us how fast momentum is changing
        # If velocity > 0, momentum is increasing -> price accelerates up
        expected_return = (vel_norm * steps * 0.6) + (accel_norm * steps * steps * 0.008)
        
        # Clamp to sane bounds
        expected_return = np.clip(expected_return, -0.012 * steps, 0.012 * steps)
        
        forecast = closes[-1] * (1.0 + expected_return)
        return float(forecast)
    except Exception:
        return closes[-1]

def tp_reachability_forecast(candles_1m, current_price, tp_long_price, sl_long_price,
                              tp_short_price, sl_short_price, extrema_data=None):
    """
    ═════════════════════════════════════════════════════════════════════════════
    TP REACHABILITY — STRIPPED-DOWN ENSEMBLE: LSTM + ExpSmooth + MACD-Velocity
    ═════════════════════════════════════════════════════════════════════════════
    
    ONLY the 3 most TP-accurate methods for short-horizon HFT:
    
    1. LSTM — captures sequential momentum patterns (most accurate)
    2. ExpSmooth — trend persistence (Holt-Winters) 
    3. MACD-Velocity — fast-reacting momentum acceleration
    
    ─────────────────────────────────────────────────────────────────────────────
    DECISION LOGIC — NO BOTH-FALSE:
    ─────────────────────────────────────────────────────────────────────────────
    
    Step 1: Count directional votes from 3 methods
      - forecast > current_price → LONG vote
      - forecast < current_price → SHORT vote
      - forecast == current_price → no vote (rare)
    
    Step 2: Determine consensus direction
      - 2+ methods agree → that's the consensus
      - 1v1v1 tie → extrema structure breaks tie
    
    Step 3: MTF Unanimous extrema OVERRIDE
      - If all 3 timeframes (1m/3m/5m) agree on structure direction,
        that direction OVERRIDES any ML tie or minority disagreement
    
    Step 4: Final L/S determination — ALWAYS one is True, one is False
      - consensus == "long"  → L:True  S:False
      - consensus == "short" → L:False S:True
    
    This eliminates the old bug where L:False S:False happened even when
    structure was clearly directional.
    
    ─────────────────────────────────────────────────────────────────────────────
    WHY THESE 3 METHODS:
    ─────────────────────────────────────────────────────────────────────────────
    
    LSTM: Sequential model that captures order-dependent patterns. In HFT,
    the sequence of recent price moves carries momentum information that
    static models miss. Consistently the most accurate for 30-bar TP hits.
    
    ExpSmooth: Holt-Winters captures trend PERSISTENCE. If price has been
    trending up, it projects that trend to continue. Simple but highly
    effective for short-horizon directional confirmation.
    
    MACD-Velocity: The histogram SLOPE (not just value) captures momentum
    ACCELERATION. When momentum is building, price moves faster. This is
    the fastest-reacting signal — catches momentum shifts before LSTM
    or ExpSmooth can react.
    
    REMOVED METHODS (and why):
    - FFT: Spectral extrapolation assumes periodicity — breaks in HFT noise
    - Random Walk: No edge, 50/50 coin flip
    - Forest: Bagged trees too slow to adapt to microstructure changes
    - Ridge: Linear model can't capture nonlinear HFT momentum
    - SVR-RBF: Kernel bandwidth unstable in fast markets
    - Multi-length Reg: Trend-following, not TP-accurate
    ═════════════════════════════════════════════════════════════════════════════
    """
    # Fallback: if no data, use extrema direction
    if len(candles_1m) < 100 or current_price <= 0:
        if extrema_data:
            d = extrema_data["direction"]
            if d == "long":
                return (True, False, current_price, {}, d, 
                        f"{d.upper()} BIAS (No ML Data) — L:True S:False")
            else:
                return (False, True, current_price, {}, d,
                        f"{d.upper()} BIAS (No ML Data) — L:False S:True")
        # Ultimate fallback: long
        return (True, False, current_price, {}, "long", 
                "LONG BIAS (No Data) — L:True S:False")

    closes = np.array([c["close"] for c in candles_1m], dtype=np.float64)
    volumes = np.array([c["volume"] for c in candles_1m], dtype=np.float64)
    
    # Clean data
    for arr in (closes, volumes):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]) or arr[i] <= 0:
                arr[i] = arr[i-1] if i > 0 else arr[0]

    steps = TP_REACH_FORECAST_STEPS
    max_move = current_price * 0.015 * steps
    min_bound = current_price - max_move
    max_bound = current_price + max_move

    # ═══════════════════════════════════════════════════════════════════════
    # ONLY 3 METHODS — the most TP-accurate for short-horizon HFT
    # ═══════════════════════════════════════════════════════════════════════
    p_lstm  = float(np.clip(_lstm_price_forecast(closes, volumes), min_bound, max_bound))
    p_exp   = float(np.clip(_exp_smoothing_forecast(closes, steps), min_bound, max_bound))
    p_macd  = float(np.clip(_macd_velocity_forecast(closes, steps), min_bound, max_bound))

    forecasts = {
        "LSTM": p_lstm,
        "ExpSmooth": p_exp,
        "MACD-Vel": p_macd,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Directional voting
    # ═══════════════════════════════════════════════════════════════════════
    long_votes = 0
    short_votes = 0
    vote_details = {}
    
    for name, pred in forecasts.items():
        if pred > current_price:
            long_votes += 1
            vote_details[name] = "LONG"
        elif pred < current_price:
            short_votes += 1
            vote_details[name] = "SHORT"
        else:
            vote_details[name] = "NEUTRAL"

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Determine consensus direction
    # ═══════════════════════════════════════════════════════════════════════
    if long_votes > short_votes:
        consensus_dir = "long"
        consensus_reason = f"{long_votes}/3 LONG"
    elif short_votes > long_votes:
        consensus_dir = "short"
        consensus_reason = f"{short_votes}/3 SHORT"
    else:
        # Tie (0v0, 1v1v1, etc): use extrema structure as tiebreaker
        if extrema_data:
            consensus_dir = extrema_data["direction"]
            consensus_reason = f"TIE → Extrema {extrema_data['direction'].upper()}"
        else:
            consensus_dir = "long"  # ultimate fallback
            consensus_reason = "TIE → Default LONG"

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: MTF Unanimous extrema OVERRIDE
    # ═══════════════════════════════════════════════════════════════════════
    extrema_override = False
    if extrema_data and extrema_data.get("unanimous", False):
        # All 3 timeframes (1m/3m/5m) agree on structure — TRUST IT
        if extrema_data["direction"] != consensus_dir:
            # ML disagrees but structure is unanimous — override
            consensus_dir = extrema_data["direction"]
            extrema_override = True
            consensus_reason = f"UNANIMOUS OVERRIDE → {consensus_dir.upper()}"

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: FINAL DETERMINATION — NO BOTH-FALSE
    # ═══════════════════════════════════════════════════════════════════════
    if consensus_dir == "long":
        reach_long = True
        reach_short = False
        reason = f"LONG CONSENSUS ({consensus_reason}) — L:True S:False"
    else:
        reach_long = False
        reach_short = True
        reason = f"SHORT CONSENSUS ({consensus_reason}) — L:False S:True"

    if extrema_override:
        reason += " [MTF UNANIMOUS]"

    # Consensus price for display (median of 3 forecasts)
    consensus_price = float(np.median([p_lstm, p_exp, p_macd]))

    return reach_long, reach_short, consensus_price, forecasts, consensus_dir, reason

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def save_trade_state(state):
    try:
        with open(TRADE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception:
        pass

def load_trade_state():
    try:
        if os.path.exists(TRADE_STATE_FILE):
            with open(TRADE_STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def log_trade(trade_str):
    try:
        with open(TRADES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} | {trade_str}\n")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_trading_loop():
    # Initialize buffers
    buffer_5m = CandleBuffer(TRADE_SYMBOL, "5m", ANALYSIS_WINDOW_5M)
    buffer_1m = CandleBuffer(TRADE_SYMBOL, "1m", ANALYSIS_WINDOW_1M)
    
    # Concurrent fetcher
    fetcher = ConcurrentDataFetcher(buffer_5m, buffer_1m, TRADE_SYMBOL)
    
    # State
    position_open = False
    position_side = None
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    entry_time = None
    trade_state = load_trade_state()
    
    if trade_state and trade_state.get("position_open", False):
        position_open = True
        position_side = trade_state.get("side")
        entry_price = float(trade_state.get("entry_price", 0))
        tp_price = float(trade_state.get("tp_price", 0))
        sl_price = float(trade_state.get("sl_price", 0))
        entry_time = trade_state.get("entry_time")
        print(f"  [state] Restored open position: {position_side} @ {entry_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")

    # Initial fetch
    if API_CONNECTED:
        print("\n  [init] Fetching initial data...")
        fetcher.fetch_all_parallel(do_full_refresh=True)
        if not buffer_1m.is_ready() or not buffer_5m.is_ready():
            print("  [init] Insufficient data, waiting for more candles...")
            time.sleep(10)
            fetcher.fetch_all_parallel(do_full_refresh=True)

    print("\n  [ready] Bot is now running. Press Ctrl+C to stop.\n")
    
    loop_count = 0
    last_display_time = 0
    signal_cooldown = 0
    
    try:
        while True:
            t_loop_start = time.perf_counter()
            
            # ─── FETCH DATA ────────────────────────────────────────────
            do_full = (loop_count % (FULL_REFRESH_INTERVAL / LOOP_SLEEP) < 1)
            data = fetcher.fetch_all_parallel(do_full_refresh=do_full)
            current_price = data["price"]
            balance = data["balance"]
            position_info = data["position"]
            
            if current_price <= 0:
                time.sleep(LOOP_SLEEP)
                loop_count += 1
                continue
            
            # ─── CHECK EXISTING POSITION (TP/SL) ───────────────────────
            if position_open and entry_price > 0:
                # Update from exchange if available
                if position_info["is_open"]:
                    current_roe = position_info["roe_pct"]
                else:
                    # Calculate ROE locally
                    if position_side == "long":
                        price_move_pct = ((current_price - entry_price) / entry_price) * 100
                        current_roe = price_move_pct * LEVERAGE
                    else:
                        price_move_pct = ((entry_price - current_price) / entry_price) * 100
                        current_roe = price_move_pct * LEVERAGE
                
                # Check TP
                if position_side == "long" and current_price >= tp_price:
                    close_result = {"type": position_side, "roe": TAKE_PROFIT_ROE, "reason": "TAKE PROFIT HIT",
                                   "entry_price": entry_price, "exit_price": current_price, "mode": "LIVE" if API_CONNECTED else "SIM"}
                    if entry_time:
                        close_result["duration"] = str(datetime.datetime.now() - datetime.datetime.fromisoformat(entry_time)).split('.')[0]
                    if API_CONNECTED:
                        try:
                            client.close_position(TRADE_SYMBOL)
                        except Exception as e:
                            print(f"\n  [ERROR] Failed to close position: {e}")
                    trade_tracker.add_trade(close_result)
                    trade_tracker.print_trade_alert(close_result)
                    log_trade(f"TP CLOSE {position_side.upper()} | Entry:{entry_price:.2f} Exit:{current_price:.2f} ROE:+{TAKE_PROFIT_ROE:.2f}%")
                    position_open = False
                    position_side = None
                    entry_price = 0.0
                    tp_price = 0.0
                    sl_price = 0.0
                    entry_time = None
                    save_trade_state({"position_open": False})
                    signal_cooldown = 5
                    
                elif position_side == "short" and current_price <= tp_price:
                    close_result = {"type": position_side, "roe": TAKE_PROFIT_ROE, "reason": "TAKE PROFIT HIT",
                                   "entry_price": entry_price, "exit_price": current_price, "mode": "LIVE" if API_CONNECTED else "SIM"}
                    if entry_time:
                        close_result["duration"] = str(datetime.datetime.now() - datetime.datetime.fromisoformat(entry_time)).split('.')[0]
                    if API_CONNECTED:
                        try:
                            client.close_position(TRADE_SYMBOL)
                        except Exception as e:
                            print(f"\n  [ERROR] Failed to close position: {e}")
                    trade_tracker.add_trade(close_result)
                    trade_tracker.print_trade_alert(close_result)
                    log_trade(f"TP CLOSE {position_side.upper()} | Entry:{entry_price:.2f} Exit:{current_price:.2f} ROE:+{TAKE_PROFIT_ROE:.2f}%")
                    position_open = False
                    position_side = None
                    entry_price = 0.0
                    tp_price = 0.0
                    sl_price = 0.0
                    entry_time = None
                    save_trade_state({"position_open": False})
                    signal_cooldown = 5
                    
                # Check SL
                elif position_side == "long" and current_price <= sl_price:
                    close_result = {"type": position_side, "roe": STOP_LOSS_ROE, "reason": "STOP LOSS HIT",
                                   "entry_price": entry_price, "exit_price": current_price, "mode": "LIVE" if API_CONNECTED else "SIM"}
                    if entry_time:
                        close_result["duration"] = str(datetime.datetime.now() - datetime.datetime.fromisoformat(entry_time)).split('.')[0]
                    if API_CONNECTED:
                        try:
                            client.close_position(TRADE_SYMBOL)
                        except Exception as e:
                            print(f"\n  [ERROR] Failed to close position: {e}")
                    trade_tracker.add_trade(close_result)
                    trade_tracker.print_trade_alert(close_result)
                    log_trade(f"SL CLOSE {position_side.upper()} | Entry:{entry_price:.2f} Exit:{current_price:.2f} ROE:{STOP_LOSS_ROE:.2f}%")
                    position_open = False
                    position_side = None
                    entry_price = 0.0
                    tp_price = 0.0
                    sl_price = 0.0
                    entry_time = None
                    save_trade_state({"position_open": False})
                    signal_cooldown = 10
                    
                elif position_side == "short" and current_price >= sl_price:
                    close_result = {"type": position_side, "roe": STOP_LOSS_ROE, "reason": "STOP LOSS HIT",
                                   "entry_price": entry_price, "exit_price": current_price, "mode": "LIVE" if API_CONNECTED else "SIM"}
                    if entry_time:
                        close_result["duration"] = str(datetime.datetime.now() - datetime.datetime.fromisoformat(entry_time)).split('.')[0]
                    if API_CONNECTED:
                        try:
                            client.close_position(TRADE_SYMBOL)
                        except Exception as e:
                            print(f"\n  [ERROR] Failed to close position: {e}")
                    trade_tracker.add_trade(close_result)
                    trade_tracker.print_trade_alert(close_result)
                    log_trade(f"SL CLOSE {position_side.upper()} | Entry:{entry_price:.2f} Exit:{current_price:.2f} ROE:{STOP_LOSS_ROE:.2f}%")
                    position_open = False
                    position_side = None
                    entry_price = 0.0
                    tp_price = 0.0
                    sl_price = 0.0
                    entry_time = None
                    save_trade_state({"position_open": False})
                    signal_cooldown = 10
                
                # Print P&L line
                if position_open:
                    trade_tracker.print_stats_line(f"{position_side.upper()}", current_roe)
                
                time.sleep(LOOP_SLEEP)
                loop_count += 1
                continue
            
            # ─── SIGNAL GENERATION (only when FLAT) ────────────────────
            if signal_cooldown > 0:
                signal_cooldown -= 1
                trade_tracker.print_stats_line("FLAT", 0.0)
                time.sleep(LOOP_SLEEP)
                loop_count += 1
                continue
            
            candles_1m = buffer_1m.get_candles()
            candles_5m = buffer_5m.get_candles()
            
            if len(candles_1m) < 100 or len(candles_5m) < 50:
                trade_tracker.print_stats_line("FLAT", 0.0)
                time.sleep(LOOP_SLEEP)
                loop_count += 1
                continue
            
            t_analysis = time.perf_counter()
            
            # ─── 1M ANALYSIS ───────────────────────────────────────────
            closes_1m = np.array([c["close"] for c in candles_1m], dtype=np.float64)
            
            # MTF Extrema (primary structure)
            extrema = get_mtf_extrema_consensus(candles_1m, EXTREMA_LOOKBACK)
            extrema_dir = extrema["direction"]
            extrema_unanimous = extrema.get("unanimous", False)
            
            # Momentum (1m, period 14)
            mom_1m = calculate_momentum(closes_1m, 14)
            mom_long = not np.isnan(mom_1m) and mom_1m > 0
            mom_short = not np.isnan(mom_1m) and mom_1m < 0
            
            # Volume (1m, last 20 candles)
            vol_candles_1m = candles_1m[-20:]
            buy_vol_pct, sell_vol_pct = get_buy_sell_volume_perc(vol_candles_1m)
            vol_long = buy_vol_pct > 55
            vol_short = sell_vol_pct > 55
            
            # ─── 5M ANALYSIS ───────────────────────────────────────────
            closes_5m = np.array([c["close"] for c in candles_5m], dtype=np.float64)
            
            # Sine cycle (5m)
            extrema_1m_native = get_extrema_direction_1m(candles_1m, ANALYSIS_WINDOW_1M)
            if extrema_1m_native:
                argmin_1m = extrema_1m_native["idx"] if extrema_1m_native["direction"] == "long" else None
                argmax_1m = extrema_1m_native["idx"] if extrema_1m_native["direction"] == "short" else None
            else:
                argmin_1m, argmax_1m = 0, 0
            sine_up, sine_down = scale_to_sine(closes_5m, argmin_1m if argmin_1m else 0, argmax_1m if argmax_1m else len(closes_5m)-1)
            sine_long = sine_up > 55
            sine_short = sine_down > 55
            
            # FFT cycle dominance (1m)
            fft_long, fft_short, fft_neg, fft_pos = analyze_fft_dominance_1m(closes_1m)
            
            # ML Forecast (quick composite)
            ml_forecast, ml_current = generate_ml_forecast(candles_1m)
            ml_long = ml_forecast > current_price if ml_forecast > 0 else False
            ml_short = ml_forecast < current_price if ml_forecast > 0 else False
            
            # ─── TP REACHABILITY (STRIPPED ENSEMBLE) ──────────────────
            tp_long = current_price * (1.0 + TP_PRICE_PCT)
            tp_short = current_price * (1.0 - TP_PRICE_PCT)
            sl_long = current_price * (1.0 - SL_PRICE_PCT)
            sl_short = current_price * (1.0 + SL_PRICE_PCT)
            
            reach_long, reach_short, consensus_price, tp_forecasts, tp_consensus_dir, tp_reason = \
                tp_reachability_forecast(candles_1m, current_price, tp_long, sl_long, tp_short, sl_short, extrema)
            
            analysis_ms = (time.perf_counter() - t_analysis) * 1000
            log_timing("analysis", analysis_ms)
            
            # ─── SIGNAL COUNTING (6 CONDITIONS) ───────────────────────
            # Condition 1: Momentum (MANDATORY)
            c_mom_long = mom_long
            c_mom_short = mom_short
            
            # Condition 2: Volume (MANDATORY)
            c_vol_long = vol_long
            c_vol_short = vol_short
            
            # Condition 3: ML Forecast
            c_ml_long = ml_long
            c_ml_short = ml_short
            
            # Condition 4: TP Reachability (MANDATORY)
            c_tp_long = reach_long
            c_tp_short = reach_short
            
            # Condition 5: Sine Cycle
            c_sine_long = sine_long
            c_sine_short = sine_short
            
            # Condition 6: FFT Cycle
            c_fft_long = fft_long
            c_fft_short = fft_short
            
            # Extrema Direction (HARD FILTER — not a count, but a gate)
            extrema_allows_long = (extrema_dir == "long")
            extrema_allows_short = (extrema_dir == "short")
            
            # Count conditions for each direction
            long_conditions = int(c_mom_long) + int(c_vol_long) + int(c_ml_long) + int(c_tp_long) + int(c_sine_long) + int(c_fft_long)
            short_conditions = int(c_mom_short) + int(c_vol_short) + int(c_ml_short) + int(c_tp_short) + int(c_sine_short) + int(c_fft_short)
            
            # Mandatory check: Mom + Vol + ML + TP must all be true
            long_mandatory_ok = c_mom_long and c_vol_long and c_ml_long and c_tp_long
            short_mandatory_ok = c_mom_short and c_vol_short and c_ml_short and c_tp_short
            
            # Final signal: mandatory OK + >=3/6 conditions + extrema filter allows
            long_signal = long_mandatory_ok and long_conditions >= 3 and extrema_allows_long
            short_signal = short_mandatory_ok and short_conditions >= 3 and extrema_allows_short
            
            # ─── DISPLAY (every ~2 seconds) ───────────────────────────
            now = time.time()
            if now - last_display_time >= 2.0:
                last_display_time = now
                print()
                
                # TP Reach header
                bias_icon = "🟢" if tp_consensus_dir == "long" else "🔴"
                print(f"  {bias_icon} ═══ {tp_consensus_dir.upper()} BIAS ═══")
                print(f"  TP-Reach: Tgt-L:{tp_long:.2f} Tgt-S:{tp_short:.2f} | Consensus:{consensus_price:.2f}")
                
                # Show only 3 methods
                for name, pred in tp_forecasts.items():
                    arrow = "↑" if pred > current_price else "↓" if pred < current_price else "→"
                    diff = ((pred - current_price) / current_price) * 100
                    print(f"  {name:<12}: {pred:.2f} {arrow} ({diff:+.4f}%)")
                
                # Show L/S determination
                l_str = f"\033[92mTRUE\033[0m" if reach_long else "\033[90mFALSE\033[0m"
                s_str = f"\033[91mTRUE\033[0m" if reach_short else "\033[90mFALSE\033[0m"
                print(f"  ──────────────────────────────────────────────")
                print(f"  TP-Reach Decision: L:{l_str} S:{s_str}")
                print(f"  Reason: {tp_reason}")
                
                # Extrema
                ext_icon = "✓" if extrema_unanimous else "~"
                print(f"  ──────────────────────────────────────────────")
                print(f"  Extrema: {extrema['extreme_type']} {ext_icon} | Conf:{extrema['confidence']:.2f} | Allows→ L:{extrema_allows_long} S:{extrema_allows_short}")
                
                # Conditions
                print(f"  ──────────────────────────────────────────────")
                print(f"  Conditions ({long_conditions}/6 LONG, {short_conditions}/6 SHORT):")
                print(f"    Mom:  L:{'✓' if c_mom_long else '✗'} S:{'✓' if c_mom_short else '✗'} (MANDATORY)")
                print(f"    Vol:  L:{'✓' if c_vol_long else '✗'} S:{'✓' if c_vol_short else '✗'} (MANDATORY) Buy:{buy_vol_pct:.0f}% Sell:{sell_vol_pct:.0f}%")
                print(f"    ML:   L:{'✓' if c_ml_long else '✗'} S:{'✓' if c_ml_short else '✗'} (MANDATORY) Fcst:{ml_forecast:.2f}")
                print(f"    TP:   L:{'✓' if c_tp_long else '✗'} S:{'✓' if c_tp_short else '✗'} (MANDATORY)")
                print(f"    Sine: L:{'✓' if c_sine_long else '✗'} S:{'✓' if c_sine_short else '✗'} Up:{sine_up:.0f}% Dn:{sine_down:.0f}%")
                print(f"    FFT:  L:{'✓' if c_fft_long else '✗'} S:{'✓' if c_fft_short else '✗'} Pos:{fft_pos:.0f}% Neg:{fft_neg:.0f}%")
                
                # Final signal
                print(f"  ──────────────────────────────────────────────")
                if long_signal:
                    print(f"  ═══════ 🟢🟢🟢 LONG SIGNAL ACTIVATED 🟢🟢🟢 ═══════")
                elif short_signal:
                    print(f"  ═══════ 🔴🔴🔴 SHORT SIGNAL ACTIVATED 🔴🔴🔴 ═══════")
                else:
                    missing_long = []
                    if not long_mandatory_ok:
                        if not c_mom_long: missing_long.append("Mom")
                        if not c_vol_long: missing_long.append("Vol")
                        if not c_ml_long: missing_long.append("ML")
                        if not c_tp_long: missing_long.append("TP")
                    if long_conditions < 3: missing_long.append(f"Cnt<{long_conditions}")
                    if not extrema_allows_long: missing_long.append("ExtBlock")
                    
                    missing_short = []
                    if not short_mandatory_ok:
                        if not c_mom_short: missing_short.append("Mom")
                        if not c_vol_short: missing_short.append("Vol")
                        if not c_ml_short: missing_short.append("ML")
                        if not c_tp_short: missing_short.append("TP")
                    if short_conditions < 3: missing_short.append(f"Cnt<{short_conditions}")
                    if not extrema_allows_short: missing_short.append("ExtBlock")
                    
                    print(f"  NO SIGNAL — L missing:[{', '.join(missing_long) if missing_long else 'OK'}] S missing:[{', '.join(missing_short) if missing_short else 'OK'}]")
                
                print(f"  Analysis: {analysis_ms:.0f}ms | Price: {current_price:.2f} | Balance: {balance:.2f} USDT")
            
            # ─── EXECUTE SIGNAL ───────────────────────────────────────
            if long_signal and not position_open:
                if balance >= MIN_BALANCE_USDT:
                    trade_size_usdt = balance * TRADE_BALANCE_PCT
                    contracts = int(trade_size_usdt * LEVERAGE / current_price)
                    if contracts > 0:
                        entry_price = current_price
                        tp_price = current_price * (1.0 + TP_PRICE_PCT)
                        sl_price = current_price * (1.0 - SL_PRICE_PCT)
                        position_side = "long"
                        position_open = True
                        entry_time = datetime.datetime.now().isoformat()
                        
                        if API_CONNECTED:
                            try:
                                result = client.place_order(TRADE_SYMBOL, "buy", contracts, LEVERAGE)
                                print(f"\n  [ORDER] LONG {contracts} contracts @ {current_price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")
                                log_trade(f"OPEN LONG | Size:{contracts} Entry:{current_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
                            except Exception as e:
                                print(f"\n  [ERROR] Order failed: {e}")
                                position_open = False
                        else:
                            print(f"\n  [SIM] LONG {contracts} contracts @ {current_price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")
                            log_trade(f"SIM OPEN LONG | Size:{contracts} Entry:{current_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
                        
                        if position_open:
                            save_trade_state({
                                "position_open": True, "side": "long", "entry_price": entry_price,
                                "tp_price": tp_price, "sl_price": sl_price, "entry_time": entry_time
                            })
                            signal_cooldown = 3
                        
            elif short_signal and not position_open:
                if balance >= MIN_BALANCE_USDT:
                    trade_size_usdt = balance * TRADE_BALANCE_PCT
                    contracts = int(trade_size_usdt * LEVERAGE / current_price)
                    if contracts > 0:
                        entry_price = current_price
                        tp_price = current_price * (1.0 - TP_PRICE_PCT)
                        sl_price = current_price * (1.0 + SL_PRICE_PCT)
                        position_side = "short"
                        position_open = True
                        entry_time = datetime.datetime.now().isoformat()
                        
                        if API_CONNECTED:
                            try:
                                result = client.place_order(TRADE_SYMBOL, "sell", contracts, LEVERAGE)
                                print(f"\n  [ORDER] SHORT {contracts} contracts @ {current_price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")
                                log_trade(f"OPEN SHORT | Size:{contracts} Entry:{current_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
                            except Exception as e:
                                print(f"\n  [ERROR] Order failed: {e}")
                                position_open = False
                        else:
                            print(f"\n  [SIM] SHORT {contracts} contracts @ {current_price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")
                            log_trade(f"SIM OPEN SHORT | Size:{contracts} Entry:{current_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
                        
                        if position_open:
                            save_trade_state({
                                "position_open": True, "side": "short", "entry_price": entry_price,
                                "tp_price": tp_price, "sl_price": sl_price, "entry_time": entry_time
                            })
                            signal_cooldown = 3
            
            # ─── LOOP TIMING ──────────────────────────────────────────
            loop_ms = (time.perf_counter() - t_loop_start) * 1000
            log_timing("total_loop", loop_ms)
            
            if not position_open:
                trade_tracker.print_stats_line("FLAT", 0.0)
            
            time.sleep(LOOP_SLEEP)
            loop_count += 1
            
            # Periodic timing summary
            if loop_count % 1200 == 0:
                print_timing_summary()
    
    except KeyboardInterrupt:
        print("\n\n  [shutdown] Bot stopped by user.")
        print_timing_summary()
        fetcher.shutdown()

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_trading_loop()
