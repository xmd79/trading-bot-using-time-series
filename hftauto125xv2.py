"""
HFT Auto Trading Bot v5.1 — KuCoin Futures Edition
===================================================
Strictly 5 Conditions (1m TF):
  1. Sine Scale: Uses argmin/argmax of last 500 values as cycle boundaries
     - Most recent extrema as reversal starting point
     - argmin more recent → up cycle | argmax more recent → down cycle
     - dist_to_min < dist_to_max (LONG) | dist_to_max < dist_to_min (SHORT)
  2. Extrema Cycle: argmin > argmax (LONG) | argmax > argmin (SHORT) [Last 500 values]
  3. Momentum: MOM > 0 (LONG) | MOM < 0 (SHORT)
  4. Volume: Bullish > Bearish (LONG) | Bearish > Bullish (SHORT) [Total 1m Volume]
  5. FFT Frequency Analysis: Predominant frequencies between reversal extremas
     - Mostly negative predominant -> LONG | Mostly positive predominant -> SHORT
     - (Always strictly evaluates to one or the other)

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

# ═══════════════════════════════════════════════════════════════════════════════
# FILE SETUP - Clean trades.txt
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_STATE_FILE = "trade_state.json"
TRADES_LOG_FILE = "trades.txt"
ANALYTICS_FILE = "analytics.json"

def clean_trades_file():
    """Create or empty the trades.txt file."""
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

print("HFT KuCoin Bot v5.1 (STRICT 5-COND / 25x / 2.55% NET TP) initialising...")
if _cleaned:
    print(f"  [cleanup] Wiped: {', '.join(_cleaned)}")
del _cleaned

# ═══════════════════════════════════════════════════════════════════════════════
# KUCOIN FUTURES REST CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

KUCOIN_FUTURES_BASE = "https://api-futures.kucoin.com"

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

    def _headers(self, method, signed_endpoint, body=""):
        ts = str(int(time.time() * 1000))
        sign = _kucoin_sign(self.api_secret, ts, method, signed_endpoint, body)
        return {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": sign,
            "KC-API-TIMESTAMP": ts,
            "KC-API-PASSPHRASE": self._signed_passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    def get(self, endpoint, params=None):
        if params:
            qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            signed_path = f"{endpoint}?{qs}"
        else:
            signed_path = endpoint
        url = KUCOIN_FUTURES_BASE + signed_path
        hdrs = self._headers("GET", signed_path)
        resp = requests.get(url, headers=hdrs, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint, payload):
        body = json.dumps(payload, separators=(",", ":"))
        url = KUCOIN_FUTURES_BASE + endpoint
        hdrs = self._headers("POST", endpoint, body)
        resp = requests.post(url, headers=hdrs, data=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_account_overview(self):
        return self.get("/api/v1/account-overview", {"currency": "USDT"})

    def get_klines(self, symbol, granularity, start_ms=None, end_ms=None):
        params = {"symbol": symbol, "granularity": granularity}
        if start_ms:
            params["from"] = start_ms
        if end_ms:
            params["to"] = end_ms
        data = self.get("/api/v1/kline/query", params)
        return data.get("data", [])

    def get_ticker(self, symbol):
        return self.get("/api/v1/ticker", {"symbol": symbol})

    def get_position(self, symbol):
        return self.get("/api/v1/position", {"symbol": symbol})

    def place_order(self, symbol, side, size, leverage):
        payload = {
            "clientOid": hashlib.md5(f"{time.time()}{side}".encode()).hexdigest(),
            "symbol": symbol,
            "side": side,
            "type": "market",
            "size": size,
            "leverage": str(leverage),
        }
        return self.post("/api/v1/orders", payload)

    def close_position(self, symbol):
        payload = {
            "clientOid": hashlib.md5(str(time.time()).encode()).hexdigest(),
            "symbol": symbol,
            "type": "market",
            "closeOrder": True,
        }
        return self.post("/api/v1/orders", payload)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_SYMBOL = "XBTUSDTM"
LEVERAGE = 25
LOOP_SLEEP = 3
MIN_BALANCE_USDT = 5.0  # Minimum balance to trade live

KUCOIN_TAKER_FEE = 0.0006
RT_FEE_ROE_PCT = KUCOIN_TAKER_FEE * 2 * LEVERAGE * 100  # 0.0012 * 25 * 100 = 3.0%
NET_PROFIT_ROE = 2.55                                  # 2.55% Clean Net Profit Target
TAKE_PROFIT_ROE = NET_PROFIT_ROE + RT_FEE_ROE_PCT      # 2.55 + 3.0 = 5.55% Gross
STOP_LOSS_ROE = -25.0                                  # -25% ROE Stop Loss

# Price move percentages for TP/SL
TP_PRICE_PCT = TAKE_PROFIT_ROE / LEVERAGE / 100.0  # 5.55 / 25 / 100 = 0.00222 (0.222%)
SL_PRICE_PCT = abs(STOP_LOSS_ROE) / LEVERAGE / 100.0  # 25.0 / 25 / 100 = 0.01 (1.0%)

try:
    with open("credentials_kucoin.txt", "r") as _f:
        _lines = _f.readlines()
        _API_KEY = _lines[0].strip()
        _API_SECRET = _lines[1].strip()
        _API_PASSPHRASE = _lines[2].strip()
    client = KuCoinFuturesClient(_API_KEY, _API_SECRET, _API_PASSPHRASE)
    API_CONNECTED = True
except Exception as e:
    print(f"  [WARN] API credentials not found or invalid: {e}")
    print("  [WARN] Running in SIMULATION-ONLY mode")
    client = None
    API_CONNECTED = False

# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_account_balance():
    if not API_CONNECTED:
        return 0.0
    try:
        data = client.get_account_overview()
        return float(data["data"]["availableBalance"])
    except Exception as e:
        print(f"  [balance] error: {e}")
        return 0.0

def get_price(symbol):
    if not API_CONNECTED:
        return 0.0
    try:
        data = client.get_ticker(symbol)
        return float(data["data"]["price"])
    except Exception as e:
        print(f"  [price] error: {e}")
        return 0.0

def fetch_candle_map(symbol):
    """Fetches up to 500 1m candles (3 requests of 200 due to API limit)."""
    candle_map = {"1m": []}
    if not API_CONNECTED:
        return candle_map
    now_ms = int(time.time() * 1000)
    limit = 200
    for i in range(3):
        end_ms = now_ms - (i * limit * 60 * 1000)
        start_ms = end_ms - (limit * 60 * 1000)
        try:
            raw = client.get_klines(symbol, 1, start_ms=start_ms, end_ms=end_ms)
            for k in reversed(raw):  # Append chronologically
                try:
                    candle_map["1m"].append({
                        "time": int(k[0]) / 1000,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                except (IndexError, TypeError, ValueError):
                    continue
        except Exception as e:
            print(f"  kline fetch error [1m]: {e}")
            
    candle_map["1m"] = candle_map["1m"][-500:]
    return candle_map

def get_buy_sell_volume_1m(candle_map):
    """Calculates total bullish vs bearish volume from all fetched 1m candles."""
    candles = candle_map.get("1m", [])
    buy_vol = sell_vol = 0.0
    for c in candles:
        if c["close"] >= c["open"]:
            buy_vol += c["volume"]
        else:
            sell_vol += c["volume"]
    return buy_vol, sell_vol

# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS (Strictly the 5 required conditions)
# ═══════════════════════════════════════════════════════════════════════════════

def scale_to_sine(close_prices, argmin_idx, argmax_idx):
    """
    Scale to Sine using argmin and argmax of last 500 values as cycle boundaries.
    """
    if len(close_prices) < 32:
        return 50.0, 50.0, 0.0, "none", np.array([])
    
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave  # Invert for standard peak/trough alignment
    
    sine_500 = sine_wave[-500:] if len(sine_wave) >= 500 else sine_wave
    sine_at_argmin = sine_500[argmin_idx]
    sine_at_argmax = sine_500[argmax_idx]
    
    if argmin_idx > argmax_idx:
        cycle_min = sine_at_argmin
        cycle_max = sine_at_argmax
        cycle_direction = "up"
    else:
        cycle_max = sine_at_argmax
        cycle_min = sine_at_argmin
        cycle_direction = "down"
    
    current_sine = sine_wave[-1]
    rng = cycle_max - cycle_min if cycle_max != cycle_min else 1e-9
    
    dist_to_min = ((current_sine - cycle_min) / rng) * 100
    dist_to_max = ((cycle_max - current_sine) / rng) * 100
    
    dist_to_min = max(0, min(100, dist_to_min))
    dist_to_max = max(0, min(100, dist_to_max))
    
    return dist_to_min, dist_to_max, current_sine, cycle_direction, sine_wave


def analyze_fft_frequencies(sine_wave, argmin_idx, argmax_idx, cycle_direction):
    """
    FFT Frequency Analysis: Since total power must equal neg + pos power,
    it strictly evaluates to LONG (mostly negative) or SHORT (mostly positive).
    There is NO 'NONE' state.
    """
    if len(sine_wave) < 16:
        return True, False, 0.0, 1.0, 0.0, False 
    
    sine_500 = sine_wave[-500:] if len(sine_wave) >= 500 else sine_wave
    start_idx = min(argmin_idx, argmax_idx)
    end_idx = max(argmin_idx, argmax_idx)
    
    if end_idx - start_idx < 8:
        segment = sine_500[-64:] if len(sine_500) >= 64 else sine_500
    else:
        segment = sine_500[start_idx:end_idx + 1]
    
    n = len(segment)
    if n < 8:
        return True, False, 0.0, 1.0, 0.0, False
    
    fft_result = fft(segment)
    fft_magnitude = np.abs(fft_result)
    freq_bins = fftfreq(n)
    
    magnitudes = fft_magnitude[1:]
    frequencies = freq_bins[1:]
    
    if len(magnitudes) == 0:
        return True, False, 0.0, 1.0, 0.0, False
    
    total_power = np.sum(magnitudes)
    if total_power == 0:
        return True, False, 0.0, 1.0, 0.0, False
    
    neg_mask = frequencies < 0
    pos_mask = frequencies > 0
    
    neg_power = np.sum(magnitudes[neg_mask])
    pos_power = np.sum(magnitudes[pos_mask])
    
    neg_power_ratio = neg_power / total_power
    pos_power_ratio = pos_power / total_power
    
    predom_idx = np.argmax(magnitudes)
    predom_freq = frequencies[predom_idx]
    
    grad_spread_neg = neg_power_ratio > 0.45
    
    # STRICT EVALUATION: One MUST be true. No 'NONE'.
    # If mostly negative predominant -> LONG
    # If mostly positive predominant -> SHORT
    fft_long_signal = neg_power_ratio >= pos_power_ratio
    fft_short_signal = pos_power_ratio > neg_power_ratio
    
    return fft_long_signal, fft_short_signal, predom_freq, neg_power_ratio, pos_power_ratio, grad_spread_neg


def calculate_momentum(close_arr, period=14):
    if len(close_arr) < period + 1:
        return np.nan
    mom = talib.MOM(close_arr, timeperiod=period)
    return float(mom[-1])


def compute_signals(candle_map):
    closes_1m_raw = [c["close"] for c in candle_map.get("1m", [])]
    live_price = get_price(TRADE_SYMBOL)
    
    if len(closes_1m_raw) < 50:
        return None
        
    close_arr = np.array(closes_1m_raw, dtype=float)
    
    # 2. Extrema Cycle
    last_500 = close_arr[-500:]
    argmin_idx = int(np.argmin(last_500))
    argmax_idx = int(np.argmax(last_500))
    cond_cycle_long = argmin_idx > argmax_idx
    cond_cycle_short = argmax_idx > argmin_idx
    
    # 1. Sine Scale
    dist_to_min, dist_to_max, current_sine, cycle_direction, sine_wave = scale_to_sine(
        close_arr, argmin_idx, argmax_idx
    )
    cond_sine_long = dist_to_min < dist_to_max
    cond_sine_short = dist_to_max < dist_to_min
    
    # 5. FFT Frequency Analysis
    fft_long, fft_short, predom_freq, neg_ratio, pos_ratio, grad_spread = analyze_fft_frequencies(
        sine_wave, argmin_idx, argmax_idx, cycle_direction
    )
    cond_fft_long = fft_long
    cond_fft_short = fft_short
    
    # 3. Momentum
    mom = calculate_momentum(close_arr)
    if np.isnan(mom):
        return None
    cond_mom_long = mom > 0
    cond_mom_short = mom < 0
    
    # 4. Volume
    buy_vol, sell_vol = get_buy_sell_volume_1m(candle_map)
    cond_vol_long = buy_vol > sell_vol
    cond_vol_short = sell_vol > buy_vol
    
    # ALL 5 CONDITIONS MUST BE TRUE
    is_long = cond_sine_long and cond_cycle_long and cond_mom_long and cond_vol_long and cond_fft_long
    is_short = cond_sine_short and cond_cycle_short and cond_mom_short and cond_vol_short and cond_fft_short
    
    return {
        "price": live_price,
        "is_long": is_long,
        "is_short": is_short,
        "cond_flags": {
            "sine_long": cond_sine_long, "sine_short": cond_sine_short,
            "cycle_long": cond_cycle_long, "cycle_short": cond_cycle_short,
            "mom_long": cond_mom_long, "mom_short": cond_mom_short,
            "vol_long": cond_vol_long, "vol_short": cond_vol_short,
            "fft_long": cond_fft_long, "fft_short": cond_fft_short,
        },
        "dist_to_min": dist_to_min,
        "dist_to_max": dist_to_max,
        "cycle_direction": cycle_direction,
        "argmin_idx": argmin_idx,
        "argmax_idx": argmax_idx,
        "momentum": mom,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "predom_freq": predom_freq,
        "neg_freq_ratio": neg_ratio,
        "pos_freq_ratio": pos_ratio,
        "grad_spread_neg": grad_spread,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SL & TP CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(entry_price, side):
    tp_dist = entry_price * TP_PRICE_PCT
    sl_dist = entry_price * SL_PRICE_PCT
    
    if side == "long":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
        
    print(f"  [Risk Calc] TP Dist: {tp_dist:.2f} ({TAKE_PROFIT_ROE}% ROE gross / {NET_PROFIT_ROE}% net)")
    print(f"  [Risk Calc] SL Dist: {sl_dist:.2f} ({STOP_LOSS_ROE}% ROE)")
    
    return float(sl_price), float(tp_price)

def check_sim_tp_sl(entry_price, current_price, side):
    """Check if simulated trade hit TP or SL. Returns (hit, reason, roe_pct)"""
    if side == "long":
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        roe_pct = price_change_pct * LEVERAGE - RT_FEE_ROE_PCT
        if roe_pct >= TAKE_PROFIT_ROE:
            return True, "TAKE PROFIT", roe_pct
        elif roe_pct <= STOP_LOSS_ROE:
            return True, "STOP LOSS", roe_pct
        return False, None, roe_pct
    else:  # short
        price_change_pct = ((entry_price - current_price) / entry_price) * 100
        roe_pct = price_change_pct * LEVERAGE - RT_FEE_ROE_PCT
        if roe_pct >= TAKE_PROFIT_ROE:
            return True, "TAKE PROFIT", roe_pct
        elif roe_pct <= STOP_LOSS_ROE:
            return True, "STOP LOSS", roe_pct
        return False, None, roe_pct

# ═══════════════════════════════════════════════════════════════════════════════
# STATE & JOURNALING
# ═══════════════════════════════════════════════════════════════════════════════

def save_trade_state(state):
    try:
        with open(TRADE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"  [state] save error: {e}")

def load_trade_state():
    if not os.path.exists(TRADE_STATE_FILE):
        return None
    try:
        with open(TRADE_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def clear_trade_state():
    try:
        if os.path.exists(TRADE_STATE_FILE):
            os.remove(TRADE_STATE_FILE)
    except Exception:
        pass

def write_trade_to_journal(trade_result):
    """Write trade result to trades.txt in clean format."""
    mode = trade_result.get("mode", "LIVE")
    line = (
        f"{'='*60}\n"
        f"MODE: {mode}\n"
        f"TYPE: {trade_result.get('type', 'N/A').upper()}\n"
        f"ENTRY TIME: {trade_result.get('entry_time', 'N/A')}\n"
        f"EXIT TIME: {trade_result.get('exit_time', 'N/A')}\n"
        f"DURATION: {trade_result.get('duration', 'N/A')}\n"
        f"ENTRY PRICE: {trade_result.get('entry_price', 0):.2f}\n"
        f"EXIT PRICE: {trade_result.get('exit_price', 0):.2f}\n"
        f"PRICE CHANGE: {trade_result.get('price_change_pct', 0):+.4f}%\n"
        f"GROSS ROE: {trade_result.get('gross_roe', 0):+.2f}%\n"
        f"RT FEE DEDUCTED: {trade_result.get('rt_fee_pct', 0):.2f}%\n"
        f"NET ROE: {trade_result.get('roe', 0):+.2f}%\n"
        f"REASON: {trade_result.get('reason', 'N/A')}\n"
        f"LEVERAGE: {LEVERAGE}x\n"
        f"{'='*60}\n\n"
    )
    try:
        with open(TRADES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"  [journal] write error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_contract_size(symbol):
    if not API_CONNECTED:
        return 0.001
    try:
        data = client.get(f"/api/v1/contracts/{symbol}")
        return float(data["data"]["multiplier"])
    except Exception:
        return 0.001

def execute_entry(symbol, side, balance, price):
    try:
        multiplier = get_contract_size(symbol)
        notional = balance * LEVERAGE
        contracts = max(1, int(notional / (price * multiplier)))
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
    if not API_CONNECTED:
        return empty
    try:
        raw = client.get_position(symbol)
        data = raw.get("data", {})
        size = float(data.get("currentQty", 0) or 0)
        if size == 0:
            return empty
        upnl = float(data.get("unrealisedPnl", 0) or 0)
        margin = float(data.get("posMargin", 0) or 0)
        roe = (upnl / margin * 100) if margin else 0.0
        return {
            "is_open": True, "side": "long" if size > 0 else "short",
            "entry_price": float(data.get("avgEntryPrice", 0)), 
            "mark_price": float(data.get("markPrice", 0)), 
            "roe_pct": roe,
            "size": size,
        }
    except Exception as e:
        print(f"    [pos/error] {e}")
        return empty

def format_duration(start_dt, end_dt):
    delta = end_dt - start_dt
    total_sec = int(delta.total_seconds())
    h, remainder = divmod(total_sec, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_conditions(flags, is_long, is_short):
    """Helper to strictly format conditions as True/False for both LONG and SHORT, plus counts."""
    print(f"  1. Sine Scale          LONG: {str(flags['sine_long']):<5} | SHORT: {str(flags['sine_short']):<5}")
    print(f"  2. Extrema Cycle       LONG: {str(flags['cycle_long']):<5} | SHORT: {str(flags['cycle_short']):<5}")
    print(f"  3. Momentum            LONG: {str(flags['mom_long']):<5} | SHORT: {str(flags['mom_short']):<5}")
    print(f"  4. Volume              LONG: {str(flags['vol_long']):<5} | SHORT: {str(flags['vol_short']):<5}")
    print(f"  5. FFT Predom Freq     LONG: {str(flags['fft_long']):<5} | SHORT: {str(flags['fft_short']):<5}")
    
    # Calculate how many are true for each side
    long_true_count = sum([flags['sine_long'], flags['cycle_long'], flags['mom_long'], flags['vol_long'], flags['fft_long']])
    short_true_count = sum([flags['sine_short'], flags['cycle_short'], flags['mom_short'], flags['vol_short'], flags['fft_short']])
    
    print(f"  ──────────────────────────────────────────────────")
    print(f"  True Count:         LONG: {long_true_count}/5      | SHORT: {short_true_count}/5")
    print(f"  Overall LONG:  {is_long}")
    print(f"  Overall SHORT: {is_short}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"HFT KuCoin Bot v5.1 Configuration:")
    print(f"{'='*60}")
    print(f"Symbol:        {TRADE_SYMBOL}")
    print(f"Leverage:      {LEVERAGE}x")
    print(f"RT Fee ROE:    {RT_FEE_ROE_PCT}%")
    print(f"Target TP:     {TAKE_PROFIT_ROE}% Gross ROE ({NET_PROFIT_ROE}% Net ROE)")
    print(f"Stop Loss:     {STOP_LOSS_ROE}% ROE")
    print(f"TP Price Move: {TP_PRICE_PCT*100:.3f}%")
    print(f"SL Price Move: {SL_PRICE_PCT*100:.3f}%")
    print(f"Min Balance:   {MIN_BALANCE_USDT} USDT")
    print(f"Conditions:    Strictly 5 (Sine Scale, Extrema Cycle, Momentum, Volume, FFT)")
    print(f"Loop:          {LOOP_SLEEP}s")
    print(f"API Connected: {API_CONNECTED}")
    print(f"{'='*60}\n")

    saved_state = load_trade_state()
    if saved_state:
        print("  [recovery] Found leftover state — verifying...")
        print(f"    State: {saved_state.get('mode', 'unknown')} {saved_state.get('side', 'unknown')}")
        print(f"    Entry: {saved_state.get('entry_price', 0):.2f}")
        
    loop_count = 0

    while True:
        try:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            loop_count += 1

            # Fetch data
            candle_map = fetch_candle_map(TRADE_SYMBOL)
            
            # Check for live position on exchange
            pos = get_position_info(TRADE_SYMBOL)
            current_price = get_price(TRADE_SYMBOL)
            
            # Determine balance
            balance = get_account_balance()
            has_sufficient_balance = balance >= MIN_BALANCE_USDT
            
            # ═══════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT PHASE - LIVE POSITION ON EXCHANGE
            # ═══════════════════════════════════════════════════════════════
            if pos["is_open"]:
                roe = pos["roe_pct"]
                side = pos["side"]
                entry_price = pos["entry_price"]
                mark_price = pos["mark_price"]
                
                sig = compute_signals(candle_map)
                
                if loop_count % 5 == 0:  # Print status every 15 seconds
                    print(f"\n[{now_str}] ╔══════ LIVE POSITION ══════╗")
                    print(f"  Side:       {side.upper()}")
                    print(f"  Entry:      {entry_price:.2f}")
                    print(f"  Mark:       {mark_price:.2f}")
                    print(f"  ROE:        {roe:+.2f}%")
                    print(f"  TP Target:  {TAKE_PROFIT_ROE}% ROE")
                    print(f"  SL Target:  {STOP_LOSS_ROE}% ROE")
                    print(f"  ─────────────────────────────")
                    
                    if sig:
                        print(f"  CONDITIONS (monitoring only - no new entry):")
                        print_conditions(sig["cond_flags"], sig["is_long"], sig["is_short"])
                    
                    print(f"  ═════════════════════════════")
                
                # Check TP/SL
                reason = None
                if roe >= TAKE_PROFIT_ROE:
                    reason = "TAKE PROFIT"
                elif roe <= STOP_LOSS_ROE:
                    reason = "STOP LOSS"
                    
                if reason:
                    print(f"  >>> [{reason}] Triggered at {roe:+.2f}% ROE. Closing position...")
                    client.close_position(TRADE_SYMBOL)
                    
                    if side == "long":
                        price_change_pct = ((mark_price - entry_price) / entry_price) * 100
                    else:
                        price_change_pct = ((entry_price - mark_price) / entry_price) * 100
                    gross_roe = price_change_pct * LEVERAGE
                    net_roe = gross_roe - RT_FEE_ROE_PCT
                    
                    start_dt = datetime.datetime.strptime(saved_state["entry_time"], "%Y-%m-%d %H:%M:%S")
                    duration = format_duration(start_dt, datetime.datetime.now())
                    
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
                        "duration": duration,
                        "reason": reason
                    }
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                    print(f"  [TRADE CLOSED] Written to {TRADES_LOG_FILE}")
                    
                time.sleep(LOOP_SLEEP)
                continue

            # ═══════════════════════════════════════════════════════════════
            # SIMULATION POSITION MANAGEMENT
            # ═══════════════════════════════════════════════════════════════
            if saved_state and saved_state.get("mode") == "SIMULATION":
                sim_entry_price = saved_state["entry_price"]
                sim_side = saved_state["side"]
                sim_entry_time = saved_state["entry_time"]
                
                hit, reason, sim_roe = check_sim_tp_sl(sim_entry_price, current_price, sim_side)
                sig = compute_signals(candle_map)
                
                if loop_count % 5 == 0:
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    elapsed = format_duration(start_dt, now)
                    
                    print(f"\n[{now_str}] ╔══════ SIMULATION POSITION ══════╗")
                    print(f"  Side:       {sim_side.upper()}")
                    print(f"  Entry:      {sim_entry_price:.2f}")
                    print(f"  Current:    {current_price:.2f}")
                    print(f"  Sim ROE:    {sim_roe:+.2f}% (Net)")
                    print(f"  Elapsed:    {elapsed}")
                    print(f"  TP Target:  {TAKE_PROFIT_ROE}% ROE")
                    print(f"  SL Target:  {STOP_LOSS_ROE}% ROE")
                    print(f"  ─────────────────────────────")
                    
                    if sig:
                        print(f"  CONDITIONS (monitoring only):")
                        print_conditions(sig["cond_flags"], sig["is_long"], sig["is_short"])
                    
                    print(f"  ═══════════════════════════════")
                
                if hit and reason:
                    print(f"  >>> [SIM {reason}] Triggered at {sim_roe:+.2f}% ROE")
                    
                    if sim_side == "long":
                        price_change_pct = ((current_price - sim_entry_price) / sim_entry_price) * 100
                    else:
                        price_change_pct = ((sim_entry_price - current_price) / sim_entry_price) * 100
                    gross_roe = price_change_pct * LEVERAGE
                    
                    start_dt = datetime.datetime.strptime(sim_entry_time, "%Y-%m-%d %H:%M:%S")
                    duration = format_duration(start_dt, now)
                    
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
                        "roe": sim_roe,
                        "duration": duration,
                        "reason": reason
                    }
                    write_trade_to_journal(trade_result)
                    clear_trade_state()
                    saved_state = None
                    print(f"  [SIM TRADE CLOSED] Written to {TRADES_LOG_FILE}")
                    
                time.sleep(LOOP_SLEEP)
                continue

            # ═══════════════════════════════════════════════════════════════
            # ORPHANED STATE CLEANUP
            # ═══════════════════════════════════════════════════════════════
            if saved_state and saved_state.get("mode") == "LIVE" and not pos["is_open"]:
                print("  [recovery] Live state active but no position on exchange. Clearing state.")
                clear_trade_state()
                saved_state = None

            # ═══════════════════════════════════════════════════════════════
            # SIGNAL SCANNING PHASE (Only if FLAT - no position anywhere)
            # ═══════════════════════════════════════════════════════════════
            if not pos["is_open"] and not saved_state:
                sig = compute_signals(candle_map)
                if not sig:
                    if loop_count % 20 == 0: 
                        print(f"[{now_str}] Gathering 1m data...")
                    time.sleep(LOOP_SLEEP)
                    continue

                # Print condition analysis with TRUE/FALSE formatting and Counts
                if loop_count % 5 == 0:
                    print(f"\n[{now_str}] Scanning Conditions (FLAT):")
                    print_conditions(sig["cond_flags"], sig["is_long"], sig["is_short"])
                    print(f"  ──────────────────────────────────────────────────")
                    print(f"  Balance: {balance:.2f} USDT | Sufficient: {has_sufficient_balance}")

                # ENTRY LOGIC - LONG
                if sig["is_long"]:
                    print(f"\n  *** ALL 5 CONDITIONS MET FOR LONG ***")
                    
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing LONG entry...")
                        if execute_entry(TRADE_SYMBOL, "buy", balance, sig["price"]):
                            sl_price, tp_price = calculate_sl_tp(sig["price"], "long")
                            saved_state = {
                                "active": True, 
                                "mode": "LIVE",
                                "side": "long", 
                                "entry_price": sig["price"],
                                "sl": sl_price, 
                                "tp": tp_price, 
                                "entry_time": now_str
                            }
                            save_trade_state(saved_state)
                            print(f"  [LIVE] State saved. Monitoring position...")
                    else:
                        print(f"  [SIMULATION] Insufficient balance ({balance:.2f} < {MIN_BALANCE_USDT} USDT)")
                        print(f"  [SIMULATION] Starting SIMULATED LONG trade...")
                        sl_price, tp_price = calculate_sl_tp(sig["price"], "long")
                        saved_state = {
                            "active": True,
                            "mode": "SIMULATION",
                            "side": "long",
                            "entry_price": sig["price"],
                            "sl": sl_price,
                            "tp": tp_price,
                            "entry_time": now_str
                        }
                        save_trade_state(saved_state)
                        print(f"  [SIMULATION] State saved. Monitoring simulated position...")

                # ENTRY LOGIC - SHORT
                elif sig["is_short"]:
                    print(f"\n  *** ALL 5 CONDITIONS MET FOR SHORT ***")
                    
                    if has_sufficient_balance:
                        print(f"  [LIVE] Executing SHORT entry...")
                        if execute_entry(TRADE_SYMBOL, "sell", balance, sig["price"]):
                            sl_price, tp_price = calculate_sl_tp(sig["price"], "short")
                            saved_state = {
                                "active": True,
                                "mode": "LIVE",
                                "side": "short",
                                "entry_price": sig["price"],
                                "sl": sl_price,
                                "tp": tp_price,
                                "entry_time": now_str
                            }
                            save_trade_state(saved_state)
                            print(f"  [LIVE] State saved. Monitoring position...")
                    else:
                        print(f"  [SIMULATION] Insufficient balance ({balance:.2f} < {MIN_BALANCE_USDT} USDT)")
                        print(f"  [SIMULATION] Starting SIMULATED SHORT trade...")
                        sl_price, tp_price = calculate_sl_tp(sig["price"], "short")
                        saved_state = {
                            "active": True,
                            "mode": "SIMULATION",
                            "side": "short",
                            "entry_price": sig["price"],
                            "sl": sl_price,
                            "tp": tp_price,
                            "entry_time": now_str
                        }
                        save_trade_state(saved_state)
                        print(f"  [SIMULATION] State saved. Monitoring simulated position...")

            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            print("\n[Bot] Shutting down safely.")
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()