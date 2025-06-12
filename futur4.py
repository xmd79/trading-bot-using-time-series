import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from decimal import Decimal, getcontext
import requests

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"  # Futures use USDC
LEVERAGE = 20

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})
client.API_URL = 'https://fapi.binance.com'  # Futures API endpoint

# Set leverage for the symbol
try:
    client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
    print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
except BinanceAPIException as e:
    print(f"Error setting leverage: {e.message}")

# Utility Functions
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1000):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=1000, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = []
            for k in klines:
                candle = {
                    "time": k[0] / 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "timeframe": timeframe
                }
                candles.append(candle)
            return candles
        except BinanceAPIException as e:
            print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.Timeout as e:
            print(f"Read Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts. Skipping timeframe.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                return price
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            print(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    return Decimal('0.0')

def get_balance(asset='USDC'):
    try:
        account = client.futures_account()
        if 'assets' not in account:
            print("No 'assets' key in futures account response. Full response:")
            print(account)
            return Decimal('0.0')

        for asset_info in account['assets']:
            if asset_info.get('asset') == asset:
                wallet = Decimal(str(asset_info.get('walletBalance', '0.0')))
                print(f"{asset} wallet balance: {wallet:.25f}")
                return wallet

        print(f"{asset} not found in futures account balances.")
        return Decimal('0.0')

    except BinanceAPIException as e:
        print(f"Binance API exception while fetching {asset} balance: {e.message}")
    except Exception as e:
        print(f"Unexpected error fetching {asset} balance: {e}")
    return Decimal('0.0')

def get_position():
    try:
        positions = client.futures_position_information(symbol=TRADE_SYMBOL)
        if not positions:
            print(f"No position data returned for {TRADE_SYMBOL}. Assuming no open position.")
            return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0')}
        position = positions[0]
        quantity = Decimal(str(position['positionAmt']))
        entry_price = Decimal(str(position['entryPrice']))
        return {
            "quantity": quantity,
            "entry_price": entry_price,
            "side": "LONG" if quantity > Decimal('0') else "SHORT" if quantity < Decimal('0') else "NONE",
            "unrealized_pnl": Decimal(str(position['unrealizedProfit']))
        }
    except BinanceAPIException as e:
        print(f"Error fetching position info: {e.message}")
        return {"quantity": Decimal('0.0'), "entry_price": Decimal('0.0'), "side": "NONE", "unrealized_pnl": Decimal('0.0')}

# Analysis Functions
def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in candle_map:
        buy_volume[timeframe] = []
        sell_volume[timeframe] = []
        for candle in candle_map[timeframe]:
            total_volume = Decimal(str(candle["volume"]))
            close_price = Decimal(str(candle["close"]))
            open_price = Decimal(str(candle["open"]))
            high_price = Decimal(str(candle["high"]))
            low_price = Decimal(str(candle["low"]))
            if high_price == low_price:
                buy_vol = total_volume / Decimal('2')
                sell_vol = total_volume / Decimal('2')
            else:
                price_range = high_price - low_price
                bullish_strength = (close_price - low_price) / price_range if price_range > Decimal('0') else Decimal('0.5')
                bearish_strength = (high_price - close_price) / price_range if price_range > Decimal('0') else Decimal('0.5')
                buy_vol = total_volume * bullish_strength
                sell_vol = total_volume * bearish_strength
            buy_volume[timeframe].append(buy_vol)
            sell_volume[timeframe].append(sell_vol)
    return buy_volume, sell_volume

def calculate_volume_ratio(buy_volume, sell_volume):
    volume_ratio = {}
    for timeframe in buy_volume.keys():
        buy_vols = np.array([float(v) for v in buy_volume[timeframe][-3:]], dtype=np.float64)
        sell_vols = np.array([float(v) for v in sell_volume[timeframe][-3:]], dtype=np.float64)
        if len(buy_vols) >= 3:
            buy_ema = Decimal(str(talib.EMA(buy_vols, timeperiod=3)[-1]))
            sell_ema = Decimal(str(talib.EMA(sell_vols, timeperiod=3)[-1]))
        else:
            buy_ema = sum(buy_volume[timeframe][-3:]) / Decimal(str(len(buy_volume[timeframe][-3:]))) if buy_volume[timeframe] else Decimal('0')
            sell_ema = sum(sell_volume[timeframe][-3:]) / Decimal(str(len(sell_volume[timeframe][-3:]))) if sell_volume[timeframe] else Decimal('0')
        total_ema = buy_ema + sell_ema
        if total_ema > Decimal('0'):
            buy_ratio = (buy_ema / total_ema) * Decimal('100')
            sell_ratio = Decimal('100') - buy_ratio
            status = "Bullish" if buy_ratio > Decimal('50') else "Bearish" if buy_ratio < Decimal('50') else "Neutral"
        else:
            buy_ratio = Decimal('0')
            sell_ratio = Decimal('0')
            status = "No Activity"
        volume_ratio[timeframe] = {"buy_ratio": buy_ratio, "sell_ratio": sell_ratio, "status": status}
        print(f"{timeframe} - Smoothed Buy Ratio: {buy_ratio:.25f}%, Sell Ratio: {sell_ratio:.25f}%, Status: {status}")
    return volume_ratio

# Main Analysis Loop
def main():
    timeframes = ["1m", "3m", "5m"]
    print("Futures Analysis Bot Initialized!")
    
    while True:
        try:
            current_local_time = datetime.datetime.now()
            current_local_time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nCurrent Time: {current_local_time_str}")

            candle_map = fetch_candles_in_parallel(timeframes)
            if not candle_map or not any(candle_map.values()):
                print("No candle data available. Retrying in 60 seconds.")
                time.sleep(60)
                continue

            current_price = get_current_price()
            if current_price <= Decimal('0'):
                print(f"Warning: Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")
                time.sleep(60)
                continue

            usdc_balance = get_balance('USDC')
            position = get_position()

            # Initialize conditions (single layer)
            conditions_long = {
                "volume_bullish_1m": False,
                "volume_bullish_3m": False,
                "volume_bullish_5m": False,
                "momentum_positive_1m": False,
                "sma_bullish_1m": False
            }
            conditions_short = {
                "volume_bearish_1m": False,
                "volume_bearish_3m": False,
                "volume_bearish_5m": False,
                "momentum_negative_1m": False,
                "sma_bearish_1m": False
            }

            # Volume analysis (across all timeframes)
            buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
            volume_ratio = calculate_volume_ratio(buy_volume, sell_volume)

            # 1m-specific analysis for momentum and SMA
            if "1m" in candle_map and candle_map["1m"]:
                print("\n--- 1m Timeframe Analysis (Momentum and SMA) ---")
                closes = [candle["close"] for candle in candle_map["1m"]]
                current_close = Decimal(str(closes[-1]))

                # Momentum
                valid_closes = np.array([float(c) for c in closes if not np.isnan(c) and c > 0], dtype=np.float64)
                momentum = talib.MOM(valid_closes, timeperiod=14)
                if len(momentum) > 0 and not np.isnan(momentum[-1]):
                    current_momentum = Decimal(str(momentum[-1]))
                    conditions_long["momentum_positive_1m"] = current_momentum > Decimal('0')
                    conditions_short["momentum_negative_1m"] = current_momentum < Decimal('0')
                    print(f"1m Momentum: {current_momentum:.25f} - {'Positive' if conditions_long['momentum_positive_1m'] else 'Negative' if conditions_short['momentum_negative_1m'] else 'Neutral'}")
                else:
                    conditions_long["momentum_positive_1m"] = False
                    conditions_short["momentum_negative_1m"] = False
                    print("1m Momentum: Insufficient data or invalid")

                # SMA stack
                sma_lengths = [5, 56, 360]
                smas = {length: Decimal(str(talib.SMA(valid_closes, timeperiod=length)[-1])) for length in sma_lengths if len(valid_closes) >= length and not np.isnan(talib.SMA(valid_closes, timeperiod=length)[-1])}
                print("Simple Moving Averages (1m):")
                for length, sma in smas.items():
                    print(f"SMA-{length}: {sma:.25f}, Current Close: {current_close:.25f} - {'Above' if current_close > sma else 'Below'}")
                if all(length in smas for length in [56, 360]):
                    sma56 = smas[56]
                    sma360 = smas[360]
                    if current_close > sma56 and sma56 > sma360:
                        conditions_long["sma_bullish_1m"] = True
                        print("1m SMA Bullish Signal: close > SMA56 > SMA360")
                    elif current_close < sma56 and sma56 < sma360:
                        conditions_short["sma_bearish_1m"] = True
                        print("1m SMA Bearish Signal: close < SMA56 < SMA360")
                    else:
                        print("1m SMA: No clear bullish or bearish signal")

            # Volume conditions for all timeframes
            for timeframe in timeframes:
                buy_vol = buy_volume.get(timeframe, [Decimal('0')])[-1]
                sell_vol = sell_volume.get(timeframe, [Decimal('0')])[-1]
                conditions_long[f"volume_bullish_{timeframe}"] = buy_vol > sell_vol
                conditions_short[f"volume_bearish_{timeframe}"] = sell_vol > buy_vol
                print(f"Volume Bullish ({timeframe}): {buy_vol:.25f}, Bearish: {sell_vol:.25f}, Bullish Condition: {conditions_long[f'volume_bullish_{timeframe}']}, Bearish Condition: {conditions_short[f'volume_bearish_{timeframe}']}")

            # Validate condition pairs for symmetry
            condition_pairs = [
                ("volume_bullish_1m", "volume_bearish_1m"),
                ("volume_bullish_3m", "volume_bearish_3m"),
                ("volume_bullish_5m", "volume_bearish_5m"),
                ("momentum_positive_1m", "momentum_negative_1m"),
                ("sma_bullish_1m", "sma_bearish_1m")
            ]
            for long_cond, short_cond in condition_pairs:
                long_val = conditions_long[long_cond]
                short_val = conditions_short[short_cond]
                if long_val and short_val:
                    print(f"Conflict in {long_cond}/{short_cond}: Both True. Setting both to False.")
                    conditions_long[long_cond] = False
                    conditions_short[short_cond] = False
                elif not long_val and not short_val:
                    tf = long_cond.split('_')[-1]
                    if "momentum" in long_cond or "sma" in long_cond:
                        # Already handled in 1m analysis
                        pass
                    else:
                        buy_vol = buy_volume.get(tf, [Decimal('0')])[-1]
                        sell_vol = sell_volume.get(tf, [Decimal('0')])[-1]
                        conditions_long[f"volume_bullish_{tf}"] = buy_vol > sell_vol
                        conditions_short[f"volume_bearish_{tf}"] = sell_vol > buy_vol

            # Evaluate signals
            long_signal = all(conditions_long.values())
            short_signal = all(conditions_short.values())

            # Print condition states
            print("\nTrade Signal Status:")
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'}")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'}")
            print("\nLong Conditions Status:")
            for condition, status in conditions_long.items():
                print(f"{condition}: {'True' if status else 'False'}")
            print("\nShort Conditions Status:")
            for condition, status in conditions_short.items():
                print(f"{condition}: {'True' if status else 'False'}")

            # Calculate condition summary
            long_true = sum(1 for val in conditions_long.values() if val)
            long_false = len(conditions_long) - long_true
            short_true = sum(1 for val in conditions_short.values() if val)
            short_false = len(conditions_short) - short_true
            print(f"\nLong Conditions Summary: {long_true} True, {long_false} False")
            print(f"Short Conditions Summary: {short_true} True, {short_false} False")

            # Determine final signal
            signal = "NO_SIGNAL"
            if long_signal and not short_signal:
                signal = "LONG"
            elif short_signal and not long_signal:
                signal = "SHORT"
            elif long_signal and short_signal:
                signal = "NO_SIGNAL"
                print("Conflict: Both LONG and SHORT signals active. Setting to NO_SIGNAL.")
            print(f"Final Signal: {signal}")

            # Position status
            if position["side"] != "NONE":
                print("\nCurrent Position Status:")
                print(f"Position Side: {position['side']}")
                print(f"Quantity: {position['quantity']:.25f} BTC")
                print(f"Entry Price: {position['entry_price']:.25f} USDC")
                print(f"Current Price: {current_price:.25f} USDC")
                print(f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC")
                current_balance = usdc_balance + position['unrealized_pnl']
                print(f"Current Total Balance: {current_balance:.25f} USDC")
            else:
                print(f"\nNo open position. USDC Balance: {usdc_balance:.25f}")

            print(f"\nCurrent USDC Balance: {usdc_balance:.25f}")
            print(f"Current Position: {position['side']}, Quantity: {position['quantity']:.25f} BTC")
            print(f"Current Price: {current_price:.25f}\n")

            # Cleanup
            del candle_map
            gc.collect()
            time.sleep(5)

        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()