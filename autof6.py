import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.fftpack as fftpack
import math
from decimal import Decimal, getcontext
import requests

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"
LEVERAGE = 20
TAKE_PROFIT_ROI = Decimal('2.55')
STOP_LOSS_ROI = Decimal('-25.50')

# Load credentials
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Initialize Binance client
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})
client.API_URL = 'https://fapi.binance.com'

# Validate symbol
def validate_symbol(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols']]
        if symbol not in symbols:
            print(f"Error: {symbol} not found in futures market. Available symbols: {symbols[:10]}...")
            return False
        return True
    except BinanceAPIException as e:
        print(f"Error validating symbol {symbol}: {e.message}")
        return False

# Set leverage
try:
    if validate_symbol(TRADE_SYMBOL):
        client.futures_change_leverage(symbol=TRADE_SYMBOL, leverage=LEVERAGE)
        print(f"Leverage set to {LEVERAGE}x for {TRADE_SYMBOL}")
    else:
        raise Exception(f"Invalid symbol {TRADE_SYMBOL}")
except BinanceAPIException as e:
    print(f"Error setting leverage: {e.message}")

# Utility Functions
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=100, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = [{
                "time": k[0] / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "timeframe": timeframe
            } for k in klines]
            return candles
        except (BinanceAPIException, requests.exceptions.ReadTimeout) as e:
            print(f"Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                return price
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except (BinanceAPIException, requests.exceptions.ReadTimeout) as e:
            print(f"Error fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print("Failed to fetch price.")
    return Decimal('0.0')

def get_balance(asset='USDC', retries=5, delay=5):
    for attempt in range(retries):
        try:
            account = client.futures_account()
            for asset_info in account['assets']:
                if asset_info['asset'] == asset:
                    balance = Decimal(str(asset_info['availableBalance']))
                    print(f"Balance: {balance:.25f} {asset}")
                    if balance <= Decimal('0'):
                        print("Zero balance detected. Running in simulation mode.")
                    return balance
            print(f"No {asset} found.")
            return Decimal('0.0')
        except (BinanceAPIException, requests.exceptions.ReadTimeout) as e:
            print(f"Error fetching balance (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print("Failed to fetch balance.")
    return Decimal('0.0')

def get_position(retries=5, delay=5):
    for attempt in range(retries):
        try:
            positions = client.futures_position_information(symbol=TRADE_SYMBOL)
            if not positions:
                print(f"No positions found for {TRADE_SYMBOL} (attempt {attempt + 1}/{retries}).")
                return {
                    "quantity": Decimal('0.0'),
                    "entry_price": Decimal('0.0'),
                    "side": "NONE",
                    "unrealized_pnl": Decimal('0.0')
                }
            position = positions[0]
            quantity = Decimal(str(position['positionAmt']))
            return {
                "quantity": quantity,
                "entry_price": Decimal(str(position['entryPrice'])),
                "side": "LONG" if quantity > Decimal('0') else "SHORT" if quantity < Decimal('0') else "NONE",
                "unrealized_pnl": Decimal(str(position['unrealizedProfit']))
            }
        except (BinanceAPIException, requests.exceptions.ReadTimeout) as e:
            print(f"Error fetching position (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch position after {retries} attempts.")
    return {
        "quantity": Decimal('0.0'),
        "entry_price": Decimal('0.0'),
        "side": "NONE",
        "unrealized_pnl": Decimal('0.0')
    }

def get_symbol_lot_size_info(symbol, retries=5, delay=5):
    for attempt in range(retries):
        try:
            exchange_info = client.futures_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    for f in symbol_info['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            return {
                                'minQty': Decimal(str(f['minQty'])),
                                'stepSize': Decimal(str(f['stepSize']))
                            }
            print(f"No lot size info for {symbol}. Using defaults.")
            return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}
        except (BinanceAPIException, requests.exceptions.ReadTimeout) as e:
            print(f"Error fetching lot size (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print("Failed to fetch lot size.")
    return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}

def open_long_position(balance):
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print("Invalid price for long position.")
            return None, None, None, None
        if balance <= Decimal('0'):
            print("Simulating LONG: No balance available.")
            notional = Decimal('1000.0') * Decimal(str(LEVERAGE))  # Simulate with $1000
            quantity = notional / current_price
            step_precision = int(-math.log10(float(step_size))) if step_size > 0 else 8
            quantity = (quantity // step_size) * step_size
            quantity = quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = quantity * current_price / Decimal(str(LEVERAGE))
            print(f"Simulated LONG: Qty {quantity:.25f}, Cost {cost:.25f}, Price {current_price:.25f}")
            return current_price, quantity, datetime.datetime.now(), cost
        notional = balance * Decimal(str(LEVERAGE))
        raw_quantity = notional / current_price
        step_precision = int(-math.log10(float(step_size))) if step_size > 0 else 8
        quantity = (raw_quantity // step_size) * step_size
        quantity = quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = quantity * current_price / Decimal(str(LEVERAGE))
        if quantity < min_trade_size:
            print(f"Quantity {quantity:.25f} below min trade size {min_trade_size:.25f}.")
            return None, None, None, None
        if cost > balance:
            print(f"Cost {cost:.25f} exceeds balance {balance:.25f}.")
            return None, None, None, None
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side='BUY',
            type='MARKET',
            quantity=float(quantity)
        )
        print(f"Long opened: {order}")
        return Decimal(str(order['avgPrice'])), quantity, datetime.datetime.now(), cost
    except BinanceAPIException as e:
        print(f"Error opening long: {e.message}")
        return None, None, None, None

def open_short_position(balance):
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print("Invalid price for short position.")
            return None, None, None, None
        if balance <= Decimal('0'):
            print("Simulating SHORT: No balance available.")
            notional = Decimal('1000.0') * Decimal(str(LEVERAGE))  # Simulate with $1000
            quantity = notional / current_price
            step_precision = int(-math.log10(float(step_size))) if step_size > 0 else 8
            quantity = (quantity // step_size) * step_size
            quantity = quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = quantity * current_price / Decimal(str(LEVERAGE))
            print(f"Simulated SHORT: Qty {quantity:.25f}, Cost {cost:.25f}, Price {current_price:.25f}")
            return current_price, quantity, datetime.datetime.now(), cost
        notional = balance * Decimal(str(LEVERAGE))
        raw_quantity = notional / current_price
        step_precision = int(-math.log10(float(step_size))) if step_size > 0 else 8
        quantity = (raw_quantity // step_size) * step_size
        quantity = quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = quantity * current_price / Decimal(str(LEVERAGE))
        if quantity < min_trade_size:
            print(f"Quantity {quantity:.25f} below min trade size {min_trade_size:.25f}.")
            return None, None, None, None
        if cost > balance:
            print(f"Cost {cost:.25f} exceeds balance {balance:.25f}.")
            return None, None, None, None
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side='SELL',
            type='MARKET',
            quantity=float(quantity)
        )
        print(f"Short opened: {order}")
        return Decimal(str(order['avgPrice'])), quantity, datetime.datetime.now(), cost
    except BinanceAPIException as e:
        print(f"Error opening short: {e.message}")
        return None, None, None, None

def close_position(side, quantity):
    try:
        if quantity == Decimal('0'):
            print("No position to close (quantity is zero).")
            return False
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print("Invalid price for closing.")
            return False
        step_precision = int(-math.log10(float(step_size))) if step_size > 0 else 8
        close_quantity = (abs(quantity) // step_size) * step_size
        close_quantity = close_quantity.quantize(Decimal('0.' + '0' * step_precision))
        if close_quantity < min_trade_size:
            print(f"Close quantity {close_quantity:.25f} below min {min_trade_size:.25f}.")
            return False
        usdc_balance = get_balance('USDC')
        if usdc_balance <= Decimal('0'):
            print(f"Simulating CLOSE {side}: Qty {close_quantity:.25f}, Price {current_price:.25f}")
            return True
        order_side = 'SELL' if side == 'LONG' else 'BUY'
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side=order_side,
            type='MARKET',
            quantity=float(close_quantity)
        )
        print(f"Position closed ({side}): {order}")
        return True
    except BinanceAPIException as e:
        print(f"Error closing {side}: {e.message}")
        return False

def check_exit_condition(initial_investment, position, current_price):
    if initial_investment <= Decimal('0') or position['quantity'] == Decimal('0'):
        print("Invalid investment or position for exit check.")
        return False, None
    unrealized_pnl = position['unrealized_pnl']
    current_balance = get_balance('USDC') + unrealized_pnl
    roi = ((current_balance - initial_investment) / initial_investment) * Decimal('100')
    take_profit = roi >= TAKE_PROFIT_ROI
    stop_loss = roi <= STOP_LOSS_ROI
    print(f"Exit Check: ROI: {roi:.2f}%, TP: {TAKE_PROFIT_ROI:.2f}%, SL: {STOP_LOSS_ROI:.2f}%")
    return take_profit or stop_loss, "TP" if take_profit else "SL" if stop_loss else None

# Analysis Functions
def backtest_model(candles):
    closes = np.array([float(candle["close"]) for candle in candles], dtype=np.float64)
    X = np.arange(len(closes)).reshape(-1, 1)
    y = closes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = Decimal(str(mean_absolute_error(y_test, predictions)))
    return model, mae, predictions, y_test

def forecast_next_price(model, num_steps=1):
    last_index = model.n_features_in_
    future_steps = np.arange(last_index, last_index + num_steps).reshape(-1, 1)
    return [Decimal(str(price)) for price in model.predict(future_steps)]

def calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=Decimal('0.05')):
    close_prices = np.array([float(x) for x in close_prices if not np.isnan(x) and x > 0], dtype=np.float64)
    if len(close_prices) == 0:
        return None, None, None, None, None, None, None
    min_close = Decimal(str(np.nanmin(close_prices)))
    max_close = Decimal(str(np.nanmax(close_prices)))
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = Decimal(str(np.nanmin(momentum)))
    max_momentum = Decimal(str(np.nanmax(momentum)))
    min_percentage_custom = Decimal(str(minimum_percentage)) / Decimal('100')
    max_percentage_custom = Decimal(str(maximum_percentage)) / Decimal('100')
    min_threshold = min(min_close - (max_close - min_close) * min_percentage_custom, Decimal(str(close_prices[-1])))
    max_threshold = max(max_close + (max_close - min_close) * max_percentage_custom, Decimal(str(close_prices[-1])))
    range_price = [Decimal(str(x)) for x in np.linspace(float(close_prices[-1]) * (1 - float(range_distance)), float(close_prices[-1]) * (1 + float(range_distance)), num=50)]
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < float(min_threshold), float(min_threshold), close_prices)
        filtered_close = np.where(filtered_close > float(max_threshold), float(max_threshold), filtered_close)
    avg_mtf = Decimal(str(np.nanmean(filtered_close)))
    current_momentum = Decimal(str(momentum[-1]))
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * Decimal('100') if max_momentum != min_momentum else Decimal('NaN')
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * Decimal('100') if max_momentum != min_momentum else Decimal('NaN')
    percent_to_min_combined = (Decimal(str(minimum_percentage)) + percent_to_min_momentum) / Decimal('2')
    percent_to_max_combined = (Decimal(str(maximum_percentage)) + percent_to_max_momentum) / Decimal('2')
    momentum_signal = percent_to_max_combined - percent_to_min_combined
    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum

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
            print(f"{timeframe} - Candle Time: {datetime.datetime.fromtimestamp(candle['time'])} - Buy Volume: {buy_vol:.25f}, Sell Volume: {sell_vol:.25f}")
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

def find_major_reversals(candles, current_close, min_threshold, max_threshold):
    lows = [Decimal(str(candle['low'])) for candle in candles 
            if min_threshold <= Decimal(str(candle['low'])) <= max_threshold]
    highs = [Decimal(str(candle['high'])) for candle in candles 
             if min_threshold <= Decimal(str(candle['high'])) <= max_threshold]
    last_bottom = min(lows) if lows else None
    last_top = max(highs) if highs else None
    closest_reversal = None
    closest_type = None
    current_close_dec = Decimal(str(current_close))
    if not lows and not highs:
        lows = [Decimal(str(candle['low'])) for candle in candles]
        highs = [Decimal(str(candle['high'])) for candle in candles]
        last_bottom = min(lows) if lows else None
        last_top = max(highs) if highs else None
    min_distance = Decimal('Infinity')
    if last_bottom is not None:
        distance_to_bottom = abs(current_close_dec - last_bottom)
        if distance_to_bottom < min_distance:
            min_distance = distance_to_bottom
            closest_reversal = last_bottom
            closest_type = 'DIP'
    if last_top is not None:
        distance_to_top = abs(current_close_dec - last_top)
        if distance_to_top < min_distance:
            min_distance = distance_to_top
            closest_reversal = last_top
            closest_type = 'TOP'
    if closest_type == 'DIP' and closest_reversal >= current_close_dec:
        closest_reversal = last_top
        closest_type = 'TOP' if last_top is not None and last_top <= current_close_dec else None
    elif closest_type == 'TOP' and closest_reversal <= current_close_dec:
        closest_reversal = last_bottom
        closest_type = 'DIP' if last_bottom is not None and last_bottom >= current_close_dec else None
    if closest_type is None and (last_bottom is not None or last_top is not None):
        if last_bottom is not None and last_top is not None:
            closest_reversal = last_bottom if abs(current_close_dec - last_bottom) <= abs(current_close_dec - last_top) else last_top
            closest_type = 'DIP' if closest_reversal == last_bottom else 'TOP'
        elif last_bottom is not None:
            closest_reversal = last_bottom
            closest_type = 'DIP'
        elif last_top is not None:
            closest_reversal = last_top
            closest_type = 'TOP'
    return last_bottom, last_top, closest_reversal, closest_type

def scale_to_sine(close_prices):
    close_prices_np = np.array([float(x) for x in close_prices], dtype=np.float64)
    sine_wave, _ = talib.HT_SINE(close_prices_np)
    current_sine = Decimal(str(np.nan_to_num(sine_wave)[-1]))
    sine_wave_min = Decimal(str(np.nanmin(sine_wave)))
    sine_wave_max = Decimal(str(np.nanmax(sine_wave)))
    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * Decimal('100') if sine_wave_max != sine_wave_min else Decimal('0')
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * Decimal('100') if sine_wave_max != sine_wave_min else Decimal('0')
    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_spectral_analysis(prices):
    prices_np = np.array([float(x) for x in prices], dtype=np.float64)
    fft_result = np.fft.fft(prices_np)
    power_spectrum = np.abs(fft_result) ** 2
    frequencies = np.fft.fftfreq(len(prices_np))
    neg_freqs = [Decimal(str(f)) for f in frequencies[frequencies < 0]]
    neg_powers = [Decimal(str(p)) for p in power_spectrum[frequencies < 0]]
    pos_freqs = [Decimal(str(f)) for f in frequencies[frequencies >= 0]]
    pos_powers = [Decimal(str(p)) for p in power_spectrum[frequencies >= 0]]
    return neg_freqs, neg_powers, pos_freqs, pos_powers

def determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, last_major_reversal_type, buy_volume, sell_volume):
    total_negative_power = sum(negative_powers)
    total_positive_power = sum(positive_powers)
    if last_major_reversal_type == 'DIP':
        return "Bullish" if buy_volume > sell_volume else "Accumulation"
    elif last_major_reversal_type == 'TOP':
        if sell_volume > buy_volume:
            return "Bearish" if total_positive_power > total_negative_power else "Distribution"
        return "Bullish"
    return "Neutral"

def calculate_45_degree_projection(last_bottom, last_top):
    if last_bottom is not None and last_top is not None:
        distance = last_top - last_bottom
        return last_top + distance
    return None

def calculate_wave_price(t, avg, min_price, max_price, omega, phi):
    t_dec = Decimal(str(t))
    avg_dec = Decimal(str(avg))
    min_price_dec = Decimal(str(min_price))
    max_price_dec = Decimal(str(max_price))
    omega_dec = Decimal(str(omega))
    phi_dec = Decimal(str(phi))
    amplitude = (max_price_dec - min_price_dec) / Decimal('2')
    return avg_dec + amplitude * Decimal(str(math.sin(float(omega_dec * t_dec + phi_dec))))

def calculate_independent_wave_price(current_price, avg, min_price, max_price, range_distance):
    current_price_dec = Decimal(str(current_price))
    avg_dec = Decimal(str(avg))
    min_price_dec = Decimal(str(min_price))
    max_price_dec = Decimal(str(max_price))
    range_distance_dec = Decimal(str(range_distance))
    noise = Decimal(str(np.random.uniform(-1, 1))) * range_distance_dec
    return avg_dec + (noise * (max_price_dec - min_price_dec) / Decimal('2'))

def find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_close):
    support_levels = []
    resistance_levels = []
    current_close_dec = Decimal(str(current_close))
    for timeframe in candle_map:
        for candle in candle_map[timeframe]:
            close_dec = Decimal(str(candle["close"]))
            if close_dec < current_close_dec and close_dec >= min_threshold:
                support_levels.append((close_dec, Decimal(str(candle["volume"])), candle["time"]))
            if close_dec > current_close_dec and close_dec <= max_threshold:
                resistance_levels.append((close_dec, Decimal(str(candle["volume"])), candle["time"]))
    support_levels.sort(key=lambda x: x[1], reverse=True)
    resistance_levels.sort(key=lambda x: x[1], reverse=True)
    significant_support = support_levels[:3]
    significant_resistance = resistance_levels[:3]
    print("Most Significant Support Levels (Price, Volume):")
    for price, volume, _ in significant_support:
        print(f"Support Price: {price:.25f}, Volume: {volume:.25f}")
    print("Most Significant Resistance Levels (Price, Volume):")
    for price, volume, _ in significant_resistance:
        print(f"Resistance Price: {price:.25f}, Volume: {volume:.25f}")
    return [level[0] for level in significant_support], [level[0] for level in significant_resistance]

def calculate_stochastic_rsi(close_prices, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3):
    close_prices_np = np.array([float(x) for x in close_prices], dtype=np.float64)
    rsi = talib.RSI(close_prices_np, timeperiod=length_rsi)
    min_rsi = talib.MIN(rsi, timeperiod=length_stoch)
    max_rsi = talib.MAX(rsi, timeperiod=length_stoch)
    stoch_k = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    stoch_k_smooth = talib.EMA(stoch_k, timeperiod=smooth_k)
    stoch_d = talib.EMA(stoch_k_smooth, timeperiod=smooth_d)
    return [Decimal(str(k)) for k in stoch_k_smooth], [Decimal(str(d)) for d in stoch_d]

def calculate_bullish_bearish_volume_ratios(candle_map):
    volume_ratios = {}
    for timeframe, candles in candle_map.items():
        bullish_volume = Decimal('0')
        bearish_volume = Decimal('0')
        recent_volumes = []
        for candle in candles[-5:]:
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
                bearish_volume += total_volume * (Decimal('1') - bullish_strength)
                bullish_volume += total_volume * bullish_strength
            recent_volumes.append((bullish_volume, bearish_volume))
        total_volume = bullish_volume + bearish_volume
        ratio = bullish_volume / bearish_volume if bearish_volume > Decimal('0') else Decimal('Infinity')
        volume_trend = "Increasing Bullish" if len(recent_volumes) >= 2 and recent_volumes[-1][0] > recent_volumes[-2][0] else \
                      "Increasing Bearish" if len(recent_volumes) >= 2 and recent_volumes[-1][1] > recent_volumes[-2][1] else "Stable"
        volume_ratios[timeframe] = {
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "ratio": ratio,
            "status": "Bullish" if ratio > Decimal('1') else "Bearish" if ratio < Decimal('1') else "Neutral",
            "trend": volume_trend
        }
        print(f"{timeframe} - Bullish Volume: {bullish_volume:.25f}, Bearish Volume: {bearish_volume:.25f}, Ratio: {ratio:.25f}, Status: {volume_ratios[timeframe]['status']}, Trend: {volume_trend}")
    return volume_ratios

def analyze_volume_changes_over_time(candle_map):
    volume_trends = {}
    for timeframe, candles in candle_map.items():
        if len(candles) < 3:
            print(f"{timeframe} - Not enough data to analyze volume changes.")
            continue
        volumes = [Decimal(str(candle["volume"])) for candle in candles[-3:]]
        bullish_volumes = []
        bearish_volumes = []
        for candle in candles[-3:]:
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
                buy_vol = total_volume * bullish_strength
                sell_vol = total_volume * (Decimal('1') - bullish_strength)
            bullish_volumes.append(buy_vol)
            bearish_volumes.append(sell_vol)
        last_volume = volumes[-1]
        previous_volume = volumes[-2]
        volume_change = last_volume - previous_volume
        percent_change = (volume_change / previous_volume * Decimal('100')) if previous_volume > Decimal('0') else Decimal('0')
        trend_direction = "Bullish" if bullish_volumes[-1] > bearish_volumes[-1] and bullish_volumes[-1] > bullish_volumes[-2] else \
                         "Bearish" if bearish_volumes[-1] > bullish_volumes[-1] and bearish_volumes[-1] > bearish_volumes[-2] else "Neutral"
        volume_trends[timeframe] = {
            "last_volume": last_volume,
            "previous_volume": previous_volume,
            "change": volume_change,
            "percent_change": percent_change,
            "trend_direction": trend_direction
        }
        print(f"{timeframe} - Last Volume: {last_volume:.25f}, Previous Volume: {previous_volume:.25f}, Change: {volume_change:.25f}, Percent Change: {percent_change:.25f}%, Trend: {trend_direction}")
    return volume_trends

def calculate_fibonacci_levels_from_reversal(last_reversal, max_threshold, min_threshold, last_major_reversal_type, current_price):
    levels = {}
    current_price_dec = Decimal(str(current_price))
    if last_reversal is None:
        last_reversal = current_price_dec
        last_major_reversal_type = 'DIP'
    else:
        last_reversal = Decimal(str(last_reversal))
    if last_major_reversal_type == 'DIP':
        levels['0.0'] = last_reversal
        levels['1.0'] = Decimal(str(max_threshold)) if max_threshold is not None else current_price_dec * Decimal('1.05')
        levels['0.5'] = last_reversal + (levels['1.0'] - last_reversal) / Decimal('2')
    elif last_major_reversal_type == 'TOP':
        levels['0.0'] = last_reversal
        levels['1.0'] = Decimal(str(min_threshold)) if min_threshold is not None else current_price_dec * Decimal('0.95')
        levels['0.5'] = last_reversal - (last_reversal - levels['1.0']) / Decimal('2')
    return levels

def forecast_fibo_target_price(fib_levels):
    return fib_levels.get('1.0', None)

def forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_price):
    current_price_dec = Decimal(str(current_price))
    if '1m' in volume_ratios and volume_ratios['1m']['buy_ratio'] > Decimal('50'):
        forecasted_price = current_price_dec + (current_price_dec * Decimal('0.0267'))
        print(f"Forecasting a bullish price increase to {forecasted_price:.25f}.")
        return forecasted_price
    elif '1m' in volume_ratios and volume_ratios['1m']['sell_ratio'] > Decimal('50'):
        forecasted_price = current_price_dec - (current_price_dec * Decimal('0.0267'))
        print(f"Forecasting a bearish price decrease to {forecasted_price:.25f}.")
        return forecasted_price
    print("No clear forecast direction based on volume ratios.")
    return None

def check_market_conditions_and_forecast(support_levels, resistance_levels, current_price):
    current_price_dec = Decimal(str(current_price))
    if not support_levels and not resistance_levels:
        print("No support or resistance levels found; trade cautiously.")
        return "No trading signals available."
    first_support = support_levels[0] if support_levels else None
    first_resistance = resistance_levels[0] if resistance_levels else None
    if first_support and current_price_dec < first_support:
        print(f"Current price {current_price_dec:.25f} is below support {first_support:.25f}.")
        return "Current price below key support level; consider shorting."
    elif first_resistance and current_price_dec > first_resistance:
        print(f"Current price {current_price_dec:.25f} is above resistance {first_resistance:.25f}.")
        return "Current price above key resistance level; consider going long."
    print(f"Current price {current_price_dec:.25f} is within support {first_support} and resistance {first_resistance}.")
    return None

def get_target(closes, n_components, last_major_reversal_type, buy_volume, sell_volume):
    closes_np = np.array([float(x) for x in closes], dtype=np.float64)
    fft = fftpack.rfft(closes_np)
    frequencies = fftpack.rfftfreq(len(closes_np))
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    current_close = Decimal(str(closes_np[-1]))
    target_price = Decimal(str(np.nanmax(filtered_signal)))
    current_time = datetime.datetime.now()
    stop_loss = current_close - Decimal(str(np.std(closes_np)))
    market_mood = "Neutral"
    if last_major_reversal_type == 'DIP' and buy_volume > sell_volume:
        market_mood = "Bullish"
    elif last_major_reversal_type == 'TOP' and sell_volume > buy_volume:
        market_mood = "Bearish" if float(sell_volume) >= float(buy_volume) else "Choppy"
    return current_time, current_close, stop_loss, target_price, market_mood

def forecast_price_per_time_pythagorean(timeframe, candles, min_threshold, max_threshold, current_price, time_window_minutes, last_reversal, last_reversal_type):
    current_price_dec = Decimal(str(current_price))
    if min_threshold is None or max_threshold is None:
        min_threshold = current_price_dec * Decimal('0.95')
        max_threshold = current_price_dec * Decimal('1.05')
    else:
        min_threshold = Decimal(str(min_threshold))
        max_threshold = Decimal(str(max_threshold))
    threshold_range = max_threshold - min_threshold
    time_leg = Decimal(str(time_window_minutes))
    hypotenuse = Decimal(str(math.sqrt(float(time_leg**2 + threshold_range**2))))
    price_per_minute = threshold_range / time_leg if time_leg > Decimal('0') else Decimal('0.0')
    forecast_price = max_threshold
    if last_reversal_type == 'DIP' and last_reversal is not None:
        last_reversal = Decimal(str(last_reversal))
        dist_from_dip = current_price_dec - last_reversal
        if dist_from_dip >= threshold_range:
            print(f"Pump incoming detected for {timeframe}: Distance from DIP ({dist_from_dip:.25f}) >= Threshold Range ({threshold_range:.25f})")
            forecast_price = last_reversal + (threshold_range * Decimal('1.5'))
        else:
            forecast_price = max_threshold
    elif last_reversal is None:
        forecast_price = current_price_dec * Decimal('1.05')
    print(f"\n--- Fixed Pythagorean Forecast for {timeframe} ---")
    print(f"Min Threshold: {min_threshold:.25f}")
    print(f"Max Threshold: {max_threshold:.25f}")
    print(f"Threshold Range (Leg B): {threshold_range:.25f}")
    print(f"Time Window (Leg A): {time_leg:.25f} minutes")
    print(f"Hypotenuse (Price-Time Distance): {hypotenuse:.25f}")
    print(f"Base Price Change Rate: {price_per_minute:.25f} USDC per minute")
    if last_reversal_type == 'DIP' and last_reversal is not None:
        print(f"Last DIP Reversal: {last_reversal:.25f}")
        print(f"Distance from DIP to Current Price: {dist_from_dip:.25f}")
    print(f"Fixed Forecast Price: {forecast_price:.25f}")
    return forecast_price, price_per_minute

# Initialize lot size
lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
min_trade_size = lot_size_info['minQty']
step_size = lot_size_info['stepSize']
print(f"Min Trade Size: {min_trade_size:.25f}, Step Size: {step_size:.25f}")

# Trade state
position_open = False
initial_investment = Decimal('0.0')
position_side = "NONE"
quantity = Decimal('0.0')
entry_price = Decimal('0.0')
entry_datetime = None

# Main loop
while True:
    current_time = datetime.datetime.now()
    print(f"\nTime: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    usdc_balance = get_balance('USDC')
    current_price = get_current_price()
    position = get_position()
    position_side = position['side']
    quantity = position['quantity']

    if position_side != "NONE" and not position_open:
        position_open = True
        entry_price = position['entry_price']
        initial_investment = abs(quantity) * entry_price / Decimal(str(LEVERAGE))
        entry_datetime = current_time
        print(f"Position detected: {position_side}, Qty: {quantity:.25f}, Entry: {entry_price:.25f}")

    candle_map = fetch_candles_in_parallel(['1m', '3m', '5m'])
    if not candle_map.get('1m'):
        print("Error: No 1m candles. Skipping cycle.")
        time.sleep(60)
        continue

    # Initialize conditions
    long_conditions = {
        "ML_Forecasted_Price_over_Current_Close": False,
        "current_close_below_average_threshold_5m": False,
        "dip_confirmed_1m": False,
        "dip_confirmed_3m": False,
        "dip_confirmed_5m": False,
        "volume_bullish_1m": False,
        "volume_bullish_3m": False,
        "volume_bullish_5m": False,
        "dist_to_min_1m_less_10pct": False,
        "dist_to_min_3m_less_10pct": False,
        "dist_to_min_5m_less_10pct": False
    }
    short_conditions = {
        "ML_Forecasted_Price_below_Current_Close": False,
        "current_close_above_average_threshold_5m": False,
        "top_confirmed_1m": False,
        "top_confirmed_3m": False,
        "top_confirmed_5m": False,
        "volume_bearish_1m": False,
        "volume_bearish_3m": False,
        "volume_bearish_5m": False,
        "dist_to_max_1m_less_10pct": False,
        "dist_to_max_3m_less_10pct": False,
        "dist_to_max_5m_less_10pct": False
    }

    # Analyze data
    buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
    volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)
    min_threshold, max_threshold, adjusted_forecasted_price = None, None, None
    if "1m" in candle_map and candle_map['1m']:
        model, backtest_mae, last_predictions, actuals = backtest_model(candle_map["1m"])
        print(f"Backtest MAE: {backtest_mae:.25f}")
        forecasted_prices = forecast_next_price(model, num_steps=1)
        closes = [candle['close'] for candle in candle_map["1m"]]
        min_threshold, max_threshold, _, _, _, _, _ = calculate_thresholds(closes)
        adjusted_forecasted_price = min(max(forecasted_prices[-1], min_threshold), max_threshold) if min_threshold is not None and max_threshold is not None else forecasted_prices[-1]
        print(f"Forecasted Price: {adjusted_forecasted_price:.25f}")

    support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold or current_price * Decimal('0.95'), max_threshold or current_price * Decimal('1.05'), current_price)
    volume_ratios_details = calculate_bullish_bearish_volume_ratios(candle_map)
    volume_trends_details = analyze_volume_changes_over_time(candle_map)

    fib_info = {}
    pythagorean_forecasts = {'1m': {'price': None, 'rate': None}, '3m': {'price': None, 'rate': None}, '5m': {'price': None, 'rate': None}}
    last_major_reversal_type = None

    # Calculate confidence scores
    confidence_scores = {}
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map and candle_map[timeframe]:
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = Decimal(str(closes[-1]))
            high_tf = Decimal(str(np.nanmax([float(x) for x in closes])))
            low_tf = Decimal(str(np.nanmin([float(x) for x in closes])))
            min_threshold_tf, max_threshold_tf, avg_mtf, momentum_signal, _, _, _ = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)

            # Confidence for ML forecast
            if timeframe == '1m':
                forecast_diff = abs(adjusted_forecasted_price - current_close) / current_close if adjusted_forecasted_price is not None else Decimal('0')
                confidence_scores['ML_Forecasted_Price'] = float(forecast_diff)

            # Confidence for threshold conditions (5m)
            if timeframe == '5m':
                threshold_diff = abs(current_close - (min_threshold_tf + max_threshold_tf) / Decimal('2')) / current_close if min_threshold_tf is not None and max_threshold_tf is not None else Decimal('0')
                confidence_scores['current_close_threshold_5m'] = float(threshold_diff)

            # Confidence for dip/top confirmation
            _, _, closest_reversal, _ = find_major_reversals(candle_map[timeframe], current_price, min_threshold_tf, max_threshold_tf)
            reversal_confidence = abs(current_close - closest_reversal) / current_close if closest_reversal is not None else Decimal('0')
            confidence_scores[f'reversal_{timeframe}'] = float(reversal_confidence)

            # Confidence for volume conditions
            buy_vol = buy_volume.get(timeframe, [Decimal('0')])[-1]
            sell_vol = sell_volume.get(timeframe, [Decimal('0')])[-1]
            volume_diff = abs(buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > Decimal('0') else Decimal('0')
            confidence_scores[f'volume_{timeframe}'] = float(volume_diff)

            # Confidence for distance to min/max
            dist_to_min = ((current_price - low_tf) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
            dist_to_max = ((high_tf - current_price) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
            confidence_scores[f'dist_{timeframe}'] = float(min(dist_to_min, dist_to_max))

    for timeframe in ['1m', '3m', '5m']:
        if timeframe in candle_map and candle_map[timeframe]:
            print(f"--- {timeframe} ---")
            closes = [candle['close'] for candle in candle_map[timeframe]]
            current_close = Decimal(str(closes[-1]))
            high_tf = Decimal(str(np.nanmax([float(x) for x in closes])))
            low_tf = Decimal(str(np.nanmin([float(x) for x in closes])))
            min_threshold_tf, max_threshold_tf, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_price, min_threshold_tf, max_threshold_tf)
            if closest_reversal is not None:
                print(f"Most Recent Major Reversal Type: {closest_type}")
                print(f"Last Major Reversal Found at Price: {closest_reversal:.25f}")
                last_major_reversal_type = closest_type
            else:
                print("No Major Reversal Found")
            dip_confirmed = closest_type == 'DIP'
            top_confirmed = closest_type == 'TOP'
            long_conditions[f'dip_confirmed_{timeframe}'] = dip_confirmed
            short_conditions[f'top_confirmed_{timeframe}'] = top_confirmed
            if timeframe == '1m':
                long_conditions["ML_Forecasted_Price_over_Current_Close"] = adjusted_forecasted_price is not None and adjusted_forecasted_price > current_close
                short_conditions["ML_Forecasted_Price_below_Current_Close"] = adjusted_forecasted_price is not None and adjusted_forecasted_price < current_close
                long_conditions["volume_bullish_1m"] = buy_volume.get('1m', [Decimal('0')])[-1] > sell_volume.get('1m', [Decimal('0')])[-1]
                short_conditions["volume_bearish_1m"] = sell_volume.get('1m', [Decimal('0')])[-1] > buy_volume.get('1m', [Decimal('0')])[-1]
                print(f"Debug: ML Forecast: {adjusted_forecasted_price:.2f}, Close: {current_close:.2f}, Long: {long_conditions['ML_Forecasted_Price_over_Current_Close']}, Short: {short_conditions['ML_Forecasted_Price_below_Current_Close']}")
            elif timeframe == '3m':
                long_conditions["volume_bullish_3m"] = buy_volume.get('3m', [Decimal('0')])[-1] > sell_volume.get('3m', [Decimal('0')])[-1]
                short_conditions["volume_bearish_3m"] = sell_volume.get('3m', [Decimal('0')])[-1] > buy_volume.get('3m', [Decimal('0')])[-1]
            elif timeframe == '5m':
                long_conditions["current_close_below_average_threshold_5m"] = current_close < avg_mtf if avg_mtf is not None else False
                short_conditions["current_close_above_average_threshold_5m"] = current_close > avg_mtf if avg_mtf is not None else False
                long_conditions["volume_bullish_5m"] = buy_volume.get('5m', [Decimal('0')])[-1] > sell_volume.get('5m', [Decimal('0')])[-1]
                short_conditions["volume_bearish_5m"] = sell_volume.get('5m', [Decimal('0')])[-1] > buy_volume.get('5m', [Decimal('0')])[-1]
            dist_to_min = ((current_price - low_tf) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
            dist_to_max = ((high_tf - current_price) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
            long_conditions[f"dist_to_min_{timeframe}_less_10pct"] = dist_to_min < Decimal('10.0')
            short_conditions[f"dist_to_max_{timeframe}_less_10pct"] = dist_to_max < Decimal('10.0')
            print(f"Debug: {timeframe} - Dip: {dip_confirmed}, Top: {top_confirmed}")
            print(f"Debug: {timeframe} - Buy Vol: {buy_volume.get(timeframe, [Decimal('0')])[-1]:.2f}, Sell Vol: {sell_volume.get(timeframe, [Decimal('0')])[-1]:.2f}")
            print(f"Debug: {timeframe} - Dist Min: {dist_to_min:.2f}%, Dist Max: {dist_to_max:.2f}%")
            valid_closes = np.array([float(c) for c in closes if not np.isnan(c) and c > 0], dtype=np.float64)
            sma_lengths = [5, 7, 9, 12]
            smas = {length: Decimal(str(talib.SMA(valid_closes, timeperiod=length)[-1])) for length in sma_lengths if not np.isnan(talib.SMA(valid_closes, timeperiod=length)[-1])}
            print("Simple Moving Averages:")
            for length, sma in smas.items():
                print(f"SMA-{length}: {sma:.25f}, Current Close: {current_close:.25f} - {'Above' if current_close > sma else 'Below'}")
            if all(current_close > sma for sma in smas.values()):
                print("SELL Signal: Current close is above all SMAs.")
            elif all(current_close < sma for sma in smas.values()):
                print("BUY Signal: Current close is below all SMAs.")
            stoch_k, stoch_d = calculate_stochastic_rsi(closes)
            print(f"Current Stochastic K: {stoch_k[-1]:.25f}")
            print(f"Current Stochastic D: {stoch_d[-1]:.25f}")
            negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
            market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, closest_type, buy_volume[timeframe][-1], sell_volume[timeframe][-1])
            print(f"Market Sentiment: {market_sentiment}")
            projected_price = calculate_45_degree_projection(last_bottom, last_top)
            print(f"Projected Price Using 45-Degree Angle: {projected_price:.25f}" if projected_price is not None else "No projection available")
            print(f"Current Close: {current_close:.25f}")
            print(f"Minimum Threshold: {min_threshold_tf:.25f}" if min_threshold_tf is not None else "Minimum Threshold: Not available")
            print(f"Maximum Threshold: {max_threshold_tf:.25f}" if max_threshold_tf is not None else "Maximum Threshold: Not available")
            print(f"Average MTF: {avg_mtf:.25f}" if avg_mtf is not None else "Average MTF: Not available")
            print(f"Momentum Signal: {momentum_signal:.25f}" if momentum_signal is not None else "Momentum Signal: Not available")
            print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bullish Ratio: Not available")
            print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bearish Ratio: Not available")
            print(f"Status: {volume_ratios[timeframe]['status']}" if timeframe in volume_ratios else "Status: Not available")
            avg = (min_threshold_tf + max_threshold_tf) / Decimal('2') if min_threshold_tf is not None and max_threshold_tf is not None else current_price
            wave_price = calculate_wave_price(len(closes), avg, min_threshold_tf or Decimal('0'), max_threshold_tf or Decimal('Infinity'), omega=Decimal('0.1'), phi=Decimal('0'))
            print(f"Calculated Wave Price: {wave_price:.25f}")
            independent_wave_price = calculate_independent_wave_price(current_price, avg, min_threshold_tf or Decimal('0'), max_threshold_tf or Decimal('Infinity'), range_distance=Decimal('0.1'))
            print(f"Calculated Independent Wave Price: {independent_wave_price:.25f}")
            current_time, entry_price_usdc, stop_loss, reversal_target, market_mood = get_target(closes, n_components=5, last_major_reversal_type=last_major_reversal_type, buy_volume=buy_volume[timeframe][-1], sell_volume=sell_volume[timeframe][-1])
            print(f"Current Time: {current_time}")
            print(f"Entry Price: {entry_price_usdc:.25f}")
            print(f"Stop Loss: {stop_loss:.25f}")
            print(f"Reversal Target: {reversal_target:.25f}")
            print(f"Market Mood (from FFT): {market_mood}")
            fib_levels = calculate_fibonacci_levels_from_reversal(closest_reversal, max_threshold_tf, min_threshold_tf, closest_type, current_price)
            fib_info[timeframe] = fib_levels
            print(f"Fibonacci Levels for {timeframe}:")
            for level, price in fib_levels.items():
                print(f"Level {level}: {price:.25f}")
            fib_reversal_price = forecast_fibo_target_price(fib_info[timeframe])
            print(f"{timeframe} Incoming Fibonacci Reversal Target (Forecast): {fib_reversal_price:.25f}" if fib_reversal_price is not None else f"{timeframe} Incoming Fibonacci Reversal Target: price not available.")
            time_window = {'1m': 1, '3m': 3, '5m': 5}[timeframe]
            forecast_price, price_rate = forecast_price_per_time_pythagorean(timeframe, candle_map[timeframe], min_threshold_tf, max_threshold_tf, current_price, time_window, closest_reversal, closest_type)
            pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
            print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
            print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")
        else:
            print(f"--- {timeframe} --- No data available.")
            fib_levels = calculate_fibonacci_levels_from_reversal(None, None, None, 'DIP', current_price)
            fib_info[timeframe] = fib_levels
            print(f"Fibonacci Levels for {timeframe} (Fallback):")
            for level, price in fib_levels.items():
                print(f"Level {level}: {price:.25f}")
            time_window = {'1m': 1, '3m': 3, '5m': 5}[timeframe]
            forecast_price, price_rate = forecast_price_per_time_pythagorean(timeframe, [], None, None, current_price, time_window, None, 'DIP')
            pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
            print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
            print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")

    # Verify condition mirroring
    condition_pairs = [
        ("ML_Forecasted_Price_over_Current_Close", "ML_Forecasted_Price_below_Current_Close", 'ML_Forecasted_Price'),
        ("current_close_below_average_threshold_5m", "current_close_above_average_threshold_5m", 'current_close_threshold_5m'),
        ("dip_confirmed_1m", "top_confirmed_1m", 'reversal_1m'),
        ("dip_confirmed_3m", "top_confirmed_3m", 'reversal_3m'),
        ("dip_confirmed_5m", "top_confirmed_5m", 'reversal_5m'),
        ("volume_bullish_1m", "volume_bearish_1m", 'volume_1m'),
        ("volume_bullish_3m", "volume_bearish_3m", 'volume_3m'),
        ("volume_bullish_5m", "volume_bearish_5m", 'volume_5m'),
        ("dist_to_min_1m_less_10pct", "dist_to_max_1m_less_10pct", 'dist_1m'),
        ("dist_to_min_3m_less_10pct", "dist_to_max_3m_less_10pct", 'dist_3m'),
        ("dist_to_min_5m_less_10pct", "dist_to_max_5m_less_10pct", 'dist_5m')
    ]

    # Ensure mutual exclusivity
    for long_key, short_key, conf_key in condition_pairs:
        long_val = long_conditions[long_key]
        short_val = short_conditions[short_key]
        if long_val and short_val:
            # Choose the condition with higher confidence
            confidence = confidence_scores.get(conf_key, 0)
            if confidence > 0.5:  # Arbitrary threshold for confidence
                short_conditions[short_key] = False
                print(f"Resolved conflict: Kept {long_key}, set {short_key} to False (Confidence: {confidence:.2f})")
            else:
                long_conditions[long_key] = False
                print(f"Resolved conflict: Kept {short_key}, set {long_key} to False (Confidence: {confidence:.2f})")
        elif not long_val and not short_val:
            # If both are False, set one to True based on market sentiment
            market_sentiment = determine_market_sentiment(
                *calculate_spectral_analysis([candle['close'] for candle in candle_map.get('1m', [])]),
                last_major_reversal_type,
                buy_volume.get('1m', [Decimal('0')])[-1],
                sell_volume.get('1m', [Decimal('0')])[-1]
            )
            if market_sentiment in ["Bullish", "Accumulation"]:
                long_conditions[long_key] = True
                print(f"Set {long_key} to True based on {market_sentiment} sentiment")
            elif market_sentiment in ["Bearish", "Distribution"]:
                short_conditions[short_key] = True
                print(f"Set {short_key} to True based on {market_sentiment} sentiment")

    # Adjust to enforce mirror condition
    long_true = sum(1 for v in long_conditions.values() if v)
    short_true = sum(1 for v in short_conditions.values() if v)
    target_long_true = long_true
    target_short_true = 11 - long_true

    if long_true + short_true != 11:
        print(f"Adjusting conditions: Long True={long_true}, Short True={short_true}")
        # Sort pairs by confidence to decide which to flip
        sorted_pairs = sorted(
            [(long_key, short_key, confidence_scores.get(conf_key, 0)) for long_key, short_key, conf_key in condition_pairs],
            key=lambda x: x[2]
        )
        current_long_true = long_true
        current_short_true = short_true
        for long_key, short_key, _ in sorted_pairs:
            if current_long_true > target_long_true:
                if long_conditions[long_key]:
                    long_conditions[long_key] = False
                    short_conditions[short_key] = True
                    current_long_true -= 1
                    current_short_true += 1
                    print(f"Flipped {long_key} to False, {short_key} to True")
            elif current_short_true > target_short_true:
                if short_conditions[short_key]:
                    short_conditions[short_key] = False
                    long_conditions[long_key] = True
                    current_short_true -= 1
                    current_long_true += 1
                    print(f"Flipped {short_key} to False, {long_key} to True")
            if current_long_true == target_long_true and current_short_true == target_short_true:
                break

    # Print conditions
    print("\nLong Conditions:")
    for k, v in long_conditions.items():
        print(f"{k}: {v}")
    print("\nShort Conditions:")
    for k, v in short_conditions.items():
        print(f"{k}: {v}")

    # Final counts
    long_true = sum(1 for v in long_conditions.values() if v)
    short_true = sum(1 for v in short_conditions.values() if v)
    print(f"\nFinal Long Summary: {long_true} True, {len(long_conditions) - long_true} False")
    print(f"Final Short Summary: {short_true} True, {len(short_conditions) - short_true} False")
    if long_true > 0 and short_true > 0:
        pass  # Removed the print statement as requested
    elif long_true > short_true:
        print("Note: Bullish bias detected.")
    elif short_true > long_true:
        print("Note: Bearish bias detected.")

    # Position management
    if position_open:
        should_exit, reason = check_exit_condition(initial_investment, position, current_price)
        if should_exit:
            if close_position(position_side, quantity):
                final_balance = get_balance('USDC')
                profit = final_balance - initial_investment
                print(f"Closed {position_side}: Profit {profit:.25f} USDC")
                position_open = False
                initial_investment = Decimal('0.0')
                position_side = "NONE"
                quantity = Decimal('0.0')
                entry_price = Decimal('0.0')
                entry_datetime = None
    else:
        long_signal = all(long_conditions.values())
        short_signal = all(short_conditions.values())
        print(f"\nLONG Signal: {'Active' if long_signal else 'Inactive'}")
        print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'}")
        if long_signal and not short_signal:
            print(f"LONG signal detected!")
            entry_price, quantity, entry_datetime, cost = open_long_position(usdc_balance)
            if entry_price and cost:
                position_open = True
                position_side = "LONG"
                initial_investment = cost
                print(f"Opened LONG: Qty {quantity:.25f}, Entry {entry_price:.25f}")
            else:
                print("Simulated LONG signal due to zero balance or error.")
        elif short_signal and not long_signal:
            print(f"SHORT signal detected!")
            entry_price, quantity, entry_datetime, cost = open_short_position(usdc_balance)
            if entry_price and cost:
                position_open = True
                position_side = "SHORT"
                initial_investment = cost
                print(f"Opened SHORT: Qty {quantity:.25f}, Entry {entry_price:.25f}")
            else:
                print("Simulated SHORT signal due to zero balance or error.")
        elif long_signal and short_signal:
            print("Error: Both signals active. Skipping trade.")
        else:
            print("No trade signals.")

    print(f"\nCurrent USDC Balance: {usdc_balance:.25f}")
    print(f"Current Position: {position_side}, Quantity: {quantity:.25f} BTC")
    print(f"Current {TRADE_SYMBOL} Price: {current_price:.25f}\n")

    del candle_map
    gc.collect()
    time.sleep(5)
