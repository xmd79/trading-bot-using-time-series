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
TRADE_SYMBOL = "BTCUSDC"  # Futures use USDC
LEVERAGE = 20
TAKE_PROFIT_ROI = Decimal('2.55')  # % TP based on initial USDC balance
STOP_LOSS_ROI = Decimal('-2.55')   # % SL based on initial USDC balance
MIN_NOTIONAL = Decimal('10.0')  # Minimum notional value for trades

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
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=100):
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
        except requests.exceptions.ReadTimeout as e:
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

        print("Futures Account Assets:")
        for a in account['assets']:
            print(f"Asset: {a['asset']}, Wallet Balance: {a.get('walletBalance')}")

        for asset_info in account['assets']:
            if asset_info.get('asset') == asset:
                wallet = Decimal(str(asset_info.get('walletBalance', '0.0')))
                print(f"{asset} wallet balance (used for trading): {wallet:.25f}")
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

def get_symbol_lot_size_info(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                for filter in symbol_info['filters']:
                    if filter['filterType'] == 'LOT_SIZE':
                        return {
                            'minQty': Decimal(str(filter['minQty'])),
                            'stepSize': Decimal(str(filter['stepSize']))
                        }
        print(f"Could not find LOT_SIZE filter for {symbol}. Using defaults.")
        return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}
    except BinanceAPIException as e:
        print(f"Error fetching symbol info for {symbol}: {e.message}")
        return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}

def open_long_position():
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print(f"Invalid current price {current_price:.25f} for long position.")
            return None, None, None, None
        usdc_balance = get_balance('USDC')
        if usdc_balance <= Decimal('0'):
            print("No USDC balance available to open long position.")
            return None, None, None, None
        notional = usdc_balance * Decimal(str(LEVERAGE))
        raw_quantity = notional / current_price
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        adjusted_quantity = (raw_quantity // step_size) * step_size
        adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = adjusted_quantity * current_price / Decimal(str(LEVERAGE))
        if cost < MIN_NOTIONAL:
            print(f"Cost {cost:.25f} USDC is below minimum notional value {MIN_NOTIONAL:.25f}. Adjusting quantity.")
            min_quantity_for_notional = MIN_NOTIONAL * Decimal(str(LEVERAGE)) / current_price
            adjusted_quantity = ((min_quantity_for_notional + step_size - Decimal('1E-25')) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price / Decimal(str(LEVERAGE))
        if adjusted_quantity < min_trade_size:
            print(f"Adjusted quantity {adjusted_quantity:.25f} is below minimum trade size {min_trade_size:.25f}. Cannot execute trade.")
            return None, None, None, None
        if cost > usdc_balance:
            print(f"Cost {cost:.25f} exceeds available balance {usdc_balance:.25f}. Re-adjusting.")
            adjusted_quantity = ((usdc_balance * Decimal(str(LEVERAGE)) / current_price) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price / Decimal(str(LEVERAGE))
        if adjusted_quantity < min_trade_size:
            print(f"Final adjusted quantity {adjusted_quantity:.25f} still below minimum trade size {min_trade_size:.25f}. Cannot execute trade.")
            return None, None, None, None
        remaining_balance = usdc_balance - cost
        print(f"Opening LONG with {cost:.25f} of {usdc_balance:.25f} USDC, Remaining Balance: {remaining_balance:.25f} USDC")
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side='BUY',
            type='MARKET',
            quantity=float(adjusted_quantity)
        )
        print(f"Long position opened: {order}")
        entry_price = Decimal(str(order['avgPrice'])) if 'avgPrice' in order else current_price
        entry_datetime = datetime.datetime.now()
        return entry_price, adjusted_quantity, entry_datetime, cost
    except BinanceAPIException as e:
        print(f"Error opening long position: {e.message}")
        return None, None, None, None

def open_short_position():
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print(f"Invalid current price {current_price:.25f} for short position.")
            return None, None, None, None
        usdc_balance = get_balance('USDC')
        if usdc_balance <= Decimal('0'):
            print("No USDC balance available to open short position.")
            return None, None, None, None
        notional = usdc_balance * Decimal(str(LEVERAGE))
        raw_quantity = notional / current_price
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        adjusted_quantity = (raw_quantity // step_size) * step_size
        adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = adjusted_quantity * current_price / Decimal(str(LEVERAGE))
        if cost < MIN_NOTIONAL:
            print(f"Cost {cost:.25f} USDC is below minimum notional value {MIN_NOTIONAL:.25f}. Adjusting quantity.")
            min_quantity_for_notional = MIN_NOTIONAL * Decimal(str(LEVERAGE)) / current_price
            adjusted_quantity = ((min_quantity_for_notional + step_size - Decimal('1E-25')) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price / Decimal(str(LEVERAGE))
        if adjusted_quantity < min_trade_size:
            print(f"Adjusted quantity {adjusted_quantity:.25f} is below minimum trade size {min_trade_size:.25f}. Cannot execute trade.")
            return None, None, None, None
        if cost > usdc_balance:
            print(f"Cost {cost:.25f} exceeds available balance {usdc_balance:.25f}. Re-adjusting.")
            adjusted_quantity = ((usdc_balance * Decimal(str(LEVERAGE)) / current_price) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price / Decimal(str(LEVERAGE))
        if adjusted_quantity < min_trade_size:
            print(f"Final adjusted quantity {adjusted_quantity:.25f} still below minimum trade size {min_trade_size:.25f}. Cannot execute trade.")
            return None, None, None, None
        remaining_balance = usdc_balance - cost
        print(f"Opening SHORT with {cost:.25f} of {usdc_balance:.25f} USDC, Remaining Balance: {remaining_balance:.25f} USDC")
        order = client.futures_create_order(
            symbol=TRADE_SYMBOL,
            side='SELL',
            type='MARKET',
            quantity=float(adjusted_quantity)
        )
        print(f"Short position opened: {order}")
        entry_price = Decimal(str(order['avgPrice'])) if 'avgPrice' in order else current_price
        entry_datetime = datetime.datetime.now()
        return entry_price, adjusted_quantity, entry_datetime, cost
    except BinanceAPIException as e:
        print(f"Error opening short position: {e.message}")
        return None, None, None, None

def close_position(side, quantity):
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print(f"Invalid current price {current_price:.25f} for closing position.")
            return False
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        close_quantity = (abs(quantity) // step_size) * step_size
        close_quantity = close_quantity.quantize(Decimal('0.' + '0' * step_precision))
        if close_quantity < min_trade_size:
            print(f"Cannot close: Quantity {close_quantity:.25f} is below minimum trade size {min_trade_size:.25f}.")
            return False
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
        print(f"Error closing {side} position: {e.message}")
        return False

def check_exit_condition(initial_investment, position, current_price):
    if initial_investment <= Decimal('0.0') or position['quantity'] == Decimal('0.0'):
        print("Invalid initial investment or position quantity for exit condition check.")
        return False, None
    unrealized_pnl = position['unrealized_pnl']
    current_balance = get_balance('USDC') + unrealized_pnl
    roi = ((current_balance - initial_investment) / initial_investment) * Decimal('100')
    take_profit_met = roi >= TAKE_PROFIT_ROI
    stop_loss_met = roi <= STOP_LOSS_ROI
    print(f"Exit Check: ROI: {roi:.25f}%, TP: {TAKE_PROFIT_ROI:.25f}%, SL: {STOP_LOSS_ROI:.25f}%")
    return take_profit_met or stop_loss_met, "TP" if take_profit_met else "SL" if stop_loss_met else None

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
    print(f"Backtest MAE: {mae:.25f}")
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
    print(f"Momentum Signal: {momentum_signal:.25f}")
    print(f"Minimum Threshold: {min_threshold:.25f}")
    print(f"Maximum Threshold: {max_threshold:.25f}")
    print(f"Average MTF: {avg_mtf:.25f}")
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
    if closest_reversal is not None:
        print(f"Most Recent Major Reversal Type: {closest_type}")
        print(f"Last Major Reversal Found at Price: {closest_reversal:.25f}")
    else:
        print("No Major Reversal Found")
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
        sentiment = "Bullish" if buy_volume > sell_volume else "Accumulation"
    elif last_major_reversal_type == 'TOP':
        sentiment = "Bearish" if sell_volume > buy_volume and total_positive_power > total_negative_power else "Distribution" if sell_volume > buy_volume else "Bullish"
    else:
        sentiment = "Neutral"
    print(f"Market Sentiment: {sentiment}")
    return sentiment

def calculate_45_degree_projection(last_bottom, last_top):
    if last_bottom is not None and last_top is not None:
        distance = last_top - last_bottom
        projected_price = last_top + distance
        print(f"Projected Price Using 45-Degree Angle: {projected_price:.25f}")
        return projected_price
    print("No projection available")
    return None

def calculate_wave_price(t, avg, min_price, max_price, omega, phi):
    t_dec = Decimal(str(t))
    avg_dec = Decimal(str(avg))
    min_price_dec = Decimal(str(min_price))
    max_price_dec = Decimal(str(max_price))
    omega_dec = Decimal(str(omega))
    phi_dec = Decimal(str(phi))
    amplitude = (max_price_dec - min_price_dec) / Decimal('2')
    wave_price = avg_dec + amplitude * Decimal(str(math.sin(float(omega_dec * t_dec + phi_dec))))
    print(f"Calculated Wave Price: {wave_price:.25f}")
    return wave_price

def calculate_independent_wave_price(current_price, avg, min_price, max_price, range_distance):
    current_price_dec = Decimal(str(current_price))
    avg_dec = Decimal(str(avg))
    min_price_dec = Decimal(str(min_price))
    max_price_dec = Decimal(str(max_price))
    range_distance_dec = Decimal(str(range_distance))
    noise = Decimal(str(np.random.uniform(-1, 1))) * range_distance_dec
    independent_wave_price = avg_dec + (noise * (max_price_dec - min_price_dec) / Decimal('2'))
    print(f"Calculated Independent Wave Price: {independent_wave_price:.25f}")
    return independent_wave_price

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
    print(f"Current Stochastic K: {Decimal(str(stoch_k_smooth[-1])):.25f}")
    print(f"Current Stochastic D: {Decimal(str(stoch_d[-1])):.25f}")
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

def calculate_volume_changes_over_time(candle_map):
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
    fib_reversal_price = fib_levels.get('1.0', None)
    print(f"Incoming Fibonacci Reversal Target (Forecast): {fib_reversal_price:.25f}" if fib_reversal_price is not None else "Incoming Fibonacci Reversal Target: price not available.")
    return fib_reversal_price

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
    print(f"Current Time: {current_time}")
    print(f"Entry Price: {current_close:.25f}")
    print(f"Stop Loss: {stop_loss:.25f}")
    print(f"Reversal Target: {target_price:.25f}")
    print(f"Market Mood (from FFT): {market_mood}")
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

# Main Trading Loop
def main():
    global min_trade_size, step_size
    lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
    min_trade_size = lot_size_info['minQty']
    step_size = lot_size_info['stepSize']
    print(f"Initialized {TRADE_SYMBOL} - Min Trade Size: {min_trade_size:.25f}, Step Size: {step_size:.25f}")
    timeframes = ["1m", "3m", "5m", "15m", "30m"]
    initial_balance = get_balance('USDC')
    print("Futures Trading Bot Initialized!")
    last_position = {"side": "NONE", "entry_price": None, "quantity": None, "entry_time": None, "cost": None}
    
    while True:
        try:
            current_local_time = datetime.datetime.now()
            current_local_time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nCurrent Local Time: {current_local_time_str}")

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
            position_value = abs(position['quantity']) * current_price
            if position['side'] != "NONE" and last_position["side"] == "NONE":
                print(f"Position detected ({position['side']}): Quantity {position['quantity']:.25f}, Value {position_value:.25f} USDC")
                last_position["side"] = position['side']
                last_position["entry_price"] = position['entry_price']
                last_position["quantity"] = position['quantity']
                last_position["entry_time"] = current_local_time
                last_position["cost"] = usdc_balance - position['unrealized_pnl']
                if last_position["cost"] <= Decimal('0'):
                    last_position["cost"] = abs(position['quantity']) * position['entry_price'] / Decimal(str(LEVERAGE))
                print(f"Estimated Initial Investment: {last_position['cost']:.25f} USDC, Entry Price: {last_position['entry_price']:.25f}")
                print(f"Entry Datetime set to current time: {current_local_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Initialize conditions
            conditions_long = {
                "forecast_price_above_close": False,
                "dip_confirmed_1m": False,
                "dip_confirmed_3m": False,
                "dip_confirmed_5m": False,
                "dip_confirmed_15m": False,
                "dip_confirmed_30m": False,
                "volume_bullish_1m": False,
                "volume_bullish_3m": False,
                "volume_bullish_5m": False,
                "volume_bullish_15m": False,
                "volume_bullish_30m": False,
                "dist_to_min_closer_1m": False,
                "dist_to_min_closer_3m": False,
                "dist_to_min_closer_5m": False,
                "dist_to_min_closer_15m": False,
                "dist_to_min_closer_30m": False
            }
            conditions_short = {
                "forecast_price_below_close": False,
                "top_confirmed_1m": False,
                "top_confirmed_3m": False,
                "top_confirmed_5m": False,
                "top_confirmed_15m": False,
                "top_confirmed_30m": False,
                "volume_bearish_1m": False,
                "volume_bearish_3m": False,
                "volume_bearish_5m": False,
                "volume_bearish_15m": False,
                "volume_bearish_30m": False,
                "dist_to_max_closer_1m": False,
                "dist_to_max_closer_3m": False,
                "dist_to_max_closer_5m": False,
                "dist_to_max_closer_15m": False,
                "dist_to_max_closer_30m": False
            }

            # Backtest and forecast
            if "1m" in candle_map and candle_map["1m"]:
                model, mae, predictions, y_test = backtest_model(candle_map["1m"])
                forecasted_price = forecast_next_price(model, num_steps=1)[0]
                print(f"Forecasted Price: {forecasted_price:.25f}")
                close_prices = [candle["close"] for candle in candle_map["1m"]]
                min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(close_prices)
                adjusted_forecasted_price = min(max(forecasted_price, min_threshold), max_threshold) if min_threshold is not None and max_threshold is not None else forecasted_price
                print(f"Adjusted Forecasted Price: {adjusted_forecasted_price:.25f}")
            else:
                min_threshold, max_threshold, avg_mtf, adjusted_forecasted_price = None, None, None, None
                print("No 1m data available for forecasting.")

            # Volume analysis
            buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
            volume_ratio = calculate_volume_ratio(buy_volume, sell_volume)

            # Support and resistance
            support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold or current_price * Decimal('0.95'), max_threshold or current_price * Decimal('1.05'), current_price)

            # Volume trends and ratios
            volume_trends = calculate_volume_changes_over_time(candle_map)
            volume_ratios = calculate_bullish_bearish_volume_ratios(candle_map)

            # Additional analysis per timeframe
            fib_info = {}
            pythagorean_forecasts = {'1m': {'price': None, 'rate': None}, '3m': {'price': None, 'rate': None}, '5m': {'price': None, 'rate': None}, '15m': {'price': None, 'rate': None}, '30m': {'price': None, 'rate': None}}
            for timeframe in timeframes:
                if timeframe in candle_map and candle_map[timeframe]:
                    print(f"\n--- {timeframe} ---")
                    closes = [candle['close'] for candle in candle_map[timeframe]]
                    current_close = Decimal(str(closes[-1]))
                    high_tf = Decimal(str(np.nanmax([float(x) for x in closes])))
                    low_tf = Decimal(str(np.nanmin([float(x) for x in closes])))
                    min_threshold_tf, max_threshold_tf, avg_mtf_tf, momentum_signal_tf, _, percent_to_min_momentum_tf, percent_to_max_momentum_tf = calculate_thresholds(closes)
                    last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_price, min_threshold_tf, max_threshold_tf)
                    
                    # Reversal conditions (mutually exclusive)
                    conditions_long[f"dip_confirmed_{timeframe}"] = closest_type == 'DIP'
                    conditions_short[f"top_confirmed_{timeframe}"] = closest_type == 'TOP'
                    if conditions_long[f"dip_confirmed_{timeframe}"] and conditions_short[f"top_confirmed_{timeframe}"]:
                        print(f"Error: Both dip and top confirmed for {timeframe}. Defaulting to neither.")
                        conditions_long[f"dip_confirmed_{timeframe}"] = False
                        conditions_short[f"top_confirmed_{timeframe}"] = False

                    # Volume conditions (mutually exclusive)
                    buy_vol = buy_volume.get(timeframe, [Decimal('0')])[-1]
                    sell_vol = sell_volume.get(timeframe, [Decimal('0')])[-1]
                    if buy_vol > sell_vol:
                        conditions_long[f"volume_bullish_{timeframe}"] = True
                        conditions_short[f"volume_bearish_{timeframe}"] = False
                    else:
                        conditions_long[f"volume_bullish_{timeframe}"] = False
                        conditions_short[f"volume_bearish_{timeframe}"] = True
                    print(f"Volume Bullish ({timeframe}): {buy_vol:.25f}, Bearish: {sell_vol:.25f}, Bullish Condition: {conditions_long[f'volume_bullish_{timeframe}']}, Bearish Condition: {conditions_short[f'volume_bearish_{timeframe}']}")

                    # Distance conditions (mutually exclusive)
                    dist_to_min_price = ((current_close - min_threshold_tf) / (max_threshold_tf - min_threshold_tf)) * Decimal('100') if (max_threshold_tf - min_threshold_tf) != Decimal('0') else Decimal('0')
                    dist_to_max_price = ((max_threshold_tf - current_close) / (max_threshold_tf - min_threshold_tf)) * Decimal('100') if (max_threshold_tf - min_threshold_tf) != Decimal('0') else Decimal('0')
                    if dist_to_min_price < dist_to_max_price:
                        conditions_long[f"dist_to_min_closer_{timeframe}"] = True
                        conditions_short[f"dist_to_max_closer_{timeframe}"] = False
                    else:
                        conditions_long[f"dist_to_min_closer_{timeframe}"] = False
                        conditions_short[f"dist_to_max_closer_{timeframe}"] = True
                    print(f"Distance to Min Threshold ({timeframe}): {dist_to_min_price:.25f}%")
                    print(f"Distance to Max Threshold: {dist_to_max_price:.25f}%")

                    # Sine wave distances
                    sine_dist_to_min, sine_dist_to_max, _ = scale_to_sine(closes)
                    print(f"Sine Wave Distance to Min ({timeframe}): {sine_dist_to_min:.25f}%")
                    print(f"Sine Wave Distance to Max: {sine_dist_to_max:.25f}%")

                    # Moving averages
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

                    # Stochastic RSI
                    stoch_k, stoch_d = calculate_stochastic_rsi(closes)

                    # Spectral analysis
                    negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
                    market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, closest_type, buy_vol, sell_vol)

                    # unción de 45 grados
                    calculate_45_degree_projection(last_bottom, last_top)

                    # Precios de onda
                    avg = (min_threshold_tf + max_threshold_tf) / Decimal('2') if min_threshold_tf is not None and max_threshold_tf is not None else current_price
                    calculate_wave_price(len(closes), avg, min_threshold_tf or Decimal('0'), max_threshold_tf or Decimal('Infinity'), omega=Decimal('0.1'), phi=Decimal('0'))
                    calculate_independent_wave_price(current_price, avg, min_threshold_tf or Decimal('0'), max_threshold_tf or Decimal('Infinity'), range_distance=Decimal('0.1'))

                    # Objetivo FFT
                    current_time, entry_price_usdc, stop_loss, reversal_target, market_mood = get_target(closes, n_components=5, last_major_reversal_type=closest_type, buy_volume=buy_vol, sell_volume=sell_vol)

                    # Niveles de Fibonacci
                    fib_levels = calculate_fibonacci_levels_from_reversal(closest_reversal, max_threshold_tf, min_threshold_tf, closest_type, current_price)
                    fib_info[timeframe] = fib_levels
                    print(f"Fibonacci Levels for {timeframe}:")
                    for level, price in fib_levels.items():
                        print(f"Level {level}: {price:.25f}")
                    forecast_fibo_target_price(fib_info[timeframe])

                    # Distancia a min/max
                    dist_to_min = ((current_price - low_tf) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                    dist_to_max = ((high_tf - current_price) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                    print(f"Distance from Current Close to Min Price ({low_tf:.25f}): {dist_to_min:.25f}%")
                    print(f"Distance from Current Close to Max Price ({high_tf:.25f}): {dist_to_max:.25f}%")
                    symmetrical_min_distance = (high_tf - current_price) / (high_tf - low_tf) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                    symmetrical_max_distance = (current_price - low_tf) / (high_tf - low_tf) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                    print(f"Normalized Distance to Min Price (Symmetrical): {symmetrical_max_distance:.25f}%")
                    print(f"Normalized Distance to Max Price (Symmetrical): {symmetrical_min_distance:.25f}%")

                    # Pronóstico pitagórico
                    time_window = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30}[timeframe]
                    forecast_price, price_rate = forecast_price_per_time_pythagorean(timeframe, candle_map[timeframe], min_threshold_tf, max_threshold_tf, current_price, time_window, closest_reversal, closest_type)
                    pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
                    print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
                    print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")

                    # Impresiones adicionales
                    print(f"Current Close: {current_close:.25f}")
                    print(f"Volume Bullish Ratio: {volume_ratio[timeframe]['buy_ratio']:.25f}%")
                    print(f"Volume Bearish Ratio: {volume_ratio[timeframe]['sell_ratio']:.25f}%")
                    print(f"Status: {volume_ratio[timeframe]['status']}")

                else:
                    print(f"--- {timeframe} --- No data available.")
                    fib_levels = calculate_fibonacci_levels_from_reversal(None, None, None, 'DIP', current_price)
                    fib_info[timeframe] = fib_levels
                    print(f"Fibonacci Levels for {timeframe} (Fallback):")
                    for level, price in fib_levels.items():
                        print(f"Level {level}: {price:.25f}")
                    time_window = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30}[timeframe]
                    forecast_price, price_rate = forecast_price_per_time_pythagorean(timeframe, [], None, None, current_price, time_window, None, 'DIP')
                    pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
                    print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
                    print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")

            # Pronóstico basado en volumen
            forecast_volume_based_on_conditions(volume_ratio, min_threshold or current_price * Decimal('0.95'), current_price)
            check_market_conditions_and_forecast(support_levels, resistance_levels, current_price)

            # Evaluar condición de pronóstico (mutuamente excluyente)
            if adjusted_forecasted_price is not None:
                if adjusted_forecasted_price > current_price:
                    conditions_long["forecast_price_above_close"] = True
                    conditions_short["forecast_price_below_close"] = False
                else:
                    conditions_long["forecast_price_above_close"] = False
                    conditions_short["forecast_price_below_close"] = True
                print(f"Forecast Condition: LONG={conditions_long['forecast_price_above_close']}, SHORT={conditions_short['forecast_price_below_close']}")

            # Validar pares de condiciones
            condition_pairs = [
                ("forecast_price_above_close", "forecast_price_below_close"),
                ("dip_confirmed_1m", "top_confirmed_1m"),
                ("dip_confirmed_3m", "top_confirmed_3m"),
                ("dip_confirmed_5m", "top_confirmed_5m"),
                ("dip_confirmed_15m", "top_confirmed_15m"),
                ("dip_confirmed_30m", "top_confirmed_30m"),
                ("volume_bullish_1m", "volume_bearish_1m"),
                ("volume_bullish_3m", "volume_bearish_3m"),
                ("volume_bullish_5m", "volume_bearish_5m"),
                ("volume_bullish_15m", "volume_bearish_15m"),
                ("volume_bullish_30m", "volume_bearish_30m"),
                ("dist_to_min_closer_1m", "dist_to_max_closer_1m"),
                ("dist_to_min_closer_3m", "dist_to_max_closer_3m"),
                ("dist_to_min_closer_5m", "dist_to_max_closer_5m"),
                ("dist_to_min_closer_15m", "dist_to_max_closer_15m"),
                ("dist_to_min_closer_30m", "dist_to_max_closer_30m")
            ]
            for long_cond, short_cond in condition_pairs:
                long_val = conditions_long[long_cond]
                short_val = conditions_short[short_cond]
                if long_val == short_val:
                    print(f"Conflict in {long_cond}/{short_cond}: Both {long_val}. Resetting both to False.")
                    conditions_long[long_cond] = False
                    conditions_short[short_cond] = False

            # Imprimir estados de condición
            print("\nTrade Signal Status:")
            long_signal = all(conditions_long.values())
            short_signal = all(conditions_short.values())
            print(f"LONG Signal: {'Active' if long_signal else 'Inactive'}")
            print(f"SHORT Signal: {'Active' if short_signal else 'Inactive'}")
            print("\nLong Conditions Status:")
            for condition, status in conditions_long.items():
                print(f"{condition}: {'True' if status else 'False'}")
            print("\nShort Conditions Status:")
            for condition, status in conditions_short.items():
                print(f"{condition}: {'True' if status else 'False'}")

            # Calcular resumen de condiciones
            long_true = sum(1 for val in conditions_long.values() if val)
            long_false = len(conditions_long) - long_true
            short_true = sum(1 for val in conditions_short.values() if val)
            short_false = len(conditions_short) - short_true
            print(f"\nLong Conditions Summary: {long_true} True, {long_false} False")
            print(f"Short Conditions Summary: {short_true} True, {short_false} False")

            # Determinar la señal
            signal = "NO_SIGNAL"
            if long_signal:
                signal = "LONG"
            elif short_signal:
                signal = "SHORT"
            print(f"Signal: {signal}")

            # Estado de la posición
            if position["side"] != "NONE":
                print("\nCurrent Position Status:")
                print(f"Position Side: {position['side']}")
                print(f"Quantity: {position['quantity']:.25f} BTC")
                print(f"Entry Price: {position['entry_price']:.25f} USDC")
                print(f"Current Price: {current_price:.25f} USDC")
                print(f"Unrealized PNL: {position['unrealized_pnl']:.25f} USDC")
                current_balance = usdc_balance + position['unrealized_pnl']
                print(f"Current Total Balance: {current_balance:.25f} USDC")
                target_value = last_position["cost"] * (Decimal('1.0') + TAKE_PROFIT_ROI / Decimal('100'))
                stop_loss_value = last_position["cost"] * (Decimal('1.0') + STOP_LOSS_ROI / Decimal('100'))
                entry_time_str = last_position["entry_time"].strftime("%H:%M") if last_position["entry_time"] else "Unknown"
                time_span = (current_local_time - last_position["entry_time"]) if last_position["entry_time"] else None
                time_span_str = f"{time_span.days} days, {time_span.seconds // 3600} hours, {(time_span.seconds % 3600) // 60} minutes" if time_span else "Unknown"
                print(f"Initial Investment: {last_position['cost']:.25f} USDC")
                print(f"Target Value (2.55% ROI): {target_value:.25f} USDC")
                print(f"Stop Loss Value (25.5% Loss): {stop_loss_value:.25f} USDC")
                print(f"Entry Time (HH:MM): {entry_time_str}")
                print(f"Time Span from Entry: {time_span_str}")
                roi = ((current_balance - last_position["cost"]) / last_position["cost"]) * Decimal('100') if last_position["cost"] > Decimal('0') else Decimal('0.0')
                print(f"Current ROI: {roi:.25f}%")
            else:
                print(f"\nNo open position. USDC Balance: {usdc_balance:.25f}")
                if usdc_balance <= Decimal('0'):
                    print("No USDC balance available for trading.")

            # Gestión de posición
            if last_position["side"] != "NONE" and position["side"] == "NONE":
                print(f"Position closed externally. Resetting last_position.")
                last_position = {"side": "NONE", "entry_price": None, "quantity": None, "entry_time": None, "cost": None}

            if position["side"] != "NONE":
                should_exit, exit_reason = check_exit_condition(last_position["cost"], position, current_price)
                if should_exit:
                    if close_position(position["side"], position["quantity"]):
                        final_balance = get_balance('USDC')
                        profit = final_balance - last_position["cost"]
                        profit_percentage = (profit / last_position["cost"]) * Decimal('100') if last_position["cost"] > Decimal('0') else Decimal('0.0')
                        print(f"Position closed. Final Balance: {final_balance:.25f} USDC")
                        print(f"Trade log: Time: {current_local_time_str}, Entry Price: {last_position['entry_price']:.25f}, Final Balance: {final_balance:.25f}, Profit: {profit:.25f} USDC, Profit Percentage: {profit_percentage:.25f}%")
                        last_position = {"side": "NONE", "entry_price": None, "quantity": None, "entry_time": None, "cost": None}
                elif signal == "LONG" and position["side"] == "SHORT":
                    if close_position(position["side"], position["quantity"]):
                        print(f"Closed SHORT position to open LONG.")
                        last_position = {"side": "NONE", "entry_price": None, "quantity": None, "entry_time": None, "cost": None}
                elif signal == "SHORT" and position["side"] == "LONG":
                    if close_position(position["side"], position["quantity"]):
                        print(f"Closed LONG position to open SHORT.")
                        last_position = {"side": "NONE", "entry_price": None, "quantity": None, "entry_time": None, "cost": None}

            if position["side"] == "NONE" and last_position["side"] == "NONE":
                if signal == "LONG":
                    print(f"LONG signal detected! Opening long position with {usdc_balance:.25f} USDC at price {current_price:.25f}")
                    entry_price, quantity, entry_time, cost = open_long_position()
                    if entry_price is not None:
                        last_position = {
                            "side": "LONG",
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "entry_time": entry_time,
                            "cost": cost
                        }
                        print(f"Long position opened at {entry_price:.25f} USDC for quantity: {quantity:.25f} BTC, Cost: {cost:.25f} USDC")
                        print(f"Entry Datetime: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                elif signal == "SHORT":
                    print(f"SHORT signal detected! Opening short position with {usdc_balance:.25f} USDC at price {current_price:.25f}")
                    entry_price, quantity, entry_time, cost = open_short_position()
                    if entry_price is not None:
                        last_position = {
                            "side": "SHORT",
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "entry_time": entry_time,
                            "cost": cost
                        }
                        print(f"Short position opened at {entry_price:.25f} USDC for quantity: {quantity:.25f} BTC, Cost: {cost:.25f} USDC")
                        print(f"Entry Datetime: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")

            print(f"\nCurrent USDC Balance: {usdc_balance:.25f}")
            print(f"Current Position: {position['side']}, Quantity: {position['quantity']:.25f} BTC")
            print(f"Current {TRADE_SYMBOL} Price: {current_price:.25f}\n")

            # Limpieza
            del candle_map
            gc.collect()
            time.sleep(5)

        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
