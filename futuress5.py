import requests
import numpy as np
import talib
from binance.client import Client as BinanceClient
import pandas as pd
from scipy.fft import fft, ifft
import pywt
from datetime import datetime

def get_binance_client():
    """Instantiate Binance client using API credentials."""
    try:
        with open("credentials.txt", "r") as f:
            lines = f.readlines()
            api_key = lines[0].strip()
            api_secret = lines[1].strip()
        return BinanceClient(api_key, api_secret)
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        raise

# Initialize the Binance client
client = get_binance_client()
TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    """Fetch candlestick data from Binance for given timeframes."""
    candles = []
    try:
        for timeframe in timeframes:
            limit = 500
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
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
    except Exception as e:
        print(f"Error fetching candles: {e}")
        raise
    return candles

candles = get_candles(TRADE_SYMBOL, timeframes)
candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]
    candle_map.setdefault(timeframe, []).append(candle)

def get_price(symbol):
    """Get the current price of the specified trading symbol."""
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        data = response.json()
        if "price" in data:
            return float(data["price"])
        else:
            raise KeyError("Price key not found in API response")
    except Exception as e:
        print(f"Error fetching price: {e}")
        raise

def get_close(timeframe):
    """Fetch closing prices for a given timeframe and the current price."""
    closes = []
    try:
        candles = candle_map[timeframe]
        for c in candles:
            close = c['close']
            if not np.isnan(close):
                closes.append(close)
        current_price = get_price(TRADE_SYMBOL)
        closes.append(current_price)
    except Exception as e:
        print(f"Error getting close prices for {timeframe}: {e}")
        raise
    return np.array(closes)

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    """Calculate dynamic incoming min and max thresholds for the closing prices."""
    try:
        close_prices = np.array(close_prices)
        min_close = np.nanmin(close_prices[-period:])
        max_close = np.nanmax(close_prices[-period:])
        min_percentage_custom = minimum_percentage / 100
        max_percentage_custom = maximum_percentage / 100
        min_threshold = min_close - (max_close - min_close) * min_percentage_custom
        max_threshold = max_close + (max_close - min_close) * max_percentage_custom
        middle_threshold = (min_threshold + max_threshold) / 2
        return min_threshold, max_threshold, middle_threshold
    except Exception as e:
        print(f"Error calculating thresholds: {e}")
        raise

def calculate_enforced_support_resistance(candles):
    """Calculate enforced support and resistance levels based on weighted SMA."""
    try:
        closes = np.array([candle['close'] for candle in candles])
        volumes = np.array([candle['volume'] for candle in candles])
        weighted_sma = talib.SMA(closes * volumes, timeperiod=20) / talib.SMA(volumes, timeperiod=20)
        support = np.nanmin(weighted_sma)
        resistance = np.nanmax(weighted_sma)
        return support, resistance
    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
        raise

def calculate_buy_sell_volume(candle_map):
    """Calculate buy and sell volume for each timeframe."""
    buy_volume, sell_volume = {}, {}
    try:
        for timeframe in candle_map:
            buy_volume[timeframe] = 0
            sell_volume[timeframe] = 0
            for candle in candle_map[timeframe]:
                if candle["close"] > candle["open"]:
                    buy_volume[timeframe] += candle["volume"]
                elif candle["close"] < candle["open"]:
                    sell_volume[timeframe] += candle["volume"]
    except Exception as e:
        print(f"Error calculating buy/sell volume: {e}")
        raise
    return buy_volume, sell_volume

def forecast_fft(close_prices):
    """Perform FFT and return dominant frequencies along with their respective ratios."""
    try:
        n = len(close_prices)
        freq_components = fft(close_prices)
        pos_freq = np.abs(freq_components[:n // 2])
        total_power = np.sum(pos_freq)
        dominant_freq_index = np.argmax(pos_freq)
        positive_ratio = pos_freq[dominant_freq_index] / total_power * 100
        negative_ratio = (total_power - pos_freq[dominant_freq_index]) / total_power * 100
        return {
            "dominant_index": dominant_freq_index,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        }, pos_freq[dominant_freq_index]
    except Exception as e:
        print(f"Error in FFT forecast: {e}")
        raise

def inverse_fft(frequencies, n):
    """Convert frequencies back into price using IFFT."""
    try:
        if isinstance(frequencies, (int, float)):
            frequencies = np.array([frequencies] * n)
        elif len(frequencies) < n:
            pad_length = n - len(frequencies)
            frequencies = np.pad(frequencies, (0, pad_length), 'constant')
        price_forecast = ifft(np.fft.ifftshift(frequencies)).real
        return price_forecast
    except Exception as e:
        print(f"Error in inverse FFT: {e}")
        raise

def volume_spike_retracement(candles, vslength=89, multiplier=0.5, max_lines=10):
    """Implement Volume Spike Retracement indicator."""
    try:
        opens = np.array([c['open'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        closes = np.array([c['close'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])

        # Volume Spike Calculation
        hvs = np.array([np.max(volumes[max(0, i-vslength):i+1]) for i in range(len(volumes))])
        abs_volume = volumes * 100 / hvs * 4 / 5
        smoothing = talib.EMA(abs_volume, 21)
        equal = abs_volume - smoothing
        limit = np.array([np.max(equal[max(0, i-vslength):i+1]) for i in range(len(equal))]) * multiplier
        cum = (equal > 0) & (equal >= limit)
        beardir = closes < opens
        bulldir = closes > opens

        bearvol = np.where(beardir & cum, -1, 0)
        bullvol = np.where(bulldir & cum, 1, 0)
        bearvolprice = np.where(bearvol == -1, highs, np.nan)
        bullvolprice = np.where(bullvol == 1, lows, np.nan)

        # Retracement Lines
        upper_lines = []
        lower_lines = []
        for i in range(len(candles)):
            if bearvol[i] == -1:
                upper_ph = highs[i]
                upper_pl = closes[i] if closes[i] < opens[i] else opens[i]
                upper_lines.append((upper_ph, upper_pl))
                if len(upper_lines) > max_lines:
                    upper_lines.pop(0)
            if bullvol[i] == 1:
                lower_pl = lows[i]
                lower_ph = opens[i] if closes[i] < opens[i] else closes[i]
                lower_lines.append((lower_ph, lower_pl))
                if len(lower_lines) > max_lines:
                    lower_lines.pop(0)

        return {
            "bear_spikes": bearvol,
            "bull_spikes": bullvol,
            "bear_prices": bearvolprice,
            "bull_prices": bullvolprice,
            "upper_lines": upper_lines,
            "lower_lines": lower_lines
        }
    except Exception as e:
        print(f"Error in volume spike retracement: {e}")
        raise

def breakout_trend(candles, short_len=20, long_len=200, ma_type='SMA'):
    """Implement Breakout Trend indicator."""
    try:
        closes = np.array([c['close'] for c in candles])
        # Donchian Channel
        dcub_close = np.array([np.max(closes[max(0, i-short_len):i+1]) for i in range(len(closes))])
        dcdb_close = np.array([np.min(closes[max(0, i-short_len):i+1]) for i in range(len(closes))])
        dcavg = (dcub_close + dcdb_close) / 2

        # Breakout Logic
        memresbrk = np.zeros(len(closes), dtype=bool)
        memsupbrk = np.zeros(len(closes), dtype=bool)
        for i in range(1, len(closes)):
            new_res_breakout = closes[i] >= dcub_close[i] and not memresbrk[i-1]
            new_sup_breakout = closes[i] <= dcdb_close[i] and not memsupbrk[i-1]
            if new_res_breakout:
                memresbrk[i] = True
                memsupbrk[i] = False
            elif new_sup_breakout:
                memsupbrk[i] = True
                memresbrk[i] = False
            else:
                memresbrk[i] = memresbrk[i-1]
                memsupbrk[i] = memsupbrk[i-1]

        # Moving Averages
        ma_short = talib.EMA(closes, short_len) if ma_type == 'EMA' else talib.SMA(closes, short_len)
        ma_long = talib.EMA(closes, long_len) if ma_type == 'EMA' else talib.SMA(closes, long_len)

        # ATR
        atr_value = talib.ATR(np.array([c['high'] for c in candles]),
                            np.array([c['low'] for c in candles]),
                            closes, timeperiod=short_len)

        return {
            "dc_upper": dcub_close,
            "dc_lower": dcdb_close,
            "dc_avg": dcavg,
            "res_breakout": memresbrk,
            "sup_breakout": memsupbrk,
            "ma_short": ma_short,
            "ma_long": ma_long,
            "atr": atr_value
        }
    except Exception as e:
        print(f"Error in breakout trend: {e}")
        raise

def gann_medians(candles, method1='Median', length1=12, mult1=1, method2='Median', length2=27, mult2=1, method3='Median', length3=56, mult3=1):
    """Implement Gann Medians indicator."""
    try:
        def avg(x, length, method):
            if method == 'SMA':
                return talib.SMA(x, length)
            elif method == 'EMA':
                return talib.EMA(x, length)
            elif method == 'Median':
                return np.array([np.percentile(x[max(0, i-length+1):i+1], 50) for i in range(len(x))])
            return np.zeros(len(x))

        opens = np.array([c['open'] for c in candles])
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])

        # Gann Calculations
        a1 = np.array([np.max(highs[max(0, i-length1):i+1]) for i in range(len(highs))]) - np.maximum(closes, opens)
        b1 = np.minimum(closes, opens) - np.array([np.min(lows[max(0, i-length1):i+1]) for i in range(len(lows))])
        c1 = np.maximum(closes, opens) + a1 * mult1
        d1 = np.minimum(closes, opens) - b1 * mult1

        a2 = np.array([np.max(highs[max(0, i-length2):i+1]) for i in range(len(highs))]) - np.maximum(closes, opens)
        b2 = np.minimum(closes, opens) - np.array([np.min(lows[max(0, i-length2):i+1]) for i in range(len(lows))])
        c2 = np.maximum(closes, opens) + a2 * mult2
        d2 = np.minimum(closes, opens) - b2 * mult2

        a3 = np.array([np.max(highs[max(0, i-length3):i+1]) for i in range(len(highs))]) - np.maximum(closes, opens)
        b3 = np.minimum(closes, opens) - np.array([np.min(lows[max(0, i-length3):i+1]) for i in range(len(lows))])
        c3 = np.maximum(closes, opens) + a3 * mult3
        d3 = np.minimum(closes, opens) - b3 * mult3

        e1 = avg(c1, length1, method1)
        f1 = avg(d1, length1, method1)
        g1 = np.zeros(len(e1))
        for i in range(1, len(e1)):
            if closes[i] > e1[i] and closes[i-1] <= e1[i-1]:
                g1[i] = 1
            elif closes[i] < f1[i] and closes[i-1] >= f1[i-1]:
                g1[i] = 0
            else:
                g1[i] = g1[i-1]
        hilo1 = g1 * f1 + (1 - g1) * e1

        e2 = avg(c2, length2, method2)
        f2 = avg(d2, length2, method2)
        g2 = np.zeros(len(e2))
        for i in range(1, len(e2)):
            if closes[i] > e2[i] and closes[i-1] <= e2[i-1]:
                g2[i] = 1
            elif closes[i] < f2[i] and closes[i-1] >= f2[i-1]:
                g2[i] = 0
            else:
                g2[i] = g2[i-1]
        hilo2 = g2 * f2 + (1 - g2) * e2

        e3 = avg(c3, length3, method3)
        f3 = avg(d3, length3, method3)
        g3 = np.zeros(len(e3))
        for i in range(1, len(e3)):
            if closes[i] > e3[i] and closes[i-1] <= e3[i-1]:
                g3[i] = 1
            elif closes[i] < f3[i] and closes[i-1] >= f3[i-1]:
                g3[i] = 0
            else:
                g3[i] = g3[i-1]
        hilo3 = g3 * f3 + (1 - g3) * e3

        return {
            'hilo1': hilo1,
            'hilo2': hilo2,
            'hilo3': hilo3,
            'g1': g1,
            'g2': g2,
            'g3': g3
        }
    except Exception as e:
        print(f"Error in Gann medians: {e}")
        raise

# Main analysis loop
summary_data = []
current_prices = {}

try:
    for timeframe in timeframes:
        print(f"\n=== Analysis for Timeframe: {timeframe} ===")
        close = get_close(timeframe)
        current_prices[timeframe] = close[-1]

        # Last major reversal
        last_reversal_index_dip = np.nanargmin(close)
        last_reversal_index_top = np.nanargmax(close)
        last_reversal_price_dip = close[last_reversal_index_dip]
        last_reversal_price_top = close[last_reversal_index_top]

        distance_to_dip = abs(current_prices[timeframe] - last_reversal_price_dip)
        distance_to_top = abs(current_prices[timeframe] - last_reversal_price_top)

        if distance_to_dip < distance_to_top:
            last_major_reversal_price = last_reversal_price_dip
            last_major_reversal_type = 'DIP'
            current_cycle_direction = 'UP'
            incoming_threshold = calculate_thresholds(close)[1]  # Max threshold
        else:
            last_major_reversal_price = last_reversal_price_top
            last_major_reversal_type = 'TOP'
            current_cycle_direction = 'DOWN'
            incoming_threshold = calculate_thresholds(close)[0]  # Min threshold

        # Volume analysis
        buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
        total_volume = buy_volume[timeframe] + sell_volume[timeframe]
        bullish_ratio = (buy_volume[timeframe] / total_volume * 100) if total_volume > 0 else 0
        bearish_ratio = (sell_volume[timeframe] / total_volume * 100) if total_volume > 0 else 0

        # Thresholds
        min_threshold, max_threshold, middle_threshold = calculate_thresholds(close)

        # Support and Resistance
        support, resistance = calculate_enforced_support_resistance(candle_map[timeframe])

        # FFT Analysis
        fft_analysis, forecast_value = forecast_fft(close)
        price_forecast_target = current_prices[timeframe] + (max_threshold - current_prices[timeframe]) * 0.5 if current_cycle_direction == 'UP' else current_prices[timeframe] - (current_prices[timeframe] - min_threshold) * 0.5

        # Volume Spike Retracement
        vsr = volume_spike_retracement(candle_map[timeframe])
        # Breakout Trend
        trend = breakout_trend(candle_map[timeframe])
        # Gann Medians
        gann = gann_medians(candle_map[timeframe])

        # Trading signals
        signals = []
        if vsr['bear_spikes'][-1] == -1:
            signals.append(f"Bearish Volume Spike at ${vsr['bear_prices'][-1]:.2f}")
        if vsr['bull_spikes'][-1] == 1:
            signals.append(f"Bullish Volume Spike at ${vsr['bull_prices'][-1]:.2f}")
        if trend['res_breakout'][-1]:
            signals.append("Resistance Breakout")
        if trend['sup_breakout'][-1]:
            signals.append("Support Breakout")
        if gann['g1'][-1] == 1:
            signals.append("Gann Uptrend (Short-term)")
        elif gann['g1'][-1] == 0:
            signals.append("Gann Downtrend (Short-term)")

        # Distance to thresholds
        distance_to_min_threshold = ((current_prices[timeframe] - min_threshold) / (max_threshold - min_threshold) * 100) if (max_threshold - min_threshold) != 0 else 0
        distance_to_max_threshold = ((max_threshold - current_prices[timeframe]) / (max_threshold - min_threshold) * 100) if (max_threshold - min_threshold) != 0 else 0

        # Print detailed results
        print(f"Current Close: ${current_prices[timeframe]:.2f}")
        print(f"Last Major Reversal: {last_major_reversal_type} at ${last_major_reversal_price:.2f}")
        print(f"Cycle Direction: {current_cycle_direction}")
        print(f"Incoming Threshold: ${incoming_threshold:.2f}")
        print(f"Forecast Price Target: ${price_forecast_target:.2f}")
        print(f"Buy Volume: {buy_volume[timeframe]:.2f}, Sell Volume: {sell_volume[timeframe]:.2f}")
        print(f"Bullish Ratio: {bullish_ratio:.2f}%, Bearish Ratio: {bearish_ratio:.2f}%")
        print(f"Min Threshold: ${min_threshold:.2f}, Max Threshold: ${max_threshold:.2f}, Middle: ${middle_threshold:.2f}")
        print(f"Distance to Min Threshold: {distance_to_min_threshold:.2f}%")
        print(f"Distance to Max Threshold: {distance_to_max_threshold:.2f}%")
        print(f"Support: ${support:.2f}, Resistance: ${resistance:.2f}")
        print(f"Signals: {', '.join(signals) if signals else 'None'}")
        print(f"Volume Spike Upper Lines: {[f'${p[0]:.2f}-${p[1]:.2f}' for p in vsr['upper_lines']]}")
        print(f"Volume Spike Lower Lines: {[f'${p[0]:.2f}-${p[1]:.2f}' for p in vsr['lower_lines']]}")
        print(f"Donchian Channel: Upper ${trend['dc_upper'][-1]:.2f}, Lower ${trend['dc_lower'][-1]:.2f}")
        print(f"Moving Averages: Short ${trend['ma_short'][-1]:.2f}, Long ${trend['ma_long'][-1]:.2f}")
        print(f"Gann Medians: Short ${gann['hilo1'][-1]:.2f}, Mid ${gann['hilo2'][-1]:.2f}, Long ${gann['hilo3'][-1]:.2f}")

        # Store data for summary
        summary_data.append({
            'Timeframe': timeframe,
            'Current Close': current_prices[timeframe],
            'Last Major Reversal Price': last_major_reversal_price,
            'Last Major Reversal Type': last_major_reversal_type,
            'Cycle Direction': current_cycle_direction,
            'Incoming Threshold': incoming_threshold,
            'Forecast Price Target': price_forecast_target,
            'Min Threshold': min_threshold,
            'Max Threshold': max_threshold,
            'Middle Threshold': middle_threshold,
            'Distance to Min Threshold %': distance_to_min_threshold,
            'Distance to Max Threshold %': distance_to_max_threshold,
            'Support': support,
            'Resistance': resistance,
            'Bullish Volume': buy_volume[timeframe],
            'Bearish Volume': sell_volume[timeframe],
            'Total Volume': total_volume,
            'Bullish Ratio %': bullish_ratio,
            'Bearish Ratio %': bearish_ratio,
            'Signals': signals,
            'VSR Upper Lines': vsr['upper_lines'],
            'VSR Lower Lines': vsr['lower_lines'],
            'Donchian Upper': trend['dc_upper'][-1],
            'Donchian Lower': trend['dc_lower'][-1],
            'MA Short': trend['ma_short'][-1],
            'MA Long': trend['ma_long'][-1],
            'Gann Short': gann['hilo1'][-1],
            'Gann Mid': gann['hilo2'][-1],
            'Gann Long': gann['hilo3'][-1]
        })

    # Print summary table
    print("\n=== Summary of All Timeframes ===")
    print("=" * 180)
    header = (f"{'Timeframe':<10} {'Close':<10} {'Reversal':<10} {'Rev Price':<12} {'Cycle':<10} {'Threshold':<12} {'Forecast':<12} {'Min Th':<12} {'Max Th':<12} {'Mid Th':<12} "
              f"{'Support':<12} {'Resistance':<12} {'Bull Vol':<12} {'Bear Vol':<12} {'Total Vol':<12} {'Bull %':<10} {'Bear %':<10} {'Signals':<20}")
    print(header)
    print("=" * 180)
    for data in summary_data:
        signals_str = ', '.join(data['Signals'])[:17] + '...' if len(', '.join(data['Signals'])) > 17 else ', '.join(data['Signals'])
        print(f"{data['Timeframe']:<10} {data['Current Close']:<10.2f} {data['Last Major Reversal Type']:<10} {data['Last Major Reversal Price']:<12.2f} "
              f"{data['Cycle Direction']:<10} {data['Incoming Threshold']:<12.2f} {data['Forecast Price Target']:<12.2f} {data['Min Threshold']:<12.2f} "
              f"{data['Max Threshold']:<12.2f} {data['Middle Threshold']:<12.2f} {data['Support']:<12.2f} {data['Resistance']:<12.2f} "
              f"{data['Bullish Volume']:<12.2f} {data['Bearish Volume']:<12.2f} {data['Total Volume']:<12.2f} {data['Bullish Ratio %']:<10.2f} "
              f"{data['Bearish Ratio %']:<10.2f} {signals_str:<20}")
    print("=" * 180)

except Exception as e:
    print(f"Error in main analysis loop: {e}")
    raise