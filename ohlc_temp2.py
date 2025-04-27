import numpy as np
from binance.client import Client
import datetime
import time
import concurrent.futures
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from decimal import Decimal, getcontext
import requests
import pandas as pd
from scipy import signal
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from matplotlib import style

# Set Decimal precision
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Load credentials
try:
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
except FileNotFoundError:
    raise Exception("credentials.txt not found.")

# Initialize Binance client
client = Client(api_key, api_secret, requests_params={"timeout": 30})

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
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
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
        except Exception as e:
            print(f"Error fetching {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch {timeframe}.")
    return []

def find_reversals(highs, lows, window=5):
    high_peaks = signal.argrelextrema(highs, np.greater, order=window)[0]
    low_peaks = signal.argrelextrema(lows, np.less, order=window)[0]
    return high_peaks, low_peaks

def fit_sinusoidal_wave(times, prices, period):
    A = (np.max(prices) - np.min(prices)) / 2
    omega = 2 * np.pi / period
    offset = np.mean(prices)
    return A * np.sin(omega * (times - times[0])) + offset

def compute_technicals(df):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    technicals = {
        'RSI': talib.RSI(closes, timeperiod=14),
        'MACD': talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)[0],
        'BB_upper': talib.BBANDS(closes, timeperiod=20)[0],
        'BB_lower': talib.BBANDS(closes, timeperiod=20)[2],
        'ATR': talib.ATR(highs, lows, closes, timeperiod=14),
        'OBV': talib.OBV(closes, volumes),
        'ADX': talib.ADX(highs, lows, closes, timeperiod=14),
        'STOCH': talib.STOCH(highs, lows, closes)[0],
        'CCI': talib.CCI(highs, lows, closes, timeperiod=14),
        'MOM': talib.MOM(closes, timeperiod=10)
    }
    return technicals

def train_ml_model(features, target):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_scaled, target)
    return model, scaler

def analyze_timeframe(df, timeframe):
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    times = df.index.astype(np.int64) // 10**9  # Conversion to Unix timestamps

    # Input data validation
    if len(prices) < 50:
        print(f"{timeframe}: Insufficient data (<50 candles).")
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"{timeframe}: Invalid index type, expected DatetimeIndex.")
        return None
    if not np.isfinite(prices).all() or not np.isfinite(highs).all() or not np.isfinite(lows).all():
        print(f"{timeframe}: Non-finite values in price data.")
        return None

    # Debug input data
    print(f"Debug: {timeframe} df['time'] head: {df.index[:5]}")
    print(f"Debug: {timeframe} df['time'] last: {df.index[-1]}")

    # Find reversals and major dip/top
    high_peaks, low_peaks = find_reversals(highs, lows)
    reversal_indices = np.sort(np.concatenate([high_peaks, low_peaks]))
    reversal_times = times[reversal_indices]
    reversal_prices = np.concatenate([highs[high_peaks], lows[low_peaks]])
    reversal_types = ['High'] * len(high_peaks) + ['Low'] * len(low_peaks)
    major_top_idx = np.argmax(highs)
    major_dip_idx = np.argmin(lows)
    major_top = (times[major_top_idx], highs[major_top_idx])
    major_dip = (times[major_dip_idx], lows[major_dip_idx])

    # Sinusoidal wave
    window = min(50, len(prices))
    recent_indices = np.arange(len(prices) - window, len(prices))
    recent_times = times[recent_indices]
    recent_prices = prices[recent_indices]
    period = window
    sin_wave = fit_sinusoidal_wave(recent_indices, recent_prices, period)
    forecast_indices = np.arange(recent_indices[-1], recent_indices[-1] + 10)
    sin_forecast = fit_sinusoidal_wave(forecast_indices, recent_prices, period)

    # Debug forecast indices
    print(f"Debug: {timeframe} recent_indices[-1]: {recent_indices[-1]}, forecast_indices: {forecast_indices}")

    # FFT analysis
    yf = fftpack.fft(sin_wave)
    freq = fftpack.fftfreq(len(sin_wave), d=1)  # Time difference in seconds for hourly data
    power = np.abs(yf) ** 2
    top_indices = power.argsort()[-5:][::-1]
    yf_filtered = np.zeros_like(yf)
    yf_filtered[top_indices] = yf[top_indices]
    fft_wave = fftpack.ifft(yf_filtered).real
    fft_forecast = np.zeros_like(forecast_indices, dtype=float)
    for idx in top_indices:
        A = np.abs(yf[idx]) / len(sin_wave)
        omega = 2 * np.pi * freq[idx]
        phi = np.angle(yf[idx])
        fft_forecast += 2 * A * np.cos(omega * (forecast_indices - recent_indices[0]) + phi)

    # Volume confirmation
    obv = talib.OBV(prices, volumes)
    volume_trend = np.diff(obv)
    volume_stages = np.where(volume_trend > 0, 'Accumulation', 'Distribution')

    # Energy motion flow
    atr = talib.ATR(highs, lows, prices, timeperiod=14)
    macd = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    energy_flow = atr * macd

    # Technical indicators
    technicals = compute_technicals(df)

    # ML forecast
    feature_list = ['RSI', 'MACD', 'ATR', 'OBV', 'ADX', 'STOCH', 'CCI', 'MOM']
    features = np.column_stack([technicals[f][-window:] for f in feature_list] + [sin_wave, fft_wave, obv[-window:], energy_flow[-window:]])
    target = prices[-window:]
    if len(features) >= 10 and np.isfinite(features).all() and np.isfinite(target).all():
        model, scaler = train_ml_model(features[:-1], target[1:])
        future_features = np.column_stack([technicals[f][-1:] for f in feature_list] + [sin_forecast[:1], fft_forecast[:1], obv[-1:], energy_flow[-1:]])
        future_features_scaled = scaler.transform(future_features)
        ml_forecast = []
        last_features = future_features.copy()
        for _ in range(10):
            pred = model.predict(future_features_scaled)[0]
            ml_forecast.append(pred)
            last_features[0, -4] = pred  # Update sin_wave approximation
            last_features[0, -3] = pred  # Update fft_wave approximation
            last_features[0, -2] = last_features[0, -2] + (1 if pred > prices[-1] else -1) * volumes[-1]  # Update OBV
            last_features[0, -1] = last_features[0, -1] * 1.01  # Approximate energy flow
            future_features_scaled = scaler.transform(last_features)
    else:
        ml_forecast = np.zeros(10)
        print(f"Debug: {timeframe} ML forecast skipped due to invalid features or target.")

    # Convert times to local datetime
    local_tz = datetime.datetime.now().astimezone().tzinfo
    reversal_datetimes = pd.to_datetime(reversal_times, unit='s').tz_localize('UTC').tz_convert(local_tz)
    major_top_dt = pd.to_datetime(major_top[0], unit='s').tz_localize('UTC').tz_convert(local_tz)
    major_dip_dt = pd.to_datetime(major_dip[0], unit='s').tz_localize('UTC').tz_convert(local_tz)

    # Fix forecast datetimes
    base_time = df.index[-1]  # UTC timezone-aware
    hours_offset = forecast_indices - recent_indices[-1]
    forecast_timestamps = pd.date_range(start=base_time, periods=len(hours_offset), freq='H', tz='UTC')
    forecast_datetimes = forecast_timestamps.tz_convert(local_tz)

    # Debug forecast datetimes
    print(f"Debug: {timeframe} base_time: {base_time}, tz: {base_time.tz}")
    print(f"Debug: {timeframe} forecast_datetimes: {forecast_datetimes}")

    # Print detailed data
    print(f"\n=== {timeframe} Analysis ===")
    print("Reversals:")
    for dt, p, r_type in zip(reversal_datetimes, reversal_prices, reversal_types):
        print(f"  Time: {dt}, Price: {p:.2f}, Type: {r_type}")
    print(f"Major Top: Time: {major_top_dt}, Price: {major_top[1]:.2f}")
    print(f"Major Dip: Time: {major_dip_dt}, Price: {major_dip[1]:.2f}")
    print("Sinusoidal Forecast (10 hours):")
    for dt, p in zip(forecast_datetimes, sin_forecast):
        print(f"  Time: {dt}, Price: {p:.2f}")
    print("FFT Forecast (10 hours):")
    for dt, p in zip(forecast_datetimes, fft_forecast):
        print(f"  Time: {dt}, Price: {p:.2f}")
    print("ML Forecast (10 hours):")
    for dt, p in zip(forecast_datetimes, ml_forecast):
        print(f"  Time: {dt}, Price: {p:.2f}")
    print("Volume Stages (Last 10):")
    for i in range(-10, 0):
        print(f"  Time: {df.index[i].tz_convert(local_tz)}, Stage: {volume_stages[i]}")
    print("Energy Flow (Last 5):")
    for i in range(-5, 0):
        print(f"  Time: {df.index[i].tz_convert(local_tz)}, Energy: {energy_flow[i]:.2f}")
    print("Technicals (Last 5):")
    for key in technicals:
        print(f"  {key}: {technicals[key][-5:].tolist()}")

    return {
        'df': df,
        'times': times,
        'reversal_datetimes': reversal_datetimes,
        'reversal_prices': reversal_prices,
        'reversal_types': reversal_types,
        'major_top': (major_top_dt, major_top[1]),
        'major_dip': (major_dip_dt, major_dip[1]),
        'sin_wave': sin_wave,
        'sin_forecast': sin_forecast,
        'forecast_datetimes': forecast_datetimes,
        'fft_wave': fft_wave,
        'fft_forecast': fft_forecast,
        'ml_forecast': ml_forecast,
        'volume_stages': volume_stages,
        'energy_flow': energy_flow,
        'technicals': technicals
    }

def analyze_market_data_wave():
    style.use('ggplot')  # Valid Matplotlib style
    timeframes = [
        client.KLINE_INTERVAL_1MINUTE,
        client.KLINE_INTERVAL_5MINUTE,
        client.KLINE_INTERVAL_15MINUTE,
        client.KLINE_INTERVAL_1HOUR,
        client.KLINE_INTERVAL_4HOUR,
        client.KLINE_INTERVAL_1DAY,
        client.KLINE_INTERVAL_1WEEK
    ]
    limit = 100

    print("Fetching candlestick data...")
    candles_dict = fetch_candles_in_parallel(timeframes, TRADE_SYMBOL, limit)

    plt.figure(figsize=(16, 8 * len(timeframes)))
    results = {}

    for idx, timeframe in enumerate(timeframes):
        if not candles_dict[timeframe]:
            print(f"No data for {timeframe}.")
            continue

        candles = candles_dict[timeframe]
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)  # Ensure UTC
        df.set_index('time', inplace=True)  # Set time as index

        results[timeframe] = analyze_timeframe(df, timeframe)
        if not results[timeframe]:
            continue

        r = results[timeframe]
        plt.subplot(len(timeframes), 2, 2 * idx + 1)
        # Convert all datetimes to numerical for plotting
        time_num = mdates.date2num(df.index)
        reversal_num = mdates.date2num(r['reversal_datetimes'])
        major_top_num = mdates.date2num([r['major_top'][0]])
        major_dip_num = mdates.date2num([r['major_dip'][0]])
        window = min(50, len(df))
        sin_wave_num = mdates.date2num(df.index[-window:])
        forecast_num = mdates.date2num(r['forecast_datetimes'])

        plt.plot(time_num, df['close'], label='Close Price', color='royalblue', linewidth=2)
        for dt_num, p, r_type in zip(reversal_num, r['reversal_prices'], r['reversal_types']):
            plt.scatter(dt_num, p, color='red' if r_type == 'High' else 'green', s=50, label='Reversals' if dt_num == reversal_num[0] else "")
        plt.scatter(major_top_num, [r['major_top'][1]], color='red', marker='^', s=200, label='Major Top')
        plt.scatter(major_dip_num, [r['major_dip'][1]], color='green', marker='v', s=200, label='Major Dip')
        plt.plot(sin_wave_num, r['sin_wave'], 'g--', label='Sinusoidal Wave', alpha=0.7)
        plt.plot(forecast_num, r['sin_forecast'], 'g:', label='Sin Forecast', alpha=0.7)
        plt.plot(forecast_num, r['ml_forecast'], 'm:', label='ML Forecast', alpha=0.7)
        # Shaded areas for major top and dip
        plt.axvspan(major_top_num[0] - 30/(24*60), major_top_num[0] + 30/(24*60), color='red', alpha=0.2)
        plt.axvspan(major_dip_num[0] - 30/(24*60), major_dip_num[0] + 30/(24*60), color='green', alpha=0.2)
        plt.title(f'{timeframe} - Price and Forecasts', fontsize=12, fontweight='bold')
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Price (USDC)', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().tick_params(axis='x', rotation=45)

        plt.subplot(len(timeframes), 2, 2 * idx + 2)
        plt.plot(sin_wave_num, r['fft_wave'], label='FFT Wave', color='purple', linewidth=2)
        plt.plot(forecast_num, r['fft_forecast'], 'purple', linestyle=':', label='FFT Forecast', alpha=0.7)
        plt.plot(time_num, r['energy_flow'], 'orange', label='Energy Flow', alpha=0.7)
        plt.title(f'{timeframe} - FFT and Energy Flow', fontsize=12, fontweight='bold')
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Amplitude/Energy', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    print("Plots displayed.")

if __name__ == "__main__":
    try:
        analyze_market_data_wave()
    except Exception as e:
        print(f"Error: {e}")
