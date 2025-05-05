import numpy as np
import pandas as pd
import time
import datetime
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import concurrent.futures
import gc

# Initialize Binance client
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})


# Utility Functions

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
        except BinanceAPIException as e:
            print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    return []


def fetch_candles_in_parallel(timeframes, symbol='BTCUSDT', limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))


def normalize_angle(angle):
    return angle % 360


def determine_cycle_stage(angle):
    quadrant = (angle // (360 / 4)) % 4
    if quadrant == 0:
        return 'Dip (-+)', 'BUY Bias'
    elif quadrant == 1:
        return 'Up (++ stage)', 'HOLD'
    elif quadrant == 2:
        return 'Top (+-)', 'SELL Bias'
    elif quadrant == 3:
        return 'Down (-- stage)', 'AVOID'


def compute_fft_forecast(series, n_forecast=20):
    N = len(series)
    freq_data = fft(series)
    filtered = np.copy(freq_data)
    cutoff = int(N * 0.1)
    filtered[cutoff:-cutoff] = 0
    smoothed = np.real(ifft(filtered))
    t = np.arange(N)
    model = LinearRegression().fit(t.reshape(-1, 1), smoothed)
    forecast_t = np.arange(N, N + n_forecast)
    forecast = model.predict(forecast_t.reshape(-1, 1))
    return smoothed, forecast


def compare_bullish_bearish_volume(df, n=12):
    bullish_volume = df[df['close'] > df['open']].tail(n)['volume'].sum()
    bearish_volume = df[df['close'] < df['open']].tail(n)['volume'].sum()
    total_relevant_volume = bullish_volume + bearish_volume

    if total_relevant_volume == 0:
        return 50.0, 50.0  # Balanced when no volume

    bullish_percentage = (bullish_volume / total_relevant_volume) * 100
    bearish_percentage = (bearish_volume / total_relevant_volume) * 100
    return bullish_percentage, bearish_percentage


def calculate_momentum(df, n=12):
    price_changes = df['close'].diff().tail(n).dropna()
    momentum = price_changes.sum()
    return momentum


def normalize_distances(current_price, high, low):
    range_total = high - low
    if range_total == 0:
        return 50.0, 50.0  # Avoid division by zero
    distance_to_high_pct = ((current_price - low) / range_total) * 100
    distance_to_low_pct = ((high - current_price) / range_total) * 100
    return distance_to_high_pct, distance_to_low_pct


def handle_nan_zeros(df):
    df = df.ffill()
    df[df == 0] = np.nan
    df = df.ffill()
    df = df.dropna()
    return df


def analyze_trends(df, timeframe):
    df = handle_nan_zeros(df)
    last_12_prices = df['close'].tail(12)
    last_12_volumes = df['volume'].tail(12)

    # Price trend
    if last_12_prices.iloc[-1] > last_12_prices.iloc[0]:
        price_trend = "UP"
    elif last_12_prices.iloc[-1] < last_12_prices.iloc[0]:
        price_trend = "DOWN"
    else:
        price_trend = "No clear trend"

    # Volume trend
    if last_12_volumes.iloc[-1] > last_12_volumes.iloc[0]:
        volume_trend = "UP"
    elif last_12_volumes.iloc[-1] < last_12_volumes.iloc[0]:
        volume_trend = "DOWN"
    else:
        volume_trend = "No clear trend"

    print(f"Price Trend: {price_trend}")
    print(f"Volume Trend: {volume_trend}")


def symmetrize_wave(df, timeframe):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Find the major DIP (lowest close) and TOP (highest close)
    dip_idx = df['close'].idxmin()
    top_idx = df['close'].idxmax()
    dip_price = df.loc[dip_idx]['close']
    top_price = df.loc[top_idx]['close']

    current_price = df['close'].iloc[-1]
    latest_time = df.index[-1]

    # Determine which major reversal is closest to current price
    dist_to_dip = abs(current_price - dip_price)
    dist_to_top = abs(current_price - top_price)

    if dist_to_dip < dist_to_top:
        last_major_reversal = 'DIP'
        reversal_idx = dip_idx
        reversal_price = dip_price
        next_forecast_target = 'TOP'
    else:
        last_major_reversal = 'TOP'
        reversal_idx = top_idx
        reversal_price = top_price
        next_forecast_target = 'DIP'

    # Phase angle calculation based on the major reversal
    total_time_span = (df.index[-1] - df.index[0]).total_seconds()
    reversal_time_span = (latest_time - reversal_idx).total_seconds()
    phase_angle = (reversal_time_span / total_time_span) * 360
    phase_angle = normalize_angle(phase_angle)

    # Determine stage and action based on phase angle
    stage, action = determine_cycle_stage(phase_angle)

    # Thresholds
    min_threshold = dip_price
    max_threshold = top_price
    middle_threshold = (dip_price + top_price) / 2

    # FFT forecast
    smoothed, forecast = compute_fft_forecast(df['close'].values)
    forecast_price = forecast[-1]

    bullish_percentage, bearish_percentage = compare_bullish_bearish_volume(df)
    dist_to_high_pct, dist_to_low_pct = normalize_distances(current_price, top_price, dip_price)
    momentum = calculate_momentum(df)

    print(f"\n[{datetime.datetime.now()}] --- {timeframe} ---")
    print(f"Major Reversal Found: {last_major_reversal} at {reversal_price:.2f}")
    print(f"Phase: {('Up Cycle (Dip to Top)' if last_major_reversal == 'DIP' else 'Down Cycle (Top to Dip)')} | Angle: {phase_angle:.2f}° | Stage: {stage} | Action: {action}")
    print(f"Forecast for next {next_forecast_target}: {forecast_price:.2f}")
    print(f"Min Threshold (DIP): {min_threshold:.2f}")
    print(f"Max Threshold (TOP): {max_threshold:.2f}")
    print(f"Middle Threshold: {middle_threshold:.2f}")
    print(f"Bullish Volume Percentage: {bullish_percentage:.2f}%")
    print(f"Bearish Volume Percentage: {bearish_percentage:.2f}%")
    print(f"Distance to High: {dist_to_high_pct:.2f}%")
    print(f"Distance to Low: {dist_to_low_pct:.2f}%")
    print(f"Momentum: {'Up' if momentum > 0 else 'Down' if momentum < 0 else 'Neutral'} ({momentum:.4f})")
    print("-" * 60)

    if last_major_reversal == 'DIP':
        print(f"Incoming Reversal Expected: TOP at {forecast_price:.2f}")
    elif last_major_reversal == 'TOP':
        print(f"Incoming Reversal Expected: DIP at {forecast_price:.2f}")

    analyze_trends(df, timeframe)


def sma_ribbon(df, windows=[5, 10, 20, 40, 80]):
    for w in windows:
        df[f'SMA_{w}'] = df['close'].rolling(window=w).mean()
    return df


def real_time_update(timeframes, fetch_candles_in_parallel, interval=30):
    while True:
        data_dict = fetch_candles_in_parallel(timeframes)
        for timeframe, data in data_dict.items():
            if not data:
                continue
            df = pd.DataFrame(data)
            df = sma_ribbon(df)
            symmetrize_wave(df, timeframe)
            gc.collect()
        time.sleep(interval)


# Example: Run the real-time update with multiple timeframes
timeframes = ['1m', '3m', '5m', '15m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
real_time_update(timeframes, fetch_candles_in_parallel, interval=30)
