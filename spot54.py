import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.fftpack as fftpack

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Fetch candlestick data in parallel for multiple timeframes
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=100):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        candles = []
        for k in klines:
            candle = {
                "time": k[0] / 1000,  # Convert to seconds
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
        print(f"Error fetching candles: {e.message}")
        return []

def get_current_btc_price():
    try:
        ticker = client.get_symbol_ticker(symbol=TRADE_SYMBOL)
        return float(ticker['price'])
    except BinanceAPIException as e:
        print(f"Error fetching BTC price: {e.message}")
        return 0.0

# Gann harmonic analysis
def gann_harmonic_levels(price, base_low, base_high):
    """Calculate Gann harmonic price levels based on octaves and ratios."""
    range_size = base_high - base_low
    octave = base_low * 2  # 2:1 harmonic (octave)
    harmonic_levels = {
        "1/8": base_low + range_size * 0.125,
        "1/4": base_low + range_size * 0.25,
        "1/3": base_low + range_size * (1/3),
        "1/2": base_low + range_size * 0.5,
        "2/3": base_low + range_size * (2/3),
        "3/4": base_low + range_size * 0.75,
        "7/8": base_low + range_size * 0.875,
        "Octave": octave
    }
    return harmonic_levels

# Detect dominant cycles using FFT
def detect_cycles(prices, sample_rate=1):
    """Use Fourier Transform to identify dominant price cycles."""
    n = len(prices)
    yf = fftpack.fft(prices)
    freqs = fftpack.fftfreq(len(yf)) * sample_rate
    power = np.abs(yf) ** 2
    dominant_idx = np.argsort(power)[::-1][1:]  # Skip DC component (index 0)
    dominant_periods = [int(1 / freqs[i]) for i in dominant_idx[:3] if freqs[i] > 0]
    return dominant_periods

# Forecast price using linear regression and harmonics
def forecast_price(candles, forecast_horizon=10):
    closes = np.array([c["close"] for c in candles])
    times = np.arange(len(closes)).reshape(-1, 1)
    
    # Linear regression for trend
    model = LinearRegression()
    model.fit(times, closes)
    future_times = np.arange(len(closes), len(closes) + forecast_horizon).reshape(-1, 1)
    trend_forecast = model.predict(future_times)
    
    # Add harmonic levels based on recent range
    base_low = min(closes[-50:])  # Last 50 candles for range
    base_high = max(closes[-50:])
    harmonic_levels = gann_harmonic_levels(closes[-1], base_low, base_high)
    
    # Detect cycles
    cycles = detect_cycles(closes)
    print(f"Detected dominant cycles (in candles): {cycles}")
    
    return trend_forecast, harmonic_levels

# Main execution
def main():
    # Fetch data for multiple timeframes
    timeframes = [client.KLINE_INTERVAL_1HOUR, client.KLINE_INTERVAL_4HOUR, client.KLINE_INTERVAL_1DAY]
    candle_data = fetch_candles_in_parallel(timeframes, limit=100)
    
    # Use 1-hour data for forecast
    hourly_candles = candle_data[client.KLINE_INTERVAL_1HOUR]
    current_price = get_current_btc_price()
    print(f"Current BTC/USDC Price: {current_price}")
    
    # Generate forecast
    forecast_horizon = 10  # Next 10 hours
    trend_forecast, harmonic_levels = forecast_price(hourly_candles, forecast_horizon)
    
    # Output results
    print("\n10-Hour Price Forecast (Trend):")
    for i, price in enumerate(trend_forecast):
        print(f"Hour {i+1}: {price:.2f}")
    
    print("\nGann Harmonic Levels from Current Price:")
    for level, value in harmonic_levels.items():
        print(f"{level}: {value:.2f}")
    
    # Suggest next harmonic target
    next_target = min([v for v in harmonic_levels.values() if v > current_price], default=harmonic_levels["Octave"])
    print(f"\nNext Likely Harmonic Target: {next_target:.2f}")

if __name__ == "__main__":
    main()