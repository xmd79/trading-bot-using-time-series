import requests
import numpy as np
import talib
from binance.client import Client as BinanceClient
import datetime
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define Binance client by reading API key and secret from a local file
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

# Initialize Binance client
client = get_binance_client()

# Constants
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

def fetch_usdc_pairs():
    """Fetch all available trading pairs against USDC."""
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']
    usdc_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
    return usdc_pairs

def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 1000
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
    return candles

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    close_prices = np.array(close_prices)
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) == 0:
        raise ValueError("No valid close prices available for threshold calculation.")

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    min_percentage_custom = minimum_percentage / 100
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    avg_mtf = np.nanmean(close_prices)
    return min_threshold, max_threshold, avg_mtf

def detect_reversal(candles):
    closes = np.array([candle['close'] for candle in candles])
    closes = closes[~np.isnan(closes) & (closes > 0)]

    if len(closes) < 1:
        raise ValueError("Insufficient close prices for reversal detection.")

    sma_length = 56
    sma56 = None

    if len(closes) >= sma_length:
        valid_closes = closes[-sma_length:]
        sma56 = talib.SMA(valid_closes, timeperiod=sma_length)[-1]

    local_dip_signal = "No Local close vs SMA Dip"
    local_top_signal = "No Local close vs SMA Top"

    if sma56 and closes[-1] < sma56 and all(closes[-1] < closes[-i] for i in range(2, 5) if len(closes) >= i):
        local_dip_signal = f"Local Dip detected at price {closes[-1]}"

    if sma56 and closes[-1] > sma56 and all(closes[-1] > closes[-i] for i in range(2, 5) if len(closes) >= i):
        local_top_signal = f"Local Top detected at price {closes[-1]}"

    return local_dip_signal, local_top_signal, sma56

def sinusoidal_forecast(close_prices):
    """Calculate the sinusoidal forecast using the HT_SINE function."""
    sine_wave = talib.HT_SINE(close_prices)
    return sine_wave

def linear_regression_forecast(close_prices, forecast_steps=1):
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) == 0:
        raise ValueError("No valid close prices for linear regression.")

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(close_prices), len(close_prices) + forecast_steps).reshape(-1, 1)
    forecast_prices = model.predict(future_X)

    return forecast_prices

def fft_forecast(close_prices):
    """Calculate the FFT forecast for cycle prediction."""
    n = len(close_prices)
    if n <= 1:
        return []

    fft_result = np.fft.fft(close_prices)
    threshold = np.abs(fft_result).max() / 10
    dominant_frequencies = np.where(np.abs(fft_result) > threshold, fft_result, 0)

    forecasted_prices = np.fft.ifft(dominant_frequencies)
    return forecasted_prices.real  # Return the real part

def analyze_asset(symbol):
    """Analyze a single asset."""
    candles = get_candles(symbol, timeframes)

    # Group candles by timeframe
    candle_map = {timeframe: [] for timeframe in timeframes}
    for candle in candles:
        candle_map[candle['timeframe']].append(candle)

    analysis_results = {}
    bullish_dips = []

    for timeframe in timeframes:
        close_prices = np.array([candle['close'] for candle in candle_map[timeframe]])
        if len(close_prices) < 1:
            continue

        # Calculate thresholds
        min_threshold, max_threshold, avg_mtf = calculate_thresholds(close_prices)

        # Detect reversal signals
        dip_signal, top_signal, _ = detect_reversal(candle_map[timeframe])

        # Calculate Volume
        total_volume = sum(candle['volume'] for candle in candle_map[timeframe])
        bullish_volume = sum(candle['volume'] for candle in candle_map[timeframe] if candle['close'] > candle['open'])
        bearish_volume = total_volume - bullish_volume

        # Add closes for mood confirmation
        closes = np.array([candle['close'] for candle in candle_map[timeframe]])

        # Check if the volume is bullish and the dip signal was detected
        if "Local Dip detected" in dip_signal and bullish_volume > bearish_volume:
            bullish_dips.append((timeframe, closes[-1], bullish_volume, dip_signal))

        # Forecast the price using linear regression
        projected_price = linear_regression_forecast(close_prices)[-1]

        # Add sinusoidal forecast
        sine_forecast = sinusoidal_forecast(close_prices)

        # Confirm market mood
        overall_market_mood = "Bullish" if (bullish_volume > bearish_volume and closes[-1] >= avg_mtf) else "Bearish"

        # Dynamic adjustment of the projected price based on market conditions
        if overall_market_mood == "Bullish":
            projected_price = max(projected_price, max_threshold)
        else:
            projected_price = min(projected_price, min_threshold)

        # Calculate the price validation against a middle price
        middle_price = (min_threshold + max_threshold) / 2
        price_validation = "Above Middle Price" if closes[-1] >= middle_price else "Below Middle Price"

        analysis_results[timeframe] = {
            "Min Threshold": min_threshold,
            "Max Threshold": max_threshold,
            "Avg MTF": avg_mtf,
            "Projected Price": projected_price,
            "Sine Forecast": sine_forecast[-1] if len(sine_forecast) > 0 else None,
            "Market Mood": overall_market_mood,
            "Dip Signal": dip_signal,
            "Top Signal": top_signal,
            "Middle Price Validation": price_validation
        }

    return symbol, analysis_results, bullish_dips

def main():
    current_time = datetime.datetime.now()
    print("Current local Time is now at: ", current_time)

    # Fetch all tradable assets against USDC
    usdc_pairs = fetch_usdc_pairs()
    
    overall_analysis = {}
    bullish_dips_found = []

    # Use thread pool to fetch data concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_asset, symbol): symbol for symbol in usdc_pairs}
        for future in as_completed(future_to_symbol):
            symbol, results, dips = future.result()
            overall_analysis[symbol] = results

            # Collect bullish dips
            bullish_dips_found.extend([(symbol, timeframe, price, volume, dip_signal) for timeframe, price, volume, dip_signal in dips])

    # Display results in a dynamic table format
    print("\n" + "=" * 60)
    print(f"{'Symbol':<15}{'Timeframe':<10}{'Min Threshold':<45}{'Max Threshold':<45}{'Projected Price':<45}{'Market Mood':<15}")
    print("-" * 60)
    
    for symbol, results in overall_analysis.items():
        for timeframe, data in results.items():
            print(f"{symbol:<15}{timeframe:<10}{data['Min Threshold']:<45.25f}{data['Max Threshold']:<45.25f}{data['Projected Price']:<45.25f}{data['Market Mood']:<15}")

    print("=" * 60)

    # Print bullish dips found
    print("\nBullish Dips Found:")
    print(f"{'Symbol':<15}{'Timeframe':<10}{'Dip Price':<30}{'Volume (Bullish)':<25}{'Dip Signal':<30}")
    print("-" * 60)
    for symbol, timeframe, price, volume, dip_signal in bullish_dips_found:
        print(f"{symbol:<15}{timeframe:<10}{price:<30.25f}{volume:<25.25f}{dip_signal:<30}")

    print("=" * 60)

if __name__ == "__main__":
    main()
