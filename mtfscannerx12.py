import requests
import numpy as np
import talib
from binance.client import Client as BinanceClient
import datetime
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.fftpack import fft

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

# Constants for timeframes
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

def calculate_thresholds(close_prices, min_percentage=3, max_percentage=3):
    close_prices = np.array(close_prices)
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) == 0:
        raise ValueError("No valid close prices available for threshold calculation.")

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    min_threshold = min_close - ((max_close - min_close) * (min_percentage / 100))
    max_threshold = max_close + ((max_close - min_close) * (max_percentage / 100))
    return min_threshold, max_threshold, min_close, max_close

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
        local_dip_signal = f"Local Dip detected at price {closes[-1]:.25f}"

    if sma56 and closes[-1] > sma56 and all(closes[-1] > closes[-i] for i in range(2, 5) if len(closes) >= i):
        local_top_signal = f"Local Top detected at price {closes[-1]:.25f}"

    return local_dip_signal, local_top_signal, sma56

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
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) < 2:
        return None

    # Calculate FFT
    freq_components = fft(close_prices)
    return freq_components

def calculate_regression_channel(close_prices):
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) == 0:
        raise ValueError("No valid close prices for regression channel.")

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices

    model = LinearRegression()
    model.fit(X, y)

    # Predictions for regression line
    regression_line = model.predict(X)

    # Calculate upper and lower channel bounds
    residuals = close_prices - regression_line
    std_dev = np.std(residuals)

    upper_channel = regression_line + (std_dev * 2)
    lower_channel = regression_line - (std_dev * 2)

    return regression_line, upper_channel, lower_channel

def analyze_asset(symbol):
    """Analyze a single asset and return analysis results."""
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
        min_threshold, max_threshold, min_close, max_close = calculate_thresholds(close_prices)

        # Detect reversal signals
        dip_signal, top_signal, _ = detect_reversal(candle_map[timeframe])

        # Calculate Volume
        total_volume = sum(candle['volume'] for candle in candle_map[timeframe])
        bullish_volume = sum(candle['volume'] for candle in candle_map[timeframe] if candle['close'] > candle['open'])
        bearish_volume = total_volume - bullish_volume
        
        # Ensure bullish volume confirms potential dips
        if bullish_volume <= bearish_volume:
            continue

        # Confirm daily volume dips
        if close_prices[-1] > min_close:  # Current close should be above last major reversal
            current_min_dist = abs(close_prices[-1] - min_close)
            current_max_dist = abs(max_close - close_prices[-1])
            total_distance = current_min_dist + current_max_dist

            if total_distance > 0:
                distance_to_min_perc = (current_min_dist / total_distance) * 100
                distance_to_max_perc = (current_max_dist / total_distance) * 100

                # Add conditions for confirming dips
                if distance_to_min_perc < 45:  # Under 45-degree condition
                    bullish_dips.append((symbol, timeframe, dip_signal, close_prices[-1]))

        # Calculate regression channel
        regression_line, upper_channel, lower_channel = calculate_regression_channel(close_prices)

        # Forecast the price using linear regression
        projected_price = linear_regression_forecast(close_prices, forecast_steps=1)[-1]

        # FFT Calculation
        fft_values = fft_forecast(close_prices)
        major_frequency = np.max(np.abs(fft_values)) if fft_values is not None else None

        # Store the analysis results
        analysis_results[timeframe] = {
            "Min Threshold": f"{min_threshold:.25f}",
            "Max Threshold": f"{max_threshold:.25f}",
            "Projected Price": f"{projected_price:.25f}",
            "Upper Channel": f"{upper_channel[-1]:.25f}",
            "Lower Channel": f"{lower_channel[-1]:.25f}",
            "Trend": "Bullish" if regression_line[-1] < projected_price else "Bearish",
            "Market Mood": "Bullish" if (bullish_volume > bearish_volume) else "Bearish",
            "Dip Signal": dip_signal,
            "Top Signal": top_signal,
            "Distance to Min Percentage": f"{distance_to_min_perc:.2f}%",
            "Distance to Max Percentage": f"{distance_to_max_perc:.2f}%",
            "Major FFT Frequency": major_frequency
        }

    return symbol, analysis_results, bullish_dips

def main():
    current_time = datetime.datetime.now()
    print("Current local Time is now at: ", current_time)

    # Fetch all tradable assets against USDC
    usdc_pairs = fetch_usdc_pairs()
    
    overall_analysis = {}
    all_bullish_dips = []

    # Use thread pool to fetch data concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_asset, symbol): symbol for symbol in usdc_pairs}
        for future in as_completed(future_to_symbol):
            symbol, results, bullish_dips = future.result()
            overall_analysis[symbol] = results
            all_bullish_dips.extend(bullish_dips)
            
    # Display detailed analysis results
    print("\n" + "=" * 60)
    print(f"{'Symbol':<15}{'Timeframe':<10}{'Min Threshold':<45}{'Max Threshold':<45}{'Projected Price':<45}{'Upper Channel':<45}{'Lower Channel':<45}{'Market Mood':<15}")
    print("-" * 60)
    
    for symbol, results in overall_analysis.items():
        for timeframe, data in results.items():
            print(f"{symbol:<15}{timeframe:<10}{data['Min Threshold']:<45}{data['Max Threshold']:<45}{data['Projected Price']:<45}{data['Upper Channel']:<45}{data['Lower Channel']:<45}{data['Market Mood']:<15}")

    print("=" * 60)
    
    # Display bullish dips
    print("\nBullish Dips Detected:")
    for dip in all_bullish_dips:
        print(f"Symbol: {dip[0]}, Timeframe: {dip[1]}, Dip Signal: {dip[2]}, Current Price: {dip[3]:.25f}")

if __name__ == "__main__":
    main()