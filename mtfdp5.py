import numpy as np
import talib
import datetime
import concurrent.futures
from sklearn.linear_model import LinearRegression
from binance.client import Client as BinanceClient

# Function to create a Binance client
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = [line.strip() for line in f]
        api_key, api_secret = lines[0], lines[1]
    return BinanceClient(api_key, api_secret)

# Initialize Binance client
client = get_binance_client()
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

def fetch_usdc_pairs():
    """Fetch all available trading pairs against USDC."""
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']
    return [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']

def get_candles(symbol, timeframes):
    """Get kline data for a symbol across specified timeframes."""
    candles = []
    for timeframe in timeframes:
        klines = client.get_klines(symbol=symbol, interval=timeframe)
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

def calculate_thresholds(close_prices, min_percentage=5, max_percentage=5):
    """Calculate min and max price thresholds for candlestick closes."""
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices) if len(close_prices) > 0 else np.nan
    max_close = np.nanmax(close_prices) if len(close_prices) > 0 else np.nan
    
    min_threshold = min_close - ((max_close - min_close) * (min_percentage / 100))
    max_threshold = max_close + ((max_close - min_close) * (max_percentage / 100))
    return min_threshold, max_threshold

def detect_reversals(candles):
    """Detect local dips and tops based on SMA."""
    closes = np.array([candle['close'] for candle in candles])
    if len(closes) < 1:
        return "No local reversal signals", None

    sma_length = 56
    sma56 = talib.SMA(closes, timeperiod=sma_length)[-1] if len(closes) >= sma_length else None

    dip_signal = top_signal = "No Local close vs SMA Dip/Top"
    if sma56:
        if closes[-1] < sma56 and all(closes[-1] < closes[-i] for i in range(2, 5)):
            dip_signal = f"Local Dip detected at price {closes[-1]}"
        elif closes[-1] > sma56 and all(closes[-1] > closes[-i] for i in range(2, 5)):
            top_signal = f"Local Top detected at price {closes[-1]}"
    
    return dip_signal, top_signal, sma56

def linear_regression_forecast(close_prices, forecast_steps=1):
    """Forecast future close prices using linear regression."""
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) < 1:
        return None

    X = np.arange(len(close_prices)).reshape(-1, 1)
    model = LinearRegression().fit(X, close_prices)
    future_X = np.arange(len(close_prices), len(close_prices) + forecast_steps).reshape(-1, 1)
    return model.predict(future_X)

def calculate_regression_channel(close_prices):
    """Calculate regression channel for the given prices."""
    valid_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(valid_prices) < 2:
        return None, None, None

    X = np.arange(len(valid_prices)).reshape(-1, 1)
    model = LinearRegression().fit(X, valid_prices)
    regression_line = model.predict(X)

    residuals = valid_prices - regression_line
    std_dev = np.std(residuals)

    upper_channel = regression_line + (std_dev * 2)
    lower_channel = regression_line - (std_dev * 2)

    return regression_line, upper_channel, lower_channel

def calculate_distances(current_price, min_threshold, max_threshold):
    """Calculate the percentage distances to the min and max thresholds."""
    distance_to_min = ((current_price - min_threshold) / (max_threshold - min_threshold)) * 100
    distance_to_max = ((max_threshold - current_price) / (max_threshold - min_threshold)) * 100
    return distance_to_min, distance_to_max

def analyze_asset(symbol):
    """Analyze a symbol, returning detailed market information."""
    candles = get_candles(symbol, timeframes)

    analysis_results = {}
    for timeframe in timeframes:
        close_prices = np.array([candle['close'] for candle in candles if candle['timeframe'] == timeframe])

        min_threshold, max_threshold = calculate_thresholds(close_prices)
        dip_signal, top_signal, _ = detect_reversals(candles)
        
        total_volume = np.sum([candle['volume'] for candle in candles if candle['timeframe'] == timeframe])
        bullish_volume = np.sum([candle['volume'] for candle in candles if candle['close'] > candle['open'] and candle['timeframe'] == timeframe])
        bearish_volume = total_volume - bullish_volume

        market_mood = "Bearish" if bearish_volume > bullish_volume else "Bullish"

        if len(close_prices) > 0:
            projected_price = linear_regression_forecast(close_prices, forecast_steps=1)[-1]
            distance_to_min, distance_to_max = calculate_distances(close_prices[-1], min_threshold, max_threshold)

            analysis_results[timeframe] = {
                "Min Threshold": f"{min_threshold:.25f}",
                "Max Threshold": f"{max_threshold:.25f}",
                "Projected Price": f"{projected_price:.25f}",
                "Market Mood": market_mood,
                "Dip Signal": dip_signal,
                "Top Signal": top_signal,
                "Distance to Min (%)": f"{distance_to_min:.2f}%",
                "Distance to Max (%)": f"{distance_to_max:.2f}%"
            }

    return symbol, analysis_results

def main():
    print("Current local Time is now at: ", datetime.datetime.now())
    
    usdc_pairs = fetch_usdc_pairs()
    overall_analysis = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_asset, symbol): symbol for symbol in usdc_pairs}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, results = future.result()
            overall_analysis[symbol] = results

    # Display detailed analysis results
    print("\n" + "=" * 60)
    print(f"{'Symbol':<15}{'Timeframe':<10}{'Min Threshold':<45}{'Max Threshold':<45}{'Projected Price':<45}{'Market Mood':<15}{'Distance to Min (%)':<20}{'Distance to Max (%)':<20}")
    print("-" * 60)

    for symbol, results in overall_analysis.items():
        for timeframe, data in results.items():
            print(f"{symbol:<15}{timeframe:<10}{data['Min Threshold']:<45}{data['Max Threshold']:<45}{data['Projected Price']:<45}{data['Market Mood']:<15}{data['Distance to Min (%)']:<20}{data['Distance to Max (%)']:<20}")

    print("=" * 60)

if __name__ == "__main__":
    main()