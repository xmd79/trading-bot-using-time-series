import requests
import numpy as np
import talib
from binance.client import Client as BinanceClient
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Binance client using credentials
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

client = get_binance_client()
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Function to fetch tradable pairs for USDC
def fetch_usdc_pairs():
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']
    usdc_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
    return usdc_pairs

# Function to fetch candles for the given symbol and timeframes
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

# Function to calculate min, avg, and max thresholds from close prices
def calculate_thresholds(close_prices):
    close_prices = np.array(close_prices)
    close_prices = close_prices[~np.isnan(close_prices) & (close_prices > 0)]
    if len(close_prices) == 0:
        raise ValueError("No valid close prices available for threshold calculation.")

    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    min_threshold = max(0, min_close)
    max_threshold = max_close  
    average_threshold = (min_threshold + max_threshold) / 2

    return min_threshold, average_threshold, max_threshold

# Function to detect major reversals based on closing prices
def detect_major_reversal(candles):
    closes = np.array([candle['close'] for candle in candles])
    closes = closes[~np.isnan(closes) & (closes > 0)]

    if len(closes) < 1:
        return [], None, None  # Return empty signals and None values

    recent_price = closes[-1]
    sma_length = 56  
    sma56 = talib.SMA(closes, timeperiod=sma_length)[-1] if len(closes) >= sma_length else None

    unique_signals = []
    if sma56 is not None:
        if recent_price < sma56:
            unique_signals.append('Major Dip detected')
        elif recent_price > sma56:
            unique_signals.append('Major Top detected')

    return unique_signals, recent_price, sma56

# Function to calculate the 45-degree angle price based on average threshold
def calculate_45_degree_price(average_threshold):
    angle_price = average_threshold + (average_threshold * 1.618)
    return angle_price

# Function to analyze volume trends within the given candles
def analyze_volume(candles):
    bullish_volume = 0
    bearish_volume = 0

    for candle in candles:
        # Classify the volume based on the price action
        if candle['close'] > candle['open']:
            bullish_volume += candle['volume']
        elif candle['close'] < candle['open']:
            bearish_volume += candle['volume']

    total_volume = bullish_volume + bearish_volume
    if total_volume > 0:
        bullish_percentage = (bullish_volume / total_volume) * 100
        bearish_percentage = (bearish_volume / total_volume) * 100
    else:
        bullish_percentage = 0
        bearish_percentage = 0

    volume_dominance = "Bullish" if bullish_volume >= bearish_volume else "Bearish"

    return bullish_percentage, bearish_percentage, volume_dominance

# Function to find volume support and resistance levels
def volume_support_resistance(candles):
    volume_levels = [candle['volume'] for candle in candles]
    max_volume = max(volume_levels, default=0)
    support_level = np.percentile(volume_levels, 25) if volume_levels else 0
    resistance_level = np.percentile(volume_levels, 75) if volume_levels else 0
    
    return support_level, resistance_level, max_volume

# Function to analyze an asset based on its candles
def analyze_asset(symbol):
    candles = get_candles(symbol, timeframes)

    candle_map = {timeframe: [] for timeframe in timeframes}
    for candle in candles:
        candle_map[candle['timeframe']].append(candle)

    analysis_results = {}
    
    for timeframe in timeframes:
        close_prices = np.array([candle['close'] for candle in candle_map[timeframe]])
        # Extract volume data from each timeframe's candle
        volumes = np.array([candle['volume'] for candle in candle_map[timeframe]])
        if len(close_prices) < 1:
            continue

        min_threshold, average_threshold, max_threshold = calculate_thresholds(close_prices)
        unique_signals, recent_price, sma56 = detect_major_reversal(candle_map[timeframe])  
        angle_price = calculate_45_degree_price(average_threshold)

        is_above_angle = "Above 45 Degree" if recent_price > angle_price else "Below 45 Degree"
        forecast = "N/A"
        if 'Major Dip detected' in unique_signals:
            forecast = f"Upward to {max_threshold:.25f}" if recent_price < angle_price else "Stable"
        elif 'Major Top detected' in unique_signals:
            forecast = f"Downward to {min_threshold:.25f}" if recent_price > angle_price else "Stable"

        # Analyze volume with the new logic
        bullish_ratio, bearish_ratio, volume_status = analyze_volume(candle_map[timeframe])
        support_level, resistance_level, max_volume = volume_support_resistance(candle_map[timeframe])

        # Prepare analysis summary for the timeframe
        analysis_results[timeframe] = {
            "Min Threshold": f"{min_threshold:.25f}",
            "Average Threshold": f"{average_threshold:.25f}",
            "Max Threshold": f"{max_threshold:.25f}",
            "45 Degree Angle Price": f"{angle_price:.25f}",
            "Forecast": forecast,
            "SMA": f"{sma56:.25f}" if sma56 is not None else "N/A",
            "Above/Below Angle": is_above_angle,
            "Volume Current Status": f"Bullish: {bullish_ratio:.2f}%, Bearish: {bearish_ratio:.2f}%",
            "Support Level": f"{support_level:.25f}",
            "Resistance Level": f"{resistance_level:.25f}",
            "Max Volume": f"{max_volume:.25f}",
            "Dominant Volume": volume_status,
            "Price Relation to Average Threshold": "Above Average" if recent_price > average_threshold else "Below Average"
        }

    return symbol, analysis_results

# Main function to execute the trading bot
def main():
    current_time = datetime.datetime.now()
    print("Current local Time is now at: ", current_time)

    usdc_pairs = fetch_usdc_pairs()
    overall_analysis = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_asset, symbol): symbol for symbol in usdc_pairs}
        for future in as_completed(future_to_symbol):
            try:
                symbol, results = future.result()
                overall_analysis[symbol] = results
            except Exception as e:
                print(f"Error analyzing symbol: {e}")

    # Print the results neatly
    print("\n" + "=" * 80)
    print(f"{'Symbol':<15}{'Timeframe':<10}{'Min Threshold':<45}{'Avg Threshold':<45}{'Max Threshold':<45}{'45 Degree Angle Price':<45}{'Forecast':<45}{'Volume Status':<50}{'Above/Below Angle':<20}{'Price Relation to Average Threshold':<30}")
    print("-" * 180)

    for symbol, results in overall_analysis.items():
        for timeframe, data in results.items():
            # Print each row continuously
            print(f"{symbol:<15}{timeframe:<10}{data['Min Threshold']:<45}{data['Average Threshold']:<45}{data['Max Threshold']:<45}{data['45 Degree Angle Price']:<45}{data['Forecast']:<45}{data['Volume Current Status']:<50}{data['Above/Below Angle']:<20}{data['Price Relation to Average Threshold']:<30}")

    print("=" * 150)

if __name__ == "__main__":
    main()