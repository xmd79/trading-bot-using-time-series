import requests
import numpy as np
import talib  # Make sure to have TA-Lib installed
from binance.client import Client as BinanceClient
import datetime
from sklearn.linear_model import LinearRegression

# Define Binance client by reading API key and secret from local file
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

client = get_binance_client()

TRADE_SYMBOL = "TONUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

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

# Get candle data for each timeframe and create a map
candles = get_candles(TRADE_SYMBOL, timeframes)
candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]
    candle_map.setdefault(timeframe, []).append(candle)

def get_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("price key not found in API response")

def get_close(timeframe):
    closes = []
    candles = candle_map[timeframe]
    for c in candles:
        close = c['close']
        if not np.isnan(close):
            closes.append(close)
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)
    return np.array(closes)

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    close_prices = np.array(close_prices)
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
    sma7 = talib.SMA(closes, timeperiod=7)
    sma12 = talib.SMA(closes, timeperiod=12)
    sma26 = talib.SMA(closes, timeperiod=26)
    sma45 = talib.SMA(closes, timeperiod=45)
    sma56 = talib.SMA(closes, timeperiod=56)
    sma100 = talib.SMA(closes, timeperiod=100)
    sma200 = talib.SMA(closes, timeperiod=200)
    sma369 = talib.SMA(closes, timeperiod=369)

    local_dip_signal = "No Local close vs SMA Dip"
    local_top_signal = "No Local close vs SMA Top"

    if closes[-1] < sma7[-1] and sma7[-1] < sma12[-1] and sma12[-1] < sma26[-1] and sma26[-1] < sma45[-1]:
        local_dip_signal = f"Local Dip detected at price {closes[-1]}"

    if closes[-1] > sma7[-1] and sma7[-1] > sma12[-1] and sma12[-1] > sma26[-1] and sma26[-1] > sma45[-1]:
        local_top_signal = f"Local Top detected at price {closes[-1]}"

    return local_dip_signal, local_top_signal, sma56[-1], sma100[-1], sma200[-1], sma369[-1]

def calculate_45_degree_angle(prices):
    """ Calculate the price expected at a 45-degree angle specific to the timeframe. """
    if len(prices) < 2:
        return np.nan
    current_price = prices[-1]
    price_range = np.max(prices) - np.min(prices)  # Calculate the range of prices in this timeframe
    return current_price + (price_range / len(prices))  # Calculate the 45-degree increment

def scale_to_sine(thresholds, last_reversal, cycles=5):
    min_threshold, max_threshold, avg_threshold = thresholds
    num_points = 100
    t = np.linspace(0, 2 * np.pi * cycles, num_points)

    amplitude = (max_threshold - min_threshold) / 2

    if last_reversal == "dip":
        baseline = min_threshold + amplitude
        sine_wave = baseline - amplitude * np.sin(t)  # Inverted sine wave for dip
    else:
        baseline = max_threshold - amplitude
        sine_wave = baseline + amplitude * np.sin(t)  # Normal sine wave for peak

    return sine_wave

def find_next_reversal_target(sine_wave, last_reversal):
    if last_reversal == "dip":
        return np.min(sine_wave)
    else:
        return np.max(sine_wave)

def check_open_close_reversals(candle):
    daily_open = candle['open']
    current_close = candle['close']

    # Get support and resistance levels safely
    timeframe = candle['timeframe']
    lows = [c['low'] for c in candle_map[timeframe]]
    highs = [c['high'] for c in candle_map[timeframe]]

    if len(lows) == 0 or len(highs) == 0:
        return "Insufficient data for support and resistance levels."

    support_level = np.min(lows)
    resistance_level = np.max(highs)

    if current_close < daily_open and current_close > support_level:
        return "Below daily open"
    elif current_close > daily_open and current_close < resistance_level:
        return "Above daily open"
    
    return "No reversal signal detected."

def analyze_market_mood(sine_wave, min_threshold, max_threshold, current_close):
    last_max = max_threshold
    last_min = min_threshold

    if abs(current_close - last_max) < abs(current_close - last_min):
        market_mood = "Bearish"
    else:
        market_mood = "Bullish"

    return market_mood, last_max, last_min

def calculate_distance_percentages(closes):
    min_close = np.min(closes)
    max_close = np.max(closes)
    current_close = closes[-1]

    distance_to_min = ((current_close - min_close) / (max_close - min_close)) * 100 if (max_close - min_close) > 0 else 0
    distance_to_max = ((max_close - current_close) / (max_close - min_close)) * 100 if (max_close - min_close) > 0 else 0

    distance_to_min_normalized = distance_to_min / (distance_to_min + distance_to_max) * 100 if (distance_to_min + distance_to_max) > 0 else 0
    distance_to_max_normalized = distance_to_max / (distance_to_min + distance_to_max) * 100 if (distance_to_min + distance_to_max) > 0 else 0

    return distance_to_min_normalized, distance_to_max_normalized

def linear_regression_forecast(close_prices, forecast_steps=1):
    """
    Forecast future prices using Linear Regression.
    """
    X = np.arange(len(close_prices)).reshape(-1, 1)  # Create an array of time
    y = close_prices  # Close prices as target variable

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(close_prices), len(close_prices) + forecast_steps).reshape(-1, 1)
    forecast_prices = model.predict(future_X)

    return forecast_prices

def generate_sinusoidal_envelope(thresholds, frequency_bands=25):
    min_threshold, max_threshold, avg_mtf = thresholds
    envelope_points = 100

    t = np.linspace(0, 2 * np.pi, envelope_points)
    amplitude = (max_threshold - min_threshold) / 2
    frequency = np.linspace(1, frequency_bands, frequency_bands)

    sinusoidal_envelope = np.zeros((frequency_bands, envelope_points))
    for i in range(frequency_bands):
        sinusoidal_envelope[i] = min_threshold + amplitude * np.sin(frequency[i] * t)

    return sinusoidal_envelope

def calculate_fear_greed_intensity(positive_indices, negative_indices):
    total_indices = len(positive_indices) + len(negative_indices)
    if total_indices == 0:
        return 0  # Avoid division by zero

    intensity = (len(positive_indices) - len(negative_indices)) / total_indices
    return intensity * 100  # Scale to percentage

def map_intensity_to_degrees(intensity):
    if intensity > 70:
        return "Extreme Greed"
    elif intensity > 50:
        return "Greed"
    elif intensity > 30:
        return "Fear"
    else:
        return "Extreme Fear"

def find_most_significant_frequency(sinusoidal_envelope):
    # Find the frequency with the maximum amplitude in the sinusoidal envelope
    # The envelope is shaped (frequency_bands, envelope_points)
    max_amplitudes = np.max(sinusoidal_envelope, axis=1)
    significant_frequency_index = np.argmax(max_amplitudes)

    return significant_frequency_index + 1  # Returning 1-based index for readability

# Main Logic for Timeframes
current_time = datetime.datetime.now()
print("Current local Time is now at: ", current_time)
print()

for timeframe in timeframes:
    candles = candle_map[timeframe]
    close_prices = get_close(timeframe)

    # Calculate thresholds with custom percentage settings
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close_prices, period=14, minimum_percentage=2, maximum_percentage=2)
    
    # Calculate the price at a 45-degree angle specific to each timeframe
    angle_price = calculate_45_degree_angle(close_prices)

    print(f"=== Timeframe: {timeframe} ===")
    print("Minimum threshold:", min_threshold)
    print("Maximum threshold:", max_threshold)
    print("Average MTF:", avg_mtf)
    print("Price at 45-degree angle:", angle_price)

    # Linear Regression Forecast
    forecasted_prices = linear_regression_forecast(close_prices, forecast_steps=1)
    
    # Print forecasted price for the next period
    print(f"Forecasted price for next period: {forecasted_prices[-1]:.2f}")

    # Check the distance of current close to angle price
    current_close = close_prices[-1]
    if angle_price is not None:
        if current_close < angle_price:
            difference_percentage = ((angle_price - current_close) / angle_price) * 100
            print(f"Current close is below the 45-degree angle price by {difference_percentage:.2f}%")
        elif current_close > angle_price:
            difference_percentage = ((current_close - angle_price) / angle_price) * 100
            print(f"Current close is above the 45-degree angle price by {difference_percentage:.2f}%")
        else:
            print("Current close is exactly at the 45-degree angle price.")

    # Check which threshold is closer to the last close price
    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - current_close))
    
    if closest_threshold == min_threshold:
        print("The last minimum value is closest to the current close.")
        print("The last minimum value is", closest_threshold)
    elif closest_threshold == max_threshold:
        print("The last maximum value is closest to the current close.")
        print("The last maximum value is", closest_threshold)
    else:
        print("No threshold value found.")

    # Calculate remaining indicators
    thresholds = (min_threshold, max_threshold, avg_mtf)
    dip_signal, top_signal, sma56, sma100, sma200, sma369 = detect_reversal(candles)

    last_reversal = "dip" if "Dip" in dip_signal else "top"
    sine_wave = scale_to_sine(thresholds, last_reversal)

    # Generate sinusoidal envelope (for internal calculations only)
    sinusoidal_envelope = generate_sinusoidal_envelope(thresholds)

    dist_to_min_normalized, dist_to_max_normalized = calculate_distance_percentages(close_prices)

    market_mood, last_max, last_min = analyze_market_mood(sine_wave, thresholds[0], thresholds[1], current_close)

    # Analyze predominant frequency and current status
    negative_range_indices = []
    positive_range_indices = []

    # Assuming sinusoidal_envelope is (frequency_bands, envelope_points)
    for band in sinusoidal_envelope:
        negative_indices = np.where(band < 0)[0]  # Indices in this band that are negative
        positive_indices = np.where(band > 0)[0]  # Indices in this band that are positive

        negative_range_indices.extend(negative_indices)  # Collect all negative indices
        positive_range_indices.extend(positive_indices)  # Collect all positive indices

    intensity = calculate_fear_greed_intensity(positive_range_indices, negative_range_indices)
    intensity_label = map_intensity_to_degrees(intensity)

    most_significant_frequency = find_most_significant_frequency(sinusoidal_envelope)

    # Determine predominant frequency and cycle direction
    if most_significant_frequency > 0:
        predominant_message = "Positive"
    else:
        predominant_message = "Negative"

    print(f"The most significant predominant frequency is: {predominant_message}")
    print(f"Current cycle is: {'down cycle' if predominant_message == 'Positive' else 'up cycle'}")

    if len(negative_range_indices) > 0 and current_close < angle_price:
        major_reversal = "Negative Predominance"
        print(f"Major reversal found: {major_reversal} below current close.")
    elif len(positive_range_indices) > 0 and current_close > angle_price:
        major_reversal = "Positive Predominance"
        print(f"Major reversal found: {major_reversal} above current close.")
    else:
        major_reversal = "Neutral"
        print("No major reversal detected.")

    # Print intensity and advice
    print(f"Current Intensity: {intensity:.2f}% - Market Mood: {intensity_label}")
    
    forecast_price = forecasted_prices[-1]
    if intensity_label in ["Extreme Greed", "Greed"]:
        print(f"Forecast price suggests potential downside towards {forecast_price:.2f}.")
    elif intensity_label in ["Fear", "Extreme Fear"]:
        print(f"Forecast price suggests potential upside towards {forecast_price:.2f}.")
    else:
        print(f"Forecast price holds steady around {forecast_price:.2f}.")

    # Print additional signals
    print(dip_signal)
    print(top_signal)
    print(f"SMA56: {sma56:.2f}")
    print(f"SMA100: {sma100:.2f}")
    print(f"SMA200: {sma200:.2f}")
    print(f"SMA369: {sma369:.2f}")
    print(f"Distance to Min (Normalized): {dist_to_min_normalized:.2f}%")
    print(f"Distance to Max (Normalized): {dist_to_max_normalized:.2f}%")

    next_reversal_target = find_next_reversal_target(sine_wave, last_reversal)

    if closest_threshold == thresholds[1]:
        print(f"Starting a down cycle from last maximum price: {thresholds[1]:.2f}, expected next dip: {next_reversal_target:.2f}")
    else:
        print(f"Starting an up cycle from last minimum price: {thresholds[0]:.2f}, expected next top: {next_reversal_target:.2f}")

    last_candle = candles[-1]
    reversal_signal = check_open_close_reversals(last_candle)

    print(f"Next target reversal price expected: {next_reversal_target:.2f}")
    print(reversal_signal)
    print(f"Current Market Mood: {market_mood}")
    print("\n" + "=" * 30 + "\n")

print("All timeframe calculations completed.")
