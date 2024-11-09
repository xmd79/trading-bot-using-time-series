import requests
import numpy as np
import talib  
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

# Initialize Binance client
client = get_binance_client()

TRADE_SYMBOL = "BTCUSDC"
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

# Get candles data
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
        if not np.isnan(close) and close > 0:
            closes.append(close)
    current_price = get_price(TRADE_SYMBOL)
    if current_price > 0:
        closes.append(current_price)
    return np.array(closes)

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
    sma56 = np.nan

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

def calculate_45_degree_angle(prices):
    if len(prices) < 2:
        return np.nan
    current_price = prices[-1]
    price_range = np.max(prices) - np.min(prices)
    return current_price + (price_range / len(prices))

def scale_to_sine(thresholds, last_reversal, cycles=5):
    min_threshold, max_threshold, avg_threshold = thresholds
    num_points = 100
    t = np.linspace(0, 2 * np.pi * cycles, num_points)

    amplitude = (max_threshold - min_threshold) / 2

    if last_reversal == "dip":
        baseline = min_threshold + amplitude
        sine_wave = baseline - amplitude * np.sin(t)  # Inverted sine wave for dip
    else:  # "top"
        baseline = max_threshold - amplitude
        sine_wave = baseline + amplitude * np.sin(t)

    return sine_wave

def find_next_reversal_target(sine_wave, last_reversal):
    if last_reversal == "dip":
        return np.max(sine_wave)  # Expecting an upward movement
    else:
        return np.min(sine_wave)  # Expecting a downward movement

def check_open_close_reversals(candle):
    daily_open = candle['open']
    current_close = candle['close']

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
    closes = closes[~np.isnan(closes) & (closes > 0)]
    if len(closes) == 0:
        return 0, 0

    min_close = np.min(closes)
    max_close = np.max(closes)
    current_close = closes[-1]

    distance_to_min = ((current_close - min_close) / (max_close - min_close)) * 100 if (max_close - min_close) > 0 else 0
    distance_to_max = ((max_close - current_close) / (max_close - min_close)) * 100 if (max_close - min_close) > 0 else 0

    distance_to_min_normalized = distance_to_min / (distance_to_min + distance_to_max) * 100 if (distance_to_min + distance_to_max) > 0 else 0
    distance_to_max_normalized = distance_to_max / (distance_to_min + distance_to_max) * 100 if (distance_to_min + distance_to_max) > 0 else 0

    return distance_to_min_normalized, distance_to_max_normalized

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
        return 0

    intensity = (len(positive_indices) - len(negative_indices)) / total_indices
    return intensity * 100

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
    max_amplitudes = np.max(sinusoidal_envelope, axis=1)
    significant_frequency_index = np.argmax(max_amplitudes)

    return significant_frequency_index + 1  # Returning 1-based index for readability

def calculate_channel(swing_low, swing_high, num_periods):
    """
    Calculate support/resistance reversal channel using the Pythagorean theorem with a 45-degree middle threshold.

    Parameters:
    swing_low (float): The price of the swing low.
    swing_high (float): The price of the swing high.
    num_periods (int): The time periods (horizontal distance) to project the channel.

    Returns:
    dict: A dictionary containing lower_channel, middle_channel, and upper_channel.
    """
    # Calculate distance a (price change)
    a = swing_high - swing_low  # vertical distance
    b = num_periods  # horizontal distance
    
    # Calculate hypotenuse c
    c = np.sqrt(a**2 + b**2)

    # Define the middle channel based on the swing low
    middle_channel = swing_low + (a + b) / 2  # This ensures it forms a threshold at a 45-degree angle.

    # Calculate channel lines based on the 45-degree mid threshold
    lower_channel = middle_channel - c / 2  # Half of the hypotenuse below the middle
    upper_channel = middle_channel + c / 2  # Half of the hypotenuse above the middle

    return {
        'lower_channel': lower_channel,
        'middle_channel': middle_channel,
        'upper_channel': upper_channel
    }

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
    print("Price at 45-degree angle specific to timeframe:", angle_price)

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

    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - current_close))

    if closest_threshold == min_threshold:
        print("The last minimum value is closest to the current close.")
        print("The last minimum value is", closest_threshold)
    elif closest_threshold == max_threshold:
        print("The last maximum value is closest to the current close.")
        print("The last maximum value is", closest_threshold)
    else:
        print("No threshold value found.")

    thresholds = (min_threshold, max_threshold, avg_mtf)
    dip_signal, top_signal, sma56 = detect_reversal(candles)

    # Now calculate channel using min and max threshold
    channel = calculate_channel(min_threshold, max_threshold, num_periods=10)  # Example num_periods
    print(f"Lower Channel: {channel['lower_channel']:.2f}")
    print(f"Middle Channel: {channel['middle_channel']:.2f}")
    print(f"Upper Channel: {channel['upper_channel']:.2f}")

    last_reversal = "dip" if "Dip" in dip_signal else "top"
    sine_wave = scale_to_sine(thresholds, last_reversal)

    sinusoidal_envelope = generate_sinusoidal_envelope(thresholds)

    dist_to_min_normalized, dist_to_max_normalized = calculate_distance_percentages(close_prices)

    market_mood, last_max, last_min = analyze_market_mood(sine_wave, thresholds[0], thresholds[1], current_close)

    # Analyze the predominant frequency and current status
    negative_range_indices = []
    positive_range_indices = []

    for band in sinusoidal_envelope:
        negative_indices = np.where(band < 0)[0]
        positive_indices = np.where(band > 0)[0]

        negative_range_indices.extend(negative_indices)
        positive_range_indices.extend(positive_indices)

    intensity = calculate_fear_greed_intensity(positive_range_indices, negative_range_indices)
    intensity_label = map_intensity_to_degrees(intensity)

    most_significant_frequency = find_most_significant_frequency(sinusoidal_envelope)

    # Determine predominant frequency and adjust for cycle direction
    if last_reversal == "top":
        predominant_frequency_direction = "Positive"  # Recently decreased peak
    else:  # last_reversal == "dip"
        predominant_frequency_direction = "Negative"  # Recently increased trough

    print("The most significant predominant frequency is:", predominant_frequency_direction)

    print(f"Current cycle is: {'up cycle' if last_reversal == 'dip' else 'down cycle'}")

    # Adjust intensity calculation based on frequency signals
    adjusted_intensity = intensity if last_reversal == 'dip' else -intensity  

    # New: Incorporate distance of current close to min and max of sine wave
    min_sine = np.min(sine_wave)
    max_sine = np.max(sine_wave)
    
    distance_to_min_sine = abs(current_close - min_sine) / (max_sine - min_sine) * 100 if (max_sine - min_sine) > 0 else 0
    distance_to_max_sine = abs(current_close - max_sine) / (max_sine - min_sine) * 100 if (max_sine - min_sine) > 0 else 0
    
    adjusted_distance_intensity = (distance_to_min_sine - distance_to_max_sine + 100) / 2
    adjusted_intensity = adjusted_intensity * adjusted_distance_intensity / 100  # Normalize to influence

    adjusted_intensity = np.clip(adjusted_intensity, -100, 100)  # Ensures intensity is within reasonable limits

    print(f"Adjusted Intensity: {adjusted_intensity:.2f}%")

    # Determine market mood based on adjusted intensity
    if adjusted_intensity > 70:
        market_mood = "Extreme Greed"
    elif adjusted_intensity > 50:
        market_mood = "Greed"
    elif adjusted_intensity > 30:
        market_mood = "Fear"
    else:
        market_mood = "Extreme Fear"

    # Print overall market mood
    print(f"Current Market Mood: {market_mood}")

    forecast_price = forecasted_prices[-1]
    if market_mood in ["Extreme Greed", "Greed"]:
        print(f"Forecast price suggests potential downside towards {forecast_price:.2f}.")
    elif market_mood in ["Fear", "Extreme Fear"]:
        print(f"Forecast price suggests potential upside towards {forecast_price:.2f}.")
    else:
        print(f"Forecast price holds steady around {forecast_price:.2f}.")

    print(dip_signal)
    print(top_signal)
    print(f"SMA56: {sma56:.2f}" if not np.isnan(sma56) else "SMA56: Not enough data")
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
    print("\n" + "=" * 30 + "\n")

print("All timeframe calculations completed.")
