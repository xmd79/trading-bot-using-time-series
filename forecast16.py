import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import talib
import requests
from binance.client import Client as BinanceClient
import datetime
from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft

# Function to create a harmonic sinusoidal stationary oscillator wave
def harmonic_wave(frequency, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = np.sin(2 * np.pi * frequency * t)
    return t, y

# Function to create a square wave
def square_wave(frequency, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = np.sign(np.sin(2 * np.pi * frequency * t))
    return t, y

# Function for Gann progression
def gann_progression(value):
    return [value * i for i in range(1, 8)]  # Gann levels up to 7 times

# Logistic Regression function
def logistic_regression(x, y):
    x = x.reshape(-1, 1)
    model = LogisticRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    accuracy = accuracy_score(y, predictions)
    return model, predictions, accuracy

# Function to filter wave using TA-Lib's HT_SINE
def filter_with_talib(y):
    ht_sine = talib.HT_SINE(y)  # returns a tuple of (sine, lead)
    sine_component = ht_sine[0]  # get the sine component
    lead_component = ht_sine[1]  # get the lead component
    sine_component = -sine_component  # negate the sine component
    return sine_component, lead_component  # return both components

# Utility functions for signal processing
def analyze_wave_properties(y, frequency, sample_rate):
    momentum_ratio = np.sum(y > 0) / y.size  # ratio of positive values
    energy_value = np.mean(np.square(y)) * sample_rate  # energy as mean amplitude squared
    oscillating_angle = np.arctan2(np.sin(y), np.cos(y))  # angle of oscillation
    rotational_symmetry = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0  # symmetry measure
    polynomial_fit = np.polyfit(np.arange(len(y)), y, 2)  # quadratic fit for general shape
    inner_harmonics = np.fft.fft(y)[:len(y)//2].real  # inner harmonics analysis

    return momentum_ratio, energy_value, oscillating_angle, rotational_symmetry, polynomial_fit, inner_harmonics

# Function to derive Gann cycles based on current state
def gann_cycles_and_fear_greed_index(y, frequency):
    max_val = np.max(y)
    min_val = np.min(y)
    current_value = y[-1]

    max_close_ratio = (current_value - min_val) / (max_val - min_val) * 100  # upward momentum
    min_close_ratio = (max_val - current_value) / (max_val - min_val) * 100  # downward momentum

    return max_close_ratio, min_close_ratio

# Function to get Binance client credentials
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

# Initialize Binance client
client = get_binance_client()

TRADE_SYMBOL = "ACTUSDC"
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

def find_next_reversal_target(channel, last_reversal):
    if last_reversal == "dip":
        return channel['lower_channel']  # Use lower channel for next dip
    else:
        return channel['upper_channel']  # Use upper channel for next top

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

def calculate_channel(swing_low, swing_high, num_periods):
    a = swing_high - swing_low  # vertical distance
    b = num_periods  # horizontal distance
    c = np.sqrt(a ** 2 + b ** 2)

    middle_channel = swing_low + (a + b) / 2  # threshold at a 45-degree angle
    lower_channel = middle_channel - c / 2  # half of the hypotenuse below the middle
    upper_channel = middle_channel + c / 2  # half of the hypotenuse above the middle

    return {
        'lower_channel': lower_channel,
        'middle_channel': middle_channel,
        'upper_channel': upper_channel
    }

def random_walk_with_pi_influence(close_prices, steps=100):
    pi_str = str(np.pi)[2:]  # get the decimal part of pi
    fictitious_prices = []

    for i in range(steps):
        if int(pi_str[i % len(pi_str)]) % 2 == 0:
            change = np.random.rand() * 0.01  # random increase
            new_price = close_prices[-1] * (1 + change)
        else:
            change = np.random.rand() * 0.01  # random decrease
            new_price = close_prices[-1] * (1 - change)

        fictitious_prices.append(new_price)
        close_prices = np.append(close_prices, new_price)

    return close_prices

def hurst_exponent(ts):
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[l:], ts[:-l])) for l in lags]
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    hurst_exponent = np.polyfit(log_lags, log_tau, 1)[0]
    return hurst_exponent

def merge_hurst_and_random_walk(candles):
    closes = np.array([candle['close'] for candle in candles])
    fictitious_prices = random_walk_with_pi_influence(closes, steps=100)

    hurst_value = hurst_exponent(fictitious_prices)
    forecasted_price = linear_regression_forecast(fictitious_prices, forecast_steps=1)[-1]

    return hurst_value, fictitious_prices[-1], forecasted_price

def analyze_fft(close_prices, n_frequencies=25):
    freq_data = fft(close_prices)
    
    magnitudes = np.abs(freq_data[:n_frequencies])

    num_magnitudes = len(magnitudes)

    if num_magnitudes < 13:
        neutral_freq = 0 if num_magnitudes < 13 else magnitudes[12]
        positive_freqs = magnitudes[13:num_magnitudes]
    else:
        neutral_freq = magnitudes[12]
        positive_freqs = magnitudes[13:25]

    negative_freqs = magnitudes[:min(12, num_magnitudes)]

    avg_negative = np.mean(negative_freqs) if len(negative_freqs) > 0 else 0
    avg_neutral = neutral_freq
    avg_positive = np.mean(positive_freqs) if len(positive_freqs) > 0 else 0

    if avg_positive > avg_negative and avg_positive > avg_neutral:
        significant_category = "Positive"
    elif avg_negative > avg_positive and avg_negative > avg_neutral:
        significant_category = "Negative"
    else:
        significant_category = "Neutral"

    return significant_category, avg_negative, avg_neutral, avg_positive

def forecast_price_from_fft(close_prices):
    freq_data = fft(close_prices)
    freq_data[12:] = 0  # Zero out the higher frequencies to reduce noise
    forecasted_close = np.real(np.fft.ifft(freq_data))

    return forecasted_close[-1]

# Modified function to analyze bullish volume with confirmed dips
def analyze_bullish_volume(candles, last_dip):
    total_volume = sum(candle['volume'] for candle in candles)
    bullish_volume = sum(candle['volume'] for candle in candles if candle['close'] > candle['open'])
    bearish_volume = total_volume - bullish_volume

    return bullish_volume, bearish_volume, total_volume

# Function to identify significant highs and lows
def identify_significant_highs_lows(candles):
    closing_prices = np.array([candle['close'] for candle in candles])
    highs = np.array([candle['high'] for candle in candles])
    lows = np.array([candle['low'] for candle in candles])
    
    significant_low = np.min(lows)
    significant_high = np.max(highs)
    recent_close = closing_prices[-1]

    return significant_low, significant_high, recent_close

# Main Logic for Timeframes
current_time = datetime.datetime.now()
print("Current local Time is now at:", current_time)
print()

# Initialize an empty list for MTF results
mtf_summary = []

for timeframe in timeframes:
    candles = candle_map[timeframe]
    close_prices = get_close(timeframe)

    # Identify significant lows and highs
    significant_low, significant_high, recent_close = identify_significant_highs_lows(candles)

    # Determine if there was a recent dip or peak
    dip_found = significant_low < recent_close and (recent_close - significant_low) > 0.02 * recent_close  # Example logic for dip
    peak_found = significant_high > recent_close and (significant_high - recent_close) > 0.02 * recent_close  # Example logic for peak
    
    min_threshold, max_threshold, avg_mtf = calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3)

    # Calculate the channel
    channel = calculate_channel(min_threshold, max_threshold, num_periods=10)

    angle_price = calculate_45_degree_angle(close_prices)

    print(f"=== Timeframe: {timeframe} ===")
    print(f"Minimum threshold: {min_threshold:.25f}")
    print(f"Maximum threshold: {max_threshold:.25f}")
    print(f"Average MTF: {avg_mtf:.25f}")
    print(f"Price at 45-degree angle specific to timeframe: {angle_price:.25f}")

    # Analyze volume
    bullish_volume, bearish_volume, total_volume = analyze_bullish_volume(candles, dip_found)
    
    print(f"Total Volume: {total_volume:.25f}, Bullish Volume: {bullish_volume:.25f}, Bearish Volume: {bearish_volume:.25f}")

    sentiment = "Bullish" if bullish_volume > bearish_volume else "Bearish"
    print(f"Current sentiment: {sentiment}")

    # Merge Hurst cycles and random walk logic
    hurst_value, last_price, forecasted_price = merge_hurst_and_random_walk(candles)

    print(f"Hurst Exponent: {hurst_value:.25f}")
    print(f"Forecasted Price based on Random Walk: {forecasted_price:.25f}")

    significant_category, avg_negative, avg_neutral, avg_positive = analyze_fft(close_prices)

    fft_forecasted_price = forecast_price_from_fft(close_prices)

    print(f"Currently mostly significant frequencies are: {significant_category}")
    print(f"Average Negative Frequencies Magnitude: {avg_negative:.25f}")
    print(f"Neutral Frequency Magnitude: {avg_neutral:.25f}")
    print(f"Average Positive Frequencies Magnitude: {avg_positive:.25f}")
    print(f"Forecasted price from FFT into the future: {fft_forecasted_price:.25f}")

    forecasted_prices = linear_regression_forecast(close_prices, forecast_steps=1)

    print(f"Forecasted price for next period (Linear Regression): {forecasted_prices[-1]:.25f}")

    current_close = close_prices[-1]
    if angle_price is not None:
        if current_close < angle_price:
            difference_percentage = ((angle_price - current_close) / angle_price) * 100
            print(f"Current close is below the 45-degree angle price by {difference_percentage:.25f}%")
        elif current_close > angle_price:
            difference_percentage = ((current_close - angle_price) / angle_price) * 100
            print(f"Current close is above the 45-degree angle price by {difference_percentage:.25f}%")
        else:
            print("Current close is exactly at the 45-degree angle price.")

    closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - current_close))

    if closest_threshold == min_threshold:
        print(f"The last minimum value is closest to the current close: {closest_threshold:.25f}")
    elif closest_threshold == max_threshold:
        print(f"The last maximum value is closest to the current close: {closest_threshold:.25f}")
    else:
        print("No threshold value found.")

    thresholds = (min_threshold, max_threshold, avg_mtf)
    dip_signal, top_signal, sma56 = detect_reversal(candles)

    print(f"Lower Channel: {channel['lower_channel']:.25f}")
    print(f"Middle Channel: {channel['middle_channel']:.25f}")
    print(f"Upper Channel: {channel['upper_channel']:.25f}")

    last_reversal = "dip" if dip_found else "top" if peak_found else None
    sine_wave = scale_to_sine(thresholds, last_reversal)

    # Calculate distance percentages
    dist_to_min_normalized, dist_to_max_normalized = calculate_distance_percentages(close_prices)

    current_closes = np.array(close_prices)
    min_sine = np.min(sine_wave)
    max_sine = np.max(sine_wave)

    distance_to_min_sine = ((current_close - min_sine) / (max_sine - min_sine)) * 100 if (max_sine - min_sine) > 0 else 0
    distance_to_max_sine = ((max_sine - current_close) / (max_sine - min_sine)) * 100 if (max_sine - min_sine) > 0 else 0

    print(f"Distance to min of sine wave: {distance_to_min_sine:.25f}%")
    print(f"Distance to max of sine wave: {distance_to_max_sine:.25f}%")

    market_mood, last_max, last_min = analyze_market_mood(sine_wave, min_threshold, max_threshold, current_close)

    next_reversal_target = find_next_reversal_target(channel, last_reversal)

    last_candle = candles[-1]
    reversal_signal = check_open_close_reversals(last_candle)

    print(f"Next target reversal price expected: {next_reversal_target:.25f}")
    print(reversal_signal)

    mtf_summary.append({
        "timeframe": timeframe,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "avg_mtf": avg_mtf,
        "angle_price": angle_price,
        "total_volume": total_volume,
        "bullish_volume": bullish_volume,
        "bearish_volume": bearish_volume,
        "sentiment": sentiment,
        "hurst_value": hurst_value,
        "forecasted_price": forecasted_price,
        "fft_forecasted_price": fft_forecasted_price,
        "linear_regression_forecast": forecasted_prices[-1],
        "market_mood": market_mood,
    })

    print("\n" + "=" * 30 + "\n")

# MTF Summary
print("=== MTF Summary ===")
for summary in mtf_summary:
    print(f"Timeframe: {summary['timeframe']}, Min Threshold: {summary['min_threshold']:.25f}, Max Threshold: {summary['max_threshold']:.25f}, "
          f"Avg MTF: {summary['avg_mtf']:.25f}, Sentiment: {summary['sentiment']}, Hurst: {summary['hurst_value']:.25f}, "
          f"Forecasted Price: {summary['forecasted_price']:.25f}, FFT Forecast: {summary['fft_forecasted_price']:.25f}, "
          f"Linear Regression Forecast: {summary['linear_regression_forecast']:.25f}, Market Mood: {summary['market_mood']}")

print("All timeframe calculations completed.")