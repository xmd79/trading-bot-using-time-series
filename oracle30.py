import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tzlocal
from binance.client import Client as BinanceClient
import talib

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Configuration
TICKER = "BTCUSDT"  # Binance symbol
LOCAL_TZ = tzlocal.get_localzone()
INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]  # Different timeframes
LOOKBACK = {
    "1m": "7 days",
    "5m": "14 days",
    "15m": "30 days",
    "30m": "60 days",
    "1h": "90 days",
    "4h": "180 days",
    "1d": "1 year",
    "1w": "3 years",
    "1M": "5 years"
}

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

def get_historical_data(symbol, interval='1h', lookback='30 days'):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                        'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    df['Open Time'] = df['Open Time'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ)
    df.set_index('Open Time', inplace=True)
    return df[['Close']]

def find_last_major_reversal(prices):
    max_price = prices.max()
    min_price = prices.min()
    last_max_date = prices[prices == max_price].index[-1]
    last_min_date = prices[prices == min_price].index[-1]
    return (max_price, last_max_date, 'high'), (min_price, last_min_date, 'low')

def sine_wave_analysis(length, amplitude):
    x = np.linspace(0, 2 * np.pi, length)
    sine_wave = amplitude * np.sin(x)
    return x, sine_wave

def plot_sine_waves(df, interval, last_reversal_time):
    # Calculate the sine wave projection for forecasting
    length = len(df)
    avg_price = df['Close'].mean()
    amplitude = (df['Close'].max() - df['Close'].min()) / 2
    x, sine_wave = sine_wave_analysis(length, amplitude)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Prices', color='gray')
    sine_wave_projection = df['Close'] + sine_wave[:len(df)]
    plt.plot(df.index, sine_wave_projection, label='Sine Wave Projection', color='purple')

    # Mark the last major reversal
    last_high = find_last_major_reversal(df['Close'])[0]
    last_max_price, last_max_time, _ = last_high
    plt.scatter(last_max_time, last_max_price, color='red', label='Last Major High')

    last_low = find_last_major_reversal(df['Close'])[1]
    last_min_price, last_min_time, _ = last_low
    plt.scatter(last_min_time, last_min_price, color='blue', label='Last Major Low')

    # Draw the 45-degree angle threshold line
    mid_price = (last_max_price + last_min_price) / 2
    threshold_line = mid_price + (x * (last_max_price - last_min_price) / (2 * length))  # 45-degree line
    plt.plot(df.index, threshold_line, color='orange', linestyle='--', label='45-degree Threshold')

    # Calculate and plot support and resistance levels
    support_level = last_min_price - (last_max_price - last_min_price) * 0.05
    resistance_level = last_max_price + (last_max_price - last_min_price) * 0.05
    plt.axhline(support_level, color='green', linestyle=':', label='Support Level')
    plt.axhline(resistance_level, color='magenta', linestyle=':', label='Resistance Level')

    # Price Forecast
    forecasted_price = np.sin(length * np.pi / 2) * amplitude + avg_price  # Forecast based on sine
    print(f"Forecasted Price: {forecasted_price:.2f}")

    # Determine if it suggests a dip or a top based on the prediction
    if forecasted_price > current_close_price:
        trend_prediction = "Forecast suggests an upcoming top (upward trend)."
    else:
        trend_prediction = "Forecast suggests an upcoming dip (downward trend)."

    print(trend_prediction)

    # Normalize distances
    total_distance = (last_max_price - last_min_price)  # Total price range

    if total_distance == 0:  # Avoid division by zero
        percentage_distance_high = 0
        percentage_distance_low = 0
    else:
        distance_to_high = (last_max_price - forecasted_price) / total_distance
        distance_to_low = (forecasted_price - last_min_price) / total_distance
        
        # Normalize such that both distances sum to 100%
        percentage_distance_high = max(0, min(1, distance_to_high)) * 100
        percentage_distance_low = max(0, min(1, distance_to_low)) * 100
        
        # Adjust percentage distances to ensure they sum to 100%
        total_percentage = percentage_distance_high + percentage_distance_low
        if total_percentage > 100:
            percentage_distance_high = (percentage_distance_high / total_percentage) * 100
            percentage_distance_low = (percentage_distance_low / total_percentage) * 100

    print(f"Symmetrical Distance to Last Major High: {percentage_distance_high:.2f}%")
    print(f"Symmetrical Distance to Last Major Low: {percentage_distance_low:.2f}%")

    # Calculate distance to the 45-degree threshold
    distance_to_mid_threshold = (forecasted_price - mid_price) / total_distance
    print(f"Distance to 45-degree Threshold: {abs(distance_to_mid_threshold) * 100:.2f}%")

    # Calculate Frequency Spectrum Index
    frequency_value = 2 * ((forecasted_price - last_min_price) / (last_max_price - last_min_price) - 0.5) * 4  # Transform to range [-4, 4]
    print(f"Frequency Spectrum Index Value: {frequency_value:.2f}")

    # Calculate symmetrical distances to the sine wave's min and max
    sine_wave_max = avg_price + amplitude
    sine_wave_min = avg_price - amplitude

    distance_to_sine_max = abs(current_close_price - sine_wave_max)
    distance_to_sine_min = abs(current_close_price - sine_wave_min)
    
    max_distance_normalized = (sine_wave_max - sine_wave_min)  # Sine wave range for normalization

    symmetrical_distance_to_sine_max = distance_to_sine_max / max_distance_normalized * 100
    symmetrical_distance_to_sine_min = distance_to_sine_min / max_distance_normalized * 100

    print(f"Symmetrical Distance from Current Close to Sine Wave Max: {symmetrical_distance_to_sine_max:.2f}%")
    print(f"Symmetrical Distance from Current Close to Sine Wave Min: {symmetrical_distance_to_sine_min:.2f}%")

    plt.title(f'Sine Wave and Last Major Reversal for {interval}', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_technical_indicators(df):
    # Example of technical indicators using TA-Lib
    df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    return df

# Main execution flow
if __name__ == "__main__":
    # Get current local time
    current_local_time = datetime.now(LOCAL_TZ)
    print("Current Local Date and Time:", current_local_time.strftime('%Y-%m-%d %H:%M:%S'))

    for interval in INTERVALS:
        print(f"\nFetching data for {interval} interval...")
        
        # Get historical price data with appropriate lookback
        lookback_period = LOOKBACK[interval]
        historic_prices = get_historical_data(TICKER, interval, lookback=lookback_period)

        # Check if we have sufficient data
        if len(historic_prices) < 24:
            print(f"Not enough data for {interval}. Expected at least 24 data points.")
            continue

        # Calculate technical indicators
        historic_prices = calculate_technical_indicators(historic_prices)

        # Print the latest values of the indicators
        print(f"Latest SMA20: {historic_prices['SMA20'].iloc[-1]:.2f}")
        print(f"Latest SMA50: {historic_prices['SMA50'].iloc[-1]:.2f}")
        print(f"Latest RSI: {historic_prices['RSI'].iloc[-1]:.2f}")
        
        # Find and print last major reversals
        last_reversal_high, last_reversal_low = find_last_major_reversal(historic_prices['Close'])
        
        # Last major high and low details
        last_reversal_price_high, last_reversal_time_high, reversal_type_high = last_reversal_high
        last_reversal_price_low, last_reversal_time_low, reversal_type_low = last_reversal_low
        
        # Current closing price
        current_close_price = historic_prices['Close'].iloc[-1]

        # Calculate distances to current close
        distance_to_high = abs(current_close_price - last_reversal_price_high)
        distance_to_low = abs(current_close_price - last_reversal_price_low)

        # Determine which is closer and print the result
        closer_reversal = 'high' if distance_to_high < distance_to_low else 'low'
        
        print(f"Last Major High: {reversal_type_high.capitalize()} at {last_reversal_price_high} on {last_reversal_time_high.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Last Major Low: {reversal_type_low.capitalize()} at {last_reversal_price_low} on {last_reversal_time_low.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Closing Price: {current_close_price:.2f}")
        print(f"Distance to Last Major High: {distance_to_high:.2f}, Distance to Last Major Low: {distance_to_low:.2f}")
        print(f"The closer reversal to the current closing price is the {closer_reversal}.")

        # Plot sine waves and reversals
        plot_sine_waves(historic_prices, interval, last_reversal_time_high)

    print("Analysis complete.")