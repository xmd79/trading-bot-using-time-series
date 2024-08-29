import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ephem
from datetime import datetime, timedelta
from binance.client import Client as BinanceClient
import tzlocal  # Automatically get the local timezone

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Configuration
TICKER = "BTCUSDT"  # Symbol for Binance
INTERVAL = "1h"  # 1-hour interval
LOOKBACK_PERIOD = "7 days"  # Fetch last 7 days of data
LOCAL_TZ = tzlocal.get_localzone()  # Automatically get the local timezone

# Print the current local time
current_local_time = datetime.now(LOCAL_TZ)
print("Current local time:", current_local_time.strftime("%Y-%m-%d %H:%M:%S"))

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

def get_historical_data(symbol, interval='1h', lookback='7 days'):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                        'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    
    # Localizing the Open Time to the local timezone
    df['Open Time'] = df['Open Time'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ)
    
    df.set_index('Open Time', inplace=True)
    return df[['Close']]

def get_planet_positions(observer):
    planets = {
        'Sun': ephem.Sun(observer),
        'Moon': ephem.Moon(observer),
        'Mercury': ephem.Mercury(observer),
        'Venus': ephem.Venus(observer),
        'Mars': ephem.Mars(observer),
        'Jupiter': ephem.Jupiter(observer),
        'Saturn': ephem.Saturn(observer),
        'Uranus': ephem.Uranus(observer),
        'Pluto': ephem.Pluto(observer),
    }

    positions = {}
    for planet_name, planet in planets.items():
        planet.compute(observer)
        positions[planet_name] = planet.alt, planet.az
    return positions

def find_major_reversals(df):
    """ Find the last major peak or dip before the current close. """
    local_max = df['Close'].rolling(window=3, min_periods=1).max()
    local_min = df['Close'].rolling(window=3, min_periods=1).min()
    peaks = df[local_max == df['Close']].index
    dips = df[local_min == df['Close']].index
  
    last_close_time = df.index[-1]
  
    last_reversal_index = max(peaks[peaks < last_close_time].tolist() or [None] + \
                               dips[dips < last_close_time].tolist() or [None])
  
    if last_reversal_index is None:
        return None, None, None

    last_reversal_value = df['Close'].loc[last_reversal_index]
    
    is_peak = last_reversal_index in peaks.tolist()

    return last_reversal_index, last_reversal_value, is_peak

def forecast_prices(df, hours=24):
    forecasts = []
    last_price = df['Close'].iloc[-1]
    for i in range(hours):
        next_hour = (df.index[-1] + timedelta(hours=i + 1)).astimezone(LOCAL_TZ)  # Convert to local timezone
        # Simulating price change
        forecast_price = last_price * (1 + np.random.uniform(-0.01, 0.01))  
        forecasts.append((next_hour, forecast_price))
    return forecasts

def sine_wave_analysis(length):
    x = np.linspace(0, 2 * np.pi, length)  # Creating x values for the forecast length
    sine_wave = np.sin(x)  # Calculate the sine wave
    return sine_wave

def plot_results(df, forecasts, sine_wave, astrological_data, last_reversal):
    plt.figure(figsize=(14, 10))

    # Plot historical data
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Historical Prices', color='blue')
    plt.axvline(x=last_reversal[0], color='red', linestyle='--', label='Last Reversal')
    plt.title('Historical Prices with Last Major Reversal')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    forecast_times, forecast_prices = zip(*forecasts)
    
    # Sine Wave
    # Find sine max and min for normalization
    sine_max = last_reversal[1] * 1.1  # Example max
    sine_min = last_reversal[1] * 0.9  # Example min

    # Adjust sine wave for the last reversal
    sine_wave_fit = sine_wave * (last_reversal[1] * 0.1) + last_reversal[1]

    # Calculate distances
    distance_to_max = abs(forecast_prices[-1] - sine_max)
    distance_to_min = abs(forecast_prices[-1] - sine_min)

    # Calculate percentages
    total_distance = distance_to_max + distance_to_min
    if total_distance > 0:
        percent_to_max = (distance_to_max / total_distance) * 100
        percent_to_min = (distance_to_min / total_distance) * 100
    else:
        percent_to_max = 0.0
        percent_to_min = 0.0

    # Print distances and percentages
    print(f"Distance to Sine Max: {distance_to_max:.2f} -> {percent_to_max:.2f}%")
    print(f"Distance to Sine Min: {distance_to_min:.2f} -> {percent_to_min:.2f}%")

    # Key Points
    key_points = {
        "Last Reversal": last_reversal[1],
        "Sine Max": sine_max,
        "Sine Min": sine_min,
        "Latest Forecast": forecast_prices[-1]
    }

    # Add projections for key points
    for key, point in key_points.items():
        plt.scatter(df.index[-1], point, label=f'{key}: {point:.2f}', s=100)

    plt.subplot(3, 1, 2)
    plt.plot(forecast_times, forecast_prices, label='Forecasted Prices', color='orange')
    plt.plot(forecast_times, sine_wave_fit, label='Sine Wave Fit', color='purple')
    plt.title('Forecasted Prices with Sine Wave Projection')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(forecast_times, sine_wave_fit, color='purple', label='Sine Wave Projection')
    plt.title('Sine Wave Analysis')
    plt.ylabel('Sine Value')
    plt.xlabel('Time')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Main execution flow
if __name__ == "__main__":
    # Get current observer time
    observer = ephem.Observer()
    observer.date = current_local_time  # Use local time

    # Get 7 days of historical data
    historical_data = get_historical_data(TICKER, INTERVAL, LOOKBACK_PERIOD)

    # Get planetary positions
    positions = get_planet_positions(observer)
    astrological_data = {planet: (alt, az) for planet, (alt, az) in positions.items()}

    # Find the last major reversal
    last_reversal = find_major_reversals(historical_data)

    # Perform price forecasting
    forecasted_prices = forecast_prices(historical_data, 24)

    # Analyze sine wave based on the length of the forecasted prices
    forecast_length = len(forecasted_prices)
    sine_wave = sine_wave_analysis(forecast_length)

    # Plot results
    plot_results(historical_data, forecasted_prices, sine_wave, astrological_data, last_reversal)