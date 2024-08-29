import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ephem
from datetime import datetime, timedelta
from binance.client import Client as BinanceClient
import pytz
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
    return sine_wave, sine_wave.max(), sine_wave.min(), (sine_wave.max() + sine_wave.min()) / 2

def plot_results(df, forecasts, sine_wave, astrological_data):
    plt.figure(figsize=(14, 7))

    # Plot historical data
    plt.plot(df.index, df['Close'], label='Historical Prices', color='blue')

    # Add forecast data
    forecast_times, forecast_prices = zip(*forecasts)
    plt.plot(forecast_times, forecast_prices, label='Forecasted Prices', color='orange')

    # Sine wave visualization
    plt.plot(forecast_times, sine_wave * 10 + np.mean(forecast_prices), label='Sine Wave', color='purple')

    # Print astrological positions and forecast data
    print("\nAstrological Positions and Forecast Data:\n")
    for time, price in forecasts:
        print(f"At {time}: Forecasted Price = {price:.2f}, Astrological Data = {astrological_data}")

    plt.title('Price Forecast with Sine Wave and Astrological Overlay')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
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

    # Perform price forecasting
    forecasted_prices = forecast_prices(historical_data, 24)

    # Analyze sine wave based on the length of the forecasted prices
    forecast_length = len(forecasted_prices)
    sine_wave, max_val, min_val, mid = sine_wave_analysis(forecast_length)

    # Plot results
    plot_results(historical_data, forecasted_prices, sine_wave, astrological_data)