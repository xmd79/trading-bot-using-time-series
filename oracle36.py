import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Importing pyplot for plotting
import ephem
from datetime import datetime, timedelta
import tzlocal
from binance.client import Client as BinanceClient
import talib  # Ensure you have TA-Lib installed for technical analysis functions

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Configuration
TICKER = "BTCUSDT"  # Binance symbol
LOCAL_TZ = tzlocal.get_localzone()

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
        'Neptune': ephem.Neptune(observer),
        'Pluto': ephem.Pluto(observer),
    }
    aspects = []
    for planet_name, planet in planets.items():
        planet.compute(observer)
        aspects.append((planet_name, planet.alt, planet.az))
    return aspects

def forecast_prices_and_astrology(current_time, prices, hours=24):
    forecasts = []
    astrological_data = []
    for i in range(hours):
        next_hour = current_time + timedelta(hours=i)
        forecast_price = prices.iloc[i]  # Use actual prices from historical data
        forecasts.append((next_hour, forecast_price))
        # Fetching astrological positions for current hour
        observer = ephem.Observer()
        observer.date = next_hour
        aspects = get_planet_positions(observer)
        astrological_data.append((next_hour, aspects))
    return forecasts, astrological_data

def sine_wave_analysis(length, amplitude):
    x = np.linspace(0, 2 * np.pi, length)
    sine_wave = amplitude * np.sin(x)  # Calculate the sine wave with given amplitude
    return x, sine_wave

def calculate_gann_progression(minima, maxima):
    # Example implementation of Gann levels based on sine wave analysis or historical data
    gann_levels = {}
    for level in range(1, 6):  # Example levels: 1-5
        price_level = minima + (maxima - minima) * (level / 6)
        gann_levels[f'Level {level}'] = price_level
    return gann_levels

# Function to remove NaNs and zeros from arrays
def remove_nans_and_zeros(*arrays):
    arrays = [np.array(array) for array in arrays]
    valid_mask = ~np.isnan(np.column_stack(arrays)).any(axis=1) & (np.column_stack(arrays) != 0).all(axis=1)
    return [array[valid_mask] for array in arrays]

# Function to scale close prices to a sine range
def scale_to_sine(close_prices):
    close_prices, = remove_nans_and_zeros(close_prices)
    sine_wave, _ = talib.HT_SINE(close_prices)
    sine_wave = np.nan_to_num(sine_wave)
    if len(sine_wave) == 0:
        print("Sine wave could not be computed – no valid data.")
        return
    current_sine = sine_wave[-1]
    sine_wave_min = np.nanmin(sine_wave)
    sine_wave_max = np.nanmax(sine_wave)
    dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0)
    dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min) * 100 if (sine_wave_max - sine_wave_min) != 0 else 0)
    print(f"Current Sine Value: {current_sine:.2f}, Distance to Min: {dist_min:.2f}%, Distance to Max: {dist_max:.2f}%")

def calculate_moving_averages(prices, short_window=5, long_window=10):
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices['Close']
    signals['short_mavg'] = prices['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = prices['Close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0
    signals.iloc[short_window:, signals.columns.get_loc('signal')] = np.where(
        signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1, 0
    )
    signals['positions'] = signals['signal'].diff()
    return signals

# Main execution flow
if __name__ == "__main__":
    # Get current local time
    current_local_time = datetime.now(LOCAL_TZ)
    print("Current Local Date and Time:", current_local_time.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Get historical price data
    historic_prices = get_historical_data(TICKER)
    
    # Ensure we have data for at least 24 hours
    if len(historic_prices) < 24:
        raise Exception("Not enough data to forecast for the next day.")

    # Current time observer
    observer = ephem.Observer()
    observer.date = current_local_time

    # Hourly forecast for today with astrological data
    forecasts_today, astrological_data_today = forecast_prices_and_astrology(current_local_time, historic_prices['Close'], 24)

    # Print hourly forecasts
    print("\nHourly Forecasts for Today:")
    for time, price in forecasts_today:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {price:.2f}")

    # Forecast for the next week
    forecast_start_time = current_local_time + timedelta(days=1)
    forecasts_next_week, astrological_data_next_week = forecast_prices_and_astrology(forecast_start_time, historic_prices['Close'], 168)

    # Print weekly forecasts
    print("\nWeekly Forecasts:")
    for time, price in forecasts_next_week:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {price:.2f}")

    # Sine wave analysis
    x, sine_wave = sine_wave_analysis(24, amplitude=200)  # Example amplitude, modify as needed!

    # Calculate and scale to sine for the specified timeframe
    scale_to_sine(historic_prices['Close'].values)  # Pass the close prices directly

    # Calculate Gann progression levels based on last found minima and maxima
    minima = min(forecast[1] for forecast in forecasts_today)
    maxima = max(forecast[1] for forecast in forecasts_today)
    gann_levels = calculate_gann_progression(minima, maxima)

    # New feature: calculate moving averages
    moving_average_signals = calculate_moving_averages(historic_prices)

    # Print moving average signals
    print("\nMoving Average Signals:")
    print(moving_average_signals.tail(10))  # Print last 10 signals for review

    # Detailed plotting
    plt.figure(figsize=(14, 12))

    # Today's forecast plot
    plt.subplot(3, 1, 1)
    forecast_times_today, forecast_prices_today = zip(*forecasts_today)
    plt.plot(forecast_times_today, forecast_prices_today, label='Hourly Forecast Prices', color='orange')
    plt.scatter(forecast_times_today, np.array(forecast_prices_today) + sine_wave[:24], label='Sine Wave Projection', color='purple')
    plt.title('Price Forecast for Today with Sine Wave Projection', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # Add Gann Levels to the plot
    for label, level in gann_levels.items():
        plt.axhline(y=level, linestyle='--', label=label)

    # Reduce the number of astrological annotations to avoid clutter
    for time, aspects in astrological_data_today[::3]:  # Display every third aspect
        plt.annotate(f"{time.strftime('%H:%M')}: {aspects}",
                     xy=(time, forecast_prices_today[forecast_times_today.index(time)] + 50), fontsize=7, rotation=45)
    plt.legend(fontsize=8)
    plt.grid()

    # Sine wave plot with reversal points
    plt.subplot(3, 1, 2)
    plt.plot(x, sine_wave, label='Sine Wave', color='blue')
    # Calculate and mark key reversal points (max, min, and threshold)
    sine_wave_max_index = np.argmax(sine_wave)
    sine_wave_min_index = np.argmin(sine_wave)
    plt.scatter(x[sine_wave_max_index], sine_wave.max(), color='red', label='Max Point', zorder=5)
    plt.scatter(x[sine_wave_min_index], sine_wave.min(), color='green', label='Min Point', zorder=5)
    # Add threshold level (middle 45-degree)
    plt.axhline(y=0, color='orange', linestyle='--', label='45° Threshold')
    plt.title('Sine Wave with Reversal Points', fontsize=14)
    plt.xlabel('X (Angle Radians)', fontsize=12)
    plt.ylabel('Sine Value', fontsize=12)
    plt.legend()
    plt.grid()

    # Next week forecast plot
    plt.subplot(3, 1, 3)
    forecast_times_week, forecast_prices_week = zip(*forecasts_next_week)
    plt.plot(forecast_times_week, forecast_prices_week, label='Weekly Forecast Prices', color='green')
    plt.title('Price Forecast for Next Week', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid()

    # Adjust layout manually to create padding
    plt.subplots_adjust(hspace=0.5, top=0.93, left=0.1, right=0.9, bottom=0.1)
    plt.show()