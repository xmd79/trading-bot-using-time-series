import requests
import numpy as np
import talib  # Ensure TA-Lib is installed
from binance.client import Client as BinanceClient
import matplotlib.pyplot as plt  # For visualization
from matplotlib.animation import FuncAnimation
from scipy.fft import fft
import pandas as pd  # For handling datetime
import time

def get_binance_client():
    """Instantiate Binance client using API credentials."""
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    return BinanceClient(api_key, api_secret)

# Initialize the Binance client
client = get_binance_client()

TRADE_SYMBOL = "BTCUSDT"  # Trading symbol
TIMEFRAME = '1m'  # Specify the timeframe to use for fetching data
UPDATE_INTERVAL = 5  # Update interval in seconds

def get_candles(symbol, timeframe, limit=1000):
    """Fetch candlestick data from Binance for the specified timeframe."""
    klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
    candles = []
    for k in klines:
        candle = {
            "time": k[0] / 1000,  # Convert to seconds
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        }
        candles.append(candle)
    return candles

def get_price(symbol):
    """Get the current price of the specified trading symbol."""
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("Price key not found in API response")

def calculate_thresholds(close_prices, minimum_percentage=3, maximum_percentage=3):
    """Calculate incoming min and max thresholds for the closing prices."""
    close_prices = np.array(close_prices)
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)

    min_threshold = min_close - (max_close - min_close) * (minimum_percentage / 100)
    max_threshold = max_close + (max_close - min_close) * (maximum_percentage / 100)
    middle_threshold = (min_threshold + max_threshold) / 2

    return min_threshold, max_threshold, middle_threshold

def forecast_sine_wave(min_threshold, max_threshold, forecast_period=15):
    """Forecast next prices using a sine wave fitting between min and max thresholds."""
    # Calculate the mean and amplitude for the sine wave
    mean = (min_threshold + max_threshold) / 2
    amplitude = (max_threshold - min_threshold) / 2

    # Generate forecasted prices using a sine wave
    forecast_prices = []
    for i in range(forecast_period):
        # Sine wave oscillation
        forecast_price = mean + amplitude * np.sin((2 * np.pi * i) / forecast_period)
        forecast_prices.append(forecast_price)
    return forecast_prices

# Setup for continuous plotting
fig, ax = plt.subplots()
xdata = []
ydata = []
forecast_data = []

def update(frame):
    global xdata, ydata, forecast_data
    # Fetch new data
    candles = get_candles(TRADE_SYMBOL, TIMEFRAME, limit=100)
    close_prices = [candle['close'] for candle in candles]
    
    # Update current price
    current_close = get_price(TRADE_SYMBOL)
    
    # Calculate thresholds and forecast
    min_threshold, max_threshold, middle_threshold = calculate_thresholds(close_prices)
    forecast_prices = forecast_sine_wave(min_threshold, max_threshold)  # Generate forecast using sine wave

    # Append data for plotting
    xdata.append(frame * UPDATE_INTERVAL)
    ydata.append(current_close)
    if forecast_prices:
        forecast_data = forecast_prices

    # Clear ax before plotting new data
    ax.clear()
    ax.plot(xdata, ydata, label='Current Price', color='b')
    forecast_x = np.arange(frame * UPDATE_INTERVAL + 1, frame * UPDATE_INTERVAL + 1 + len(forecast_data))
    ax.plot(forecast_x, forecast_data, label='Forecast Prices', color='orange', linestyle='--')

    # Indicators and thresholds
    ax.axhline(y=min_threshold, color='r', linestyle='--', label='Min Threshold')
    ax.axhline(y=max_threshold, color='g', linestyle='--', label='Max Threshold')
    ax.axhline(y=middle_threshold, color='y', linestyle='--', label='Middle Threshold')
    
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Real-time BTCUSDT Price and Forecast")
    ax.legend()

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 1000), interval=UPDATE_INTERVAL * 1000)  # Update every 5 seconds
plt.show()