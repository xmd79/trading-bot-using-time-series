import requests
import numpy as np
from binance.client import Client as BinanceClient
import matplotlib.pyplot as plt

def get_binance_client():
    """
    Instantiate Binance client using API credentials.
    Returns a Binance client object.
    """
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    return BinanceClient(api_key, api_secret)

# Initialize the Binance client
client = get_binance_client()
TRADE_SYMBOL = "BTCUSDT"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']  # Define the timeframes

def get_candles(symbol, timeframe):
    """
    Fetch candlestick data from Binance API.
    Args:
        symbol (str): Trading symbol.
        timeframe (str): Timeframe for the candlesticks.
    Returns:
        list: List of candle data.
    """
    limit = 1000
    klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
    candles = []
    for k in klines:
        candle = {
            "time": k[0] / 1000,  # Convert to seconds
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
        candles.append(candle)
    return candles

def get_close_prices(candles):
    """
    Extract closing prices from candle data.
    Args:
        candles (list): List of candle data.
    Returns:
        list: List of closing prices.
    """
    return [candle['close'] for candle in candles]

def calculate_harmonics(close_prices):
    """
    Calculate harmonic frequencies based on closing prices.
    Args:
        close_prices (list): List of closing prices.
    Returns:
        float: Fundamental frequency.
    """
    if len(close_prices) == 0:
        return 0  # Prevent division by zero
    mean_price = np.mean(close_prices)
    std_dev = np.std(close_prices)
    freq_n1 = 1 / (4 * np.pi * std_dev / (mean_price - 0.5))  # Calculate fundamental frequency
    return freq_n1  # Return the fundamental frequency

def plot_mtf_analysis():
    """
    Main function to plot the compounded symmetrical harmonic wave.
    Continuously fetches data and updates the plot.
    """
    plt.figure(figsize=(14, 10))  # Set the figure size
    plt.ion()  # Enable interactive mode

    while True:
        combined_wave = None  # Initialize combined wave variable
        
        for timeframe in timeframes:
            candles = get_candles(TRADE_SYMBOL, timeframe)
            close_prices = get_close_prices(candles)

            # Calculate harmonics based on the close prices
            freq_n1 = calculate_harmonics(close_prices)
            num_points = len(close_prices)

            if num_points == 0:
                continue  # Skip if there are no closing prices
            
            # Create harmonic wave
            x_vals = np.linspace(0, num_points - 1, num_points)
            harmonic_wave = np.sin(2 * np.pi * freq_n1 * x_vals)

            # Combine this harmonic wave into the overall combined wave
            if combined_wave is None:
                combined_wave = harmonic_wave
            else:
                combined_wave += harmonic_wave  # Sum to combine waves

        # Normalize the combined wave
        if combined_wave is not None:
            combined_wave /= len(timeframes)  # Average across timeframes
            combined_wave *= (1 / np.max(np.abs(combined_wave)))  # Normalize to range [-1, 1]

            # Prepare for plotting
            x_values_combined_wave = np.linspace(0, len(combined_wave) - 1, len(combined_wave))

            # Clear previous plot for combined wave
            plt.clf()  
            plt.plot(x_values_combined_wave, combined_wave, color='orange', label='Symmetrical Compounded Harmonic Wave', alpha=0.7)
            plt.title('Symmetrical Compounded Harmonic Wave Across Timeframes')
            plt.xlabel('Time Index')
            plt.ylabel('Normalized Wave Value')
            plt.axhline(0, color='black', lw=0.5, linestyle='--')  # Add horizontal axis for reference
            plt.legend()
            plt.grid()
            plt.pause(0.001)  # Instant reloading animation

if __name__ == "__main__":
    plot_mtf_analysis()