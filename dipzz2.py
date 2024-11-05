import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from binance.client import Client
import sys

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols'] 
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        return trading_pairs

# Function to filter timeframes for dips
def filter_timeframe(trader, pair, timeframe):
    klines = trader.client.get_klines(symbol=pair, interval=timeframe)
    close = [float(entry[4]) for entry in klines]

    if len(close) < 1:
        return None  # Skip if there are no close prices available

    # Filter out nan and zero values
    close = [x for x in close if pd.notna(x) and x > 0]
    
    if len(close) == 0:
        return None  # No valid data after filtering

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)  # Linear best fit line
    best_fit_line2 = best_fit_line1 * 1.01  # Upper boundary
    best_fit_line3 = best_fit_line1 * 0.99  # Lower boundary

    # Check if the latest close is below the lower boundary line
    if close[-1] < best_fit_line3[-1]:
        return pair  # Dips found; return the pair
    return None

def scan_pairs_for_dips(trader, pairs, timeframes):
    results = {tf: [] for tf in timeframes}

    # Scan only on the daily timeframe first to determine candidates for lower timeframe scans
    daily_results = []
    for pair in pairs:
        result = filter_timeframe(trader, pair, '1d')
        if result:
            daily_results.append(result)

    # Store daily results
    results['1d'] = daily_results

    # Scan lower timeframes for pairs identified in the daily scan
    for pair in daily_results:
        for tf in timeframes[1:]:  # Skip the daily timeframe for lower scans
            result = filter_timeframe(trader, pair, tf)
            if result:
                results[tf].append(result)

    return results

# Initialize trading bot with credentials file
filename = 'credentials.txt'
trader = Trader(filename)

# Fetch USDC trading pairs
trading_pairs = trader.get_usdc_pairs()

# Define timeframes we want to analyze
timeframes = ['1d', '4h', '1h', '15m', '5m']

# Start scanning for dips
dip_results = scan_pairs_for_dips(trader, trading_pairs, timeframes)

# Create a list to store dip information
dip_info_list = []

# Populate the list with found dips for the daily timeframe only
for pair in trading_pairs:
    dip_info = {
        'Pair': pair,
        '1d': pair in dip_results.get('1d', []),
    }
    dip_info_list.append(dip_info)

# Create the summary DataFrame from the list
summary_df = pd.DataFrame(dip_info_list)

# Filter the summary to include only daily dips
daily_dip_summary_df = summary_df[summary_df['1d'] == True]

# Set Pandas display options to show all rows
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent breaking the line for long DataFrames

# Print the complete summary of daily dips found
print("Daily Dips Summary Table:\n", daily_dip_summary_df)

# Determine the pairs to fetch 1m data only from daily dips
final_pairs = daily_dip_summary_df['Pair'].tolist()

# Initialize the dictionary for lowest momentum values
lowest_momentum_values = {}

# Process 1-minute data for selected pairs (if any)
if final_pairs:
    for pair in final_pairs:
        klines = trader.client.get_klines(symbol=pair, interval='1m')
        close = [float(entry[4]) for entry in klines]

        # Filter out nan and zero values
        close = [x for x in close if pd.notna(x) and x > 0]

        if len(close) < 14:  # Ensure enough data for momentum calculation
            print(f"Not enough valid data for momentum calculation on {pair}")
            continue 

        # Calculate momentum - using the default 14-period momentum
        momentum = pd.Series(close).diff(14)

        # Drop nan values
        valid_momentum = momentum.dropna()

        if len(valid_momentum) > 0:
            # Find the most negative momentum value and corresponding index
            most_negative_momentum = valid_momentum.min()
            index_of_most_negative = valid_momentum.idxmin()  # Find the index of the most negative value

            lowest_momentum_values[pair] = (most_negative_momentum, close[index_of_most_negative])
            print(f"Momentum for {pair}: Most Negative Value: {most_negative_momentum:.10f}, Price: {close[index_of_most_negative]:.10f}")
        else:
            # If no valid momentum values, set a default or indicate
            lowest_momentum_values[pair] = (0, close[-1])  # Set default momentum to 0, use the last price if available
            print(f"No valid momentum calculated for {pair}, using default.")

        # Apply FFT to the closing prices for the sinusoidal forecast
        N = len(close)
        if N < 4:
            print(f"Not enough data for FFT on {pair}")
            continue

        # Perform FFT
        yf = fft(close)

        # Create a sinusoidal forecast based on the frequencies
        forecast_length = 60  # Forecast for the next 60 minutes
        future_close = np.zeros(N + forecast_length)

        # Using the existing frequency representation to reconstruct values
        future_close[:N] = close  # Fill existing prices

        # Extend the frequencies for forecasting
        yf_extended = np.zeros(N + forecast_length, dtype=complex)
        yf_extended[:N] = yf

        # Perform the Inverse FFT to obtain the forecasted values
        forecast = ifft(yf_extended).real  # Taking only the real part

        # Get the most significant value (highest forecasted price)
        most_significant_value = max(forecast[-forecast_length:])

        # Print the most significant forecasted price as a long float
        print(f"Most significant forecasted price for {pair} for the next {forecast_length} minutes: {most_significant_value:.10f}")

else:
    print("No daily pairs available for further analysis.")

# Final result output for lowest momentum values
if lowest_momentum_values:
    # Sort the momentum values from lowest to highest (i.e., most negative first)
    sorted_momentum = sorted(lowest_momentum_values.items(), key=lambda item: item[1][0])

    print("Pairs with the lowest momentum values and their prices:")
    for pair, (momentum, price) in sorted_momentum:
        print(f"Pair: {pair}, Momentum: {momentum:.10f}, Price: {price:.10f}")
else:
    print("No pairs processed for momentum calculations.")

# Exit the script
sys.exit(0)