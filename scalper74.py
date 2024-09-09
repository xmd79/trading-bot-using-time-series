import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
import time

# Load Binance credentials
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Initialize Binance Client
client = Client(api_key, api_secret)

symbol = "BTCUSDC"  # The trading pair
timeframes = ["1h", "15m", "3m", "1m"]

class BinanceWrapper:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
    
    def get_exchange_rate(self, symbol):
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def get_binance_balance(self):
        return self.client.get_account()['balances']

def get_candles(symbol, timeframe, limit=1000):
    klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
    return [{"time": k[0] / 1000, "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])} for k in klines]

def calculate_ema(prices, period):
    return pd.Series(prices).ewm(span=period, adjust=False).mean().to_numpy()

def evaluate_ema_pattern(candles):
    closes = np.array([c["close"] for c in candles])
    short_ema = calculate_ema(closes, 12)
    long_ema = calculate_ema(closes, 26)
    return closes[-1], short_ema[-1], long_ema[-1]

def find_last_reversals(candles):
    closes = np.array([c["close"] for c in candles])
    highest_high = np.max(closes)
    lowest_low = np.min(closes)
    return highest_high, lowest_low

def calculate_oscillation(current_close, highest_high, lowest_low):
    distance_to_low = abs(current_close - lowest_low)
    distance_to_high = abs(current_close - highest_high)
    
    total_range = highest_high - lowest_low

    # Symmetrical percentages
    from_low_percentage = distance_to_low / total_range * 100
    from_high_percentage = distance_to_high / total_range * 100

    return from_low_percentage, from_high_percentage

def generate_cycle_report(candle_data):
    plt.figure(figsize=(12, 6))

    for timeframe, candles in candle_data.items():
        current_close, _, _ = evaluate_ema_pattern(candles)
        
        highest_high, lowest_low = find_last_reversals(candles)
        from_low, from_high = calculate_oscillation(current_close, highest_high, lowest_low)

        # Determine cycle direction
        cycle_direction = "Up" if current_close > (lowest_low + (highest_high - lowest_low) / 2) else "Down"

        # Create a sine wave based on the number of candles
        cycle_length = len(candles)
        sine_wave = np.sin(np.linspace(0, 2 * np.pi, cycle_length))  # Full sine wave oscillation
        oscillation_mid = (highest_high + lowest_low) / 2
        
        # Adjust sine wave between the last reversals
        adjusted_wave = oscillation_mid + (highest_high - lowest_low) / 2 * sine_wave
        
        # Price for middle threshold
        middle_threshold_price = oscillation_mid
        
        # Print additional information
        print(f"\nTimeframe: {timeframe}")
        print(f"Current Close: {current_close:.2f}")
        print(f"Last High: {highest_high:.2f}, Last Low: {lowest_low:.2f}")
        print(f"Distance from Low: {from_low:.2f}%, Distance from High: {from_high:.2f}%")
        print(f"Cycle Direction: {cycle_direction}")
        print(f"Middle Threshold Price: {middle_threshold_price:.2f} (Current Close is {'above' if current_close > middle_threshold_price else 'below'} the threshold)\n")

        x_values = np.arange(len(candles))
        plt.plot(x_values, adjusted_wave, label=f'Sine Wave ({timeframe})')

        # Plotting the middle threshold as a horizontal line
        plt.axhline(y=middle_threshold_price, color='r', linestyle='--', label='Middle Threshold (45°)')
    
    plt.title("Cycle Oscillation Sine Wave for Different Timeframes")
    plt.xlabel("Candle Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    
    # Pause for 5 seconds
    plt.pause(5)  
    
    # Close the plot after the pause
    plt.close()  # This will kill the plot after showing it

# Main loop for continuous analysis
if __name__ == "__main__":
    while True:
        candle_map = {}
        for timeframe in timeframes:
            candle_map[timeframe] = get_candles(symbol, timeframe)

        generate_cycle_report(candle_map)

        # Delay for the next iteration
        time.sleep(5)  # 5 seconds waiting time before fetching new data