import requests
import numpy as np
import talib
from binance.client import Client as BinanceClient
import datetime
import matplotlib.pyplot as plt

# Define Binance client
def get_binance_client():
    with open("credentials.txt", "r") as f:   
        lines = f.readlines()
        api_key = lines[0].strip()  
        api_secret = lines[1].strip()  
    client = BinanceClient(api_key, api_secret)
    return client

client = get_binance_client()

TRADE_SYMBOL = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
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

candles = get_candles(TRADE_SYMBOL, timeframes)
candle_map = {}
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

def get_close(timeframe):
    closes = []
    candles = candle_map[timeframe]
    
    for c in candles:
        close = c['close']
        if not np.isnan(close):
            closes.append(close)
    
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)
    
    return closes

def get_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    if "price" in data:
        return float(data["price"])
    else:
        raise KeyError("price key not found in API response")

# Define the Elliott Wave Analysis function:
def identify_elliott_waves(close_prices):
    prices = np.array(close_prices)

    # Placeholder for identified wave patterns
    waves = []

    # Simple logic for illustrative purposes
    # This is where you'd implement the real analysis logic for identifying wave structure
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:  # Local Min (Potential Wave Bottom)
            waves.append({'type': 'trough', 'index': i, 'price': prices[i]})
        elif prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:  # Local Max (Potential Wave Top)
            waves.append({'type': 'peak', 'index': i, 'price': prices[i]})

    return waves

# This function can determine symmetry within the identified waves
def assess_wave_symmetry(wave_list):
    symmetry_results = []
    for i in range(len(wave_list) - 2):
        # Look for symmetry patterns (e.g., distance between waves)
        if (wave_list[i]['type'] == 'trough' and 
            wave_list[i + 2]['type'] == 'trough'):
            distance1 = wave_list[i + 1]['index'] - wave_list[i]['index']
            distance2 = wave_list[i + 1]['index'] - wave_list[i + 2]['index']

            symmetry_score = abs(distance1 - distance2)
            symmetry_results.append({
                'Wave Pair': (wave_list[i]['price'], wave_list[i + 2]['price']),
                'Symmetry Score': symmetry_score
            })

    return symmetry_results

# Calculate and assess TSI, then identify waves
for timeframe in timeframes:
    close_prices = get_close(timeframe)
    waves = identify_elliott_waves(close_prices)
    symmetry_results = assess_wave_symmetry(waves)

    print(f"Timeframe: {timeframe}")
    for wave in waves:
        print(f"Identified {wave['type']} at index {wave['index']} with price {wave['price']:.2f}")
    
    print("Symmetry Results:")
    for result in symmetry_results:
        print(f"Wave Pair: {result['Wave Pair']}, Symmetry Score: {result['Symmetry Score']}")

    print("\n" + "=" * 30 + "\n")

# Optional: Plotting function to visualize the identified waves
def plot_waves(close_prices, waves):
    plt.figure(figsize=(10, 5))
    plt.plot(close_prices, label='Close Prices')
    for wave in waves:
        color = 'green' if wave['type'] == 'trough' else 'red'
        plt.scatter(wave['index'], wave['price'], color=color, label=f"{wave['type'].capitalize()} at {wave['price']:.2f}")

    plt.title('Elliott Wave Identification')
    plt.xlabel('Time')
    plt.ylabel('Price Level')
    plt.legend()
    plt.grid()
    plt.show()

# Uncomment to plot waves for a specific timeframe
# plot_waves(close_prices, waves)