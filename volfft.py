import numpy as np
import sys
from datetime import datetime
from binance.client import Client
import pandas as pd
import concurrent.futures

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        try:
            with open(file, 'r') as f:
                key = f.readline().strip()
                secret = f.readline().strip()
            self.client = Client(key, secret)
            print("Connected to Binance API successfully.")
        except Exception as e:
            print("Error connecting to Binance API:", e)
            sys.exit(1)

def get_active_trading_pairs(client):
    tickers = client.get_all_tickers()
    exchange_info = client.get_exchange_info()
    symbols_info = exchange_info['symbols']
    active_trading_pairs = [symbol['symbol'] for symbol in symbols_info if symbol['status'] == 'TRADING' and symbol['symbol'].endswith("USDC")]
    return active_trading_pairs

def get_klines(client, symbol, interval, limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = {
        'Date': [datetime.fromtimestamp(entry[0] / 1000.0) for entry in klines],
        'Open': [float(entry[1]) for entry in klines],
        'High': [float(entry[2]) for entry in klines],
        'Low': [float(entry[3]) for entry in klines],
        'Close': [float(entry[4]) for entry in klines],
        'Volume': [float(entry[5]) for entry in klines],
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def calculate_volume_stats(df):
    buy_volume = df[df['Close'] > df['Open']]['Volume'].sum()
    sell_volume = df[df['Close'] < df['Open']]['Volume'].sum()
    total_volume = df['Volume'].sum()
    return buy_volume, sell_volume, total_volume

def perform_fft(close_prices, num_frequencies):
    n = len(close_prices)
    close_fft = np.fft.fft(close_prices)
    
    # Function to filter frequencies
    def filter_frequencies(freq_band):
        filtered_fft = np.copy(close_fft)
        filtered_fft[freq_band[0]:freq_band[1]] = 0
        filtered_fft[-freq_band[1]:-freq_band[0]] = 0
        return np.fft.ifft(filtered_fft).real
    
    # Define frequency bands for filtering
    small_band = (num_frequencies, n - num_frequencies)
    medium_band = (num_frequencies * 2, n - num_frequencies * 2)
    large_band = (num_frequencies * 3, n - num_frequencies * 3)
    
    # Perform filtering
    small_wave_forecast = filter_frequencies(small_band)
    medium_wave_forecast = filter_frequencies(medium_band)
    large_wave_forecast = filter_frequencies(large_band)
    
    return small_wave_forecast, medium_wave_forecast, large_wave_forecast

def main(file):
    trader = Trader(file)
    client = trader.client
    active_trading_pairs = get_active_trading_pairs(client)

    volume_stats = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_klines, client, symbol, '1m'): symbol for symbol in active_trading_pairs}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                buy_volume, sell_volume, total_volume = calculate_volume_stats(df)
                volume_stats.append((symbol, df, buy_volume, sell_volume, total_volume))
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    volume_stats.sort(key=lambda x: x[4], reverse=True)  # Sort by total volume descending

    top_volume_stats = volume_stats[:10]  # Get top 10 by volume

    for symbol, df, buy_volume, sell_volume, total_volume in top_volume_stats:
        close_prices = df['Close'].values
        small_wave_forecast, medium_wave_forecast, large_wave_forecast = perform_fft(close_prices, 20)

        df['Small_Wave_Forecast'] = small_wave_forecast
        df['Medium_Wave_Forecast'] = medium_wave_forecast
        df['Large_Wave_Forecast'] = large_wave_forecast
        trend = 'Bullish' if buy_volume > sell_volume else 'Bearish'
        print(f"\nSymbol: {symbol}, Total Volume: {total_volume}, Buy Volume: {buy_volume}, Sell Volume: {sell_volume}, Trend: {trend}")

        print("\nSmall Wave Forecast:")
        print(df[['Close', 'Small_Wave_Forecast']].tail(10))  # Print the last 10 values for comparison
        
        print("\nMedium Wave Forecast:")
        print(df[['Close', 'Medium_Wave_Forecast']].tail(10))  # Print the last 10 values for comparison
        
        print("\nLarge Wave Forecast:")
        print(df[['Close', 'Large_Wave_Forecast']].tail(10))  # Print the last 10 values for comparison

if __name__ == "__main__":
    api_keys_file = "credentials.txt"  # Replace with the path to your API keys file
    main(api_keys_file)
