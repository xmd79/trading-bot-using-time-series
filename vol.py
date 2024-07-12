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

def get_klines(client, symbol, interval):
    klines = client.get_klines(symbol=symbol, interval=interval)
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

def main(file):
    trader = Trader(file)
    client = trader.client
    active_trading_pairs = get_active_trading_pairs(client)

    volume_stats = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_klines, client, symbol, '1d'): symbol for symbol in active_trading_pairs}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                buy_volume, sell_volume, total_volume = calculate_volume_stats(df)
                volume_stats.append((symbol, buy_volume, sell_volume, total_volume))
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    volume_stats.sort(key=lambda x: x[3], reverse=True)  # Sort by total volume descending

    for symbol, buy_volume, sell_volume, total_volume in volume_stats:
        trend = 'Bullish' if buy_volume > sell_volume else 'Bearish'
        print(f"Symbol: {symbol}, Total Volume: {total_volume}, Buy Volume: {buy_volume}, Sell Volume: {sell_volume}, Trend: {trend}")

if __name__ == "__main__":
    api_keys_file = "credentials.txt"  # Replace with the path to your API keys file
    main(api_keys_file)
