import numpy as np
import sys
from datetime import datetime
from binance.client import Client
import pandas as pd
import concurrent.futures
from scipy.signal import hilbert
import itertools

class Trader:
    def __init__(self, file):
        self.connect(file)
        self.triangles = []

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

    def find_arbitrage_triangles(self):
        active_pairs = self.get_active_trading_pairs()
        for pair in active_pairs:
            base, quote = pair.split('USDT')
            for middle in set(active_pairs) - {pair}:
                if middle.endswith(base):
                    third_pair = f"{quote}USDT"
                    if third_pair in active_pairs:
                        self.triangles.append((pair, middle, third_pair))
                elif middle.endswith(quote):
                    third_pair = f"{base}USDT"
                    if third_pair in active_pairs:
                        self.triangles.append((pair, middle, third_pair))

    def get_active_trading_pairs(self):
        tickers = self.client.get_all_tickers()
        exchange_info = self.client.get_exchange_info()
        symbols_info = exchange_info['symbols']
        active_trading_pairs = [symbol['symbol'] for symbol in symbols_info if symbol['status'] == 'TRADING' and symbol['symbol'].endswith("USDT")]
        return active_trading_pairs

    def print_triangles(self):
        for triangle in self.triangles:
            print(triangle)

    def get_klines(self, symbol, interval):
        klines = self.client.get_klines(symbol=symbol, interval=interval)
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

# Main execution
if __name__ == "__main__":
    file = 'credentials.txt'  # Replace with your API keys file path
    trader = Trader(file)
    trader.find_arbitrage_triangles()
    trader.print_triangles()
