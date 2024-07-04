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
        usdt_pairs = [pair for pair in active_pairs if pair.endswith('USDT')]
        for usdt_pair in usdt_pairs:
            a = usdt_pair[:-4]  # Remove 'USDT' to get the base currency
            for pair in active_pairs:
                if pair.startswith(a) and pair != usdt_pair:
                    b = pair[len(a):]
                    third_pair = f"{b}USDT"
                    if third_pair in active_pairs:
                        self.triangles.append((usdt_pair, pair, third_pair))

    def get_active_trading_pairs(self):
        exchange_info = self.client.get_exchange_info()
        symbols_info = exchange_info['symbols']
        active_trading_pairs = [symbol['symbol'] for symbol in symbols_info if symbol['status'] == 'TRADING']
        return active_trading_pairs

    def get_current_prices(self):
        prices = self.client.get_all_tickers()
        price_dict = {item['symbol']: float(item['price']) for item in prices}
        return price_dict

    def calculate_profit(self, triangle, prices):
        pair1, pair2, pair3 = triangle
        if pair1 in prices and pair2 in prices and pair3 in prices:
            rate1 = prices[pair1]  # USDT to A
            rate2 = prices[pair2]  # A to B
            rate3 = prices[pair3]  # B to USDT

            # Calculate cross rates
            if pair2.startswith(pair1[:-4]):
                cross_rate2 = rate2
            else:
                cross_rate2 = 1 / rate2

            if pair3.startswith(pair2[len(pair1[:-4]):]):
                cross_rate3 = rate3
            else:
                cross_rate3 = 1 / rate3

            # Profit calculation
            initial_amount = 1
            amount_a = initial_amount / rate1
            amount_b = amount_a * cross_rate2
            final_amount = amount_b * cross_rate3
            profit_percentage = ((final_amount - initial_amount) / initial_amount) * 100
            return profit_percentage, rate1, cross_rate2, cross_rate3
        return None

    def find_profitable_triangles(self):
        prices = self.get_current_prices()
        profitable_triangles = []
        for triangle in self.triangles:
            result = self.calculate_profit(triangle, prices)
            if result:
                profit_percentage = result[0]
                if profit_percentage > 0:
                    profitable_triangles.append((triangle, result))
        return profitable_triangles

    def print_profitable_triangles(self):
        profitable_triangles = self.find_profitable_triangles()
        # Sort triangles by profit percentage in descending order
        profitable_triangles.sort(key=lambda x: x[1][0], reverse=True)
        for triangle, (profit_percentage, rate1, cross_rate2, cross_rate3) in profitable_triangles:
            print(f"Triangle: {triangle} | Potential Profit: {profit_percentage:.2f}%")
            pair1, pair2, pair3 = triangle
            print(f"1. Trade {pair1}: 1 USDT -> {1 / rate1:.6f} {pair1[:-4]}")
            print(f"2. Trade {pair2}: {1 / rate1:.6f} {pair1[:-4]} -> {1 / rate1 * cross_rate2:.6f} {pair2[len(pair1[:-4]):]}")
            print(f"3. Trade {pair3}: {1 / rate1 * cross_rate2:.6f} {pair2[len(pair1[:-4]):]} -> {1 / rate1 * cross_rate2 * cross_rate3:.6f} USDT")
            print(f"Overall Profit Percentage: {profit_percentage:.2f}%")
            print("--------------------------------------------------")

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
    trader.print_profitable_triangles()
