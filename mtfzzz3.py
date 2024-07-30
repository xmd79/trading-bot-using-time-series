import numpy as np
import sys
from datetime import datetime
from binance.client import Client
import pandas as pd
import talib

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

def calculate_sma(prices, period):
    return prices.rolling(window=period).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_percentage_deviation(close_price, sma):
    return ((close_price - sma) / sma) * 100

def filter_pair_details(client, pair, interval):
    df = get_klines(client, pair, interval)
    close_prices = df['Close']
    
    # Calculate SMAs
    s_sma = calculate_sma(close_prices, 5).iloc[-1]
    m_sma = calculate_sma(close_prices, 20).iloc[-1]
    l_sma = calculate_sma(close_prices, 50).iloc[-1]
    
    # Calculate RSI
    rsi = calculate_rsi(close_prices).iloc[-1]

    current_price = close_prices.iloc[-1]

    # Calculate percentage deviations
    s_sma_dev = calculate_percentage_deviation(current_price, s_sma)
    m_sma_dev = calculate_percentage_deviation(current_price, m_sma)
    l_sma_dev = calculate_percentage_deviation(current_price, l_sma)
    
    # Determine if conditions are met for dip
    dip_condition = s_sma < m_sma < l_sma

    return {
        'pair': pair,
        'current_price': current_price,
        's_sma': s_sma,
        'm_sma': m_sma,
        'l_sma': l_sma,
        'rsi': rsi,
        's_sma_dev': s_sma_dev,
        'm_sma_dev': m_sma_dev,
        'l_sma_dev': l_sma_dev,
        'dip_condition': dip_condition
    }

def find_top_asset(client, pairs, interval):
    best_asset = None
    lowest_rsi = float('inf')
    best_details = {}
    
    for pair in pairs:
        details = filter_pair_details(client, pair, interval)
        
        # Select asset with the lowest RSI and meet the dip condition
        if details['rsi'] < lowest_rsi and details['dip_condition']:
            lowest_rsi = details['rsi']
            best_asset = pair
            best_details = details

    return best_asset, best_details

# Main execution
filename = 'credentials.txt'
trader = Trader(filename)

try:
    active_trading_pairs = get_active_trading_pairs(trader.client)

    print("All active trading pairs vs USDT (Spot trading only):")
    for pair in active_trading_pairs:
        print(f"Symbol: {pair}")

    print('\nFinding top asset with the lowest RSI and dip condition...')
    top_asset, top_asset_details = find_top_asset(trader.client, active_trading_pairs, '2h')

    if top_asset:
        print(f"\nTop asset with the lowest RSI and dip condition: {top_asset}")
        print(f"Details: {top_asset_details}")
    else:
        print("No suitable asset found.")

except Exception as e:
    print("Error during execution:", e)
