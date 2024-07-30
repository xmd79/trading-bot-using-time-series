import numpy as np
import sys
from datetime import datetime
from binance.client import Client
import pandas as pd
import concurrent.futures
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

def calculate_volume_stats(df):
    buy_volume = df[df['Close'] > df['Open']]['Volume'].sum()
    sell_volume = df[df['Close'] < df['Open']]['Volume'].sum()
    total_volume = df['Volume'].sum()
    return buy_volume, sell_volume, total_volume

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def forecast_reversal(close, s_sma, m_sma, l_sma):
    if close < s_sma < m_sma < l_sma:
        return 'Dip forecasted'
    elif close > s_sma > m_sma > l_sma:
        return 'Top forecasted'
    else:
        return 'No clear forecast'

def determine_cycle(close, m_sma):
    return 'up' if close > m_sma else 'down'

def filter_pair_with_sma_and_rsi(client, pair, interval):
    df = get_klines(client, pair, interval)
    close_prices = df['Close']
    
    # Calculate SMAs
    s_sma = calculate_sma(close_prices, 5).iloc[-1]
    m_sma = calculate_sma(close_prices, 20).iloc[-1]
    l_sma = calculate_sma(close_prices, 50).iloc[-1]
    
    # Calculate RSI
    rsi = calculate_rsi(close_prices).iloc[-1]

    current_price = close_prices.iloc[-1]
    forecast = forecast_reversal(current_price, s_sma, m_sma, l_sma)
    current_cycle = determine_cycle(current_price, m_sma)

    # Determine status
    status = forecast
    if forecast == 'Dip forecasted' and current_cycle == 'down':
        status = 'Confirmed Dip'
    elif forecast == 'Top forecasted' and current_cycle == 'up':
        status = 'Confirmed Top'
    
    buy_volume, sell_volume, total_volume = calculate_volume_stats(df)
    volume_status = 'Bullish' if buy_volume > sell_volume else 'Bearish'

    return {
        'pair': pair,
        'current_price': current_price,
        's_sma': s_sma,
        'm_sma': m_sma,
        'l_sma': l_sma,
        'rsi': rsi,
        'status': status,
        'volume_status': volume_status
    }

def find_top_asset_with_lowest_position(client, pairs, interval):
    best_asset = None
    lowest_position = float('inf')
    best_details = {}

    for pair in pairs:
        details = filter_pair_with_sma_and_rsi(client, pair, interval)
        current_price = details['current_price']
        s_sma = details['s_sma']
        m_sma = details['m_sma']
        l_sma = details['l_sma']
        
        # Compute position based on SMAs
        position = abs(current_price - s_sma) + abs(current_price - m_sma) + abs(current_price - l_sma)

        if position < lowest_position:
            lowest_position = position
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

    print('Finding top asset with the lowest position...')
    top_asset, top_asset_details = find_top_asset_with_lowest_position(trader.client, active_trading_pairs, '2h')

    if top_asset:
        print(f"Top asset with the lowest position: {top_asset}")
        print(f"Details: {top_asset_details}")
    else:
        print("No suitable asset found.")

except Exception as e:
    print("Error during execution:", e)
