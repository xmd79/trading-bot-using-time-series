from binance.client import Client
import pandas as pd
import concurrent.futures
import time

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
            symbol['symbol']
            for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        return trading_pairs

    def get_ohlc_data(self, symbol, interval='1m'):
        klines = self.client.get_historical_klines(symbol, interval, limit=1)
        if klines:
            # Kline data format: [Open time, Open, High, Low, Close, ...]
            ohlc = {
                'Open Time': pd.to_datetime(klines[0][0], unit='ms'),
                'Open': float(klines[0][1]),
                'High': float(klines[0][2]),
                'Low': float(klines[0][3]),
                'Close': float(klines[0][4]),
            }
            return symbol, interval, ohlc
        return None

def fetch_ohlc(trader, pair, timeframe):
    return trader.get_ohlc_data(pair, timeframe)

def main():
    filename = 'credentials.txt'
    trader = Trader(filename)

    usdc_pairs = trader.get_usdc_pairs()
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for pair in usdc_pairs:
            for timeframe in timeframes:
                futures.append(executor.submit(fetch_ohlc, trader, pair, timeframe))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                pair, timeframe, ohlc = result
                print(f"OHLC for {pair} at {timeframe}: {ohlc}")

if __name__ == "__main__":
    main()