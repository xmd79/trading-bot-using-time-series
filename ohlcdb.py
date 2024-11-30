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
        klines = self.client.get_historical_klines(symbol, interval, limit=100)  # Fetch 100 latest
        if klines:
            # Kline data format: [Open time, Open, High, Low, Close, Volume, ...]
            ohlc_data = []
            for kline in klines:
                ohlc_data.append({
                    'Open Time': pd.to_datetime(kline[0], unit='ms'),
                    'Open': float(kline[1]),
                    'High': float(kline[2]),
                    'Low': float(kline[3]),
                    'Close': float(kline[4]),
                    'Volume': float(kline[5]),
                })
            return ohlc_data
        return None

    def significant_volume_levels(self, ohlc_data):
        total_volume = sum(d['Volume'] for d in ohlc_data)
        bullish_volume = sum(d['Volume'] for d in ohlc_data if d['Close'] > d['Open'])
        bearish_volume = sum(d['Volume'] for d in ohlc_data if d['Close'] < d['Open'])
        
        significant_levels = {
            'Support': min(d['Low'] for d in ohlc_data),  # Potential support level
            'Resistance': max(d['High'] for d in ohlc_data),  # Potential resistance level
        }

        volume_percentage = {
            'Bullish Volume Percentage': 100 * (bullish_volume / total_volume) if total_volume > 0 else 0,
            'Bearish Volume Percentage': 100 * (bearish_volume / total_volume) if total_volume > 0 else 0,
        }

        if bullish_volume > bearish_volume:
            volume_mood = "BULLISH"
        else:
            volume_mood = "BEARISH"

        return significant_levels, volume_percentage, volume_mood

def fetch_ohlc(trader, pair, timeframe):
    ohlc_data = trader.get_ohlc_data(pair, timeframe)
    if ohlc_data:
        significant_levels, volume_percentage, volume_mood = trader.significant_volume_levels(ohlc_data)
        return (pair, timeframe, ohlc_data, significant_levels, volume_percentage, volume_mood)
    return None

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
                pair, timeframe, ohlc_data, significant_levels, volume_percentage, volume_mood = result
                print(f"--- {pair} at {timeframe} ---")
                print(f"Significant Levels: {significant_levels}")
                print(f"Volume Percentages: {volume_percentage}")
                print(f"Volume Mood: {volume_mood}")
                print("Recent OHLC data:")
                for entry in ohlc_data:
                    print(f"Open: {entry['Open']:.25f}, High: {entry['High']:.25f}, Low: {entry['Low']:.25f}, Close: {entry['Close']:.25f}, Volume: {entry['Volume']:.25f}")
                print("\n")

if __name__ == "__main__":
    main()
