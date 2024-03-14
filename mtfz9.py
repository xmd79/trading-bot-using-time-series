from binance.client import Client
import numpy as np
import talib as ta
import asyncio
from binance.exceptions import BinanceAPIException

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        with open(file) as f:
            lines = f.readlines()
        key = lines[0].strip()
        secret = lines[1].strip()
        self.client = Client(key, secret)

    def get_latest_price(self, symbol):
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

trader = Trader('credentials.txt')

filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
filtered_pairs4 = []
selected_pair = []
selected_pairCMO = []

async def filter_trading_pairs():
    trading_pairs = [symbol['symbol'] for symbol in trader.client.futures_exchange_info()['symbols'] if 'USDT' in symbol['symbol']]
    print("Trading pairs:", trading_pairs)  # Print out trading pairs for debugging
    for pair in trading_pairs:
        await filter1(pair)
    return True

async def filter1(pair):
    interval = '4h'
    try:
        klines = trader.client.get_klines(symbol=pair, interval=interval)
        close = [float(entry[4]) for entry in klines]

        x = close
        y = range(len(x))

        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

        if x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
            filtered_pairs1.append(pair)
            print(f'Found dip on 4h timeframe for {pair}')
        elif x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
            filtered_pairs1.append(pair)
            print(f'Found top on 4h timeframe for {pair}')
        else:
            print(f'Searching for {pair}')
    except BinanceAPIException as e:
        if e.code == -1121:
            pass  # Ignore invalid symbol error
        else:
            print(f"Error processing {pair}: {e}")

async def main():
    await filter_trading_pairs()

asyncio.run(main())
