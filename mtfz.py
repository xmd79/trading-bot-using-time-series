from binance.client import Client
import numpy as np
import matplotlib.pyplot as plt
import talib
 
class Trader:
    def __init__(self, file):
        self.connect(file)
 
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        self.key = lines[0]
        self.secret = lines[1]
        self.client = Client(self.key, self.secret)
 
        
class TechnicalAnalyzer:
    def __init__(self, client):
        self.client = client
 
    def get_historical_data(self, symbol, interval):
        klines = self.client.futures_historical_klines(symbol, interval, '1 day ago')
        close_prices = np.array([float(entry[4]) for entry in klines])
        return close_prices
 
    def fit_line(self, data):
        x = np.arange(len(data))
        z = np.polyfit(x, data, 1)
        p = np.poly1d(z)
        return p(x)
    
    def identify_dip(self, close_prices):
        fit_line = self.fit_line(close_prices)
        if close_prices[-1] < fit_line[-1] and fit_line[0] <= fit_line[-1]:
            return True
        return False
    
    def identify_top(self, close_prices):
        fit_line = self.fit_line(close_prices)
        if close_prices[-1] > fit_line[-1] and fit_line[0] >= fit_line[-1]:
            return True
        return False
 
 
filename = 'credentials.txt'
trader = Trader(filename)
analyzer = TechnicalAnalyzer(trader.client)
 
trading_pairs = [symbol['symbol'] for symbol in trader.client.futures_exchange_info()['symbols'] if 'USDT' in symbol['symbol']]
 
for pair in trading_pairs:
    close_prices_2h = analyzer.get_historical_data(pair, '2h')
    close_prices_15min = analyzer.get_historical_data(pair, '15m')
    close_prices_5min = analyzer.get_historical_data(pair, '5m')
    close_prices_1min = analyzer.get_historical_data(pair, '1m')
    
    if (analyzer.identify_dip(close_prices_2h) and
        analyzer.identify_dip(close_prices_15min) and
        analyzer.identify_dip(close_prices_5min) and
        analyzer.identify_dip(close_prices_1min)):
        print(f'{pair}: Dip confirmed on all timeframes')
            
    elif (analyzer.identify_top(close_prices_2h) and
          analyzer.identify_top(close_prices_15min) and
          analyzer.identify_top(close_prices_5min) and
          analyzer.identify_top(close_prices_1min)):
        print(f'{pair}: Top confirmed on all timeframes')
