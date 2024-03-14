from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import talib as ta
import statsmodels.api as sm
import os, sys

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def getBalances(self):
        account_info = self.client.get_account()
        balances = account_info['balances']
        return balances

    def getCurrentPrices(self):
        ticker_prices = self.client.get_all_tickers()
        return {ticker['symbol']: float(ticker['price']) for ticker in ticker_prices}

filename = 'credentials.txt'
trader = Trader(filename)

# Fetch all trading pairs
exchange_info = trader.client.get_exchange_info()
symbols = exchange_info['symbols']

# Extract trading pairs with USDT as the quote asset
trading_pairs = [symbol['symbol'] for symbol in symbols if symbol['quoteAsset'] == 'USDT']

filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pairCMO = []
mom_dip = []
selected_pairSINE = []

# Example: Get and print current prices for USDT parity assets
current_prices = trader.getCurrentPrices()
usdt_parity_assets = []

for symbol, price in current_prices.items():
    if symbol.endswith("USDT"):
        usdt_parity_assets.append(symbol)
        print(f"Symbol: {symbol}, Current Price: {price}")

print()

def filter1(pair):

    interval = '15m'
    symbol = pair
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 15m timeframe " + symbol)

    x = close
    y = range(len(x))

    poly_model = np.polyfit(y, x, 1)
    best_fit_line1 = np.poly1d(poly_model)(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    min_poly_value = min(best_fit_line1)
    max_poly_value = max(best_fit_line1)
    min_poly_price = np.interp(min_poly_value, best_fit_line1, x)
    max_poly_price = np.interp(max_poly_value, best_fit_line1, x)

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs1.append(symbol)
        print('found')
        print("Current Poly Value:", poly_model)
        print("Min Poly Value:", min_poly_value, "Price:", min_poly_price)
        print("Max Poly Value:", max_poly_value, "Price:", max_poly_price)

    elif x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
        filtered_pairs1.append(symbol)
        print('found')
        print("Current Poly Value:", poly_model)
        print("Min Poly Value:", min_poly_value, "Price:", min_poly_price)
        print("Max Poly Value:", max_poly_value, "Price:", max_poly_price)

    else:
        print('searching') 

for pair in trading_pairs:
    filter1(pair)

def filter2(filtered_pairs1):
    interval = '5m'
    symbol = filtered_pairs1
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 15min timeframe " + symbol)

    x = close
    y = range(len(x))

    poly_model = np.polyfit(y, x, 1)
    best_fit_line1 = np.poly1d(poly_model)(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    min_poly_value = min(best_fit_line1)
    max_poly_value = max(best_fit_line1)
    min_poly_price = np.interp(min_poly_value, best_fit_line1, x)
    max_poly_price = np.interp(max_poly_value, best_fit_line1, x)

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] < best_fit_line1[-1]:
        filtered_pairs2.append(symbol)
        print('found')
        print("Current Poly Value:", poly_model)
        print("Min Poly Value:", min_poly_value, "Price:", min_poly_price)
        print("Max Poly Value:", max_poly_value, "Price:", max_poly_price)

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
        filtered_pairs2.append(symbol)
        print('found') 
        print("Current Poly Value:", poly_model)
        print("Min Poly Value:", min_poly_value, "Price:", min_poly_price)
        print("Max Poly Value:", max_poly_value, "Price:", max_poly_price)

for pair in filtered_pairs1:
    filter2(pair)

def filter3(filtered_pairs2):
    interval = '3m'
    symbol = filtered_pairs2
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 3m timeframe " + symbol)

    x = close
    y = range(len(x))

    poly_model = np.polyfit(y, x, 1)
    best_fit_line1 = np.poly1d(poly_model)(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99

    min_poly_value = min(best_fit_line1)
    max_poly_value = max(best_fit_line1)
    min_poly_price = np.interp(min_poly_value, best_fit_line1, x)
    max_poly_price = np.interp(max_poly_value, best_fit_line1, x)

    if x[-1] < best_fit_line1[-1]:
        filtered_pairs3.append(symbol)
        print('found')
        print("Current Poly Value:", poly_model)
        print("Min Poly Value:", min_poly_value, "Price:", min_poly_price)
        print("Max Poly Value:", max_poly_value, "Price:", max_poly_price)

    else:
        print('searching') 

for pair in filtered_pairs2:
    filter3(pair)

def momentum(filtered_pairs3):
    interval = '1m'
    symbol = filtered_pairs3
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 1m timeframe " + symbol)

    real = ta.CMO(close_array, timeperiod=14)

    if real[-1] < -50:
        print('oversold dip found')
        selected_pair.append(symbol)
        selected_pairCMO.append(real[-1])

    else:
        print('searching')

def trigger(selected_pair):
    interval = '1m'
    symbol = selected_pair
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) if entry[4] != 'NaN' else np.nan for entry in klines]
    close_array = np.asarray(close)

    close_array = np.nan_to_num(close_array, nan=np.nanmean(close_array))  # Replace NaN with mean

    print("on 1m timeframe " + symbol)

    sine, leadsine = ta.HT_SINE(close_array)
    
    if sine[-1] <= -0.5:
        print("found dip on time series sinewave")
        mom_dip.append(symbol)
        selected_pairSINE.append(sine[-1])
        print(sine[-1])
        
    else:
        print('dip is not low on time series...')


for pair in filtered_pairs3:
    momentum(pair)

for pair in selected_pair:
    trigger(pair)

if len(selected_pair) > 1:
    print('dips are more than 1 oversold') 
    print(selected_pair) 
    #print(selected_pairCMO)
    
    if min(selected_pairCMO) in selected_pairCMO:
        #print(selected_pairCMO.index(min(selected_pairCMO)))
        position = selected_pairCMO.index(min(selected_pairCMO))

    for id, value in enumerate(selected_pair):
        if id == position:
            print(selected_pair[id])
    #sys.exit()

elif len(selected_pair) == 1:
    print('1 dip found')   
    print(selected_pair) 
    #print(selected_pairCMO)
    #sys.exit()

else:
    print('no oversold dips for the moment, rescan dips...')

    #print(selected_pair) 
    #print(selected_pairCMO) 
    #os.execl(sys.executable, sys.executable, *sys.argv)

if len(mom_dip) > 1:
    print('dips are more than 1 on time series') 
    print(mom_dip)
    
    if min(mom_dip) in selected_pairSINE:
        #print(selected_pairSINE.index(min(selected_pairSINE)))
        position = selected_pairSINE.index(min(selected_pairSINE))

        for id, value in enumerate(selected_pair):
            if id == position:
                print(mom_dip[id])
        #print(mom_dip[id])
    #sys.exit()


elif len(mom_dip) == 1:
    print('1 dip found on time series')
    print(mom_dip)

else:
    print('no momentum dips on time series')
