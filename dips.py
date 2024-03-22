from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import talib as ta
import statsmodels.api as sm
import os, sys

import asyncio


class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """
    def connect(self,file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    """ Gets all account balances """
    def getBalances(self):
        prices = self.client.get_withdraw_history()
        return prices

filename = 'credentials.txt'
trader = Trader(filename)



filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pairCMO = []


trading_pairs = []

# Assuming 'trader' and 'client' are predefined objects
info = trader.client.get_exchange_info()
symbols = info['symbols']

for s in symbols:
    # Check if the symbol's quote asset is 'USDT' and is active for trading
    if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING' and 'SPOT' in s['permissions']:
        trading_pairs.append(s['symbol'])

with open('alts.txt', 'w') as f:
    for pair in trading_pairs:
        f.write(pair + '\n')


def filter1(pair):

    interval = '2h'
    symbol = pair
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 1h timeframe " + symbol)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99
 
    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs1.append(symbol)
        print('found')

        #plt.figure(figsize=(8,6))
        #plt.grid(True)
        #plt.plot(x)
        #plt.plot(best_fit_line1, '--', color='r')
        #plt.plot(best_fit_line2, '--', color='r')
        #plt.plot(best_fit_line3, '--', color='r')
        #plt.show(block=False)
        #plt.pause(6)
        #plt.close()

    elif x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
        filtered_pairs1.append(symbol)
        print('found')

        #plt.figure(figsize=(8,6))
        #plt.grid(True)
        #plt.plot(x)
        #plt.plot(best_fit_line1, '--', color='r')
        #plt.plot(best_fit_line2, '--', color='r')
        #plt.plot(best_fit_line3, '--', color='r')
        #plt.show(block=False)
        #plt.pause(6)
        #plt.close()

    else:
        print('searching') 

def filter2(filtered_pairs1):
    interval = '15m'
    symbol = filtered_pairs1
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 15min timeframe " + symbol)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] < best_fit_line1[-1]:
        filtered_pairs2.append(symbol)
        print('found')

        #plt.figure(figsize=(8,6))
        #plt.grid(True)
        #plt.plot(x)
        #plt.plot(best_fit_line1, '--', color='r')
        #plt.plot(best_fit_line2, '--', color='r')
        #plt.plot(best_fit_line3, '--', color='r')
        #plt.show(block=False)
        #plt.pause(6)
        #plt.close()

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
        filtered_pairs2.append(symbol)
        print('found') 

        #plt.figure(figsize=(8,6))
        #plt.grid(True)
        #plt.plot(x)
        #plt.plot(best_fit_line1, '--', color='r')
        #plt.plot(best_fit_line2, '--', color='r')
        #plt.plot(best_fit_line3, '--', color='r')
        #plt.show(block=False)
        #plt.pause(6)
        #plt.close()

def filter3(filtered_pairs2):
    interval = '5m'
    symbol = filtered_pairs2
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 5m timeframe " + symbol)

    #min = ta.MIN(close_array, timeperiod=30)
    #max = ta.MAX(close_array, timeperiod=30)

    #real = ta.HT_TRENDLINE(close_array)
    #wcl = ta.WCLPRICE(max, min, close_array)
    
    print(close[-1])
    print()
    #print(min[-1])
    #print(max[-1])
    #print(real[-1])    

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if x[-1] < best_fit_line1[-1]:
        filtered_pairs3.append(symbol)
        print('found')

        #plt.figure(figsize=(8,6))
        #plt.title(symbol)
        #plt.grid(True)
        #plt.plot(close)
        #plt.plot(best_fit_line1, '--', color='r')
        #plt.plot(best_fit_line2, '--', color='r')
        #plt.plot(best_fit_line3, '--', color='r')
        #plt.plot(close)
        #plt.plot(min)
        #plt.plot(max)
        #plt.plot(real)
        #plt.show(block=False)
        #plt.pause(5)
        #plt.close()

    else:
        print('searching') 


def momentum(filtered_pairs3):
    interval = '1m'
    symbol = filtered_pairs3
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 1m timeframe " + symbol)

    real = ta.CMO(close_array, timeperiod=14)
    #print(real[-1])

    if real[-1] < -50:
        print('oversold dip found')
        selected_pair.append(symbol)
        selected_pairCMO.append(real[-1])

    else:
        print('searching')

for i in trading_pairs:
    output = filter1(i)
    print(filtered_pairs1)

for i in filtered_pairs1:
    output = filter2(i)
    print(filtered_pairs2) 

for i in filtered_pairs2:
    output = filter3(i)
    print(filtered_pairs3) 

for i in filtered_pairs3:
    output = momentum(i)
    print(selected_pair)

if len(selected_pair) > 1:
    print('dips are more then 1 oversold') 
    print(selected_pair) 
    print(selected_pairCMO)
    
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
    print(selected_pairCMO)
    #sys.exit()

else:
    print('no oversold dips for the moment, restart script...')
    print(selected_pair) 
    print(selected_pairCMO) 
    #os.execl(sys.executable, sys.executable, *sys.argv)

sys.exit(0)
sys.exit()
exit()

