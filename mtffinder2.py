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

trading_pairs = []

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

# Now, usdt_parity_assets contains all the trading pairs with USDT parity
print("USDT Parity trading pairs:", usdt_parity_assets)



def filter1(pair):

    interval = '1h'
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

def trigger(selected_pair):
    interval = '1m'
    symbol = selected_pair
    klines = trader.client.get_klines(symbol=symbol,interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 1m timeframe " + symbol)

    sine, leadsine = ta.HT_SINE(close_array)
    
    if sine[-1] <= -0.5:
        print("found dip on time series sinewave")
        mom_dip.append(symbol)
        selected_pairSINE.append(sine[-1])
        print(sine[-1])
        
    else:
        print('dip is not low on time series...')


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

for i in selected_pair:
    output = trigger(i)
    print(mom_dip)

if len(selected_pair) > 1:
    print('dips are more then 1 oversold') 
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
    print('dips are more then 1 on time series') 
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

sys.exit(0)
sys.exit()
exit()

