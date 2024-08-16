#!/usr/bin/env python3

import numpy as np
import talib
import requests
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.stats import linregress
import time
from colorama import init, Fore, Style
from datetime import datetime, timezone
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import scipy.fftpack as fftpack

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Initialize colorama
init(autoreset=True)

timeframes = ["30m", "5m"]  # Timeframes for analysis
candle_map = {}

class BinanceWrapper:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.binance_client = BinanceClient(api_key=api_key, api_secret=api_secret)

    def get_binance_balance(self):
        """Get the balances of Binance Wallet."""
        balances = {}
        try:
            balances_response = self.binance_client.get_account()['balances']
            for balance in balances_response:
                balances[balance['asset']] = float(balance['free']) + float(balance['locked'])
            return balances
        except BinanceAPIException as e:
            print(f"Failed to get Binance balance. Exception: {e}")
            return None

    def get_available_pairs(self):
        """Get all trading pairs available against USDC."""
        exchange_info = self.binance_client.get_exchange_info()
        usdc_pairs = []
        for symbol_info in exchange_info['symbols']:
            if symbol_info['quoteAsset'] == 'USDC' and symbol_info['status'] == 'TRADING':
                usdc_pairs.append(symbol_info['symbol'])
        return usdc_pairs

    def get_candles(self, symbol, timeframe, limit=1000):
        """Fetch OHLCV data."""
        try:
            klines = self.binance_client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
            return [{
                "time": k[0] / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            } for k in klines]
        except BinanceAPIException as e:
            print(f"Error fetching candles for {symbol} at {timeframe}: {e}")
            return []

    def get_exchange_rate(self, symbol):
        """Get the exchange rate of Coins."""
        try:
            ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
            rate = float(ticker['price'])
            return rate
        except Exception as e:
            print(f"Failed to get {symbol} rate. Exception: {e}")
            return None

    def place_market_order(self, symbol, side, quantity):
        """Place a market order."""
        try:
            order = self.binance_client.order_market(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            print(f"Market Order: {side} {quantity} {symbol} placed successfully.")
            return order
        except BinanceAPIException as e:
            print(f"Error placing market order: {e}")
            return None

class Calculations:
    def __init__(self, binance_wrapper):
        self.binance_wrapper = binance_wrapper

    def calculate_dip(self, candles, threshold=0.02):
        """Check for dips in the closing price."""
        if len(candles) < 10:
            return False

        closing_prices = np.array([c["close"] for c in candles])
        moving_average = np.mean(closing_prices[-10:])  # Simple moving average
        current_price = closing_prices[-1]

        return current_price < moving_average * (1 - threshold)

    def check_profit_threshold(self, initial_price, current_price, profit_threshold=0.03):
        """Check if the price has moved above the profit threshold."""
        return current_price >= initial_price * (1 + profit_threshold)

# Initialize BinanceWrapper and Calculation instance
binance_wrapper = BinanceWrapper(api_key, api_secret)
calculations = Calculations(binance_wrapper)

# Main loop for continuous analysis
investment_amount = 1000  # Initial investment in USDC
purchase_orders = {}  # Store purchase orders and their entry prices

while True:
    # Fetch all USDC trading pairs
    usdc_pairs = binance_wrapper.get_available_pairs()
    print(f"\nAvailable USDC Pairs: {usdc_pairs}")

    # Check for dips in 30m timeframe
    assets_in_dip_30m = []
    for symbol in usdc_pairs:
        candles_30m = binance_wrapper.get_candles(symbol, "30m")
        if calculations.calculate_dip(candles_30m):
            assets_in_dip_30m.append(symbol)

    print(f"Assets in 30m Dip: {assets_in_dip_30m}")

    # Check for dips in 5m timeframe
    assets_in_dip_5m = []
    for symbol in assets_in_dip_30m:
        candles_5m = binance_wrapper.get_candles(symbol, "5m")
        if calculations.calculate_dip(candles_5m):
            assets_in_dip_5m.append(symbol)

    print(f"Assets in 5m Dip from 30m Dips: {assets_in_dip_5m}")

    # Execute trades for assets in dip
    for symbol in assets_in_dip_5m:
        usdc_balance = binance_wrapper.get_binance_balance().get('USDC', 0.0)

        # Define order quantity: invest all USDC available split by dip assets
        order_quantity = (usdc_balance / len(assets_in_dip_5m)) / binance_wrapper.get_exchange_rate(symbol)

        if order_quantity > 0:
            order = binance_wrapper.place_market_order(symbol, 'BUY', order_quantity)
            if order:
                purchase_orders[symbol] = {
                    'quantity': order_quantity,
                    'entry_price': binance_wrapper.get_exchange_rate(symbol)
                }

    # Check for take profit conditions
    for symbol, order_info in list(purchase_orders.items()):
        current_price = binance_wrapper.get_exchange_rate(symbol)
        if calculations.check_profit_threshold(order_info['entry_price'], current_price):
            # Execute convert (sell) market order
            order = binance_wrapper.place_market_order(symbol, 'SELL', order_info['quantity'])
            if order:
                print(f"Sold {order_info['quantity']} of {symbol} at profit target.")
                del purchase_orders[symbol]  # Remove from tracking after selling

    # Adding a delay before the next data fetch
    time.sleep(5)  # Adjust this based on your rate limits and needs
