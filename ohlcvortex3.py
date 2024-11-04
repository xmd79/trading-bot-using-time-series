import numpy as np
import pandas as pd
from binance.client import Client as BinanceClient
import datetime

# Define Binance client by reading API key and secret from a local file
def get_binance_client():
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
    client = BinanceClient(api_key, api_secret)
    return client

client = get_binance_client()
TRADE_SYMBOL = "BTCUSDT"  # Specify the trading pair
INTERVAL = BinanceClient.KLINE_INTERVAL_1DAY  # Specify the interval for OHLC data
LIMIT = 1000  # Number of data points to retrieve

# Function to fetch historical OHLC data from Binance
def fetch_ohlc_data(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close',
        'Volume', 'Close Time', 'Quote Asset Volume',
        'Number of Trades', 'Taker Buy Base Asset Volume',
        'Taker Buy Quote Asset Volume', 'Ignore'])
    
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
    data['Close'] = data['Close'].astype(float)
    data['Volume'] = data['Volume'].astype(float)
    
    return data[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Vortex forecast based on previous price changes
def vortex_forecast(close_price, data):
    data['Daily_Return'] = data['Close'].pct_change()
    recent_returns = data['Daily_Return'].dropna().tail(30)
    avg_daily_change = recent_returns.mean()
    
    future_price = close_price * ((1 + avg_daily_change) ** (15 / 24))  # Project for 15 hours
    return future_price, close_price  # Return both forecasted future price and current price

# Apply Vortex Math signals
def vortex_math_signals(data):
    data['Digit_Sum'] = data['Close'].apply(lambda x: sum(int(digit) for digit in str(int(x))))
    data['Vortex_Pattern'] = data['Digit_Sum'] % 9
    data['Buy_Signal'] = (data['Vortex_Pattern'] == 1).astype(int)
    data['Sell_Signal'] = (data['Vortex_Pattern'] == 5).astype(int)
    return data

# Market sentiment analysis based on recent closes
def market_mood(data):
    recent_changes = data['Close'].pct_change().tail(10)  # Last 10 price changes
    if recent_changes.sum() > 0:
        return "Bullish"
    elif recent_changes.sum() < 0:
        return "Bearish"
    return "Neutral"

# Time to target forecasting
def time_to_target():
    target_hour = 17  # Set default target hour to 5 PM
    now = datetime.datetime.now()
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    
    if now > target_time:
        target_time += datetime.timedelta(days=1)

    time_diff = target_time - now
    return time_diff.total_seconds() / 3600, time_diff  # Return time in hours and delta for minutes

# Calculate target price forecast
def target_price_forecast(data, hours_to_target):
    data['Daily_Return'] = data['Close'].pct_change()
    recent_returns = data['Daily_Return'].dropna().tail(30)
    avg_daily_change = recent_returns.mean()
    current_price = data['Close'].iloc[-1]
    
    forecast_price = current_price * ((1 + avg_daily_change) ** hours_to_target)
    return forecast_price, current_price

# Main workflow to analyze and process data from Binance
def main():
    stock_data = fetch_ohlc_data(TRADE_SYMBOL, INTERVAL, LIMIT)
    stock_data = vortex_math_signals(stock_data)

    mood = market_mood(stock_data)
    hours_to_target, time_diff = time_to_target()
    target_price, current_price = target_price_forecast(stock_data, hours_to_target)
    
    # Get Vortex forecast price
    vortex_forecast_price, current_price_vortex = vortex_forecast(stock_data['Close'].iloc[-1], stock_data)

    # Print relevant information
    print(f"Market Mood: {mood} (based on the last 10 price changes)")
    print(f"Time to target: {int(hours_to_target)} hours and {int((time_diff.seconds // 60) % 60)} minutes.")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Target Price Forecast: ${target_price:.2f}")
    print(f"Vortex Forecast Price: ${vortex_forecast_price:.2f} (based on current price of ${current_price_vortex:.2f})")

# Run the main function
if __name__ == "__main__":
    main()