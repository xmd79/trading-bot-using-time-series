import asyncio
import json
import numpy as np
import talib
import websockets
import aiohttp
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
import statsmodels.api as sm

# Constants
SYMBOL = 'btcusdc'
INTERVAL_1M = '1m'
INTERVAL_5M = '5m'
HISTORICAL_PERIOD_MINUTES = 120  # Fetch 120 minutes of historical data
NUM_SECONDS = 60  # Collect 60 seconds of real-time data
MIN_DATA_LENGTH = 120  # Minimum length of data required for decomposition

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def clean_data(data):
    """ Remove NaN and zero values from the data. """
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = data[data != 0]
    return data

def decompose_time_series(data, freq):
    """ Decompose time series data into trend, seasonal, and residual components. """
    decomposed = sm.tsa.seasonal_decompose(data, period=freq)
    return decomposed.trend, decomposed.seasonal, decomposed.resid

async def fetch_historical_data(interval, period_minutes):
    """ Fetch historical OHLC and volume data from Binance API. """
    async with aiohttp.ClientSession() as session:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=period_minutes)
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval={interval}&startTime={int(start_time.timestamp() * 1000)}&endTime={int(end_time.timestamp() * 1000)}"
        
        async with session.get(url) as response:
            data = await response.json()
            ohlc = np.array([[
                float(candle[1]),  # Open
                float(candle[2]),  # High
                float(candle[3]),  # Low
                float(candle[4]),  # Close
                float(candle[5])   # Volume
            ] for candle in data])
            return ohlc

async def process_websocket_data(websocket, historical_data_1m, historical_data_5m):
    """ Process WebSocket data and apply filters and analysis. """
    last_minute_data = []
    
    while True:
        try:
            response = await websocket.recv()
            data = json.loads(response)
            current_price = float(data['c'])
            current_volume = float(data['v'])
            
            # Update last minute of data
            last_minute_data.append(current_price)
            last_minute_data = last_minute_data[-NUM_SECONDS:]
            
            # Combine WebSocket data with historical data
            combined_data_1m = np.concatenate((historical_data_1m[:, 3], last_minute_data))
            cleaned_data_1m = clean_data(combined_data_1m)
            
            combined_data_5m = historical_data_5m[:, 3]
            cleaned_data_5m = clean_data(combined_data_5m)
            
            print(f"Combined Prices Length (1m): {len(combined_data_1m)}")
            print(f"Cleaned Prices (1m): {cleaned_data_1m}")

            if len(cleaned_data_1m) >= MIN_DATA_LENGTH:
                # Apply low bandpass filter to the data
                filtered_data = butter_lowpass_filter(cleaned_data_1m, cutoff=0.1, fs=1.0)
                
                # Decompose time series
                trend, seasonal, resid = decompose_time_series(filtered_data, freq=NUM_SECONDS)
                
                # Volume analysis
                total_volume = np.sum(historical_data_1m[:, 4]) + current_volume
                avg_volume = total_volume / (len(historical_data_1m[:, 4]) + 1)
                volume_trend = 'Bullish' if current_volume > avg_volume else 'Bearish'
                
                print(f"Current Price: {current_price}")
                print(f"Volume Trend: {volume_trend}")
                
                # Further analysis (e.g., volume profile, market profile, liquidation prices) can be added here.
                
            else:
                print("Insufficient data for decomposition")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error: {e}")

async def main():
    url = "wss://stream.binance.com:9443/ws/btcusdc@ticker"
    
    # Fetch sufficient historical data
    ohlc_data_1m = await fetch_historical_data(INTERVAL_1M, HISTORICAL_PERIOD_MINUTES)
    ohlc_data_5m = await fetch_historical_data(INTERVAL_5M, HISTORICAL_PERIOD_MINUTES)
    
    async with websockets.connect(url) as websocket:
        await process_websocket_data(websocket, ohlc_data_1m, ohlc_data_5m)

if __name__ == "__main__":
    asyncio.run(main())
