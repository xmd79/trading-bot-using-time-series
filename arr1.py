import asyncio
import json
import numpy as np
import talib
import websockets
import aiohttp
from datetime import datetime, timedelta

# Constants
SYMBOL = 'btcusdc'
INTERVAL = '1m'  # 1-minute intervals for historical data
HISTORICAL_PERIOD = 60  # Fetch 60 minutes of historical data to ensure enough data
NUM_SECONDS = 60  # Collect 60 seconds of real-time data for analysis
MIN_DATA_LENGTH = 50  # Minimum length of data required for HT_SINE calculation

def clean_data(data):
    """ Remove NaN and zero values from the data. """
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = data[data != 0]
    return data

def scale_to_sine(close_prices):
    """ Calculate and scale the sine wave using HT_SINE from TA-Lib. """
    if len(close_prices) < MIN_DATA_LENGTH:
        print("Not enough data for HT_SINE calculation.")
        return None, None, None
    
    sine_wave, leadsine = talib.HT_SINE(close_prices)
    
    # Handle NaN values
    sine_wave = np.nan_to_num(sine_wave, nan=0)
    current_sine = sine_wave[-1] if len(sine_wave) > 0 else float('nan')
    
    # Calculate min and max sine values
    sine_wave_min = np.min(sine_wave) if len(sine_wave) > 0 else float('nan')
    sine_wave_max = np.max(sine_wave) if len(sine_wave) > 0 else float('nan')
    
    # Calculate percentage distances
    dist_min, dist_max = None, None
    if not np.isnan(current_sine) and not np.isnan(sine_wave_min) and not np.isnan(sine_wave_max):
        dist_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
        dist_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * 100

    return dist_min, dist_max, current_sine

async def fetch_historical_data():
    """ Fetch historical OHLC and volume data from Binance API. """
    async with aiohttp.ClientSession() as session:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=HISTORICAL_PERIOD)
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval={INTERVAL}&startTime={int(start_time.timestamp() * 1000)}&endTime={int(end_time.timestamp() * 1000)}"
        
        async with session.get(url) as response:
            data = await response.json()
            
            # Extract OHLC and volume data
            ohlc = np.array([float(candle[4]) for candle in data])  # Close prices
            return ohlc

async def process_websocket_data(websocket, historical_data):
    """ Process WebSocket data and apply TA-Lib HT_SINE. """
    last_minute_data = []
    
    while True:
        try:
            response = await websocket.recv()
            data = json.loads(response)
            current_price = float(data['c'])
            
            # Update last minute of data
            last_minute_data.append(current_price)
            last_minute_data = last_minute_data[-NUM_SECONDS:]
            
            # Combine WebSocket data with historical data
            combined_data = np.concatenate((historical_data, last_minute_data))
            cleaned_data = clean_data(combined_data)
            
            print(f"Combined Prices Length: {len(combined_data)}")
            print(f"Cleaned Prices: {cleaned_data}")

            if len(cleaned_data) >= MIN_DATA_LENGTH:
                # Scale current close price to sine wave
                dist_min, dist_max, current_sine = scale_to_sine(cleaned_data)
                
                print(f"Current Price: {current_price}")
                print(f"HT_SINE Current Sine Value: {current_sine}")
                print(f"Distance to Min: {dist_min:.2f}%")
                print(f"Distance to Max: {dist_max:.2f}%")
                
                # Analyze harmonic wave
                print(f"Cleaned Prices (last 10): {cleaned_data[-10:]}")
                print(f"Cycle Analysis: {'Peak' if dist_min and dist_max and dist_min < 50 else 'Trough'}")
                print(f"Trend Analysis: {'Bullish' if dist_min and dist_max and dist_min < 50 else 'Bearish'}")
            else:
                print("Insufficient data for HT_SINE calculation")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error: {e}")

async def main():
    url = "wss://stream.binance.com:9443/ws/btcusdc@ticker"
    ohlc_data = await fetch_historical_data()
    
    async with websockets.connect(url) as websocket:
        await process_websocket_data(websocket, ohlc_data)

if __name__ == "__main__":
    asyncio.run(main())
