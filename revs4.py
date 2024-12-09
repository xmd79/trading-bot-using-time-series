import numpy as np
import ephem
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import concurrent.futures

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Function to fetch candles for a specific timeframe with a dynamic limit for large timeframes
def get_candles(symbol, timeframe, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        candles = []
        for k in klines:
            candle = {
                "time": k[0] / 1000,  # Convert ms to seconds
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "timeframe": timeframe
            }
            candles.append(candle)
        return candles
    except BinanceAPIException as e:
        print(f"Error fetching candles: {e.message}")
        return []

# Function to gradually adjust data fetching for larger timeframes (1d and 1w)
def adjust_data_for_large_timeframes(symbol, timeframe):
    limit = 1000
    while limit >= 100:
        candles = get_candles(symbol, timeframe, limit)
        if len(candles) > 0:
            return candles
        limit -= 50
    return []

# Function to fetch candles for multiple timeframes in parallel
def fetch_candles_in_parallel(timeframes, symbol='BTCUSDC'):
    def fetch_candles(timeframe):
        return adjust_data_for_large_timeframes(symbol, timeframe)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

# Function to get the current BTC price
def get_current_btc_price():
    try:
        ticker = client.get_symbol_ticker(symbol="BTCUSDC")
        return float(ticker['price'])
    except BinanceAPIException as e:
        print(f"Error fetching BTC price: {e.message}")
        return 0.0

# Function to find major and minor reversals with volume ratios
def find_reversals_and_volume_ratios(candles, current_close):
    if not candles:
        return None, None, None, None, None, None, None, None  # No data available

    # Major Reversals
    lows = [candle['low'] for candle in candles]
    highs = [candle['high'] for candle in candles]
    lowest_low = np.min(lows) if lows else None
    highest_high = np.max(highs) if highs else None

    major_local_dips = []
    major_local_tops = []
    minor_local_dips = []
    minor_local_tops = []

    for i in range(1, len(candles) - 1):
        # Find major local dips
        if (
            candles[i]['low'] < candles[i - 1]['low'] and
            candles[i]['low'] < candles[i + 1]['low'] and
            abs(candles[i]['low'] - current_close) > 200  # Arbitrary threshold for major
        ):
            major_local_dips.append(candles[i])
        
        # Find major local tops
        if (
            candles[i]['high'] > candles[i - 1]['high'] and
            candles[i]['high'] > candles[i + 1]['high'] and
            abs(candles[i]['high'] - current_close) > 200  # Arbitrary threshold for major
        ):
            major_local_tops.append(candles[i])

        # Find minor local dips
        if (
            candles[i]['low'] < candles[i - 1]['low'] and
            candles[i]['low'] < candles[i + 1]['low']
        ):
            minor_local_dips.append(candles[i])
        
        # Find minor local tops
        if (
            candles[i]['high'] > candles[i - 1]['high'] and
            candles[i]['high'] > candles[i + 1]['high']
        ):
            minor_local_tops.append(candles[i])

    # Determine valid local dips and tops
    valid_minor_dips = [dip for dip in minor_local_dips if dip['low'] < current_close]
    valid_minor_tops = [top for top in minor_local_tops if top['high'] > current_close]
    valid_major_dips = [dip for dip in major_local_dips if dip['low'] < current_close]
    valid_major_tops = [top for top in major_local_tops if top['high'] > current_close]

    closest_minor_dip = min(valid_minor_dips, key=lambda x: abs(x['low'] - current_close), default=None) if valid_minor_dips else None
    closest_minor_top = min(valid_minor_tops, key=lambda x: abs(x['high'] - current_close), default=None) if valid_minor_tops else None
    closest_major_dip = min(valid_major_dips, key=lambda x: abs(x['low'] - current_close), default=None) if valid_major_dips else None
    closest_major_top = min(valid_major_tops, key=lambda x: abs(x['high'] - current_close), default=None) if valid_major_tops else None

    # Volume calculations
    bullish_volume = sum(candle['volume'] for candle in candles if candle['close'] > candle['open'])
    bearish_volume = sum(candle['volume'] for candle in candles if candle['close'] < candle['open'])
    total_volume = bullish_volume + bearish_volume
    bullish_ratio = (bullish_volume / total_volume) * 100 if total_volume > 0 else 0.0
    bearish_ratio = (bearish_volume / total_volume) * 100 if total_volume > 0 else 0.0

    return (
        lowest_low,
        highest_high,
        closest_minor_dip,
        closest_minor_top,
        closest_major_dip,
        closest_major_top,
        bullish_ratio,
        bearish_ratio,
    )

# Function to retrieve astrological data for a specific datetime
def get_astrological_data(reversal_datetime):
    observer = ephem.Observer()
    observer.lat = '40.7128'  # Latitude for New York
    observer.lon = '-74.0060'  # Longitude for New York
    observer.elev = 0          # Elevation

    observer.date = reversal_datetime  # Set the date for the observer

    planets = {
        'Sun': ephem.Sun(observer),
        'Moon': ephem.Moon(observer),
        'Mercury': ephem.Mercury(observer),
        'Venus': ephem.Venus(observer),
        'Mars': ephem.Mars(observer),
        'Jupiter': ephem.Jupiter(observer),
        'Saturn': ephem.Saturn(observer),
        'Uranus': ephem.Uranus(observer),
        'Neptune': ephem.Neptune(observer),
        'Pluto': ephem.Pluto(observer),
    }

    data = {}
    for planet_name, planet in planets.items():
        data[planet_name] = {
            'azimuth': planet.az,
            'altitude': planet.alt
        }
    return data

# Function to get current status of astrological data and ratios
def get_current_astro_status():
    current_time = ephem.now()  # Current time
    cycle_start_time = ephem.Date(current_time) - 30  # Start 30 days ago
    cycle_end_time = ephem.Date(current_time) + 30   # End in 30 days

    # Retrieve astrological data for the start and end of the cycle
    start_data = get_astrological_data(cycle_start_time)
    end_data = get_astrological_data(cycle_end_time)
    current_data = get_astrological_data(current_time)

    ratio_results = {}

    for planet in current_data:
        start_azimuth = start_data[planet]['azimuth']
        end_azimuth = end_data[planet]['azimuth']
        current_azimuth = current_data[planet]['azimuth']

        start_altitude = start_data[planet]['altitude']
        end_altitude = end_data[planet]['altitude']
        current_altitude = current_data[planet]['altitude']

        # Calculate distances
        azimuth_distance_to_start = abs(current_azimuth - start_azimuth)
        azimuth_distance_to_end = abs(end_azimuth - current_azimuth)
        altitude_distance_to_start = abs(current_altitude - start_altitude)
        altitude_distance_to_end = abs(end_altitude - current_altitude)

        total_distance_azimuth = azimuth_distance_to_start + azimuth_distance_to_end
        total_distance_altitude = altitude_distance_to_start + altitude_distance_to_end
        
        # Normalize ratios
        ratio_azimuth_start = (azimuth_distance_to_start / total_distance_azimuth) * 100 if total_distance_azimuth != 0 else 0
        ratio_azimuth_end = (azimuth_distance_to_end / total_distance_azimuth) * 100 if total_distance_azimuth != 0 else 0
        
        ratio_altitude_start = (altitude_distance_to_start / total_distance_altitude) * 100 if total_distance_altitude != 0 else 0
        ratio_altitude_end = (altitude_distance_to_end / total_distance_altitude) * 100 if total_distance_altitude != 0 else 0

        ratio_results[planet] = {
            'current_azimuth': current_azimuth,
            'current_altitude': current_altitude,
            'ratio_azimuth_start': ratio_azimuth_start,
            'ratio_azimuth_end': ratio_azimuth_end,
            'ratio_altitude_start': ratio_altitude_start,
            'ratio_altitude_end': ratio_altitude_end,
        }
    
    return ratio_results

# Example Usage
def main():
    TRADE_SYMBOL = "BTCUSDC"
    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    # Fetch candlestick data with adjustments for all timeframes
    candles_by_timeframe = fetch_candles_in_parallel(TIMEFRAMES, symbol=TRADE_SYMBOL)

    # Get current BTC price
    current_close = get_current_btc_price()
    print(f"Current BTC Price: {current_close}")

    # Check each timeframe for major and minor reversals
    for timeframe, candles in candles_by_timeframe.items():
        if not candles:  # Only process non-empty candle data
            print(f"No candles found for timeframe: {timeframe}.")
            continue
        
        print(f"\nTimeframe: {timeframe}")
        lowest_low, highest_high, closest_minor_dip, closest_minor_top, closest_major_dip, closest_major_top, bullish_ratio, bearish_ratio = find_reversals_and_volume_ratios(candles, current_close)
        
        print(f"Lowest Low: {lowest_low}, Highest High: {highest_high}")

        # Print information about the closest minor dip
        if closest_minor_dip is not None:
            reversal_time = closest_minor_dip['time']
            print(f"Closest Minor Reversal Dip: {closest_minor_dip}, Price: {closest_minor_dip['low']}")
            # Retrieve and print astrological data for the minor reversal datetime
            reversal_datetime = ephem.Date(reversal_time)
            astro_data_minor_dip = get_astrological_data(reversal_datetime)
            print("Astrological Data for Minor Dip:")
            for planet, data in astro_data_minor_dip.items():
                print(f"{planet} - Azimuth: {data['azimuth']}, Altitude: {data['altitude']}")

        # Print information about the closest minor top
        if closest_minor_top is not None:
            reversal_time = closest_minor_top['time']
            print(f"Closest Minor Reversal Top: {closest_minor_top}, Price: {closest_minor_top['high']}")
            # Retrieve and print astrological data for the minor reversal datetime
            reversal_datetime = ephem.Date(reversal_time)
            astro_data_minor_top = get_astrological_data(reversal_datetime)
            print("Astrological Data for Minor Top:")
            for planet, data in astro_data_minor_top.items():
                print(f"{planet} - Azimuth: {data['azimuth']}, Altitude: {data['altitude']}")

        # Print information about the closest major dip
        if closest_major_dip is not None:
            reversal_time = closest_major_dip['time']
            print(f"Closest Major Reversal Dip: {closest_major_dip}, Price: {closest_major_dip['low']}")
            # Retrieve and print astrological data for the major reversal datetime
            reversal_datetime = ephem.Date(reversal_time)
            astro_data_major_dip = get_astrological_data(reversal_datetime)
            print("Astrological Data for Major Dip:")
            for planet, data in astro_data_major_dip.items():
                print(f"{planet} - Azimuth: {data['azimuth']}, Altitude: {data['altitude']}")

        # Print information about the closest major top
        if closest_major_top is not None:
            reversal_time = closest_major_top['time']
            print(f"Closest Major Reversal Top: {closest_major_top}, Price: {closest_major_top['high']}")
            # Retrieve and print astrological data for the major reversal datetime
            reversal_datetime = ephem.Date(reversal_time)
            astro_data_major_top = get_astrological_data(reversal_datetime)
            print("Astrological Data for Major Top:")
            for planet, data in astro_data_major_top.items():
                print(f"{planet} - Azimuth: {data['azimuth']}, Altitude: {data['altitude']}")

        print(f"Bullish Volume Ratio: {bullish_ratio:.2f}%, Bearish Volume Ratio: {bearish_ratio:.2f}%")

        # Determine if the current timeframe is more bullish or bearish
        if bullish_ratio > bearish_ratio:
            print("Current timeframe is more bullish.")
        elif bearish_ratio > bullish_ratio:
            print("Current timeframe is more bearish.")
        else:
            print("Current timeframe is neutral.")
    
    # Get and print the current status of the astrological data
    current_astro_status = get_current_astro_status()
    print("\nCurrent Astrological Status:")
    for planet, data in current_astro_status.items():
        print(f"{planet} - Current Azimuth: {data['current_azimuth']}, Current Altitude: {data['current_altitude']}")
        print(f"Azimuth Start Ratio: {data['ratio_azimuth_start']:.2f}%, Azimuth End Ratio: {data['ratio_azimuth_end']:.2f}%")
        print(f"Altitude Start Ratio: {data['ratio_altitude_start']:.2f}%, Altitude End Ratio: {data['ratio_altitude_end']:.2f}%")

if __name__ == "__main__":
    main()
