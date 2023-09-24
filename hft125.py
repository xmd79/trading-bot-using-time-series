##################################################
##################################################

# Start code:

print()

##################################################
##################################################

print("Init code: ")
print()

##################################################
##################################################

print("Test")
print()

##################################################
##################################################

# Import modules:

import math
import time
import numpy as np
import hashlib
import requests
import hmac
import talib
import json
import datetime
from datetime import timedelta
from decimal import Decimal
import decimal
import random
import statistics
from statistics import mean
import scipy.fftpack as fftpack
import gc

##################################################
##################################################

# binance module imports
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *

##################################################
##################################################

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

##################################################
##################################################

# Define a function to get the account balance in BUSD

def get_account_balance():
    accounts = client.futures_account_balance()
    for account in accounts:
        if account['asset'] == 'USDT':
            bUSD_balance = float(account['balance'])
            return bUSD_balance

# Get the USDT balance of the futures account
bUSD_balance = float(get_account_balance())

# Print account balance
print("USDT Futures balance:", bUSD_balance)
print()

##################################################
##################################################

# Define Binance client reading api key and secret from local file:

def get_binance_client():
    # Read credentials from file    
    with open("credentials.txt", "r") as f:   
         lines = f.readlines()
         api_key = lines[0].strip()  
         api_secret = lines[1].strip()  
          
    # Instantiate client        
    client = BinanceClient(api_key, api_secret)
          
    return client

# Call the function to get the client  
client = get_binance_client()

##################################################
##################################################

# Initialize variables for tracking trade state:

TRADE_SYMBOL = "BTCUSDT"

##################################################
##################################################

# Define timeframes and get candles:

timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',  '6h', '8h', '12h', '1d']

def get_candles(symbol, timeframes):
    candles = []
    for timeframe in timeframes:
        limit = 10000  # default limit
        tf_value = int(timeframe[:-1])  # extract numeric value of timeframe
        if tf_value >= 4:  # check if timeframe is 4h or above
            limit = 20000  # increase limit for 4h timeframe and above
        klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        # Convert klines to candle dict
        for k in klines:
            candle = {
                "time": k[0] / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "timeframe": timeframe
            }
            candles.append(candle)
    return candles

# Get candles  
candles = get_candles(TRADE_SYMBOL, timeframes) 
#print(candles)

# Organize candles by timeframe        
candle_map = {}  
for candle in candles:
    timeframe = candle["timeframe"]  
    candle_map.setdefault(timeframe, []).append(candle)

#print(candle_map)

##################################################
##################################################

def get_latest_candle(symbol, interval, start_time=None):
    """Retrieve the latest candle for a given symbol and interval"""
    if start_time is None:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=1)
    else:
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, limit=1)
    candle = {
        "time": klines[0][0],
        "open": float(klines[0][1]),
        "high": float(klines[0][2]),
        "low": float(klines[0][3]),
        "close": float(klines[0][4]),
        "volume": float(klines[0][5]),
        "timeframe": interval
    }
    return candle

##################################################
##################################################

# Get current price as <class 'float'>

def get_price(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
        params = {
            "symbol": symbol 
            }
        response = requests.get(url, params=params)
        data = response.json()
        if "price" in data:
            price = float(data["price"])
        else:
            raise KeyError("price key not found in API response")
        return price      
    except (BinanceAPIException, KeyError) as e:
        print(f"Error fetching price for {symbol}: {e}")
        return 0

price = get_price("BTCUSDT")

print(price)

print()

##################################################
##################################################

# Get entire list of close prices as <class 'list'> type
def get_close(timeframe):
    closes = []
    candles = candle_map[timeframe]

    for c in candles:
        close = c['close']
        if not np.isnan(close):
            closes.append(close)

    # Append current price to the list of closing prices
    current_price = get_price(TRADE_SYMBOL)
    closes.append(current_price)

    return closes

close = get_close('1m')
#print(close)

##################################################
##################################################

# Get entire list of close prices as <class 'list'> type

def get_closes(timeframe):
    closes = []
    candles = candle_map[timeframe]
    
    for c in candles:
        close = c['close']
        if not np.isnan(close):     
            closes.append(close)
            
    return closes

closes = get_closes('1m')

#print(closes)

print()

##################################################
##################################################

def calculate_volume(candles):
    total_volume = sum(candle["volume"] for candle in candles)
    return total_volume

def get_volume_5min(candles):
    total_volume_5min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "5m")
    return total_volume_5min

def get_volume_3min(candles):
    total_volume_3min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "3m")
    return total_volume_3min

def get_volume_1min(candles):
    total_volume_1min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "1m")
    return total_volume_1min

def calculate_buy_sell_volume(candles):
    buy_volume_5min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "5m" and candle["close"] > candle["open"])
    sell_volume_5min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "5m" and candle["close"] < candle["open"])
    buy_volume_3min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "3m" and candle["close"] > candle["open"])
    sell_volume_3min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "3m" and candle["close"] < candle["open"])
    buy_volume_1min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "1m" and candle["close"] > candle["open"])
    sell_volume_1min = sum(candle["volume"] for candle in candles if candle["timeframe"] == "1m" and candle["close"] < candle["open"])

    return buy_volume_5min, sell_volume_5min, buy_volume_3min, sell_volume_3min , buy_volume_1min, sell_volume_1min 

def calculate_support_resistance(candles):
    support_levels_1min = []
    resistance_levels_1min = []
    support_levels_3min = []
    resistance_levels_3min = []
    support_levels_5min = []
    resistance_levels_5min = []

    timeframes = ["1m", "3m", "5m"]
    for timeframe in timeframes:
        close_prices = [candle["close"] for candle in candles if candle["timeframe"] == timeframe]
        min_close = min(close_prices)
        max_close = max(close_prices)
        price_range = max_close - min_close
        support_spread = price_range / 11  # Divide the price range into 11 equal parts for 10 support and resistance levels
        resistance_spread = price_range / 11

        if timeframe == "1m":
            support_levels_1min.append(min_close - support_spread)
            resistance_levels_1min.append(max_close + resistance_spread)
            for _ in range(4):
                support_levels_1min.append(support_levels_1min[-1] - support_spread)
                resistance_levels_1min.append(resistance_levels_1min[-1] + resistance_spread)
        elif timeframe == "3m":
            support_levels_3min.append(min_close - support_spread)
            resistance_levels_3min.append(max_close + resistance_spread)
            for _ in range(4):
                support_levels_3min.append(support_levels_3min[-1] - support_spread)
                resistance_levels_3min.append(resistance_levels_3min[-1] + resistance_spread)
        else:
            support_levels_5min.append(min_close - support_spread)
            resistance_levels_5min.append(max_close + resistance_spread)
            for _ in range(4):
                support_levels_5min.append(support_levels_5min[-1] - support_spread)
                resistance_levels_5min.append(resistance_levels_5min[-1] + resistance_spread)

    return support_levels_1min, resistance_levels_1min, support_levels_3min, resistance_levels_3min, support_levels_5min, resistance_levels_5min

def calculate_reversal_keypoints(levels, leverage):
    reversal_points = []
    for level in levels:
        reversal = level * (1 - 0.1 * leverage)
        reversal_points.append(reversal)
    return reversal_points

def get_higher_timeframe_data(symbol, higher_timeframe):
    higher_candles = get_candles(symbol, [higher_timeframe])

    if not higher_candles or higher_timeframe not in higher_candles:
        return [], []

    higher_support_levels, higher_resistance_levels = calculate_support_resistance(higher_candles[higher_timeframe])
    return higher_support_levels, higher_resistance_levels

def calculate_bollinger_bands(candles, window=20, num_std_dev=2):
    close_prices = [candle["close"] for candle in candles if candle["timeframe"] == "5m"]
    rolling_mean = calculate_rolling_mean(close_prices, window)
    rolling_std = calculate_rolling_std(close_prices, window)
    upper_band = [mean + (std_dev * num_std_dev) for mean, std_dev in zip(rolling_mean, rolling_std)]
    lower_band = [mean - (std_dev * num_std_dev) for mean, std_dev in zip(rolling_mean, rolling_std)]
    return upper_band, lower_band

def calculate_rolling_mean(data, window):
    rolling_sum = [sum(data[i - window + 1:i + 1]) for i in range(window - 1, len(data))]
    return [sum / window for sum in rolling_sum]

def calculate_rolling_std(data, window):
    rolling_std = []
    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1:i + 1]
        mean = sum(window_data) / window
        squared_diffs = [(x - mean) ** 2 for x in window_data]
        std_dev = (sum(squared_diffs) / window) ** 0.5
        rolling_std.append(std_dev)
    return rolling_std

def calculate_poly_channel(candles, window=20):
    close_prices = [candle["close"] for candle in candles if candle["timeframe"] == "1m"]
    poly_channel = np.polyfit(range(len(close_prices)), close_prices, 1)
    channel = np.polyval(poly_channel, range(len(close_prices)))
    upper_channel = channel + np.std(channel) * window
    lower_channel = channel - np.std(channel) * window
    return upper_channel.tolist(), lower_channel.tolist()

total_volume = calculate_volume(candles)
buy_volume_5min, sell_volume_5min, buy_volume_3min, sell_volume_3min , buy_volume_1min, sell_volume_1min = calculate_buy_sell_volume(candles)

(support_levels_1min, resistance_levels_1min, support_levels_3min, resistance_levels_3min, support_levels_5min, resistance_levels_5min) = calculate_support_resistance(candles)

total_volume_5min = get_volume_5min(candles)
total_volume_3min = get_volume_3min(candles)
total_volume_1min = get_volume_1min(candles)

small_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 2)
medium_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 5)
large_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 10)

small_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 2)
medium_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 5)
large_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 10)

small_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 2)
medium_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 5)
large_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 10)

higher_support_5min, higher_resistance_5min = get_higher_timeframe_data(TRADE_SYMBOL, "5m")

print("Total Volume:", total_volume)
print("Total Volume (5min tf):", total_volume_5min)

print()

print("Buy Volume (5min tf):", buy_volume_5min)
print("Sell Volume (5min tf):", sell_volume_5min)

print()


print("Buy Volume (1min tf):", buy_volume_1min)
print("Sell Volume (1min tf):", sell_volume_1min)

print()

# Print support and resistance levels for the 5-minute timeframe
print("Support Levels (5min tf):", support_levels_5min[-1])
print("Resistance Levels (5min tf):", resistance_levels_5min[-1])

# Print support and resistance levels for the 3-minute timeframe
print("Support Levels (3min tf):", support_levels_3min[-1])
print("Resistance Levels (3min tf):", resistance_levels_3min[-1])

# Calculate and print support and resistance levels for the 1-minute timeframe
print("Support Levels (1min tf):", support_levels_1min[-1])
print("Resistance Levels (1min tf):", resistance_levels_1min[-1])

support_levels_modified = [min(support, candles[-1]["close"]) for support in support_levels_5min]
resistance_levels_modified = [max(resistance, candles[-1]["close"]) for resistance in resistance_levels_5min]

# Calculate Bollinger Bands and Poly Channel for 5-minute timeframe
upper_bb_5min, lower_bb_5min = calculate_bollinger_bands(candles)
upper_poly_5min, lower_poly_5min = calculate_poly_channel(candles)

# Calculate the spread factor and number of levels
spread_factor = 0.02
num_levels = 5

# Calculate modified support and resistance levels with spread and additional levels
price = candles[-1]["close"]
support_spread = price * spread_factor
resistance_spread = price * spread_factor

# Select the appropriate support and resistance levels based on the desired timeframe
desired_timeframe = "5m"

if desired_timeframe == "1m":
    support_levels_selected, resistance_levels_selected = support_levels_1min, resistance_levels_1min
elif desired_timeframe == "3m":
    support_levels_selected, resistance_levels_selected = support_levels_3min, resistance_levels_3min
else:
    support_levels_selected, resistance_levels_selected = support_levels_5min, resistance_levels_5min

# Modify the selected support and resistance levels
support_levels_modified = [min(support, candles[-1]["close"]) for support in support_levels_selected]
resistance_levels_modified = [max(resistance, candles[-1]["close"]) for resistance in resistance_levels_selected]

# Calculate modified support and resistance levels with spread and additional levels
modified_support_levels = [price - i * support_spread for i in range(num_levels, 0, -1)]
modified_resistance_levels = [price + i * resistance_spread for i in range(num_levels)]

# Rule for identifying reversal dips and tops
if price <= lower_bb_5min[-1] and buy_volume_5min > sell_volume_5min and modified_support_levels and modified_resistance_levels:
    if all(level < small_lvrg_levels_5min[0] for level in modified_support_levels) and all(level < medium_lvrg_levels_5min[0] for level in modified_support_levels) and all(level < large_lvrg_levels_5min[0] for level in modified_support_levels):
        print("Potential Reversal Dip (5min): Close at or below Bollinger Bands Lower Band and More Buy Volume at Support")
    elif buy_volume_5min > sell_volume_5min:
        print("Potential Reversal Dip (5min): Close at or below Bollinger Bands Lower Band")

if price >= upper_bb_5min[-1] and sell_volume_5min > buy_volume_5min and modified_support_levels and modified_resistance_levels:
    if all(level > small_lvrg_levels_5min[0] for level in modified_resistance_levels) and all(level > medium_lvrg_levels_5min[0] for level in modified_resistance_levels) and all(level > large_lvrg_levels_5min[0] for level in modified_resistance_levels):
        print("Potential Reversal Top (5min): Close at or above Bollinger Bands Upper Band and More Sell Volume at Resistance")
    elif sell_volume_5min > buy_volume_5min:
        print("Potential Reversal Top (5min): Close at or above Bollinger Bands Upper Band")

print()

##################################################
##################################################

import datetime
import pytz
import ephem
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time

def get_moon_phase_momentum(current_time):
    # Set up timezone information
    tz = pytz.timezone('Etc/GMT-3')  # Use 'Etc/GMT-3' for UTC+3
    current_time = tz.normalize(current_time.astimezone(tz))
    current_date = current_time.date()
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # Calculate the moon phase for the current date
    moon = ephem.Moon(current_date)
    moon_phase = moon.phase
    
    # Calculate the moon age in days
    previous_new_moon = ephem.previous_new_moon(current_date)
    previous_new_moon_datetime = ephem.Date(previous_new_moon).datetime()
    previous_new_moon_datetime = previous_new_moon_datetime.replace(tzinfo=pytz.timezone('Etc/GMT-3'))
    moon_age = (current_time - previous_new_moon_datetime).days
    
    # Calculate the current moon sign
    sun = ephem.Sun(current_time)
    moon_sign = ephem.constellation(ephem.Moon(current_time))[1]
    
    # Calculate the moon's position
    moon.compute(current_time)
    moon_ra = moon.ra
    moon_dec = moon.dec
    
    # Calculate the moon's distance from earth in kilometers
    moon_distance_km = moon.earth_distance * ephem.meters_per_au / 1000
    
    # Calculate the moon's angular diameter in degrees
    moon_angular_diameter = math.degrees(moon.size / moon_distance_km)
    
    # Calculate the moon's speed in kilometers per hour
    moon_speed_km_hr = moon_distance_km / (1 / 24)
    
    # Calculate the moon's energy level
    moon_energy = (moon_phase / 100) ** 2
    
    # Calculate the astrological map for the current time
    map_data = get_astro_map_data(current_time)
    
    # Create a dictionary to hold all the data
    moon_data = {
        'moon_phase': moon_phase,
        'moon_age': moon_age,
        'moon_sign': moon_sign,
        'moon_ra': moon_ra,
        'moon_dec': moon_dec,
        'moon_distance_km': moon_distance_km,
        'moon_angular_diameter': moon_angular_diameter,
        'moon_speed_km_hr': moon_speed_km_hr,
        'moon_energy': moon_energy,
        'astro_map': map_data
    }
    
    return moon_data

##################################################
##################################################

def get_astro_map_data(current_time):
    # Set up timezone information
    tz = pytz.timezone('Etc/GMT-3')  # Use 'Etc/GMT-3' for UTC+3
    current_time = tz.normalize(current_time.astimezone(tz))
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # Calculate the ascendant and midheaven signs
    obs = ephem.Observer()
    obs.lon = '-118.248405'
    obs.lat = '34.052187'
    obs.date = current_time
    obs.pressure = 0
    obs.horizon = '-0:34'
    sun = ephem.Sun(obs)
    sun.compute(obs)
    moon = ephem.Moon(obs)
    moon.compute(obs)
    
    # Create a FixedBody object from the observer's coordinates
    fixed_body = ephem.FixedBody()
    fixed_body._ra = obs.sidereal_time()
    fixed_body._dec = obs.lat
    
    # Calculate the position of the fixed body
    fixed_body.compute(current_time)
    
    # Calculate the ascendant and midheaven signs
    asc = ephem.constellation(fixed_body)[1]
    vega = ephem.star('Vega')
    vega.compute(current_time)  # Compute the position of the Vega star
    mc = ephem.constellation(vega)[1]
    
    # Create a dictionary to hold the data
    astro_map_data = {
        'ascendant': asc,
        'midheaven': mc,
        'sun': {
            'sign': ephem.constellation(sun)[1],
            'degree': math.degrees(sun.ra)
        },
        'moon': {
            'sign': ephem.constellation(moon)[1],
            'degree': math.degrees(moon.ra)
        }
    }
    
    return astro_map_data

##################################################
##################################################

def get_planet_positions():
    now = Time.now()
    
    planet_positions = {}
    sun_position = {}
    
    planets = [
        {'name': 'Mercury', 'id': '1'},
        {'name': 'Venus', 'id': '2'},
        {'name': 'Mars', 'id': '4'},
        {'name': 'Jupiter', 'id': '5'},
        {'name': 'Saturn', 'id': '6'}
    ]
    
    for planet in planets:
       obj = Horizons(id=planet['id'], location='500', epochs=now.jd)  
       eph = obj.ephemerides()[0] 
       planet_positions[planet['name']] = {'RA': eph['RA'], 'DEC': eph['DEC']}
       
    obj = Horizons(id='10', location='500', epochs=now.jd)  
    eph = obj.ephemerides()[0]  
    sun_position['RA'] = eph['RA']  
    sun_position['DEC'] = eph['DEC']
        
    return planet_positions, sun_position

planet_positions, sun_position = get_planet_positions()

##################################################
##################################################

def get_vedic_houses(date, observer):
    # Convert datetime to ephem date
    date_ephem = ephem.Date(date)

    # Set up ephem observer object
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = date_ephem

    # Calculate the sidereal time at the observer's location
    sidereal_time = obs.sidereal_time()

    # Calculate the Local Sidereal Time (LST)
    lst_hours = sidereal_time * 24 / (2 * ephem.pi)
    lst_deg = (lst_hours * 15) % 360

    # Calculate the ascendant degree
    asc_deg = (lst_deg + 180) % 360

    # Calculate the house cusps
    house_cusps = []
    for i in range(1, 13):
        cusp_deg = (asc_deg + (i - 1) * 30) % 360
        cusp_sign = get_vedic_sign(cusp_deg)
        house_cusps.append((i, cusp_sign, cusp_deg))

    house_cusps_dict = {house: (sign, deg) for house, sign, deg in house_cusps}
    return house_cusps_dict



##################################################
##################################################

def get_vedic_sign(deg):
    deg = (deg + 360) % 360
    if deg >= 0 and deg < 30:
        return 'Aries'
    elif deg >= 30 and deg < 60:
        return 'Taurus'
    elif deg >= 60 and deg < 90:
        return 'Gemini'
    elif deg >= 90 and deg < 120:
        return 'Cancer'
    elif deg >= 120 and deg < 150:
        return 'Leo'
    elif deg >= 150 and deg < 180:
        return 'Virgo'
    elif deg >= 180 and deg < 210:
        return 'Libra'
    elif deg >= 210 and deg < 240:
        return 'Scorpio'
    elif deg >= 240 and deg < 270:
        return 'Sagittarius'
    elif deg >= 270 and deg < 300:
        return 'Capricorn'
    elif deg >= 300 and deg < 330:
        return 'Aquarius'
    elif deg >= 330 and deg < 360:
        return 'Pisces'

##################################################
##################################################

# Define list of stars
stars = [
    ('Sun', ephem.Sun(), ''),    
    ('Polaris', '02:31:49.09', '+89:15:50.8'), 
    ('Vega', '18:36:56.34', '+38:47:01.3'),
    ('Betelgeuse', '05:55:10.31', '+07:24:25.4'),
    ('Rigel', '05:14:32.28', '-08:12:05.9'),  
    ('Achernar', '01:37:42.84', '-57:14:12.3'),
    ('Hadar', '14:03:49.40', '-60:22:22.3'),
    ('Altair', '19:50:46.99', '+08:52:05.9'),
    ('Deneb', '20:41:25.91', '+45:16:49.2')
]

##################################################
##################################################

def get_star_positions(date, observer):
    # Set up ephem observer object
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = ephem.Date(date)

    # Get positions of stars in list
    star_positions = []
    for star in stars:
        # Set up ephem star object
        fixed_body = ephem.FixedBody()
        fixed_body._ra = star[1]  # Set right ascension
        fixed_body._dec = star[2]  # Set declination

        # Calculate position of star for current date/time and observer location
        fixed_body.compute(obs)

        # Convert right ascension and declination to degrees
        ra_deg = math.degrees(fixed_body.ra)
        dec_deg = math.degrees(fixed_body.dec)

        # Append star name and position to list
        star_positions.append((star[0], ra_deg, dec_deg))

    return star_positions

##################################################
##################################################

# Set up the current time
current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=3)

##################################################
##################################################

# Get the moon data
moon_data = get_moon_phase_momentum(current_time)

# Print the moon data
print('Moon phase:', moon_data['moon_phase'])
print('Moon age:', moon_data['moon_age'])
print('Moon sign:', moon_data['moon_sign'])
print('Moon right ascension:', moon_data['moon_ra'])
print('Moon declination:', moon_data['moon_dec'])
print('Moon distance from Earth (km):', moon_data['moon_distance_km'])
print('Moon angular diameter:', moon_data['moon_angular_diameter'])
print('Moon speed (km/hr):', moon_data['moon_speed_km_hr'])
print('Moon energy level:', moon_data['moon_energy'])
print('Ascendant sign:', moon_data['astro_map']['ascendant'])
print('Midheaven sign:', moon_data['astro_map']['midheaven'])
print('Sun sign:', moon_data['astro_map']['sun']['sign'])
print('Sun degree:', moon_data['astro_map']['sun']['degree'])
print('Moon sign:', moon_data['astro_map']['moon']['sign'])
print('Moon degree:', moon_data['astro_map']['moon']['degree'])

print()
  
##################################################
##################################################

# Calculate fixed_body
obs = ephem.Observer()

# Set observer info
fixed_body = ephem.FixedBody()  
fixed_body._ra = obs.sidereal_time()
fixed_body._dec = obs.lat

# Call get_vedic_houses(), passing fixed_body 
observer = {
    'longitude': '-118.248405',
    'latitude': '34.052187'
}
vedic_houses = get_vedic_houses(current_time, observer)

print()

# Compute fixed_body position
fixed_body.compute(current_time)

# Print results 
for house, sign in vedic_houses.items():
    print(f"House {house}: {sign}")

print()

# Print results
for house in range(1,13):
    sign = vedic_houses[house]
    print(f"Vedic House {house}: {sign}")

print()

print("Full Results:")

for house, (sign, deg) in vedic_houses.items():
    print(f"House {house} - {sign} at {deg:.2f} degrees")

print()

##################################################
##################################################

from astroquery.jplhorizons import Horizons
from astropy.time import Time

def get_planet_positions():
    # Define the list of planets to retrieve positions for
    planets = [{'name': 'Mercury', 'id': '1'},
               {'name': 'Venus', 'id': '2'},
               {'name': 'Mars', 'id': '4'},
               {'name': 'Jupiter', 'id': '5'},
               {'name': 'Saturn', 'id': '6'},
               {'name': 'Uranus', 'id': '7'},
               {'name': 'Neptune', 'id': '8'}]

    # Get the current date and time in UTC
    now = Time.now()

    # Create empty dictionaries to store the planet and sun positions
    planet_positions = {}
    sun_position = {}

    # Loop through each planet and retrieve its position
    for planet in planets:
        # Query the JPL Horizons database to get the planet's position
        obj = Horizons(id=planet['id'], location='500', epochs=now.jd)
        eph = obj.ephemerides()[0]

        # Store the position in the dictionary
        planet_positions[planet['name']] = {'RA': eph['RA'], 'DEC': eph['DEC']}

    # Retrieve the position of the Sun
    obj = Horizons(id='10', location='500', epochs=now.jd)
    eph = obj.ephemerides()[0]

    # Store the position in the dictionary
    sun_position['RA'] = eph['RA']
    sun_position['DEC'] = eph['DEC']

    # Return the dictionaries of planet and sun positions
    return planet_positions, sun_position

# Call the function to retrieve the planet and sun positions
planet_positions, sun_position = get_planet_positions()

# Print the positions in a detailed format
print('Planet Positions:')
for planet_name, position in planet_positions.items():
    print('{}\n\tRA: {}\n\tDEC: {}'.format(planet_name, position['RA'], position['DEC']))
    
print('Sun Position:')
print('\tRA: {}\n\tDEC: {}'.format(sun_position['RA'], sun_position['DEC']))

print()

##################################################
##################################################

# Function to convert degrees to hours, minutes, seconds
def deg_to_hours(deg_str):
    deg, minute, sec = deg_str.split(':') 
    degrees = float(deg)
    minutes = float(minute) / 60  
    seconds = float(sec) / 3600    
    return degrees + minutes + seconds

##################################################
##################################################

def get_star_positions_from_sun(date):
    sun = ephem.Sun()
    sun.compute(date)
    
    obs = ephem.Observer()       
    obs.lon = math.degrees(sun.a_ra)   
    obs.lat = math.degrees(sun.a_dec)        
    obs.date = ephem.Date(date)
    
    star_positions = []    
            
    for star in stars:
        if star[0] == 'Sun':    
            star_ephem = ephem.Sun() 
        else:          
            if len(star) == 3 and star[2]:
               dec_deg = deg_to_hours(star[2])       
               fixed_body = ephem.FixedBody()        
               fixed_body._ra = star[1]
               fixed_body._dec = dec_deg
               star_ephem = fixed_body 
         
        star_ephem.compute(obs)       
       
        ra_deg = math.degrees(star_ephem.ra)     
        dec_deg = math.degrees(star_ephem.dec)      
        star_positions.append((star[0], ra_deg, dec_deg))
            
    return star_positions

##################################################
##################################################

date = datetime.datetime.now()
star_positions = get_star_positions_from_sun(date)

for name, ra, dec in star_positions:
    print(f"{name}: RA = {ra}, DEC = {dec}")  

print()

##################################################
##################################################

def get_observer():
    obs = ephem.Observer() 
    obs.lon = '21.22571'  # Longitude of Timișoara  
    obs.lat = '45.75372'  # Latitude of Timișoara
    obs.elevation = 102   # Elevation of Timișoara in meters
    obs.date = ephem.now()
    return obs

##################################################
##################################################

def get_current_aspects():
    # Create an observer at your location
    obs = get_observer()

    # Get the current date and time
    current_date = ephem.now()

    # Set the observer's date and time to the current date and time
    obs.date = current_date

    # Define the planets to check aspects for
    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                'Jupiter', 'Saturn', 'Uranus', 'Neptune']

    # Initialize list to store aspects    
    aspects = []

    # Loop through each planet           
    for planet in planets:
        p = getattr(ephem, planet)()
        p.compute(obs)  # Compute the object's fields

        # Calculate the angular separation from each other planet         
        for other_planet in planets:
            o = getattr(ephem, other_planet)()   
            o.compute(obs)  # Compute the object's fields

            # Compute the separation between the two objects
            p.compute(obs)
            o.compute(obs)
            separation = ephem.separation(p, o)
            separation_deg = ephem.degrees(separation)   
                        
            # Check if the planets form an aspect (within orb)                  
            if check_aspect(separation_deg):
                aspects.append((planet, other_planet, separation_deg))
                
    return aspects

##################################################
##################################################

def check_aspect(sep):
    orb = 6 # Degree orb for considering an aspect            
    return sep <= orb or 360-sep <= orb

print()

##################################################
##################################################

def get_predominant_frequencies(close):
    # Calculate the periodogram to find predominant frequencies
    import scipy.signal as signal
    frequencies, power = signal.periodogram(close)
    
    # Find the 3 largest peaks in the periodogram
    largest_peaks = np.argsort(power)[-3:][::-1] 
    peaks = frequencies[largest_peaks]
    
    # Map frequencies to timeframes
    timeframes = {
        peaks[0]: 'fast cycle',  # Shortest period
        peaks[1]: 'medium cycle',  
        peaks[2]: 'long cycle'
    } 
    
    return timeframes

##################################################
##################################################

def get_market_mood(close, aspects, vedic_houses):
    # Check for dip/top   
    if 'Moon' in [p[0] for p in aspects] or 'Saturn' in [p[0] for p in aspects]:
       # Reversal aspects - likely dip/top
       
        mood = determine_mood_from_aspects(aspects)
        mood = adjust_mood_from_houses(mood, vedic_houses)
      
    return mood

##################################################
##################################################

def determine_mood_from_aspects(aspects):
    mood = 'neutral'
      
    if any('Mars' in a for a in aspects):
       mood = 'aggressive'
        
    if any('Jupiter' in a for a in aspects):
       mood = 'optimistic'
       
    return mood

##################################################
##################################################

def adjust_mood_from_houses(mood, vedic_houses):  
    if 'Sun' in vedic_houses[1] or 'Jupiter' in vedic_houses[1]:
       mood = 'bullish' if mood == 'aggressive' else mood
       
    if 'Saturn' in vedic_houses[8] or 'Mars' in vedic_houses[8]:
       mood = 'bearish' if mood == 'optimistic' else mood
       
    return mood

##################################################
##################################################

def get_possible_reversals(aspects):
    reversals = []
    
    for a in aspects: 
        if a[2] <= 5: # Within 5 degree orb
            planet1 = a[0].lower()
            planet2 = a[1].lower()
            if planet1 == 'moon' or planet2 == 'moon':
                reversals.append(a)
            
    return reversals
  
print()

##################################################
##################################################

# Call the function to get the current aspects
aspects = get_current_aspects() 

print("Current aspects:")    
for planet1, planet2, separation in aspects:
    print(f"{planet1} aspecting {planet2} at {separation}°")     
      
frequencies = get_predominant_frequencies(close)

print()

print("Frequencies now: ", frequencies)  

mood = get_market_mood(close, aspects, vedic_houses)

print()

print("Market mood: ", mood)

reversals = get_possible_reversals(aspects)  

print()

print("Forecasted reversals now: ", reversals)

print()

##################################################
##################################################

# Scale current close price to sine wave       
def scale_to_sine(timeframe):  
  
    close_prices = np.array(get_close(timeframe))
  
    # Get last close price 
    current_close = close_prices[-1]      
        
    # Calculate sine wave        
    sine_wave, leadsine = talib.HT_SINE(close_prices)
            
    # Replace NaN values with 0        
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave
        
    # Get the sine value for last close      
    current_sine = sine_wave[-1]
            
    # Calculate the min and max sine           
    sine_wave_min = np.min(sine_wave)        
    sine_wave_max = np.max(sine_wave)

    # Calculate % distances            
    dist_min, dist_max = [], []
 
    for close in close_prices:    
        # Calculate distances as percentages        
        dist_from_close_to_min = ((current_sine - sine_wave_min) /  
                           (sine_wave_max - sine_wave_min)) * 100            
        dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                           (sine_wave_max - sine_wave_min)) * 100
                
        dist_min.append(dist_from_close_to_min)       
        dist_max.append(dist_from_close_to_max)

    # Take average % distances    
    avg_dist_min = sum(dist_min) / len(dist_min)
    avg_dist_max = sum(dist_max) / len(dist_max) 

    #print(f"{timeframe} Close is now at "       
          #f"dist. to min: {dist_from_close_to_min:.2f}% "
          #f"and at "
          #f"dist. to max: {dist_from_close_to_max:.2f}%")

    return dist_from_close_to_min, dist_from_close_to_max, current_sine

# Call function           
for timeframe in timeframes:        
    scale_to_sine(timeframe)

print()

##################################################
##################################################

def get_custom_sine_wave(close, moon_data, factor=0.2):
    close = np.array(close)  # Convert 'close' list to a numpy array

    # Get moon energy level
    moon_energy = moon_data['moon_energy']

    # Calculate the sine wave based on time values
    current_sine = -moon_energy  # Use the moon energy level as the current sine value
    sine_wave, _ = talib.HT_SINE(close)
    sine_wave = -np.nan_to_num(sine_wave)  # Replace NaN values with 0 and invert

    # Calculate the min and max sine values
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Scale the sine wave to the range of [min_close, max_close]
    min_close = np.min(close)
    max_close = np.max(close)
    scaled_sine_wave = min_close + (sine_wave - sine_wave_min) * (max_close - min_close) / (sine_wave_max - sine_wave_min)

    # Calculate the new custom sine value at the current time based on the astrological energy level
    custom_sine_value = min_close + (moon_energy - sine_wave_min) * (max_close - min_close) / (sine_wave_max - sine_wave_min)

    # Introduce a factor to control the magnitude of the changes
    scaled_sine_wave = custom_sine_value + factor * (scaled_sine_wave - custom_sine_value)

    # Calculate the min and max price values of the new custom sine wave
    min_price = min(min_close, np.min(scaled_sine_wave))  # Ensure minimum price is below the current close
    max_price = np.max(scaled_sine_wave)

    return min_price, max_price

##################################################
##################################################

def get_market_mood(close, moon_data, aspects, vedic_houses):
    # Check for dip/top   
    mood = 'neutral'

    if 'Moon' in [p[0] for p in aspects] or 'Saturn' in [p[0] for p in aspects]:
        # Reversal aspects - likely dip/top
        mood = determine_mood_from_aspects(aspects)
        mood = adjust_mood_from_houses(mood, vedic_houses)
      
    # Determine market direction based on sine wave
    current_close = close[-1]
    min_price, max_price = get_custom_sine_wave(close, moon_data, factor=0.2)

    if current_close < min_price:
        mood = 'bearish'
    elif current_close > max_price:
        mood = 'bullish'

    return mood

##################################################
##################################################

def determine_mood_from_aspects(aspects):
    mood = 'neutral'
    if any('Mars' in a for a in aspects):
        mood = 'aggressive'
    if any('Jupiter' in a for a in aspects):
        mood = 'optimistic'
    return mood

##################################################
##################################################

def adjust_mood_from_houses(mood, vedic_houses):  
    if 'Sun' in vedic_houses[1] or 'Jupiter' in vedic_houses[1]:
        mood = 'bullish' if mood == 'aggressive' else mood
    if 'Saturn' in vedic_houses[8] or 'Mars' in vedic_houses[8]:
        mood = 'bearish' if mood == 'optimistic' else mood
    return mood

##################################################
##################################################

import numpy as np
import talib
import datetime

def get_hft_targets(close, moon_data, aspects, vedic_houses, num_targets=10, time_interval_minutes=1):
    close = np.array(close)  # Convert 'close' list to a numpy array

    # Get moon energy level
    moon_energy = moon_data['moon_energy']

    # Calculate the sine wave based on time values
    current_sine = -moon_energy  # Use the moon energy level as the current sine value
    sine_wave, _ = talib.HT_SINE(close)
    sine_wave = -np.nan_to_num(sine_wave)  # Replace NaN values with 0 and invert

    # Calculate the min and max sine values
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Scale the sine wave to the range of [min_close, max_close]
    min_close = np.min(close)
    max_close = np.max(close)
    scaled_sine_wave = min_close + (sine_wave - sine_wave_min) * (max_close - min_close) / (sine_wave_max - sine_wave_min)

    # Calculate the new custom sine value at the current time based on the astrological energy level
    custom_sine_value = min_close + (moon_energy - sine_wave_min) * (max_close - min_close) / (sine_wave_max - sine_wave_min)

    # Use planetary positions for faster cycles
    planetary_positions = {
        'Mars': -20,
        'Venus': 35,
        'Jupiter': 70,
        'Saturn': -10,
        'Mercury': 25,
        'Uranus': 50,
        'Neptune': -5,
        'Pluto': 30,
        'Sun': 65,
        'Moon': 0
    }

    for planet, position in planetary_positions.items():
        if planet in vedic_houses[1] or planet in vedic_houses[8]:
            # Adjust the sine wave based on the planetary position
            scaled_sine_wave += position

    # Introduce a factor to control the magnitude of the changes
    scaled_sine_wave = custom_sine_value + 0.2 * (scaled_sine_wave - custom_sine_value)

    # Calculate the min and max price values of the new custom sine wave
    min_price = min(min_close, np.min(scaled_sine_wave))  # Ensure minimum price is below the current close
    max_price = np.max(scaled_sine_wave)

    # Calculate the reversal keypoint as the midpoint between min and max
    reversal_keypoint = (min_price + max_price) / 2

    # Determine market direction based on sine wave
    current_close = close[-1]
    market_mood = 'neutral'

    if current_close < min_price:
        market_mood = 'bearish'
    elif current_close > max_price:
        market_mood = 'bullish'

    # Calculate the dip and top prices (always below close for dip and above close for top)
    dip_price = min(reversal_keypoint, current_close)
    top_price = max(reversal_keypoint, current_close)

    # Calculate the incoming reversal keypoint for both dip and top
    incoming_reversal_keypoint = reversal_keypoint - (current_close - reversal_keypoint) if market_mood == 'bearish' else reversal_keypoint + (reversal_keypoint - current_close)

    # Determine if the incoming reversal is a dip or a top
    incoming_reversal = 'dip' if market_mood == 'bearish' else 'top'

    # Calculate the inner targets as percentages of the distance from the current close to the reversal keypoint
    inner_target_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Adjust these factors as per your preference

    # Determine if a dip or top is incoming based on market mood
    if market_mood == 'bearish':
        # If market mood is bearish, the path is to a dip, so inner targets are below the close
        inner_targets = [current_close - factor * (current_close - reversal_keypoint) for factor in inner_target_factors]
    else:
        # If market mood is bullish, the path is to a top, so inner targets are above the close
        inner_targets = [current_close + factor * (reversal_keypoint - current_close) for factor in inner_target_factors]

    # Calculate the next 'num_targets' targets for the next minutes
    current_time = datetime.datetime.utcnow()
    next_targets = []
    for i in range(1, num_targets + 1):
        target_time = current_time + datetime.timedelta(minutes=i * time_interval_minutes)
        target_price = (reversal_keypoint - current_close) * (num_targets + 1 - i) / (num_targets + 1) + current_close
        next_target = target_time, target_price
        next_targets.append(next_target)

    # Calculate new inner reversals for local micro cycles in between main reversals
    num_inner_reversals = 10
    inner_reversals = []
    for i in range(1, num_inner_reversals + 1):
        if market_mood == 'bearish':
            inner_reversal_price = current_close - i * (current_close - incoming_reversal_keypoint) / (num_inner_reversals + 1)
        else:
            inner_reversal_price = current_close + i * (incoming_reversal_keypoint - current_close) / (num_inner_reversals + 1)
        inner_reversal = target_time, inner_reversal_price
        inner_reversals.append(inner_reversal)

    return min_price, max_price, dip_price, top_price, inner_targets, market_mood, incoming_reversal_keypoint, incoming_reversal, next_targets, inner_reversals


##################################################
##################################################

# Example usage:
# Assuming you have already calculated the 'close' prices, 'moon_data', 'aspects', and 'vedic_houses'
min_price, max_price, dip_price, top_price, inner_targets, market_mood, incoming_reversal_keypoint, incoming_reversal, next_targets, inner_reversals = get_hft_targets(close, moon_data, aspects, vedic_houses)

print("Minimum Price:", min_price)
print("Maximum Price:", max_price)
print("Dip Price:", dip_price)
print("Top Price:", top_price)
print("Market Mood:", market_mood)
print("Incoming Reversal Keypoint:", incoming_reversal_keypoint)
print("Incoming Reversal:", incoming_reversal)

print()

print("Inner Targets:")
for i, target in enumerate(inner_targets, start=1):
    print(f"Inner Target {i}: {target}")

print()

##################################################
##################################################

def custom_sine_wave(timeframe):
    # Get close prices
    closes = get_closes(timeframe)
    
    # Get astrological data
    moon_data = get_moon_phase_momentum(datetime.datetime.utcnow())
    vedic_houses = get_vedic_houses(datetime.datetime.utcnow(), observer)
    aspects = get_current_aspects() 
    
    # Get market mood
    mood = get_market_mood(closes, aspects, vedic_houses)
            
    # Generate HT_SINE as before
    sine_wave, leadsine = talib.HT_SINE(closes)
    sine_wave = -sine_wave
    sine_wave_min = np.min(sine_wave)    
    sine_wave_max = np.max(sine_wave)   
        
    # Adjust max and min based on market mood            
    if mood == 'bullish': 
        sine_wave_max = sine_wave_max * 1.1        
    elif mood == 'bearish':     
        sine_wave_min = sine_wave_min * 0.95
        
    # Regenerate sine wave                 
    custom_sine, _ = talib.HT_SINE(closes, min=sine_wave_min, max=sine_wave_max)
      
    return custom_sine

##################################################
##################################################

def get_momentum(timeframe):
    """Calculate momentum for a single timeframe"""
    # Get candle data               
    candles = candle_map[timeframe][-100:]  
    # Calculate momentum using talib MOM
    momentum = talib.MOM(np.array([c["close"] for c in candles]), timeperiod=14)
    return momentum[-1]

##################################################
##################################################

# Calculate momentum for each timeframe
for timeframe in timeframes:
    momentum = get_momentum(timeframe)
    print(f"Momentum for {timeframe}: {momentum}")

print()

##################################################
##################################################

def generate_momentum_sinewave(timeframes):
    # Initialize variables
    momentum_sorter = []
    market_mood = []
    last_reversal = None
    last_reversal_value_on_sine = None
    last_reversal_value_on_price = None
    next_reversal = None
    next_reversal_value_on_sine = None
    next_reversal_value_on_price = None

    # Loop over timeframes
    for timeframe in timeframes:
        # Get close prices for current timeframe
        close_prices = np.array(get_closes(timeframe))

        # Get last close price
        current_close = close_prices[-1]

        # Calculate sine wave for current timeframe
        sine_wave, leadsine = talib.HT_SINE(close_prices)

        # Replace NaN values with 0
        sine_wave = np.nan_to_num(sine_wave)
        sine_wave = -sine_wave

        # Get the sine value for last close
        current_sine = sine_wave[-1]

        # Calculate the min and max sine
        sine_wave_min = np.nanmin(sine_wave) # Use nanmin to ignore NaN values
        sine_wave_max = np.nanmax(sine_wave)

        # Calculate price values at min and max sine
        sine_wave_min_price = close_prices[sine_wave == sine_wave_min][0]
        sine_wave_max_price = close_prices[sine_wave == sine_wave_max][0]
     

        # Calculate the difference between the max and min sine
        sine_wave_diff = sine_wave_max - sine_wave_min

        # If last close was the lowest, set as last reversal                                  
        if current_sine == sine_wave_min:
            last_reversal = 'dip'
            last_reversal_value_on_sine = sine_wave_min 
            last_reversal_value_on_price = sine_wave_min_price
        
        # If last close was the highest, set as last reversal                                 
        if current_sine == sine_wave_max:
            last_reversal = 'top'
            last_reversal_value_on_sine = sine_wave_max
            last_reversal_value_on_price = sine_wave_max_price

        # Calculate % distances
        newsine_dist_min, newsine_dist_max = [], []
        for close in close_prices:
            # Calculate distances as percentages
            dist_from_close_to_min = ((current_sine - sine_wave_min) /  
                                      sine_wave_diff) * 100            
            dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                                      sine_wave_diff) * 100
                
            newsine_dist_min.append(dist_from_close_to_min)       
            newsine_dist_max.append(dist_from_close_to_max)

        # Take average % distances
        avg_dist_min = sum(newsine_dist_min) / len(newsine_dist_min)
        avg_dist_max = sum(newsine_dist_max) / len(newsine_dist_max)

        # Determine market mood based on % distances
        if avg_dist_min <= 15:
            mood = "At DIP Reversal and Up to Bullish"
            if last_reversal != 'dip':
                next_reversal = 'dip'
                next_reversal_value_on_sine = sine_wave_min
                next_reversal_value_on_price = close_prices[sine_wave == sine_wave_min][0]
        elif avg_dist_max <= 15:
            mood = "At TOP Reversal and Down to Bearish"
            if last_reversal != 'top':
                next_reversal = 'top'
                next_reversal_value_on_sine = sine_wave_max
                next_reversal_value_on_price = close_prices[sine_wave == sine_wave_max][0]
        elif avg_dist_min < avg_dist_max:
            mood = "Bullish"
        else:
            mood = "Bearish"

        # Append momentum score and market mood to lists
        momentum_score = avg_dist_max - avg_dist_min
        momentum_sorter.append(momentum_score)
        market_mood.append(mood)

        # Print distances and market mood
        #print(f"{timeframe} Close is now at "       
              #f"dist. to min: {avg_dist_min:.2f}% "
              #f"and at "
              #f"dist. to max: {avg_dist_max:.2f}%. "
              #f"Market mood: {mood}")

        # Update last and next reversal info
        if next_reversal:
            last_reversal = next_reversal
            last_reversal_value_on_sine = next_reversal_value_on_sine
            last_reversal_value_on_price = next_reversal_value_on_price
            next_reversal = None
            next_reversal_value_on_sine = None
            next_reversal_value_on_price = None

    # Get close prices for the 1-minute timeframe and last 3 closes
    close_prices = np.array(get_closes('1m'))

    # Calculate sine wave
    sine_wave, leadsine = talib.HT_SINE(close_prices)

    # Replace NaN values with 0
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    # Get the sine value for last close
    current_sine = sine_wave[-1]

    # Get current date and time
    now = datetime.datetime.now()

    # Calculate the min and max sine
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate the difference between the maxand min sine
    sine_wave_diff = sine_wave_max - sine_wave_min

    # Calculate % distances
    dist_from_close_to_min = ((current_sine - sine_wave_min) / 
                              sine_wave_diff) * 100
    dist_from_close_to_max = ((sine_wave_max - current_sine) / 
                              sine_wave_diff) * 100

    # Determine market mood based on % distances
    if dist_from_close_to_min <= 15:
        mood = "At DIP Reversal and Up to Bullish"
        if last_reversal != 'dip':
            next_reversal = 'dip'
            next_reversal_value_on_sine = sine_wave_min

    elif dist_from_close_to_max <= 15:
        mood = "At TOP Reversal and Down to Bearish"
        if last_reversal != 'top':
            next_reversal = 'top'
            next_reversal_value_on_sine = sine_wave_max

    elif dist_from_close_to_min < dist_from_close_to_max:
        mood = "Bullish"
    else:
        mood = "Bearish"

    # Get the close prices that correspond to the min and max sine values
    close_prices_between_min_and_max = close_prices[(sine_wave >= sine_wave_min) & (sine_wave <= sine_wave_max)]

    print()

    # Print distances and market mood for 1-minute timeframe
    print(f"On 1min timeframe,Close is now at "       
          f"dist. to min: {dist_from_close_to_min:.2f}% "
          f"and at "
          f"dist. to max:{dist_from_close_to_max:.2f}%. "
          f"Market mood: {mood}")

    min_val = min(close_prices_between_min_and_max)
    max_val = max(close_prices_between_min_and_max)

    print("The lowest value in the array is:", min_val)
    print("The highest value in the array is:", max_val)

    print()

    # Update last and next reversal info
    #if next_reversal:
        #last_reversal = next_reversal
        #last_reversal_value_on_sine = next_reversal_value_on_sine

        #next_reversal = None
        #next_reversal_value_on_sine = None


    # Print last and next reversal info
    #if last_reversal:
        #print(f"Last reversal was at {last_reversal} on the sine wave at {last_reversal_value_on_sine:.2f} ")

    # Return the momentum sorter, market mood, close prices between min and max sine, and reversal info
    return momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max, min_val, max_val 
     
##################################################
##################################################

momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max, min_val, max_val = generate_momentum_sinewave(timeframes)

print()

#print("Close price values between last reversals on sine: ")
#print(close_prices_between_min_and_max)

print()

print("Current close on sine value now at: ", current_sine)
print("distances as percentages from close to min: ", dist_from_close_to_min, "%")
print("distances as percentages from close to max: ", dist_from_close_to_max, "%")
print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
print("Mood on 1min timeframe is now at: ", market_mood[-12])
print("The lowest close value in the array is:", min_val)
print("The highest close value in the array is:", max_val)



print()

##################################################
##################################################

def generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5):
    # Calculate the sine wave using HT_SINE
    sine_wave, _ = talib.HT_SINE(close_prices)
    
    # Replace NaN values with 0 using nan_to_num
    sine_wave = np.nan_to_num(sine_wave)
    sine_wave = -sine_wave

    print("Current close on Sine wave:", sine_wave[-1])

    # Calculate the minimum and maximum values of the sine wave
    sine_wave_min = np.min(sine_wave)
    sine_wave_max = np.max(sine_wave)

    # Calculate the distance from close to min and max as percentages on a scale from 0 to 100%
    dist_from_close_to_min = ((sine_wave[-1] - sine_wave_min) / (sine_wave_max - sine_wave_min)) * 100
    dist_from_close_to_max = ((sine_wave_max - sine_wave[-1]) / (sine_wave_max - sine_wave_min)) * 100

    print("Distance from close to min:", dist_from_close_to_min)
    print("Distance from close to max:", dist_from_close_to_max)

    # Calculate the range of values for each quadrant
    range_q1 = (sine_wave_max - sine_wave_min) / 4
    range_q2 = (sine_wave_max - sine_wave_min) / 4
    range_q3 = (sine_wave_max - sine_wave_min) / 4
    range_q4 = (sine_wave_max - sine_wave_min) / 4

    # Set the EM amplitude for each quadrant based on the range of values
    em_amp_q1 = range_q1 / percent_to_max_val
    em_amp_q2 = range_q2 / percent_to_max_val
    em_amp_q3 = range_q3 / percent_to_max_val
    em_amp_q4 = range_q4 / percent_to_max_val

    # Calculate the EM phase for each quadrant
    em_phase_q1 = 0
    em_phase_q2 = math.pi/2
    em_phase_q3 = math.pi
    em_phase_q4 = 3*math.pi/2

    # Calculate the current position of the price on the sine wave
    current_position = (sine_wave[-1] - sine_wave_min) / (sine_wave_max - sine_wave_min)
    current_quadrant = 0

    # Determine which quadrant the current position is in
    if current_position < 0.25:
        # In quadrant 1
        em_amp = em_amp_q1
        em_phase = em_phase_q1
        current_quadrant = 1
        print("Current position is in quadrant 1. Distance from 0% to 25% of range:", (current_position - 0.0) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    elif current_position < 0.5:
        # In quadrant 2
        em_amp = em_amp_q2
        em_phase = em_phase_q2
        current_quadrant = 2
        print("Current position is in quadrant 2. Distance from 25% to 50% of range:", (current_position - 0.25) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    elif current_position < 0.75:
        # In quadrant 3
        em_amp = em_amp_q3
        em_phase = em_phase_q3
        current_quadrant = 3
        print("Current position is in quadrant 3. Distance from 50% to 75% of range:", (current_position - 0.5) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)
    else:
        # In quadrant 4
        em_amp = em_amp_q4
        em_phase = em_phase_q4
        current_quadrant = 4
        print("Current position is in quadrant 4. Distance from 75% to 100% of range:", (current_position - 0.75) / 0.25 * 100, "%")
        print("Current quadrant is: ", current_quadrant)

    print("EM amplitude:", em_amp)
    print("EM phase:", em_phase)

    # Calculate the EM value
    em_value = em_amp * math.sin(em_phase)

    print("EM value:", em_value)

    # Calculate the percentage of the price range
    price_range = candles[-1]["high"] - candles[-1]["low"]
    price_range_percent = (close_prices[-1] - candles[-1]["low"]) / price_range * 100

    print("Price range percent:", price_range_percent)

    # Calculate the momentum value
    momentum = em_value * price_range_percent / 100

    print("Momentum value:", momentum)

    print()

    # Return a dictionary of all the features
    return {
        "current_close": sine_wave[-1],
        "dist_from_close_to_min": dist_from_close_to_min,
        "dist_from_close_to_max": dist_from_close_to_max,
        "current_quadrant": current_quadrant,
        "em_amplitude": em_amp,
        "em_phase": em_phase,
        "price_range_percent": price_range_percent,
        "momentum": momentum,
        "min": sine_wave_min,
        "max": sine_wave_max
    }
   
#sine_wave = generate_new_momentum_sinewave(close_prices, candles, percent_to_max_val=5, percent_to_min_val=5)
#print(sine_wave)

print()

##################################################
##################################################

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate thresholds and averages based on min and max percentages. 
    """
  
    # Get min/max close    
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    
    # Convert close_prices to numpy array
    close_prices = np.array(close_prices)
    
    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=period)
    
    # Get min/max momentum    
    min_momentum = np.nanmin(momentum)   
    max_momentum = np.nanmax(momentum)
    
    # Calculate custom percentages 
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    # Calculate thresholds       
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])

    # Calculate range of prices within a certain distance from the current close price
    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)

    # Filter close prices
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
        
    # Calculate avg    
    avg_mtf = np.nanmean(filtered_close)

    # Get current momentum       
    current_momentum = momentum[-1]

    # Calculate % to min/max momentum    
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) /   
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               

        percent_to_max_momentum = ((current_momentum - min_momentum) / 
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
 
    # Calculate combined percentages              
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    # Combined momentum signal     
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price


# Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

print("Momentum signal:", momentum_signal)
print()

print("Minimum threshold:", min_threshold)
print("Maximum threshold:", max_threshold)
print("Average MTF:", avg_mtf)

#print("Range of prices within distance from current close price:")
#print(range_price[-1])

# Determine which threshold is closest to the current close
closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close[-1]))

if closest_threshold == min_threshold:
    print("The last minimum value is closest to the current close.")
elif closest_threshold == max_threshold:
    print("The last maximum value is closest to the current close.")
else:
    print("No threshold value found.")

print()

##################################################
##################################################

# Define the current time and close price
current_time = datetime.datetime.now()
current_close = closes[-1]

print("Current local Time is now at: ", current_time)
print("Current close price is at : ", current_close)

print()


##################################################
##################################################

def get_closes_last_n_minutes(interval, n):
    """Generate mock closing prices for the last n minutes"""
    closes = []
    for i in range(n):
        closes.append(random.uniform(0, 100))
    return closes

print()

##################################################
##################################################

import numpy as np
import scipy.fftpack as fftpack
import datetime

def get_target(closes, n_components, target_distance=0.01):
    # Calculate FFT of closing prices
    fft = fftpack.rfft(closes) 
    frequencies = fftpack.rfftfreq(len(closes))
    
    # Sort frequencies by magnitude and keep only the top n_components 
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    top_frequencies = frequencies[idx]
    
    # Filter out the top frequencies and reconstruct the signal
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    
    # Calculate the target price as the next value after the last closing price, plus a small constant
    current_close = closes[-1]
    target_price = filtered_signal[-1] + target_distance
    
    # Get the current time           
    current_time = datetime.datetime.now()
    
    # Calculate the market mood based on the predicted target price and the current close price
    diff = target_price - current_close
    if diff > 0:           
        market_mood = "Bullish"
    else:
        market_mood = "Bearish"
    
    # Calculate fast cycle targets
    fastest_target = current_close + target_distance / 2
    fast_target1 = current_close + target_distance / 4
    fast_target2 = current_close + target_distance / 8
    fast_target3 = current_close + target_distance / 16
    fast_target4 = current_close + target_distance / 32
    
    # Calculate other targets
    target1 = target_price + np.std(closes) / 16
    target2 = target_price + np.std(closes) / 8
    target3 = target_price + np.std(closes) / 4
    target4 = target_price + np.std(closes) / 2
    target5 = target_price + np.std(closes)
    
    # Calculate the stop loss and target levels
    entry_price = closes[-1]    
    stop_loss = entry_price - 3 * np.std(closes)   
    target6 = target_price + np.std(closes)
    target7 = target_price + 2 * np.std(closes)
    target8 = target_price + 3 * np.std(closes)
    target9 = target_price + 4 * np.std(closes)
    target10 = target_price + 5 * np.std(closes)
    
    return current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood

closes = get_closes("1m")     
n_components = 5

current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood = get_target(closes, n_components, target_distance=56)

print("Current local Time is now at:", current_time)
print("Market mood is:", market_mood)

print()

current_close = closes[-1]
print("Current close price is at:", current_close)

print()

print("Fast target 1 is:", fast_target4)
print("Fast target 2 is:", fast_target3)
print("Fast target 3 is:", fast_target2)
print("Fast target 4 is:", fast_target1)

print()

print("Fastest target is:", fastest_target)

print()

print("Target 1 is:", target1)
print("Target 2 is:", target2)
print("Target 3 is:", target3)
print("Target 4 is:", target4)
print("Target 5 is:", target5)

print()

##################################################
##################################################

def get_current_price():
    url = "https://fapi.binance.com/fapi/v1/ticker/price"
    params = {
        "symbol": "BTCUSDT" 
    }
    response = requests.get(url, params=params)
    data = response.json()
    price = float(data["price"])
    return price

# Get the current price
price = get_current_price()

print()

##################################################
##################################################

def get_support_resistance_levels(close):
    # Convert close list to numpy array
    close_prices = np.array(close)

    # Calculate EMA50 and EMA200
    ema50 = talib.EMA(close_prices, timeperiod=50)
    ema200 = talib.EMA(close_prices, timeperiod=200)

    # Check if ema50 and ema200 have at least one element
    if len(ema50) == 0 or len(ema200) == 0:
        return []

    # Get the last element of ema50 and ema200
    ema50 = ema50[-1]
    ema200 = ema200[-1]

    # Calculate Phi Ratio levels
    range_ = ema200 - ema50
    phi_levels = [ema50, ema50 + range_/1.618, ema50 + range_]

    # Calculate Gann Square levels
    current_price = close_prices[-1]
    high_points = [current_price, ema200, max(phi_levels)]
    low_points = [min(phi_levels), ema50, current_price]

    gann_levels = []
    for i in range(1, min(4, len(high_points))):
        for j in range(1, min(4, len(low_points))):
            gann_level = ((high_points[i-1] - low_points[j-1]) * 0.25 * (i + j)) + low_points[j-1]
            gann_levels.append(gann_level)

    # Combine levels and sort
    levels = phi_levels + gann_levels
    levels.sort()

    return levels

print()

# Get the support and resistance levels
levels = get_support_resistance_levels(close_prices)

support_levels, resistance_levels = [], []

for level in levels:
    if level < close_prices[-1]:
        support_levels.append(level)
    else:
        resistance_levels.append(level)

# Determine the market mood
if len(levels) > 0:
    support_levels = []
    resistance_levels = []
    for level in levels:
        if level < close_prices[-1]:
            support_levels.append(level)
        else:
            resistance_levels.append(level)

    if len(support_levels) > 0 and len(resistance_levels) > 0:
        market_mood_sr = "Neutral"
    elif len(support_levels) > 0:
        market_mood_sr = "Bullish"
    elif len(resistance_levels) > 0:
        market_mood_sr = "Bearish"
    else:
        market_mood_sr = "Undefined"

    # Calculate support and resistance ranges
    if len(support_levels) > 0:
        support_range = max(support_levels) - min(support_levels)
        print("Support range: {:.2f}".format(support_range))
    else:
        print("Support range: None")

    if len(resistance_levels) > 0:
        resistance_range = max(resistance_levels) - min(resistance_levels)
        print("Resistance range: {:.2f}".format(resistance_range))
    else:
        print("Resistance range: None")

    # Print the levels and market mood
    print("Potential support levels:")
    if len(support_levels) > 0:
        for level in support_levels:
            print("  - {:.2f}".format(level))
    else:
        print("  None found.")

    print("Potential resistance levels:")
    if len(resistance_levels) > 0:
        for level in resistance_levels:
            print("  - {:.2f}".format(level))
    else:
        print("  None found.")

    incoming_bullish_reversal = None
    incoming_bearish_reversal = None

    if market_mood_sr == "Neutral":
        print("Market mood: {}".format(market_mood_sr))
        if len(support_levels) > 0:
            support = max(support_levels)
            support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            support = None
            support_percentage = None

        if len(resistance_levels) > 0:
            top = min(resistance_levels)
            top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            top = None
            top_percentage = None

        print("Best dip: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))

        if support_percentage >= 3.0:
            incoming_bullish_reversal = True

    elif market_mood_sr == "Bullish":
        print("Market mood: {}".format(market_mood_sr))
        if len(resistance_levels) > 0:
            top = min(resistance_levels)
            top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            top = None
            top_percentage = None

        if top is not None:
            print("Best breakout: {:.2f}% (Resistance level: {:.2f})".format(top_percentage, top))
        else:
            print("Best breakout: None")

        if top_percentage is not None and top_percentage >= 3.0:
            incoming_bullish_reversal = True

    elif market_mood_sr == "Bearish":
        print("Market mood: {}".format(market_mood_sr))
        if len(support_levels) > 0:
            support = max(support_levels)
            support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
        else:
            support = None
            support_percentage = None

        if support is not None:
            print("Best bounce: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))
        else:
            print("Best bounce: None")

        if support_percentage is not None and support_percentage >= 3.0:
            incoming_bearish_reversal = True

    else:
        print("Market mood: {}".format(market_mood_sr))

    # Print incoming reversal signals
    if incoming_bullish_reversal:
        print("Incoming bullish reversal signal!")

    if incoming_bearish_reversal:
        print("Incoming bearish reversal signal!")


print()

##################################################
##################################################

from sklearn.linear_model import LinearRegression

def price_regression(close):
    # Convert 'close' to a numpy array
    close_data = np.array(close)

    # Create timestamps based on the index (assuming each close price corresponds to a single time unit)
    timestamps = np.arange(len(close_data))

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(timestamps.reshape(-1, 1), close_data)

    # Predict future prices using the regression model
    num_targets = 1
    future_timestamps = np.arange(len(close_data), len(close_data) + num_targets)
    future_prices = model.predict(future_timestamps.reshape(-1, 1))

    return future_timestamps, future_prices

##################################################
##################################################

# Convert 'close' list to a NumPy array
close_np = np.array(close)

# Call the price_regression function with the example data
future_timestamps, future_prices = price_regression(close_np)

# Print the predicted future prices
for timestamp, f_price in zip(future_timestamps, future_prices):
    print(f"Timestamp: {timestamp}, Predicted Price: {f_price}")

print()

##################################################
##################################################

def calculate_reversal_and_forecast(close):
    # Initialize variables
    current_reversal = None
    next_reversal = None
    last_reversal = None
    forecast_dip = None
    forecast_top = None
    
    # Calculate minimum and maximum values
    min_value = np.min(close)
    max_value = np.max(close)
    
    # Calculate forecast direction and price using FFT
    fft = fftpack.rfft(close)
    frequencies = fftpack.rfftfreq(len(close))
    idx = np.argsort(np.abs(fft))[::-1][:10]
    top_frequencies = frequencies[idx]
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    
    if len(close) > 1:
        if filtered_signal[-1] > filtered_signal[-2]:
            forecast_direction = "Up"
            forecast_price_fft = filtered_signal[-1] + (filtered_signal[-1] - filtered_signal[-2]) * 0.5
        else:
            forecast_direction = "Down"
            forecast_price_fft = filtered_signal[-1] - (filtered_signal[-2] - filtered_signal[-1]) * 0.5
    else:
        forecast_direction = "Neutral"
        forecast_price_fft = close[-1]
    
    # Check the relationship between the last value and min/max
    last_value = close[-1]
    if min_value <= last_value <= max_value:
        if last_value == min_value:
            current_reversal = "DIP"
            next_reversal = "TOP"
        elif last_value == max_value:
            current_reversal = "TOP"
            next_reversal = "DIP"
    else:
        forecast_direction = "Up" if close[-1] > close[-2] else "Down"
        forecast_price_fft = price_regression(close)
    
    # Initialize variables for last reversal and distance
    distance = None
    last_reversal = None
    
    # Calculate the distance between the last reversal and the last value
    reversal_idx = None
    for i in range(len(close) - 2, -1, -1):
        if current_reversal == "DIP" and close[i] == min_value:
            reversal_idx = i
            break
        elif current_reversal == "TOP" and close[i] == max_value:
            reversal_idx = i
            break
    
    if reversal_idx is not None:
        distance = len(close) - 1 - reversal_idx
        if current_reversal == "DIP":
            last_reversal = "DIP"
        elif current_reversal == "TOP":
            last_reversal = "TOP"
    
    # Calculate forecast DIP and TOP
    if last_reversal == "DIP":
        forecast_dip = close[-1] - (distance * 0.1)
        forecast_top = forecast_dip + (forecast_dip - close[-1]) * 2
    elif last_reversal == "TOP":
        forecast_top = close[-1] + (distance * 0.1)
        forecast_dip = forecast_top - (close[-1] - forecast_top) * 2
    
    future_price_regression = price_regression(close)
    
    return current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression, last_reversal, forecast_dip, forecast_top



# Call the calculate_reversal_and_forecast function with the example data
(current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression, last_reversal, forecast_dip, forecast_top) = calculate_reversal_and_forecast(close)

# Print the results
print("Current Reversal:", current_reversal if current_reversal is not None else "None")
print("Next Reversal:", next_reversal if next_reversal is not None else "None")
print("Forecast Direction:", forecast_direction if forecast_direction is not None else "None")

# Handle NaN and None values for Forecast Price FFT
if forecast_price_fft is None or np.isnan(forecast_price_fft):
    forecast_price_fft = close_np[-1]
print("Forecast Price FFT:", forecast_price_fft)

# Handle NaN and None values for Future Price Regression
if future_price_regression is None or np.isnan(future_price_regression[1][0]):
    future_price_regression = close_np[-1]
else:
    future_price_regression = future_price_regression[1][0]
print("Future Price Regression:", future_price_regression)

print("Last Reversal:", last_reversal if last_reversal is not None else "None")

# Convert forecast_dip and forecast_top to NumPy floats if they are not None
if forecast_dip is not None and not isinstance(forecast_dip, np.float64):
    forecast_dip = np.float64(forecast_dip)
if forecast_top is not None and not isinstance(forecast_top, np.float64):
    forecast_top = np.float64(forecast_top)

print("Forecast DIP:", forecast_dip if forecast_dip is not None else "None")
print("Forecast TOP:", forecast_top if forecast_top is not None else "None")

print()

##################################################
##################################################

def calculate_elements():
    # Define PHI constant with 15 decimals
    PHI = 1.6180339887498948482045868343656381177
    # Calculate the Brun constant from the phi ratio and sqrt(5)
    brun_constant = math.sqrt(PHI * math.sqrt(5))
    # Define PI constant with 15 decimals
    PI = 3.1415926535897932384626433832795028842
    # Define e constant with 15 decimals
    e = 2.718281828459045235360287471352662498
    # Calculate sacred frequency
    sacred_freq = (432 * PHI ** 2) / 360
    # Calculate Alpha and Omega ratios
    alpha_ratio = PHI / PI
    omega_ratio = PI / PHI
    # Calculate Alpha and Omega spiral angle rates
    alpha_spiral = (2 * math.pi * sacred_freq) / alpha_ratio
    omega_spiral = (2 * math.pi * sacred_freq) / omega_ratio
    # Calculate inverse powers of PHI and fractional reciprocals
    inverse_phi = 1 / PHI
    inverse_phi_squared = 1 / (PHI ** 2)
    inverse_phi_cubed = 1 / (PHI ** 3)
    reciprocal_phi = PHI ** -1
    reciprocal_phi_squared = PHI ** -2
    reciprocal_phi_cubed = PHI ** -3

    # Calculate unit circle degrees for each quadrant, including dip reversal up and top reversal down cycles
    unit_circle_degrees = {
        1: {'angle': 135, 'polarity': ('-', '-'), 'cycle': 'dip_to_top'},  # Quadrant 1 (Dip to Top)
        2: {'angle': 45, 'polarity': ('+', '+'), 'cycle': 'top_to_dip'},   # Quadrant 2 (Top to Dip)
        3: {'angle': 315, 'polarity': ('+', '-'), 'cycle': 'dip_to_top'},  # Quadrant 3 (Dip to Top)
        4: {'angle': 225, 'polarity': ('-', '+'), 'cycle': 'top_to_dip'},   # Quadrant 4 (Top to Dip)
    }

    # Calculate ratios up to 12 ratio degrees
    ratios = [math.atan(math.radians(degrees)) for degrees in range(1, 13)]

    # Calculate arctanh values
    arctanh_values = {
        0: 0,
        1: float('inf'),
        -1: float('-inf')
    }

    # Calculate imaginary number
    imaginary_number = 1j

    return PHI, sacred_freq, unit_circle_degrees, ratios, arctanh_values, imaginary_number, brun_constant, PI, e, alpha_ratio, omega_ratio, inverse_phi, inverse_phi_squared, inverse_phi_cubed, reciprocal_phi, reciprocal_phi_squared, reciprocal_phi_cubed  

print()

def calculate_sma(close):
    if not isinstance(close, np.ndarray):
        close = np.array(close_prices, dtype=float)

    # Replace NaN    
    close = np.nan_to_num(close, nan=0.0)

    # Calculate SMAs using TA-Lib
    sma_12 = talib.SMA(close, timeperiod=5)
    sma_27 = talib.SMA(close, timeperiod=7)
    sma_56 = talib.SMA(close, timeperiod=9)
    
    return sma_12, sma_27, sma_56

# Example usage:
sma_5, sma_7, sma_9 = calculate_sma(close)

print("SMA 5:", sma_5[-1])
print("SMA 7:", sma_7[-1])
print("SMA 9:", sma_9[-1])

print()

##################################################
##################################################

def entry_long(symbol):
    try:     
        # Get balance and leverage     
        account_balance = get_account_balance()   
        trade_leverage = 20
    
        # Get symbol price    
        symbol_price = client.futures_symbol_ticker(symbol=symbol)['price']
        
        # Get step size from exchange info
        info = client.futures_exchange_info()
        filters = [f for f in info['symbols'] if f['symbol'] == symbol][0]['filters']
        step_size = [f['stepSize'] for f in filters if f['filterType']=='LOT_SIZE'][0]
                    
        # Calculate max quantity based on balance, leverage, and price
        max_qty = int(account_balance * trade_leverage / float(symbol_price) / float(step_size)) * float(step_size)  
                    
        # Create buy market order    
        order = client.futures_create_order(
            symbol=symbol,        
            side='BUY',           
            type='MARKET',         
            quantity=max_qty)          
                    
        if 'orderId' in order:
            return True
          
        else: 
            print("Error creating long order.")  
            return False
            
    except BinanceAPIException as e:
        print(f"Error creating long order: {e}")
        return False

def entry_short(symbol):
    try:     
        # Get balance and leverage     
        account_balance = get_account_balance()   
        trade_leverage = 20
    
        # Get symbol price    
        symbol_price = client.futures_symbol_ticker(symbol=symbol)['price']
        
        # Get step size from exchange info
        info = client.futures_exchange_info()
        filters = [f for f in info['symbols'] if f['symbol'] == symbol][0]['filters']
        step_size = [f['stepSize'] for f in filters if f['filterType']=='LOT_SIZE'][0]
                    
        # Calculate max quantity based on balance, leverage, and price
        max_qty = int(account_balance * trade_leverage / float(symbol_price) / float(step_size)) * float(step_size)  
                    
        # Create sell market order    
        order = client.futures_create_order(
            symbol=symbol,        
            side='SELL',           
            type='MARKET',         
            quantity=max_qty)          
                    
        if 'orderId' in order:
            return True
          
        else: 
            print("Error creating short order.")  
            return False
            
    except BinanceAPIException as e:
        print(f"Error creating short order: {e}")
        return False

def exit_trade():
    try:
        # Get account information including available margin
        account_info = client.futures_account()

        # Check if 'availableBalance' is present in the response
        if 'availableBalance' in account_info:
            available_margin = float(account_info['availableBalance'])

            # Check available margin before proceeding
            if available_margin < 0:
                print("Insufficient available margin to exit trades.")
                return

            # Get all open positions
            positions = client.futures_position_information()

            # Loop through each position
            for position in positions:
                symbol = position['symbol']
                position_amount = float(position['positionAmt'])

                # Determine order side
                if position_amount > 0:
                    order_side = 'SELL'
                elif position_amount < 0:
                    order_side = 'BUY'
                else:
                    continue  # Skip positions with zero amount

                # Place order to exit position      
                order = client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type='MARKET',
                    quantity=abs(position_amount))

                print(f"{order_side} order created to exit {abs(position_amount)} {symbol}.")

            print("All positions exited!")
        else:
            print("Error: 'availableBalance' not found in account_info.")
    except BinanceAPIException as e:
        print(f"Error exiting trade: {e}")




print()

##################################################
##################################################

def calculate_donchian_channel(close, period):
    # Calculate highest high and lowest low over the specified period
    highest_high = max(close[-period:])
    lowest_low = min(close[-period:])
    
    # Calculate middle line
    middle_line = (highest_high + lowest_low) / 2.0
    
    # Calculate upper and lower lines of the Donchian Channel
    upper_line = highest_high
    lower_line = lowest_low
    
    return float(upper_line), float(middle_line), float(lower_line)

print()

period = 56
upper, middle, lower = calculate_donchian_channel(close, period)

print("Lower Line:", lower)
print("Middle Line:", middle)
print("Upper Line:", upper)

print()

##################################################
##################################################

def calculate_bb_percent_b(close, period, std_dev_factor, min_reversal, max_reversal):
    import numpy as np

    # Calculate the moving average and standard deviation
    moving_avg = np.mean(close[-period:])
    std_dev = np.std(close[-period:])

    # Calculate upper and lower Bollinger Bands
    upper_band = moving_avg + std_dev_factor * std_dev
    lower_band = moving_avg - std_dev_factor * std_dev

    # Calculate %B
    percent_b = (close[-1] - lower_band) / (upper_band - lower_band)

    # Determine market mood based on %B
    market_mood = "Neutral"
    if percent_b < 0.2:
        market_mood = "Oversold"
    elif percent_b > 0.8:
        market_mood = "Overbought"

    # Determine reversal signals based on %B and provided min/max reversal thresholds
    reversal_signal = "No Reversal"
    if percent_b <= min_reversal:
        reversal_signal = "Positive Reversal"
    elif percent_b >= max_reversal:
        reversal_signal = "Negative Reversal"

    return percent_b, market_mood, reversal_signal

# Example usage
period = 20
std_dev_factor = 2
min_reversal = 0.1
max_reversal = 0.9

percent_b, market_mood, reversal_signal = calculate_bb_percent_b(close, period, std_dev_factor, min_reversal, max_reversal)

print("BB %B:", percent_b)
print("Market Mood:", market_mood)
print("Reversal Signal:", reversal_signal)

print()

##################################################
##################################################

def kepler_triangle(close):
    # Calculate the other side lengths using Kepler triangle properties
    long_side = close * math.sqrt(2 + math.sqrt(2))
    short_side = close * math.sqrt(2 - math.sqrt(2))
    
    # Return the calculated side lengths as a dictionary
    triangle_sides = {
        'close': close,
        'long_side': long_side,
        'short_side': short_side
    }
    
    return triangle_sides

def map_frequency_bands(triangle_sides):
    # EM field bands and their frequency ranges
    bands = {
        'gamma': [1e-18, 3e-15],    # Hz (added gamma band)
        'alpha': [1e-3, 100e3],     # Hz (alpha band)
        'delta': [100e3, 4e6],      # Hz (delta band)
        'theta': [4e6, 8e6],        # Hz (theta band)
        'beta': [12e6, 30e6],       # Hz (beta band)
        'omega': [30e6, 300e6],     # Hz (omega band)
        # Add more bands and ranges as needed
    }
    
    # Determine the EM field band for the close value
    close = triangle_sides['close']
    em_field_band = None
    for band, (lower, upper) in bands.items():
        if lower <= close <= upper:
            em_field_band = band
            break
    
    if em_field_band is not None:
        triangle_sides['em_field_band'] = em_field_band

# Example usage
close_length = 0.05  # Example close length in MHz (gamma band range)
triangle_sides = kepler_triangle(close_length)
map_frequency_bands(triangle_sides)
print(triangle_sides)


print()

##################################################
##################################################

def calculate_normalized_distance(price, close):
    min_price = np.min(close)
    max_price = np.max(close)
    
    distance_to_min = price - min_price
    distance_to_max = max_price - price
    
    normalized_distance_to_min = distance_to_min / (distance_to_min + distance_to_max) * 100
    normalized_distance_to_max = distance_to_max / (distance_to_min + distance_to_max) * 100
    
    return normalized_distance_to_min, normalized_distance_to_max

def calculate_price_distance_and_wave(price, close):
    normalized_distance_to_min, normalized_distance_to_max = calculate_normalized_distance(price, close)
    
    # Calculate HT_SINE using talib
    ht_sine, _ = talib.HT_SINE(close) 
    #ht_sine = -ht_sine

    # Initialize market_mood
    market_mood = None
    
    # Determine market mood based on HT_SINE crossings and closest reversal
    closest_to_min = np.abs(close - np.min(close)).argmin()
    closest_to_max = np.abs(close - np.max(close)).argmin()
    
    # Check if any of the elements in the close array up to the last value is the minimum or maximum
    if np.any(close[:len(close)-1] == np.min(close[:len(close)-1])):
        market_mood = "Uptrend"
    elif np.any(close[:len(close)-1] == np.max(close[:len(close)-1])):
        market_mood = "Downtrend"

    result = {
        "price": price,
        "ht_sine_value": ht_sine[-1],
        "normalized_distance_to_min": normalized_distance_to_min,
        "normalized_distance_to_max": normalized_distance_to_max,
        "min_price": np.min(close),
        "max_price": np.max(close),
        "market_mood": market_mood
    }
    
    return result

# Example close prices
close_prices = np.array(close)  # Insert your actual close prices here

result = calculate_price_distance_and_wave(price, close_prices)

# Print the detailed information for the given price
print(f"Price: {result['price']:.2f}")
print(f"HT_SINE Value: {result['ht_sine_value']:.2f}")
print(f"Normalized Distance to Min: {result['normalized_distance_to_min']:.2f}%")
print(f"Normalized Distance to Max: {result['normalized_distance_to_max']:.2f}%")
print(f"Min Price: {result['min_price']:.2f}")
print(f"Max Price: {result['max_price']:.2f}")
print(f"Market Mood: {result['market_mood']}")

print()

##################################################
##################################################

print("Init main() loop: ")

print()

##################################################
##################################################

def main():
    # Load credentials from file
    with open("credentials.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()

    # Instantiate Binance client
    client = BinanceClient(api_key, api_secret)

    get_account_balance()

    ##################################################
    ##################################################

    # Define timeframes
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    TRADE_SYMBOL = "BTCUSDT"
    symbol = TRADE_SYMBOL

    trade_entry_pnl = 0
    trade_exit_pnl = 0

    current_pnl = 0.0

    ##################################################
    ##################################################

    print()

    ##################################################
    ##################################################

    while True:

        ##################################################
        ##################################################
        
        # Get fresh closes for the current timeframe
        closes = get_closes('1m')
       
        # Get close price as <class 'float'> type
        close = get_close('1m')
        
        # Get fresh candles  
        candles = get_candles(TRADE_SYMBOL, timeframes)
                
        # Update candle_map with fresh data
        for candle in candles:
            timeframe = candle["timeframe"]  
            candle_map.setdefault(timeframe, []).append(candle)
                
        ##################################################
        ##################################################

        try:     
            ##################################################
            ##################################################

            # Calculate fresh sine wave  
            close_prices = np.array(closes)
            sine, leadsine = talib.HT_SINE(close_prices)
            sine = -sine
    
            # Call scale_to_sine() function   
            #dist_from_close_to_min, dist_from_close_to_max = scale_to_sine('1m')

            #for timeframe in timeframes:
                #dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)

                # Print results        
                #print(f"On {timeframe} Close price value on sine is now at: {current_sine})")
                #print(f"On {timeframe} Distance from close to min perc. is now at: {dist_from_close_to_min})")
                #print(f"On {timeframe} Distance from close to max perc. is now at: {dist_from_close_to_max})")

            ##################################################
            ##################################################

            print()

            momentum_sorter, market_mood, sine_wave_diff, dist_from_close_to_min, dist_from_close_to_max, now, close_prices, current_sine, close_prices_between_min_and_max, min_val, max_val = generate_momentum_sinewave(timeframes)
        
            print()

            print("Current close on sine value now at: ", current_sine)
            print("Distance as percentages from close to min: ", dist_from_close_to_min, "%")
            print("Distance as percentages from close to max: ", dist_from_close_to_max, "%")
            #print("Momentum on 1min timeframe is now at: ", momentum_sorter[-12])
            print("Mood on 1min timeframe is now at: ", market_mood[-12])
            print("The lowest close value in the array is:", min_val)
            print("The highest close value in the array is:", max_val)

            print()


            ##################################################
            ##################################################

            sine_wave = generate_new_momentum_sinewave(close_prices, candles,  
                                               percent_to_max_val=5, 
                                               percent_to_min_val=5)      

            sine_wave_max = sine_wave["max"]   
            sine_wave_min = sine_wave["min"]

            # Call the function
            results = generate_new_momentum_sinewave(
                close_prices, 
                candles,  
                percent_to_max_val=5,  
                percent_to_min_val=5
                )
  
            # Unpack the returned values    
            current_close = results["current_close"]  
            dist_from_close_to_min = results["dist_from_close_to_min"]  
            dist_from_close_to_max = results["dist_from_close_to_max"]
            current_quadrant = results["current_quadrant"]
            em_amp = results["em_amplitude"]
            em_phase = results["em_phase"]  
            price_range_percent = results["price_range_percent"] 
            momentum = results["momentum"]
            sine_wave_min = results["min"]
            sine_wave_max = results["max"]

            print()

            ##################################################
            ##################################################

            url = "https://fapi.binance.com/fapi/v1/ticker/price"

            params = {
                "symbol": "BTCUSDT" 
                }

            response = requests.get(url, params=params)
            data = response.json()

            price = data["price"]
            #print(f"Current BTCUSDT price: {price}")

            # Define the current time and close price
            current_time = datetime.datetime.now()
            current_close = price

            #print("Current local Time is now at: ", current_time)
            #print("Current close price is at : ", current_close)

            print()

            ##################################################
            ##################################################

            # Print the variables from generate_new_momentum_sinewave()
            print(f"Distance from close to min: {dist_from_close_to_min}") 
            print(f"Distance from close to max: {dist_from_close_to_max}")
            print(f"Current_quadrant now at: {current_quadrant}")

            print()

            ##################################################
            ##################################################

            timeframe = '1m'
            momentum = get_momentum(timeframe)
            print("Momentum on 1min tf is at: ", momentum)

            ##################################################
            ##################################################

            # Call function with minimum percentage of 2%, maximum percentage of 2%, and range distance of 5%
            min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05)

            print("Momentum sinewave signal:", momentum_signal)
            print()

            print("Minimum threshold:", min_threshold)
            print("Maximum threshold:", max_threshold)
            print("Average MTF:", avg_mtf)

            print()

            ##################################################
            ##################################################

            closes = get_closes("1m")     
            n_components = 5

            current_time, entry_price, stop_loss, fastest_target, fast_target1, fast_target2, fast_target3, fast_target4, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10, filtered_signal, target_price, market_mood = get_target(closes, n_components, target_distance=56)

            print("Current local Time is now at: ", current_time)
            print("Market mood is: ", market_mood)
            market_mood_fft = market_mood
 
            print()

            print("Current close price is at : ", current_close)

            print()

            print("Fast target 1 is: ", fast_target4)
            print("Fast target 2 is: ", fast_target3)
            print("Fast target 3 is: ", fast_target2)
            print("Fast target 4 is: ", fast_target1)

            print()

            print("Fastest target is: ", fastest_target)

            print()

            print("Target 1 is: ", target1)
            print("Target 2 is: ", target2)
            print("Target 3 is: ", target3)
            print("Target 4 is: ", target4)
            print("Target 5 is: ", target5)

            # Get the current price
            price = get_current_price()

            print()

            price = float(price)

            print()

            ##################################################
            ##################################################

            # Initialize variables
            trigger_long = False 
            trigger_short = False

            current_time = datetime.datetime.utcnow() + timedelta(hours=3)

            range_threshold = max_threshold - min_threshold
            dist_from_min = price - min_threshold
            dist_from_max = max_threshold - price

            pct_diff_to_min = (dist_from_min / range_threshold) * 100
            pct_diff_to_max = (dist_from_max / range_threshold) * 100

            print("Percentage difference to min threshold:", pct_diff_to_min)
            print("Percentage difference to max threshold:", pct_diff_to_max)

            print()

            # Get the support and resistance levels
            levels = get_support_resistance_levels(close_prices)

            support_levels, resistance_levels = [], []

            for level in levels:
                if level < close_prices[-1]:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)

            # Determine the market mood
            if len(levels) > 0:
                support_levels = []
                resistance_levels = []
                for level in levels:
                    if level < close_prices[-1]:
                        support_levels.append(level)
                    else:
                        resistance_levels.append(level)

            if len(support_levels) > 0 and len(resistance_levels) > 0:
                market_mood_sr = "Neutral"
            elif len(support_levels) > 0:
                market_mood_sr = "Bullish"
            elif len(resistance_levels) > 0:
                market_mood_sr = "Bearish"
            else:
                market_mood_sr = "Undefined"

            # Calculate support and resistance ranges
            if len(support_levels) > 0:
                support_range = max(support_levels) - min(support_levels)
                print("Support range: {:.2f}".format(support_range))
            else:
                print("Support range: None")

            if len(resistance_levels) > 0:
                resistance_range = max(resistance_levels) - min(resistance_levels)
                print("Resistance range: {:.2f}".format(resistance_range))
            else:
                print("Resistance range: None")

            # Print the levels and market mood
            print("Potential support levels:")
            if len(support_levels) > 0:
                for level in support_levels:
                    print("  - {:.2f}".format(level))
            else:
                print("  None found.")

            print("Potential resistance levels:")
            if len(resistance_levels) > 0:
                for level in resistance_levels:
                    print("  - {:.2f}".format(level))
            else:
                print("  None found.")

            incoming_bullish_reversal = None
            incoming_bearish_reversal = None

            if market_mood_sr == "Neutral":
                print("Market mood: {}".format(market_mood_sr))
                if len(support_levels) > 0:
                    support = max(support_levels)
                    support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    support = None
                    support_percentage = None

                if len(resistance_levels) > 0:
                    top = min(resistance_levels)
                    top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    top = None
                    top_percentage = None

                print("Best dip: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))

                if support_percentage >= 3.0:
                    incoming_bullish_reversal = True

            elif market_mood_sr == "Bullish":
                print("Market mood: {}".format(market_mood_sr))
                if len(resistance_levels) > 0:
                    top = min(resistance_levels)
                    top_percentage = round(abs(top - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    top = None
                    top_percentage = None

                if top is not None:
                    print("Best breakout: {:.2f}% (Resistance level: {:.2f})".format(top_percentage, top))
                else:
                    print("Best breakout: None")

                if top_percentage is not None and top_percentage >= 3.0:
                    incoming_bullish_reversal = True

            elif market_mood_sr == "Bearish":
                print("Market mood: {}".format(market_mood_sr))
                if len(support_levels) > 0:
                    support = max(support_levels)
                    support_percentage = round(abs(support - close_prices[-1]) / close_prices[-1] * 100, 12)
                else:
                    support = None
                    support_percentage = None

                if support is not None:
                    print("Best bounce: {:.2f}% (Support level: {:.2f})".format(support_percentage, support))
                else:
                    print("Best bounce: None")

                if support_percentage is not None and support_percentage >= 3.0:
                    incoming_bearish_reversal = True

            else:
                print("Market mood: {}".format(market_mood_sr))

            # Print incoming reversal signals
            if incoming_bullish_reversal:
                print("Incoming bullish reversal signal!")

            if incoming_bearish_reversal:
                print("Incoming bearish reversal signal!")

            print()

            ##################################################
            ##################################################

            print("current price at: ", price)
            print("fft fast target1: ", fast_target4)
            print("fft fast target2: ", fast_target3)
            print("fft fast target3: ", fast_target2)
            print("fft fast target4:", fast_target1)
            print("fft fastest target: ", fastest_target)

            print()

            print("Minimum threshold:", min_threshold)
            print("Avrg mtf price: ", avg_mtf)
            print("Maximum threshold:", max_threshold)
            print("The lowest close value in the array is:", min_val)
            print("The highest close value in the array is:", max_val)

            print()

            print("Current quadrant: ", current_quadrant)
            print("dist from close to sine min: ", dist_from_close_to_min)
            print("dist from close to sine max: ", dist_from_close_to_max)
            print("dist from close to thres min: ", pct_diff_to_min)
            print("dist from close to thres max: ", pct_diff_to_max)
            print("momentum market mood: ", market_mood_sr)
            print("momentum value: ", momentum)

            print()

            ##################################################
            ##################################################

            # Assuming you have already calculated the 'close' prices, 'moon_data', 'aspects', and 'vedic_houses'
            min_price, max_price, dip_price, top_price, inner_targets, market_mood, incoming_reversal_keypoint, incoming_reversal, next_targets, inner_reversals = get_hft_targets(close, moon_data, aspects, vedic_houses)

            print("Minimum Price:", min_price)
            print("Maximum Price:", max_price)
            print("Dip Price:", dip_price)
            print("Top Price:", top_price)
            print("Market Mood:", market_mood)
            print("Incoming Reversal Keypoint:", incoming_reversal_keypoint)

            print()

            print("Inner Targets:")
            for i, target in enumerate(inner_targets, start=1):
                print(f"Inner Target {i}: {target}")

            incoming_reversal_keypoint = float(incoming_reversal_keypoint)

            print()

            ##################################################
            ##################################################

            # Call the calculate_reversal_and_forecast function with the example data
            (current_reversal, next_reversal, forecast_direction, forecast_price_fft, future_price_regression, last_reversal, forecast_dip, forecast_top) = calculate_reversal_and_forecast(close)
            print("Forecast Direction:", forecast_direction if forecast_direction is not None else "None")

            # Handle NaN and None values for Forecast Price FFT
            if forecast_price_fft is None or np.isnan(forecast_price_fft):
                forecast_price_fft = close_np[-1]
            print("Forecast Price FFT:", forecast_price_fft)

            forecast_price_fft = float(forecast_price_fft)

            # Handle NaN and None values for Future Price Regression
            if future_price_regression is None or np.isnan(future_price_regression[1][0]):
                future_price_regression = close_np[-1]
            else:
                future_price_regression = future_price_regression[1][0]
            print("Future Price Regression:", future_price_regression)

            future_price_regression = float(future_price_regression)

            print()

            ##################################################
            ##################################################

            #if close[-1] < min(close[-5:]):
                #print("Last element is BELOW any of the last 5 elements.")
            #elif close[-1] > max(close[-5:]):
                #print("Last element is ABOVE any of the last 5 elements.")

            print()

            sma_5, sma_7, sma_9 = calculate_sma(close)

            sma_5 = float(sma_5[-1])
            sma_7 = float(sma_7[-1])
            sma_9 = float(sma_9[-1])

            print("Close is now at: ", price)

            print("SMA 5:", sma_5)
            print("SMA 7:", sma_7)
            print("SMA 9:", sma_9)

            if price < sma_5 and price < sma_7 and price < sma_9:
                print("close now below sma5, sma7, sma9")

            elif price > sma_5 and price > sma_7 and price > sma_9:
                print("close now above sma5, sma7, sma9")

            print()

            # Determine which threshold is closest to the current close
            closest_threshold = min(min_threshold, max_threshold, key=lambda x: abs(x - close[-1]))

            if closest_threshold == min_threshold:
                print("The last minimum value is closest to the current close.")
            elif closest_threshold == max_threshold:
                print("The last maximum value is closest to the current close.")
            else:
                print("No threshold value found.")

            print()

            ##################################################
            ##################################################

            total_volume = calculate_volume(candles)
            buy_volume_5min, sell_volume_5min, buy_volume_3min, sell_volume_3min , buy_volume_1min, sell_volume_1min = calculate_buy_sell_volume(candles)

            (support_levels_1min, resistance_levels_1min, support_levels_3min, resistance_levels_3min, support_levels_5min, resistance_levels_5min) = calculate_support_resistance(candles)

            total_volume_5min = get_volume_5min(candles)
            total_volume_1min = get_volume_1min(candles)

            small_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 2)
            medium_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 5)
            large_lvrg_levels_1min = calculate_reversal_keypoints(support_levels_1min, 10)

            small_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 2)
            medium_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 5)
            large_lvrg_levels_3min = calculate_reversal_keypoints(support_levels_3min, 10)

            small_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 2)
            medium_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 5)
            large_lvrg_levels_5min = calculate_reversal_keypoints(support_levels_5min, 10)

            higher_support_5min, higher_resistance_5min = get_higher_timeframe_data(TRADE_SYMBOL, "5m")

            print("Total Volume:", total_volume)
            print("Total Volume (5min tf):", total_volume_5min)

            print()

            print("Buy Volume (5min tf):", buy_volume_5min)
            print("Sell Volume (5min tf):", sell_volume_5min)

            print()

            print("Buy Volume (3min tf):", buy_volume_3min)
            print("Sell Volume (3min tf):", sell_volume_3min)

            print()

            print("Buy Volume (1min tf):", buy_volume_1min)
            print("Sell Volume (1min tf):", sell_volume_1min)

            print()

            # Print support and resistance levels for the 5-minute timeframe
            print("Support Levels (5min tf):", support_levels_5min[-1])
            print("Resistance Levels (5min tf):", resistance_levels_5min[-1])

            # Print support and resistance levels for the 3-minute timeframe
            print("Support Levels (3min tf):", support_levels_3min[-1])
            print("Resistance Levels (3min tf):", resistance_levels_3min[-1])

            # Calculate and print support and resistance levels for the 1-minute timeframe
            print("Support Levels (1min tf):", support_levels_1min[-1])
            print("Resistance Levels (1min tf):", resistance_levels_1min[-1])

            support_levels_modified = [min(support, candles[-1]["close"]) for support in support_levels_5min]
            resistance_levels_modified = [max(resistance, candles[-1]["close"]) for resistance in resistance_levels_5min]

            # Calculate Bollinger Bands and Poly Channel for 5-minute timeframe
            upper_bb_5min, lower_bb_5min = calculate_bollinger_bands(candles)
            upper_poly_5min, lower_poly_5min = calculate_poly_channel(candles)

            # Calculate the spread factor and number of levels
            spread_factor = 0.02
            num_levels = 5

            # Calculate modified support and resistance levels with spread and additional levels
            #price = candles[-1]["close"]
            support_spread = price * spread_factor
            resistance_spread = price * spread_factor

            # Select the appropriate support and resistance levels based on the desired timeframe
            desired_timeframe = "5m"

            if desired_timeframe == "1m":
                support_levels_selected, resistance_levels_selected = support_levels_1min, resistance_levels_1min
            elif desired_timeframe == "3m":
                support_levels_selected, resistance_levels_selected = support_levels_3min, resistance_levels_3min
            else:
                support_levels_selected, resistance_levels_selected = support_levels_5min, resistance_levels_5min

            # Modify the selected support and resistance levels
            support_levels_modified = [min(support, candles[-1]["close"]) for support in support_levels_selected]
            resistance_levels_modified = [max(resistance, candles[-1]["close"]) for resistance in resistance_levels_selected]

            # Calculate modified support and resistance levels with spread and additional levels
            modified_support_levels = [price - i * support_spread for i in range(num_levels, 0, -1)]
            modified_resistance_levels = [price + i * resistance_spread for i in range(num_levels)]

            # Rule for identifying reversal dips and tops
            if price <= lower_bb_5min[-1] and buy_volume_5min > sell_volume_5min and modified_support_levels and modified_resistance_levels:
                if all(level < small_lvrg_levels_5min[0] for level in modified_support_levels) and all(level < medium_lvrg_levels_5min[0] for level in modified_support_levels) and all(level < large_lvrg_levels_5min[0] for level in modified_support_levels):
                    print("Potential Reversal Dip (5min): Close at or below Bollinger Bands Lower Band and More Buy Volume at Support")
                #elif buy_volume_5min > sell_volume_5min:
                    #print("Potential Reversal Dip (5min): Close at or below Bollinger Bands Lower Band")

            if price >= upper_bb_5min[-1] and sell_volume_5min > buy_volume_5min and modified_support_levels and modified_resistance_levels:
                if all(level > small_lvrg_levels_5min[0] for level in modified_resistance_levels) and all(level > medium_lvrg_levels_5min[0] for level in modified_resistance_levels) and all(level > large_lvrg_levels_5min[0] for level in modified_resistance_levels):
                    print("Potential Reversal Top (5min): Close at or above Bollinger Bands Upper Band and More Sell Volume at Resistance")
                #elif sell_volume_5min > buy_volume_5min:
                    #print("Potential Reversal Top (5min): Close at or above Bollinger Bands Upper Band")

            print()

            print("Lower BB is now at: ", lower_bb_5min[-1])
            print("Upper BB is now at: ", upper_bb_5min[-1])
            print("Lower Poly is now at: ", lower_poly_5min[-1])
            print("Upper Poly is now at: ", upper_poly_5min[-1])

            print()

            distance_to_lower = abs(price - lower_bb_5min[-1])
            distance_to_upper = abs(price - upper_bb_5min[-1])
    
            if distance_to_lower < distance_to_upper:
                print("Price is closer to the Lower Bollinger Band")
            elif distance_to_upper < distance_to_lower:
                print("Price is closer to the upper Bollinger Band")
            else:
                print("Price is equidistant to both Bollinger Bands")

            print()

            if buy_volume_5min > sell_volume_5min:
                print("Buy vol on 5min tf is higher then sell vol: BULLISH")
            elif sell_volume_5min > buy_volume_5min:
                print("Sell vol on 5min tf is higher then buy vol: BEARISH")

            if buy_volume_3min > sell_volume_3min:
                print("Buy vol on 3min tf is higher then sell vol: BULLISH")
            elif sell_volume_3min > buy_volume_3min:
                print("Sell vol on 3min tf is higher then buy vol: BEARISH")

            if buy_volume_1min > sell_volume_1min:
                print("Buy vol on 1min tf is higher then sell vol: BULLISH")
            elif sell_volume_1min > buy_volume_1min:
                print("Sell vol on 1min tf is higher then buy vol: BEARISH")

            print()

            ##################################################
            ##################################################

            period = 256
            upper, middle, lower = calculate_donchian_channel(close, period)

            print("Lower Line:", lower)
            print("Middle Line:", middle)
            print("Upper Line:", upper)
             
            distance_to_lower = abs(price - lower)
            distance_to_upper = abs(price - upper)
    
            if distance_to_lower < distance_to_upper:
                print("Price is closer to the Lower Line")
            else:
                print("Price is closer to the Upper Line")

            print()

            ##################################################
            ##################################################

            # Example close prices
            close_prices = np.array(close)  # Insert your actual close prices here

            result = calculate_price_distance_and_wave(price, close_prices)

            # Print the detailed information for the given price
            print(f"Price: {result['price']:.2f}")
            print(f"HT_SINE Value: {result['ht_sine_value']:.2f}")
            print(f"Normalized Distance to Min: {result['normalized_distance_to_min']:.2f}%")
            print(f"Normalized Distance to Max: {result['normalized_distance_to_max']:.2f}%")
            print(f"Min Price: {result['min_price']:.2f}")
            print(f"Max Price: {result['max_price']:.2f}")
            print(f"Market Mood: {result['market_mood']}")
    
            market_mood_sine = result['market_mood']

            normalized_distance_to_min = result['normalized_distance_to_min']
            normalized_distance_to_max = result['normalized_distance_to_max']

            print()

            ##################################################
            ##################################################

            # Check for new dip and new high
            for timeframe, candles in candle_map.items():
                if len(candles) >= 4:  # Ensure there are enough candles to compare
                    last_two_lows = [candle["low"] for candle in candles[-2:]]
                    last_two_highs = [candle["high"] for candle in candles[-2:]]
                    last_closes = [candle["close"] for candle in candles[-2:]]

                    if last_two_lows[1] < last_two_lows[0]:
                        new_dip_price = last_two_lows[1]
                        print(f"New dip detected in {timeframe} timeframe! Price: {new_dip_price}")

                        if last_closes[1] < last_two_lows[0]:
                            print(f"Downtrend continuation pattern detected in {timeframe} timeframe")
                        else:
                            print(f"Reversal dip pattern detected in {timeframe} timeframe")

                    if last_two_highs[1] > last_two_highs[0]:
                        new_high_price = last_two_highs[1]
                        print(f"New high detected in {timeframe} timeframe! Price: {new_high_price}")

                        if last_closes[1] > last_two_highs[0]:
                            print(f"Uptrend continuation pattern detected in {timeframe} timeframe")
                        else:
                            print(f"Reversal top pattern detected in {timeframe} timeframe")

            print()

            ##################################################
            ##################################################
            for timeframe in timeframes:
                dist_from_close_to_min, dist_from_close_to_max, current_sine = scale_to_sine(timeframe)

                # Print results        
                print(f"On {timeframe} Close price value on sine is now at: {current_sine})")
                print(f"On {timeframe} Distance from close to min perc. is now at: {dist_from_close_to_min})")
                print(f"On {timeframe} Distance from close to max perc. is now at: {dist_from_close_to_max})")

                #if timeframe == '5m' and dist_from_close_to_min < dist_from_close_to_max:
                    #print(f"Close on {timeframe} tf in reversal dip area") 

                #elif timeframe == '5m' and dist_from_close_to_min > dist_from_close_to_max:
                    #print(f"Close on {timeframe} tf in reversal top area")

            print()

            print("Last reversal keypoint was: ", closest_threshold)
            
            print()

            ##################################################
            ##################################################

            take_profit = 5.00
            #stop_loss = -2.33

            # Current timestamp in milliseconds
            timestamp = int(time.time() * 1000)

            # Concatenate the parameters and create the signature
            params = f"symbol={TRADE_SYMBOL}&timestamp={timestamp}"
            signature = hmac.new(api_secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()

            # Construct the complete URL
            position_endpoint = f"https://fapi.binance.com/fapi/v2/positionRisk?{params}&signature={signature}"

            response = requests.get(position_endpoint, headers={"X-MBX-APIKEY": api_key})
    
            if response.status_code == 200:

                positions = response.json()
                found_position = False

                for position in positions:
                    if position['symbol'] == TRADE_SYMBOL:
                        found_position = True
                        print("Position open:", position['positionAmt'])

                        entry_price = float(position['entryPrice'])
                        mark_price = float(position['markPrice'])
                        position_amount = float(position['positionAmt'])
                        leverage = float(position['leverage'])
                        un_realized_profit = float(position['unRealizedProfit'])

                        print("entry price at:", entry_price)
                        print("mark price at:", mark_price)
                        print("position_amount:", position_amount)
                        print("Current PNL at:", un_realized_profit)

                        direction = 1 if position_amount > 0 else -1
                        unrealized_pnl = position_amount * direction * (mark_price - entry_price)
                        imr = 1 / leverage
                        entry_margin = position_amount * mark_price * imr

                        if entry_margin != 0:
                            roe_percentage = (unrealized_pnl / entry_margin) * 100
                            print(f"ROE Percentage (ROE %) for {TRADE_SYMBOL}: {roe_percentage:.2f}%")
                        else:
                            print("Initial Margin is zero, no position is open yet")
                        break
        
                    if not found_position:
                        print("Position not open.")

            else:
                print("Failed to retrieve position information. Status code:", response.status_code)

            print()

            ##################################################
            ##################################################

            with open("signals.txt", "a") as f:
                # Get data and calculate indicators here...
                timestamp = current_time.strftime("%d %H %M %S")

                if un_realized_profit == 0:  
                    # Check if a position is not open
                    print("Now not in a trade, seeking entry conditions")

                    if current_quadrant == 1 and price < fastest_target and normalized_distance_to_min < normalized_distance_to_max:
                        if market_mood_sr == "Bullish" or market_mood_sr == "Neutral" and closest_threshold < price:
                            if market_mood_sine == "Uptrend":
                                if momentum > 0:
                                    for timeframe in timeframes:
                                        if timeframe == '5m' and dist_from_close_to_min < dist_from_close_to_max:
                                            trigger_long = True

                    if current_quadrant == 4 and price > fastest_target and normalized_distance_to_min > normalized_distance_to_max:
                        if market_mood_sr == "Bearish" or market_mood_sr == "Neutral" and closest_threshold > price:
                            if market_mood_sine == "Downtrend":
                                if momentum < 0:
                                    for timeframe in timeframes:
                                        if timeframe == '5m' and dist_from_close_to_min > dist_from_close_to_max:
                                            trigger_short = True

                    if trigger_long:
                        print("LONG signal!")
                        f.write(f"{timestamp} LONG {price}\n")
                        entry_long(symbol)
                        trigger_long = False

                    if trigger_short:
                        print("SHORT signal!")
                        f.write(f"{timestamp} SHORT {price}\n")
                        entry_short(symbol)
                        trigger_short = False

                # Check stop loss and take profit conditions
                if un_realized_profit != 0:
                    print("Now in a trade, seeking exit conditions")

                    if roe_percentage >= take_profit:
                        # Call exit_trade() function
                        exit_trade() 
                    
            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################

            print()

            ##################################################
            ##################################################

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

        ##################################################
        ##################################################

        del close
        del closes
        del candles           
        del sine
        del sine_wave  
        del sine_wave_min
        del sine_wave_max
        del levels           
        del current_time 
        del em_amp
        del em_phase
        del price_range_percent   
        del filtered_signal  
        del current_quadrant
        del market_mood
        del current_close
        del n_components
        del price
        del target1
        del target2
        del target3
        del target4
        del target5
        del fastest_target
        del fast_target1
        del fast_target2
        del fast_target3
        del fast_target4
        del support_levels
        del resistance_levels
        del inner_targets
        del min_price
        del max_price
        del dip_price
        del top_price
        del forecast_direction
        del incoming_reversal_keypoint
        del incoming_reversal
        del future_price_regression
        del min_val
        del max_val
        del dist_from_close_to_min
        del dist_from_close_to_max
        del pct_diff_to_min
        del pct_diff_to_max
        del momentum_signal
        del market_mood_sr
        del min_threshold
        del max_threshold
        del avg_mtf
        del momentum
        del sma_5
        del sma_7
        del sma_9
        del trigger_long
        del trigger_short
        del closest_threshold
        del total_volume
        del buy_volume_5min
        del sell_volume_5min
        del buy_volume_1min
        del sell_volume_1min
        del total_volume_5min
        del total_volume_1min
        del support_levels_1min
        del resistance_levels_1min
        del support_levels_3min
        del resistance_levels_3min
        del support_levels_5min
        del resistance_levels_5min
        del small_lvrg_levels_1min
        del medium_lvrg_levels_1min
        del large_lvrg_levels_1min
        del small_lvrg_levels_3min
        del medium_lvrg_levels_3min
        del large_lvrg_levels_3min
        del small_lvrg_levels_5min
        del medium_lvrg_levels_5min
        del large_lvrg_levels_5min
        del higher_support_5min
        del higher_resistance_5min
        del support_levels_modified
        del resistance_levels_modified
        del upper_bb_5min
        del lower_bb_5min
        del upper_poly_5min 
        del lower_poly_5min
        del spread_factor
        del num_levels
        del support_spread
        del resistance_spread
        del desired_timeframe
        del support_levels_selected
        del resistance_levels_selected
        del modified_support_levels
        del modified_resistance_levels
        del distance_to_lower
        del distance_to_upper
        del lower
        del middle
        del upper

        gc.collect() 

        ##################################################
        ##################################################

        time.sleep(5)
        print()

        ##################################################
        ##################################################

##################################################
##################################################

print()

##################################################
##################################################

# Run the main function
if __name__ == '__main__':
    main()

##################################################
##################################################


print()
##################################################
##################################################
