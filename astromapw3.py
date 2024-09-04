import datetime
import pytz
import ephem
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np

# Gann Octaves for Market Sentiment
GANN_OCTAVES = {
    "Extreme Fear": 1,
    "Severe Fear": 2,
    "Fear": 3,
    "Neutral": 4,
    "Greed": 5,
    "Severe Greed": 6,
    "Extreme Greed": 7,
}

def get_moon_phase_momentum(current_time):
    # Set up timezone information  
    tz = pytz.timezone('Etc/GMT-3')
    current_time = tz.normalize(current_time.astimezone(tz))
    current_date = current_time.date()
    
    # Calculate the moon phase for the current date
    moon = ephem.Moon(current_date)
    moon_phase = moon.phase
    
    # Calculate the moon age in days
    previous_new_moon = ephem.previous_new_moon(current_date)
    previous_new_moon_datetime = ephem.Date(previous_new_moon).datetime()
    previous_new_moon_datetime = previous_new_moon_datetime.replace(tzinfo=pytz.timezone('Etc/GMT-3'))
    moon_age = (current_time - previous_new_moon_datetime).days

    # Calculate the current moon sign
    moon_sign = ephem.constellation(ephem.Moon(current_time))[1]
    
    # Calculate the moon's position
    moon.compute(current_time)
    moon_ra = moon.ra
    moon_dec = moon.dec
    
    # Calculate the moon's distance from Earth in kilometers
    moon_distance_km = moon.earth_distance * ephem.meters_per_au / 1000
    
    # Calculate the moon's angular diameter in degrees
    moon_angular_diameter = math.degrees(moon.size / moon_distance_km)
    
    # Calculate the moon's speed in kilometers per hour
    moon_speed_km_hr = moon_distance_km / (1 / 24)
    
    # Calculate the moon's energy level
    moon_energy = (moon_phase / 100) ** 2
    
    # Calculate the astrological map for the current time
    map_data = get_astro_map_data(current_time)
    
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

def get_astro_map_data(current_time):
    tz = pytz.timezone('Etc/GMT-3')
    current_time = tz.normalize(current_time.astimezone(tz))
    
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
    
    fixed_body = ephem.FixedBody()
    fixed_body._ra = obs.sidereal_time()
    fixed_body._dec = obs.lat
    
    fixed_body.compute(obs)
    
    asc = ephem.constellation(fixed_body)[1]
    vega = ephem.star('Vega')
    vega.compute(current_time)
    mc = ephem.constellation(vega)[1]

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

def get_vedic_houses(date, observer):
    date_ephem = ephem.Date(date)
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = date_ephem

    sidereal_time = float(obs.sidereal_time())
    asc_deg = obs.radec_of(date_ephem, 0)[0] * 180 / ephem.pi

    house_cusps_dict = {}
    for i in range(1, 13):
        cusp_deg = (asc_deg + (i - 1) * 30) % 360
        cusp_sign = get_vedic_sign(cusp_deg)
        house_cusps_dict[i] = {
            'sign': cusp_sign,
            'degree': cusp_deg
        }

    return house_cusps_dict

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

# Define a list of important stars
def get_reliable_star_data():
    stars = [
        ('Sirius', '06:45:08.92', '-16:42:58.02'),   # Sirius
        ('Canopus', '06:23:57.11', '-52:41:44.30'),  # Canopus
        ('Arcturus', '14:15:39.67', '+19:10:56.67'),  # Arcturus
        ('Vega', '18:36:56.34', '+38:47:01.3'),       # Vega
        ('Capella', '05:16:41.35', '+45:59:52.92'),   # Capella
        ('Rigel', '05:14:32.28', '-08:12:05.9'),      # Rigel
        ('Betelgeuse', '05:55:10.31', '+07:24:25.4'), # Betelgeuse
        ('Deneb', '20:41:25.91', '+45:16:49.2'),      # Deneb
        ('Polaris', '02:31:49.09', '+89:15:50.8'),    # Polaris
        ('Pluto', '17:45:22.27', '-22:59:50.0')        # Pluto
    ]
    return stars

stars = get_reliable_star_data()

def get_star_positions(date, observer):
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = ephem.Date(date)

    star_positions = []
    for star in stars:
        fixed_body = ephem.FixedBody()
        fixed_body._ra = star[1]
        fixed_body._dec = star[2]

        fixed_body.compute(obs)

        ra_deg = math.degrees(fixed_body.ra)
        dec_deg = math.degrees(fixed_body.dec)

        star_positions.append((star[0], ra_deg, dec_deg))

    return star_positions

# Function to get planet positions
def get_planet_positions():
    now = Time.now()
    
    planet_positions = {}
    sun_position = {}
    
    planets = [
        {'name': 'Mercury', 'id': '1'},
        {'name': 'Venus', 'id': '2'},
        {'name': 'Mars', 'id': '4'},
        {'name': 'Jupiter', 'id': '5'},
        {'name': 'Saturn', 'id': '6'},
        {'name': 'Uranus', 'id': '7'},
        {'name': 'Neptune', 'id': '8'},
        {'name': 'Pluto', 'id': '9'},
        {'name': 'Sun', 'id': '10'}
    ]
    
    for planet in planets:
        obj = Horizons(id=planet['id'], location='500', epochs=now.jd)  
        eph = obj.ephemerides()[0] 
        planet_positions[planet['name']] = {'RA': eph['RA'], 'DEC': eph['DEC']}
        
    return planet_positions, planet_positions['Sun']

# Print Vedic Houses
def print_vedic_houses(house_cusps_dict):
    for house, data in house_cusps_dict.items():
        print(f"Vedic House {house}: {data['sign']} at {data['degree']:.2f} degrees")

# Set up the current time
current_time = datetime.datetime.utcnow()

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

# Define observer coordinates
observer = {
    'longitude': '-118.248405',
    'latitude': '34.052187'
}

# Calculate Vedic houses
vedic_houses = get_vedic_houses(current_time, observer)

# Print Vedic houses
print("Vedic Houses:")
print_vedic_houses(vedic_houses)

# Get planet positions and print them
planet_positions, sun_position = get_planet_positions()
print('Planet Positions:')
for planet_name, position in planet_positions.items():
    print(f"{planet_name}\n\tRA: {position['RA']}\n\tDEC: {position['DEC']}")
print('Sun Position:')
print(f'\tRA: {sun_position["RA"]}\n\tDEC: {sun_position["DEC"]}')

print()

# Function to convert degrees to hours, minutes, seconds
def deg_to_hours(deg_str):
    deg, minute, sec = deg_str.split(':') 
    degrees = float(deg)
    minutes = float(minute) / 60  
    seconds = float(sec) / 3600    
    return degrees + minutes + seconds

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

date = datetime.datetime.now()
star_positions = get_star_positions_from_sun(date)

for name, ra, dec in star_positions:
    print(f"{name}: RA = {ra}, DEC = {dec}")  

print()

def get_observer():
    obs = ephem.Observer() 
    obs.lon = '21.21621'  # Longitude of Timișoara  
    obs.lat = '45.75415'  # Latitude of Timișoara
    obs.elevation = 102   # Elevation of Timișoara in meters
    obs.date = ephem.now()
    return obs

def get_current_aspects():
    obs = get_observer()
    current_date = ephem.now()
    obs.date = current_date

    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
               'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    aspects = []

    for planet in planets:        
        p = getattr(ephem, planet)()
        p.compute(obs)  

        for other_planet in planets:
            o = getattr(ephem, other_planet)()   
            o.compute(obs)

            separation = ephem.separation(p, o)
            separation_deg = ephem.degrees(separation)

            if check_aspect(separation_deg):
                aspects.append((planet, other_planet, separation_deg))
                
    return aspects

def check_aspect(sep):
    orb = 6  # Degree orb for considering an aspect            
    return sep <= orb or 360 - sep <= orb

# Call the function to get the current aspects
aspects = get_current_aspects()
    
# Print the aspects
print("Current aspects:")
for planet1, planet2, separation in aspects:
    print(f"{planet1} aspecting {planet2} at {separation}°")

print()

def get_market_mood(aspects):
    moon_aspects = [a for a in aspects if a[0] == 'Moon']
    mood = "Neutral" 
    mood_level = 4  

    if moon_aspects:
        moon_planets = [a[1] for a in moon_aspects]
        if 'Mars' in moon_planets:
            mood = "Up"
            mood_level = GANN_OCTAVES['Greed']
        elif 'Jupiter' in moon_planets:
            mood = "Up"
            mood_level = GANN_OCTAVES['Greed']
        else:
            mood = "Down"
            mood_level = GANN_OCTAVES['Fear']

    return mood, mood_level

# Call the function to get the market mood
market_mood, mood_intensity = get_market_mood(aspects)

# Print market mood and corresponding intensity
mood_description = list(GANN_OCTAVES.keys())[mood_intensity - 1]  
print(f"Market Mood: {market_mood}, Intensity Level: {mood_description}")

# Function to analyze possible reversals
def get_possible_reversals(aspects):
    reversals = []
    for a in aspects: 
        if a[2] <= 5:  # Within 5 degree orb
            reversals.append(a)
    return reversals

# Get possible reversals
reversals = get_possible_reversals(aspects)
print("Possible reversals:")
for p1, p2, sep in reversals:
    print(f"{p1} is reversing with {p2} at {sep}°")

# Analyze intensity of fear and greed
def analyze_intensity_and_forecast(moon_data, aspects, vedic_houses):
    forecasts = []
    
    # Initialize sentiment counters
    sentiment_count = {
        'Positive': 0,
        'Cautious': 0
    }
    
    for aspect in aspects:
        planet1, planet2, separation = aspect
        forecast_detail = f"{planet1} in aspect with {planet2} at separation of {separation:.2f}°. "
        
        if planet1 == "Moon":
            if planet2 in ["Mars", "Jupiter"]:
                forecast_detail += "This indicates a positive market sentiment. Expect upward momentum."
                sentiment_count['Positive'] += 1  # Count positive aspects
            elif planet2 in ["Saturn", "Uranus", "Neptune"]:
                forecast_detail += "This indicates a more cautious market. Possible downward pressure."
                sentiment_count['Cautious'] += 1  # Count cautious aspects
        
        forecasts.append(forecast_detail)

    return forecasts, sentiment_count

# Generate detailed forecasts
forecasts, sentiment_count = analyze_intensity_and_forecast(moon_data, aspects, vedic_houses)

# Print forecasts
print("\nAstrological Forecasts:")
for forecast in forecasts:
    print(forecast)

# Determine the overall dominant sentiment
dominant_sentiment = 'Neutral'
if sentiment_count['Positive'] > sentiment_count['Cautious']:
    dominant_sentiment = 'Positive'
elif sentiment_count['Cautious'] > sentiment_count['Positive']:
    dominant_sentiment = 'Cautious'

# Print dominant sentiment
print("\nDominant Sentiment:")
print(f"Positive Count: {sentiment_count['Positive']}")
print(f"Cautious Count: {sentiment_count['Cautious']}")
print(f"Overall Dominant Sentiment: {dominant_sentiment}")

# Final report on market mood
print(f'Market Mood: {market_mood}')
print(f'Intensity of Mood: {mood_description}')