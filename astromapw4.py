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

# Define planetary names
PLANETS = [
    "Sun", "Moon", "Mercury", "Venus", "Mars", 
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
]

# Function to create ephem.FixedBody from RA and DEC
def create_fixed_body(ra, dec):
    body = ephem.FixedBody()
    body._ra = ra
    body._dec = dec
    body.compute()
    return body

# Function to calculate dualities and symmetries
def calculate_symmetries_and_dualities(planet_positions):
    dualities = {}
    symmetries = {}
    
    # Calculate dualities and symmetries
    for i, planet1 in enumerate(PLANETS):
        for j, planet2 in enumerate(PLANETS):
            if i < j:  # Avoid repeating combinations
                body1 = create_fixed_body(planet_positions[planet1]['RA'], planet_positions[planet1]['DEC'])
                body2 = create_fixed_body(planet_positions[planet2]['RA'], planet_positions[planet2]['DEC'])
                
                angle = ephem.separation(body1, body2)
                dualities[(planet1, planet2)] = angle
                # Store symmetries
                if angle < 10:  # Arbitrary small angle for symmetry
                    symmetries[(planet1, planet2)] = angle
    return dualities, symmetries

# Function to calculate geometric shapes
def calculate_geometric_shapes(planet_positions):
    shape_data = {
        'Trinities': [],
        'Squares': [],
        'Equal Triangles': []
    }

    positions = list(planet_positions.values())
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            for k in range(j + 1, len(positions)):
                # Calculate the angular separation between the three planets
                ra1, dec1 = positions[i]['RA'], positions[i]['DEC']
                ra2, dec2 = positions[j]['RA'], positions[j]['DEC']
                ra3, dec3 = positions[k]['RA'], positions[k]['DEC']
                
                angle1 = ephem.separation(create_fixed_body(ra1, dec1), create_fixed_body(ra2, dec2))
                angle2 = ephem.separation(create_fixed_body(ra2, dec2), create_fixed_body(ra3, dec3))
                angle3 = ephem.separation(create_fixed_body(ra1, dec1), create_fixed_body(ra3, dec3))
                
                # Identify shapes based on defined criteria
                if abs(angle1 - 60) < 10 and abs(angle2 - 60) < 10 and abs(angle3 - 60) < 10:
                    shape_data['Equal Triangles'].append((positions[i], positions[j], positions[k]))
                if (abs(angle1 - 90) < 5 and abs(angle2 - 90) < 5) or (abs(angle1 - 90) < 5 and abs(angle3 - 90) < 5) or (abs(angle2 - 90) < 5 and abs(angle3 - 90) < 5):
                    shape_data['Squares'].append((positions[i], positions[j], positions[k]))
                if (angle1 < 10 and angle2 < 10) or (angle1 < 10 and angle3 < 10) or (angle2 < 10 and angle3 < 10):
                    shape_data['Trinities'].append((positions[i], positions[j], positions[k]))
                    
    return shape_data

# Function to get planet positions
def get_planet_positions():
    now = Time.now()
    planet_positions = {}

    planets = [
        {'name': 'Mercury', 'id': '1'},
        {'name': 'Venus', 'id': '2'},
        {'name': 'Mars', 'id': '4'},
        {'name': 'Jupiter', 'id': '5'},
        {'name': 'Saturn', 'id': '6'},
        {'name': 'Uranus', 'id': '7'},
        {'name': 'Neptune', 'id': '8'},
        {'name': 'Pluto', 'id': '9'},
        {'name': 'Sun', 'id': '10'},
        {'name': 'Moon', 'id': '11'}  # Ensure Moon is included as well.
    ]

    for planet in planets:
        obj = Horizons(id=planet['id'], location='500', epochs=now.jd)
        eph = obj.ephemerides()[0] 
        planet_positions[planet['name']] = {'RA': eph['RA'], 'DEC': eph['DEC']}
        
    return planet_positions

# Function to calculate market moods based on planetary aspects
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

# Function to analyze possible reversals
def get_possible_reversals(aspects):
    reversals = []
    for a in aspects: 
        if a[2] <= 5:  # Within 5 degree orb
            reversals.append(a)
    return reversals

def analyze_intensity_and_forecast(moon_data, aspects):
    forecasts = []
    
    sentiment_count = {
        'Positive': 0,
        'Negative': 0
    }
    
    for aspect in aspects:
        planet1, planet2, separation = aspect
        forecast_detail = f"{planet1} in aspect with {planet2} at separation of {separation:.2f}°. "
        
        if planet1 == "Moon":
            if planet2 in ["Mars", "Jupiter"]:
                forecast_detail += "This indicates a positive market sentiment. Expect upward momentum."
                sentiment_count['Positive'] += 1  # Count positive aspects
            elif planet2 in ["Saturn", "Uranus", "Neptune"]:
                forecast_detail += "This indicates a more negative market. Possible downward pressure."
                sentiment_count['Negative'] += 1  # Count negative aspects
        
        forecasts.append(forecast_detail)

    return forecasts, sentiment_count

# Function to fetch moon phase and momentum data
def get_moon_phase_momentum(current_time):
    tz = pytz.timezone('Etc/GMT-3')
    current_time = tz.normalize(current_time.astimezone(tz))
    current_date = current_time.date()
    
    moon = ephem.Moon(current_date)
    moon_phase = moon.phase
    
    previous_new_moon = ephem.previous_new_moon(current_date)
    previous_new_moon_datetime = ephem.Date(previous_new_moon).datetime()
    previous_new_moon_datetime = previous_new_moon_datetime.replace(tzinfo=pytz.timezone('Etc/GMT-3'))
    moon_age = (current_time - previous_new_moon_datetime).days

    moon_sign = ephem.constellation(ephem.Moon(current_time))[1]
    
    moon.compute(current_time)
    moon_ra = moon.ra
    moon_dec = moon.dec
    
    moon_distance_km = moon.earth_distance * ephem.meters_per_au / 1000
    moon_angular_diameter = math.degrees(moon.size / moon_distance_km)
    moon_speed_km_hr = moon_distance_km / (1 / 24)
    moon_energy = (moon_phase / 100) ** 2
    
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

# Function to get astrological map data
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

# Function to create Vedic houses
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

# Function to define a list of important stars
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
for house, data in vedic_houses.items():
    print(f"Vedic House {house}: {data['sign']} at {data['degree']:.2f} degrees")

# Get planet positions and print them
planet_positions = get_planet_positions()
print('Planet Positions:')
for planet_name, position in planet_positions.items():
    print(f"{planet_name}\n\tRA: {position['RA']}\n\tDEC: {position['DEC']}")
print()

# Calculate and print geometric shapes
geometric_shapes = calculate_geometric_shapes(planet_positions)
print("\nGeometric Shapes:")
for shape_type, value in geometric_shapes.items():
    print(f"{shape_type}: {value}")

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

# Call the function to get the market mood
market_mood, mood_intensity = get_market_mood(aspects)

# Print market mood and corresponding intensity
mood_description = list(GANN_OCTAVES.keys())[mood_intensity - 1]  
print(f"Market Mood: {market_mood}, Intensity Level: {mood_description}")

# Get possible reversals
reversals = get_possible_reversals(aspects)
print("Possible reversals:")
for p1, p2, sep in reversals:
    print(f"{p1} is reversing with {p2} at {sep}°")

# Generate detailed forecasts
forecasts, sentiment_count = analyze_intensity_and_forecast(moon_data, aspects)

# Print forecasts
print("\nAstrological Forecasts:")
for forecast in forecasts:
    print(forecast)

# Calculate and print dualities and symmetries
dualities, symmetries = calculate_symmetries_and_dualities(planet_positions)

print("\nDualities:")
for (p1, p2), angle in dualities.items():
    print(f"{p1} - {p2}: {angle:.2f} degrees")

print("\nSymmetries:")
for (p1, p2), angle in symmetries.items():
    print(f"{p1} - {p2}: {angle:.2f} degrees")

# Determine the overall dominant sentiment
dominant_sentiment = 'Neutral'
if sentiment_count['Positive'] > sentiment_count['Negative']:
    dominant_sentiment = 'Positive'
elif sentiment_count['Negative'] > sentiment_count['Positive']:
    dominant_sentiment = 'Negative'

# Print dominant sentiment
print("\nDominant Freq:")
print(f"Positive Count: {sentiment_count['Positive']}")
print(f"Negative Count: {sentiment_count['Negative']}")
print(f"Overall Dominant Freq: {dominant_sentiment}")

# Final report on market mood
print(f'Market Mood: {market_mood}')
print(f'Intensity of Mood: {mood_description}')