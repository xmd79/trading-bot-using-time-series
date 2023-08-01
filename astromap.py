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

def get_vedic_houses(date, observer):
    # Convert datetime to ephem date
    date_ephem = ephem.Date(date)

    # Set up ephem observer object
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = date_ephem

    # Calculate the sidereal time at the observer's location
    sidereal_time = float(obs.sidereal_time())

    # Calculate the ascendant degree
    asc_deg = obs.radec_of(date_ephem, 0)[0] * 180 / ephem.pi

    # Calculate the MC degree
    mc_deg = (sidereal_time - asc_deg + 180) % 360

    # Calculate the house cusps
    house_cusps = []
    for i in range(1, 13):
        cusp_deg = (i * 30 - asc_deg) % 360
        cusp_sign = get_vedic_sign(cusp_deg)
        house_cusps.append((i, cusp_sign))

    house_cusps_dict = {house: sign for house, sign in house_cusps}
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
  
# Define current_time with UTC+3 offset   
current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=3)

# Call the function to populate moon_data 
moon_data = get_moon_phase_momentum(current_time)

# Now we can access moon_data  
print("Moon Phase: {:.2f}%".format(moon_data['moon_phase']))
print("Moon Age: {} days".format(moon_data['moon_age']))
print("Moon Sign: {}".format(moon_data['moon_sign']))
print("Moon Distance from Earth: {:.2f} km".format(moon_data['moon_distance_km']))
print("Moon Angular Diameter: {:.2f} degrees".format(moon_data['moon_angular_diameter']))
print("Moon Speed: {:.2f} km/hr".format(moon_data['moon_speed_km_hr']))
print("Moon Energy Level: {:.2f}%".format(moon_data['moon_energy']))
print("Moon Astrological Map: {}".format(moon_data['astro_map']))


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

# Print results
for house in range(1,13):
    sign = vedic_houses[house]
    print(f"Vedic House {house}: {sign}")

print()

print("Full Results:")
for house, sign in vedic_houses.items():
    degree = math.degrees(fixed_body.ra)  
    print(f"House {house} - {sign} at {degree:.2f} degrees") 

print()

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
