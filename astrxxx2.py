import datetime
import pytz
import ephem
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np
import scipy.signal as signal
from geopy.geocoders import Nominatim

def get_moon_phase_momentum(current_time):
    tz = pytz.timezone('Etc/GMT-3')
    current_time = tz.normalize(current_time.astimezone(tz))
    current_date = current_time.date()

    moon = ephem.Moon(current_date)
    moon_phase = moon.phase

    previous_new_moon = ephem.previous_new_moon(current_date)
    previous_new_moon_datetime = ephem.Date(previous_new_moon).datetime().replace(tzinfo=tz)
    moon_age = (current_time - previous_new_moon_datetime).days

    sun = ephem.Sun(current_time)
    moon_sign = ephem.constellation(ephem.Moon(current_time))[1]

    moon.compute(current_time)
    moon_distance_km = moon.earth_distance * ephem.meters_per_au / 1000
    moon_angular_diameter = math.degrees(moon.size / moon_distance_km)
    moon_speed_km_hr = moon_distance_km / (1 / 24)
    moon_energy = (moon_phase / 100) ** 2

    map_data = get_astro_map_data(current_time)

    moon_data = {
        'moon_phase': moon_phase,
        'moon_age': moon_age,
        'moon_sign': moon_sign,
        'moon_ra': moon.ra,
        'moon_dec': moon.dec,
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
    fixed_body.compute(current_time)

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
        {'name': 'Pluto', 'id': '9'}
    ]

    for planet in planets:
        obj = Horizons(id=planet['id'], location='500', epochs=now.jd)
        eph = obj.ephemerides()[0]
        planet_positions[planet['name']] = {'RA': eph['RA'], 'DEC': eph['DEC']}

    obj = Horizons(id='10', location='500', epochs=now.jd)
    eph = obj.ephemerides()[0]
    sun_position = {'RA': eph['RA'], 'DEC': eph['DEC']}

    return planet_positions, sun_position

def get_vedic_houses(date, observer):
    date_ephem = ephem.Date(date)
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = date_ephem

    sidereal_time = float(obs.sidereal_time())
    asc_deg = obs.radec_of(date_ephem, 0)[0] * 180 / ephem.pi
    mc_deg = (sidereal_time - asc_deg + 180) % 360

    house_cusps = [(i, get_vedic_sign((i * 30 - asc_deg) % 360)) for i in range(1, 13)]

    return {house: sign for house, sign in house_cusps}

def get_vedic_sign(deg):
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    return signs[int(deg // 30)]

def get_star_positions(date, observer):
    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = ephem.Date(date)

    stars = [
        ('Polaris', '02:31:49.09', '+89:15:50.8'),
        ('Vega', '18:36:56.34', '+38:47:01.3'),
        ('Betelgeuse', '05:55:10.31', '+07:24:25.4'),
        ('Rigel', '05:14:32.28', '-08:12:05.9'),
        ('Achernar', '01:37:42.84', '-57:14:12.3'),
        ('Hadar', '14:03:49.40', '-60:22:22.3'),
        ('Altair', '19:50:46.99', '+08:52:05.9'),
        ('Deneb', '20:41:25.91', '+45:16:49.2')
    ]

    star_positions = []
    for name, ra, dec in stars:
        fixed_body = ephem.FixedBody()
        fixed_body._ra = ra
        fixed_body._dec = dec
        fixed_body.compute(obs)
        star_positions.append((name, math.degrees(fixed_body.ra), math.degrees(fixed_body.dec)))

    return star_positions

def get_location():
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.geocode("Timisoara, Romania")
    return location.latitude, location.longitude

def get_planetary_element(hour, latitude, longitude):
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = datetime.datetime.utcnow()

    planets = [ephem.Mars(), ephem.Venus(), ephem.Mercury(), ephem.Moon(), ephem.Sun(), ephem.Uranus(), ephem.Neptune()]
    
    # Compute the positions of each planet
    for planet in planets:
        planet.compute(observer)

    # Now you can access the altitudes
    planetary_altitudes = {planet.name: planet.alt for planet in planets}
    dominant_planet = max(planetary_altitudes, key=planetary_altitudes.get)

    elements = {
        'Sun': 'Fire',
        'Earth': 'Earth',
        'Mercury': 'Air',
        'Venus': 'Water',
        'Mars': 'Fire',
        'Jupiter': 'Air',
        'Saturn': 'Earth',
        'Uranus': 'Air',
        'Neptune': 'Water',
        'Pluto': 'Water'
    }

    return elements.get(dominant_planet, 'Unknown')

def main():
    current_time = datetime.datetime.utcnow()

    # Get the moon data
    moon_data = get_moon_phase_momentum(current_time)
    print(f"Moon Phase: {moon_data['moon_phase']:.2f}%")
    print(f"Moon Age: {moon_data['moon_age']} days")
    print(f"Moon Sign: {moon_data['moon_sign']}")
    print(f"Moon Distance from Earth: {moon_data['moon_distance_km']:.2f} km")
    print(f"Moon Angular Diameter: {moon_data['moon_angular_diameter']:.2f} degrees")
    print(f"Moon Speed: {moon_data['moon_speed_km_hr']:.2f} km/hr")
    print(f"Moon Energy Level: {moon_data['moon_energy']:.2f}%")
    print(f"Ascendant sign: {moon_data['astro_map']['ascendant']}")
    print(f"Midheaven sign: {moon_data['astro_map']['midheaven']}")
    print(f"Sun sign: {moon_data['astro_map']['sun']['sign']}")
    print(f"Sun degree: {moon_data['astro_map']['sun']['degree']:.2f}")
    print(f"Moon sign: {moon_data['astro_map']['moon']['sign']}")
    print(f"Moon degree: {moon_data['astro_map']['moon']['degree']:.2f}")

    # Get the planet positions
    planet_positions, sun_position = get_planet_positions()
    for planet, position in planet_positions.items():
        print(f"{planet} Position - RA: {position['RA']:.2f}, DEC: {position['DEC']:.2f}")
    print(f"Sun Position - RA: {sun_position['RA']:.2f}, DEC: {sun_position['DEC']:.2f}")

    # Get the Vedic houses
    location = get_location()
    observer = {
        'longitude': location[1],
        'latitude': location[0]
    }
    vedic_houses = get_vedic_houses(current_time, observer)
    for house, sign in vedic_houses.items():
        print(f"House {house}: {sign}")

    # Get the star positions
    star_positions = get_star_positions(current_time, observer)
    for star in star_positions:
        print(f"Star {star[0]} Position - RA: {star[1]:.2f}, DEC: {star[2]:.2f}")

    # Get the planetary element
    planetary_element = get_planetary_element(current_time.hour, location[0], location[1])
    print(f"Current Dominant Planetary Element: {planetary_element}")

if __name__ == "__main__":
    main()
