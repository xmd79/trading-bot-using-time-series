import datetime
import pytz
import ephem
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np
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

    moon.compute(current_time)
    moon_sign = ephem.constellation(moon)[1]
    moon_distance_km = moon.earth_distance * ephem.meters_per_au / 1000
    moon_angular_diameter = math.degrees(moon.size / moon_distance_km)
    moon_speed_km_hr = moon_distance_km / (1 / 24)
    moon_energy = (moon_phase / 100) ** 2

    map_data = get_astro_map_data(current_time)

    moon_data = {
        'moon_phase': moon_phase,
        'moon_age': moon_age,
        'moon_sign': moon_sign,
        'moon_ra': math.degrees(moon.ra),
        'moon_dec': math.degrees(moon.dec),
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
    obs.lon = '-118.248405'  # Example longitude (adjust as needed)
    obs.lat = '34.052187'    # Example latitude (adjust as needed)
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

def get_planet_positions(current_time):
    tz = pytz.timezone('Etc/GMT-3')
    current_time = tz.normalize(current_time.astimezone(tz))

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

    planet_data = {}

    for planet in planets:
        obj = Horizons(id=planet['id'], location='500', epochs=Time(current_time).jd)
        eph = obj.ephemerides()[0]

        # Compute phase angle for each planet
        phase_angle = float(eph['alpha'])

        # Compute phase percentage
        phase_percent = (1 + np.cos(np.radians(phase_angle))) / 2 * 100

        # Calculate age from previous new moon (for Mercury, Venus, Mars)
        previous_new_moon = ephem.previous_new_moon(current_time.date())
        previous_new_moon_datetime = ephem.Date(previous_new_moon).datetime().replace(tzinfo=tz)
        age_days = (current_time - previous_new_moon_datetime).days

        # Compute sign and degree
        planet_ra = float(eph['RA'])
        planet_dec = float(eph['DEC'])
        planet_body = ephem.FixedBody()
        planet_body._ra = planet_ra
        planet_body._dec = planet_dec
        planet_body.compute(current_time)
        planet_sign = ephem.constellation(planet_body)[1]
        planet_degree = planet_body.elong  # Use elong instead of hlong

        # Compute distance from Earth (in km) and angular diameter (in degrees)
        planet_distance_km = float(eph['delta']) * ephem.meters_per_au / 1000
        planet_angular_diameter = math.degrees(math.atan(1 / (2 * planet_distance_km)))

        # Compute speed (in km/hr)
        planet_speed_km_hr = planet_distance_km / (1 / 24)

        # Compute energy level
        planet_energy = (phase_percent / 100) ** 2

        # Compute current astrological cycle status
        # This can vary widely and is often based on astrological interpretations rather than direct astronomical calculations
        astrological_cycle_status = "To be defined"

        # Store data for the planet
        planet_data[planet['name']] = {
            'phase_percent': phase_percent,
            'age_days': age_days,
            'sign': planet_sign,
            'degree': planet_degree,
            'distance_km': planet_distance_km,
            'angular_diameter': planet_angular_diameter,
            'speed_km_hr': planet_speed_km_hr,
            'energy': planet_energy,
            'astro_cycle_status': astrological_cycle_status
        }

    return planet_data

def get_location():
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.geocode("Timisoara, Romania")  # Example location (adjust as needed)
    return location.latitude, location.longitude

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
    print()

    # Get the planet positions
    planet_positions = get_planet_positions(current_time)
    for planet, data in planet_positions.items():
        print(f"{planet} Phase: {data['phase_percent']:.2f}%")
        print(f"{planet} Age: {data['age_days']} days")
        print(f"{planet} Sign: {data['sign']}")
        print(f"{planet} Degree: {data['degree']:.2f}")
        print(f"{planet} Distance from Earth: {data['distance_km']:.2f} km")
        print(f"{planet} Angular Diameter: {data['angular_diameter']:.2f} degrees")
        print(f"{planet} Speed: {data['speed_km_hr']:.2f} km/hr")
        print(f"{planet} Energy Level: {data['energy']:.2f}%")
        print(f"{planet} Astro Cycle Status: {data['astro_cycle_status']}")
        print()

    # Get the Vedic houses
    location = get_location()
    observer = {
        'longitude': location[1],
        'latitude': location[0]
    }
    vedic_houses = get_vedic_houses(current_time, observer)
    for house, sign in vedic_houses.items():
        print(f"House {house}: {sign}")

    print()

    # Get the star positions
    star_positions = get_star_positions(current_time, observer)
    for star in star_positions:
        print(f"Star {star[0]} Position - RA: {star[1]:.2f}, DEC: {star[2]:.2f}")

    print()

    # Get the planetary element
    planetary_element = get_planetary_element(current_time.hour, location[0], location[1])
    print(f"Current Dominant Planetary Element: {planetary_element}")

    print()

if __name__ == "__main__":
    main()
