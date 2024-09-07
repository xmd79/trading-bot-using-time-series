import ephem
import datetime
import pytz
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time

def get_moon_phase_momentum(current_time):
    tz = pytz.timezone('Etc/GMT-3')  # Use 'Etc/GMT-3' for UTC+3
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
    moon.compute(current_time)
    moon_sign = ephem.constellation(moon)[1]

    # Calculate the moon's position
    moon_ra = moon.ra
    moon_dec = moon.dec

    # Calculate the moon's distance from earth in kilometers
    moon_distance_km = moon.earth_distance * ephem.meters_per_au / 1000

    # Calculate the moon's angular diameter in degrees
    moon_angular_diameter = math.degrees(moon.size / moon_distance_km)

    # Calculate the moon's speed in kilometers per hour
    moon_speed_km_hr = (moon_distance_km * 3600 / 24) if moon_distance_km else None  # 24 hours for km/hr calculation

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

def get_planet_data(name, planet_id, current_time):
    now = Time(current_time).jd
    obj = Horizons(id=planet_id, location='500', epochs=now)
    eph = obj.ephemerides()[0]

    # Print the full eph dictionary for diagnosis
    print(f"Data for {name}: {eph}")

    # Calculate RA and DEC in degrees
    ra = eph['RA']
    dec = eph['DEC']

    # Attempt to get distance; use a default value if not found
    distance_au = eph.get('r', None)  # Using 'r' which is the distance in AU
    if distance_au is not None:
        distance_km = distance_au * ephem.meters_per_au / 1000  # Convert to km
    else:
        distance_km = None
        print(f"Warning: 'distance' not available for {name}.")

    # Calculate speed in km/h; if distance is None, speed will also be None
    speed_km_hr = (distance_km * 3600 / 24) if distance_km else None

    # Estimate the sign based on position
    planet_position_body = ephem.FixedBody()
    planet_position_body._ra = ephem.degrees(ra)
    planet_position_body._dec = ephem.degrees(dec)
    planet_position_body.compute()  # Compute position for sign calculation
    sign = ephem.constellation(planet_position_body)[1]

    planet_data = {
        'name': name,
        'RA': ra,
        'DEC': dec,
        'distance_km': distance_km,
        'speed_km_hr': speed_km_hr,
        'energy': (eph['mag'] / 100) ** 2 if 'mag' in eph else 0,
        'sign': sign  # Zodiac sign based on current position
    }

    return planet_data

def get_astro_map_data(current_time):
    tz = pytz.timezone('Etc/GMT-3')
    current_time = tz.normalize(current_time.astimezone(tz))

    obs = ephem.Observer()
    obs.lon = '-118.248405'   # Location longitude
    obs.lat = '34.052187'      # Location latitude
    obs.date = current_time
    obs.pressure = 0
    obs.horizon = '-0:34'

    sun = ephem.Sun(obs)
    moon = ephem.Moon(obs)
    sun.compute(obs)
    moon.compute(obs)

    north_node = ephem.FixedBody()
    north_node._ra = ephem.degrees('18:00:00')
    north_node._dec = ephem.degrees('+5:00:00')
    north_node.compute(obs)
    north_node_sign = ephem.constellation(north_node)[1]

    ascendant = ephem.FixedBody()
    ascendant._ra = ephem.degrees('12:00:00')
    ascendant._dec = ephem.degrees('+0:00:00')
    ascendant.compute(obs)
    ascendant_sign = ephem.constellation(ascendant)[1]

    mc = ephem.FixedBody()
    mc._ra = ephem.degrees('6:00:00')
    mc._dec = ephem.degrees('+0:00:00')
    mc.compute(obs)
    mc_sign = ephem.constellation(mc)[1]

    astro_map_data = {
        'ascendant': ascendant_sign,
        'midheaven': mc_sign,
        'sun': {
            'sign': ephem.constellation(sun)[1],
            'degree': math.degrees(sun.ra)
        },
        'moon': {
            'sign': ephem.constellation(moon)[1],
            'degree': math.degrees(moon.ra)
        },
        'north_node': north_node_sign
    }

    return astro_map_data

def get_planet_positions(current_time):
    now = Time(current_time)
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

    planet_positions = {}
    for planet in planets:
        planet_positions[planet['name']] = get_planet_data(planet['name'], planet['id'], current_time)

    return planet_positions

def get_vedic_houses(date, observer):
    date_ephem = ephem.Date(date)

    obs = ephem.Observer()
    obs.lon = str(observer['longitude'])
    obs.lat = str(observer['latitude'])
    obs.date = date_ephem

    sidereal_time = float(obs.sidereal_time())
    asc_deg = obs.radec_of(date_ephem, 0)[0] * 180 / ephem.pi
    mc_deg = (sidereal_time - asc_deg + 180) % 360

    house_cusps = []
    for i in range(1, 13):
        cusp_deg = (i * 30 - asc_deg) % 360
        cusp_sign = get_vedic_sign(cusp_deg)
        house_cusps.append((i, cusp_sign))

    house_cusps_dict = {house: sign for house, sign in house_cusps}
    return house_cusps_dict

def get_vedic_sign(deg):
    deg = (deg + 360) % 360
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    return signs[int(deg // 30)]

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

def get_current_aspects(observer):
    obs = ephem.Observer()
    obs.lon = observer['longitude']
    obs.lat = observer['latitude']
    obs.date = ephem.now()

    planets = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                'Jupiter', 'Saturn', 'Uranus', 'Neptune']

    aspects = []
    for planet in planets:
        p = getattr(ephem, planet)()
        p.compute(obs)
        for other_planet in planets:
            if other_planet != planet:
                o = getattr(ephem, other_planet)()
                o.compute(obs)
                separation = abs(p.ra - o.ra)
                separation = separation * 180 / ephem.pi  # Convert from radians to degrees
                aspect = check_aspect(separation)
                if aspect:
                    aspects.append({
                        'planet1': planet,
                        'planet2': other_planet,
                        'aspect': aspect,
                        'separation': separation
                    })

    return aspects

def check_aspect(separation_deg):
    aspects = {
        'Conjunction': (0, 10),
        'Opposition': (180, 10),
        'Trine': (120, 8),
        'Square': (90, 8),
        'Sextile': (60, 8)
    }
    for aspect, (center, orb) in aspects.items():
        if abs(separation_deg - center) <= orb:
            return aspect
    return None

# Define a list of fixed stars with their RA and DEC
stars = [
    ('Sirius', ephem.degrees('6:45:08.9'), ephem.degrees('-16:42:58')),
    ('Betelgeuse', ephem.degrees('5:55:10.3'), ephem.degrees('+7:24:25')),
    ('Rigel', ephem.degrees('5:14:32.3'), ephem.degrees('-8:12:06'))
]

# Example usage
current_time = datetime.datetime.now(pytz.timezone('Etc/GMT-3'))

# Get moon phase and other details
moon_data = get_moon_phase_momentum(current_time)
print("Moon Data:")
print(f"Moon Phase: {moon_data['moon_phase']}")
print(f"Moon Age: {moon_data['moon_age']}")
print(f"Moon Sign: {moon_data['moon_sign']}")
print(f"Moon RA: {moon_data['moon_ra']}")
print(f"Moon DEC: {moon_data['moon_dec']}")
print(f"Moon Distance (km): {moon_data['moon_distance_km']}")
print(f"Moon Angular Diameter: {moon_data['moon_angular_diameter']}")
print(f"Moon Speed (km/hr): {moon_data['moon_speed_km_hr']}")
print(f"Moon Energy: {moon_data['moon_energy']}")
print(f"Astrological Map: {moon_data['astro_map']}")

# Get planetary positions
planet_positions = get_planet_positions(current_time)
print("\nPlanetary Positions:")
for planet, data in planet_positions.items():
    print(f"{data['name']}: RA: {data['RA']}, DEC: {data['DEC']}, Distance (km): {data['distance_km']}, Speed (km/hr): {data['speed_km_hr']}, Energy: {data['energy']}, Sign: {data['sign']}")

# Get Vedic houses
observer = {'longitude': '21.21621', 'latitude': '45.75415'}
vedic_houses = get_vedic_houses(current_time, observer)
print("\nVedic Houses:")
for house, sign in vedic_houses.items():
    print(f"House {house}: {sign}")

# Get star positions
star_positions = get_star_positions(current_time, observer)
print("\nStar Positions:")
for star in star_positions:
    print(f"{star[0]} - RA: {star[1]}, DEC: {star[2]}")

# Get current aspects
aspects = get_current_aspects(observer)
print("\nCurrent Aspects:")
for aspect in aspects:
    print(f"{aspect['planet1']} - {aspect['planet2']}: {aspect['aspect']} (Separation: {aspect['separation']}°)")