import ephem
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
import math

def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError("City not found")

def angular_distance(angle1, angle2):
    """Calculate the angular distance between two angles."""
    diff = abs(angle1 - angle2)
    return min(diff, 2 * math.pi - diff)

def get_aspect(angle1, angle2):
    """Determine the aspect between two planetary positions."""
    distance = angular_distance(angle1, angle2)
    if abs(distance) < math.radians(8):  # Conjunction
        return 'Conjunction'
    elif abs(distance - math.pi) < math.radians(8):  # Opposition
        return 'Opposition'
    elif abs(distance - math.radians(60)) < math.radians(8):  # Trine
        return 'Trine'
    elif abs(distance - math.radians(90)) < math.radians(8):  # Square
        return 'Square'
    elif abs(distance - math.radians(120)) < math.radians(8):  # Sextile
        return 'Sextile'
    else:
        return 'No significant aspect'

def get_astrological_data(city_name):
    # Get coordinates for the city
    lat, lon = get_coordinates(city_name)
    
    # Setup observer
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)
    observer.date = datetime.utcnow()

    # Planets
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
        'Pluto': ephem.Pluto(observer)
    }

    # Fetch planetary positions
    astro_data = {}
    for planet_name, planet in planets.items():
        astro_data[planet_name] = {
            'RA': planet.ra,  # Right Ascension
            'Dec': planet.dec,  # Declination
            'Constellation': ephem.constellation(planet)  # Zodiac sign
        }

    # Local Time
    utc_now = datetime.utcnow()
    local_tz = pytz.timezone("Europe/Bucharest")  # Timisoara timezone
    local_now = utc_now.astimezone(local_tz)

    # Print local time and planetary positions
    print(f"Local Time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Location: {city_name} (Lat: {lat}, Lon: {lon})\n")
    print("Planetary Positions:")
    for planet, data in astro_data.items():
        print(f"{planet}:")
        print(f"  Right Ascension: {data['RA']}")
        print(f"  Declination: {data['Dec']}")
        print(f"  Constellation: {data['Constellation']}")
        print()

    # Example of zodiac relations and aspects
    print("Example Zodiac Relations and Aspects:")
    planet_names = list(astro_data.keys())
    for i in range(len(planet_names)):
        for j in range(i + 1, len(planet_names)):
            p1 = planet_names[i]
            p2 = planet_names[j]
            ra1, ra2 = astro_data[p1]['RA'], astro_data[p2]['RA']
            aspect = get_aspect(ra1, ra2)
            print(f"Aspects between {p1} and {p2}: {aspect}")

    # Further analysis required for more detailed zodiac relations

if __name__ == "__main__":
    city = "Timisoara, Timis, Romania"
    get_astrological_data(city)
