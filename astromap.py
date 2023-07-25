import datetime
import pytz
import ephem
import math

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

# Example usage
current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=3)
moon_data = get_moon_phase_momentum(current_time)

# Print out the moon phase and momentum data
print("Moon Phase: {:.2f}%".format(moon_data['moon_phase']))
print("Moon Age: {} days".format(moon_data['moon_age']))
print("Moon Sign: {}".format(moon_data['moon_sign']))
print("Moon Distance from Earth: {:.2f} km".format(moon_data['moon_distance_km']))
print("Moon Angular Diameter: {:.2f} degrees".format(moon_data['moon_angular_diameter']))
print("Moon Speed: {:.2f} km/hr".format(moon_data['moon_speed_km_hr']))
print("Moon Energy Level: {:.2f}%".format(moon_data['moon_energy']))
print("Moon Astrological Map: {}".format(moon_data['astro_map']))

# Print out the astrological map data
print("\nAstrological Map:")
print("Ascendant: {}".format(moon_data['astro_map']['ascendant']))
print("Midheaven: {}".format(moon_data['astro_map']['midheaven']))
print("Sun Sign: {}".format(moon_data['astro_map']['sun']['sign']))
print("Sun Degree: {:.2f} degrees".format(moon_data['astro_map']['sun']['degree']))
print("Moon Sign: {}".format(moon_data['astro_map']['moon']['sign']))
print("Moon Degree: {:.2f} degrees".format(moon_data['astro_map']['moon']['degree']))

print()