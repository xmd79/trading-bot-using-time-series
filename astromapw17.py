import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor
import requests
import logging
import time

# Set up logging for monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for the observer's location
LONGITUDE = '21.21621'  # Longitude of Timișoara  
LATITUDE = '45.75415'    # Latitude of Timișoara

# Define sigil symbols and colors for planets
planet_sigils = {
    'Mercury': '☿',
    'Venus': '♀',
    'Mars': '♂',
    'Jupiter': '♃',
    'Saturn': '♄',
    'Uranus': '♅',
    'Neptune': '♆',
    'Pluto': '♇',
    'Moon': '🌙',
    'Sun': '☀'
}

planet_colors = {
    'Mercury': 'gray',
    'Venus': 'lightyellow',
    'Mars': 'red',
    'Jupiter': 'orange',
    'Saturn': 'gold',
    'Uranus': 'lightblue',
    'Neptune': 'blueviolet',
    'Pluto': 'brown',
    'Moon': 'silver',
    'Sun': 'yellow'
}

# Function to fetch planet data with retries
def fetch_planet_data(planet_name, retries=5):
    """Fetch data for a planet, retrying on failure with exponential backoff."""
    for attempt in range(retries):
        try:
            logging.info(f"Fetching data for {planet_name}")
            url = f'https://api.le-systeme-solaire.net/rest/bodies/{planet_name.lower()}'
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                return {
                    'ra': data.get('sideralRotation', 0),  # Mocking right ascension
                    'dec': 0  # Mock DEC
                }
            else:
                logging.error(f"Error fetching data for {planet_name}: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"Connection error: {e} on attempt {attempt + 1}")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout on attempt {attempt + 1} for {planet_name}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"RequestException on attempt {attempt + 1}: {e}")

        # Exponential backoff logic
        sleep_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
        logging.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)

    logging.error(f"Failed to fetch data for {planet_name} after {retries} attempts.")
    return None

# Function to get planetary positions
def get_planet_positions():
    planets = [
        'mercury', 'venus', 'mars',
        'jupiter', 'saturn', 'uranus', 'neptune',
        'pluto', 'moon', 'sun'
    ]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_planet_data, planets))

    return {planet: res for planet, res in zip(planets, results) if res is not None}

# Function to calculate cycle durations for planets
def planetary_cycle_info(planet_name):
    logging.info(f"Calculating cycle info for {planet_name}")
    cycle_durations = {
        'Moon': 27.32,  # in days
        'Mercury': 115.87,
        'Venus': 225.0,
        'Mars': 686.98,
        'Jupiter': 4331.57,
        'Saturn': 10759.22,
        'Uranus': 30688.5,
        'Neptune': 60182.0,
        'Pluto': 90560.0,
        'Sun': 365.25
    }
    duration = cycle_durations.get(planet_name, 27.32) * 86400  # Convert to seconds
    return duration

# Function to draw a circle
def draw_circle(ax, radius):
    circle = plt.Circle((0, 0), radius, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_artist(circle)

# Function to draw symmetrical octaves
def plot_octaves(ax, radius):
    octaves = 8
    for octave in range(octaves):
        angle = (360 / octaves) * octave
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        ax.plot([0, x], [0, y], linestyle='--', color='purple', lw=1)
        ax.text(x * 1.1, y * 1.1, f'Octave {octave+1}', fontsize=9, ha='center')

# Update function for the animation
def update(frame):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title(f"Current Time: {datetime.datetime.now(pytz.timezone('Etc/GMT-3')).strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14)

    # Draw the main circle representing the planetary system
    circle_radius = 2
    draw_circle(ax, radius=circle_radius)

    # Draw octaves symmetrically around the circle
    plot_octaves(ax, radius=circle_radius)

    planet_positions = get_planet_positions()

    for planet, data in planet_positions.items():
        if data is None:
            continue
        
        angle = (float(data['ra']) / 24) * 360  # Convert RA from hours to degrees
        x = circle_radius * np.cos(np.radians(angle))
        y = circle_radius * np.sin(np.radians(angle))

        # Get cycle duration and calculate performance metrics
        duration = planetary_cycle_info(planet.title())
        current_time = datetime.datetime.now(pytz.timezone('Etc/GMT-3'))
        cycle_start = current_time - datetime.timedelta(seconds=(current_time.timestamp() % duration))
        percentage_to_start = ((current_time - cycle_start).total_seconds() / duration) * 100
        percentage_to_end = 100 - percentage_to_start

        # Draw planet with logging and performance metrics
        ax.plot(x, y, 'o', markersize=12, label=planet.title(), color=planet_colors[planet.title()])
        logging.info(f"{planet.title()} Position - X: {x:.2f}, Y: {y:.2f}, Start Percentage: {percentage_to_start:.2f}%, End Percentage: {percentage_to_end:.2f}%")
        
        # Show percentages as text on the plot
        annotation_text = f'{planet.title()} {planet_sigils[planet.title()]}: Start {percentage_to_start:.2f}%, End {percentage_to_end:.2f}%'
        ax.text(x * 1.2, y * 1.2, annotation_text, fontsize=10, ha='center', color='black', 
                alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))

# Create the animated plot
fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=365, interval=1000, repeat=True)

plt.show()