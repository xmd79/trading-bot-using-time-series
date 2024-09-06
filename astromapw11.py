import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor
import requests
import logging

# Set up logging for monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for the observer's location (replace with actual trading bot logic)
LONGITUDE = '21.21621'  # Longitude of Timișoara  
LATITUDE = '45.75415'    # Latitude of Timișoara

# Define sigil symbols and colors for planets (used for trading decisions)
planet_sigils = {
    'Mercury': '☿',
    'Venus': '♀',
    'Mars': '♂',
    'Jupiter': '♃',
    'Saturn': '♄',
    'Uranus': '♅',
    'Neptune': '♆',
    'Pluto': '♇',
    'Moon': 'Moon',    # Replacing emoji with text
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

# Function to fetch planet data (mock API data, excluding Earth)
def fetch_planet_data(planet_name):
    try:
        logging.info(f"Fetching data for {planet_name}")
        
        url = f'https://api.le-systeme-solaire.net/rest/bodies/{planet_name.lower()}'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return {
                'ra': data.get('sideralRotation', 0),  # Mocking this for right ascension
                'dec': 0  # Mock DEC
            }
        else:
            logging.error(f"Error fetching data for {planet_name}: {response.status_code}")
    except Exception as e:
        logging.exception(f"Error while fetching data for {planet_name}: {e}")

    return None  # If fetching fails

# Function to get planetary positions for trading decisions
def get_planet_positions():
    planets = [
        'mercury', 'venus', 'mars',
        'jupiter', 'saturn', 'uranus', 'neptune',
        'pluto', 'moon', 'sun'
    ]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_planet_data, planets))

    return {planet: res for planet, res in zip(planets, results) if res is not None}

# Function to calculate cycle durations for each planet (enhanced for trading)
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

# Update function for the animation (enhanced with performance metrics)
def update(frame):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title(f"Current Time: {datetime.datetime.now(pytz.timezone('Etc/GMT-3')).strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14)

    # Draw circle to represent the planetary system (for aesthetics/logic)
    draw_circle(ax, radius=2)

    planet_positions = get_planet_positions()

    for planet, data in planet_positions.items():
        if not data:
            continue
        
        angle = (float(data['ra']) / 24) * 360  # Convert RA from hours to degrees
        x = 2 * np.cos(np.radians(angle))
        y = 2 * np.sin(np.radians(angle))

        # Calculate cycle duration and performance metrics
        duration = planetary_cycle_info(planet.title())
        current_time = datetime.datetime.now(pytz.timezone('Etc/GMT-3'))
        cycle_start = current_time - datetime.timedelta(seconds=(current_time.timestamp() % duration))
        percentage_to_start = ((current_time - cycle_start).total_seconds() / duration) * 100
        percentage_to_end = 100 - percentage_to_start

        # Draw planet with enhanced logging and performance metrics
        ax.plot(x, y, 'o', markersize=12, label=planet.title(), color=planet_colors[planet.title()])
        logging.info(f"{planet.title()} Position - X: {x:.2f}, Y: {y:.2f}, Start Percentage: {percentage_to_start:.2f}%, End Percentage: {percentage_to_end:.2f}%")
        
        # Show percentages as text on plot (dynamic annotation)
        annotation_text = f'{planet.title()} {planet_sigils[planet.title()]}: Start {percentage_to_start:.2f}%, End {percentage_to_end:.2f}%'
        ax.text(x * 1.2, y * 1.2, annotation_text, fontsize=10, ha='center', color='black', 
                alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))

# Circle drawing function
def draw_circle(ax, radius):
    circle = plt.Circle((0, 0), radius, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_artist(circle)

# Create the animated plot
fig, ax = plt.subplots(figsize=(10, 10))
ani = FuncAnimation(fig, update, frames=365, interval=1000, repeat=True)

plt.show()