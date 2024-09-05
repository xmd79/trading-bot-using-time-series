import numpy as np
import matplotlib.pyplot as plt
import ephem
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from tzlocal import get_localzone
from itertools import combinations

# Calculate synodic periods for major planets
def calculate_synodic_periods():
    synodic_periods = {
        'Mercury': 116.0,  # in days
        'Venus': 584.0,
        'Mars': 780.0,
        'Jupiter': 399.0,
        'Saturn': 378.0,
        'Uranus': 370.0,
        'Neptune': 367.0,
        'Pluto': 366.7,
        'Sun': 365.0,
        'Moon': 29.53
    }
    return synodic_periods

# Get local time with timezone information
def get_local_time():
    local_timezone = get_localzone()
    return datetime.now(local_timezone)

# Calculate the positions of planets
def get_planet_positions(observer):
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

    positions = {}
    for name, planet in planets.items():
        ra_hours = planet.a_ra * 12.0 / np.pi  # Convert radians to hours
        dec_degrees = planet.a_dec * 180.0 / np.pi  # Convert radians to degrees
        positions[name] = (ra_hours, dec_degrees)

    return positions

# Draw a square centered at (0, 0) that touches the circle
def draw_square(ax, center, size, angle=0, **kwargs):
    half_size = size / 2
    square = np.array([
        [-half_size, -half_size],
        [half_size, -half_size],
        [half_size, half_size],
        [-half_size, half_size],
        [-half_size, -half_size]  # Closing the square shape
    ])
    # Rotate the square
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])
    square = square @ rotation_matrix.T  # Rotate the square points

    # Translate to the center
    square += center
    ax.plot(square[:, 0], square[:, 1], **kwargs)

# Draw a circle
def draw_circle(ax, radius):
    circle = plt.Circle((0, 0), radius, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_artist(circle)

# Draw inner symmetries and align with the circular frame
def draw_inner_symmetries(ax):
    radius = 2  # Radius of the main circle

    # Draw outer equilateral triangles
    for angle in range(0, 360, 120):  # 3 triangles (0°, 120°, 240°)
        triangle = np.array([
            [radius * np.cos(np.radians(angle)), radius * np.sin(np.radians(angle))],  # Vertex 1
            [radius * np.cos(np.radians(angle + 120)), radius * np.sin(np.radians(angle + 120))],  # Vertex 2
            [radius * np.cos(np.radians(angle + 240)), radius * np.sin(np.radians(angle + 240))]  # Vertex 3
        ])
        ax.fill(*zip(*triangle), color='green', alpha=0.3)

    # Draw inner concentric triangles
    inner_radius = radius / 2  # Inner triangle radius
    for angle in range(0, 360, 120):  # 3 inner triangles
        triangle = np.array([
            [inner_radius * np.cos(np.radians(angle)), inner_radius * np.sin(np.radians(angle))],  # Vertex 1
            [inner_radius * np.cos(np.radians(angle + 120)), inner_radius * np.sin(np.radians(angle + 120))],  # Vertex 2
            [inner_radius * np.cos(np.radians(angle + 240)), inner_radius * np.sin(np.radians(angle + 240))]  # Vertex 3
        ])
        ax.fill(*zip(*triangle), color='lightgreen', alpha=0.5)

# Calculate aspects between two planets
def calculate_aspect(planet1, planet2):
    angle_diff = abs(planet1[0] - planet2[0]) % 360
    if angle_diff < 8:  # Conjunction
        return 'Conjunction'
    elif 172 < angle_diff < 188:  # Opposition
        return 'Opposition'
    elif angle_diff < 38:  # Trine
        return 'Trine'
    elif angle_diff < 62:  # Square
        return 'Square'
    return None

# Draw aspects between planets
def draw_aspects(ax, positions):
    planet_names = list(positions.keys())
    for i in range(len(planet_names)):
        for j in range(i + 1, len(planet_names)):
            aspect = calculate_aspect(positions[planet_names[i]], positions[planet_names[j]])
            if aspect is not None:
                x1, y1 = np.cos(np.radians(positions[planet_names[i]][0] * 15)), np.sin(np.radians(positions[planet_names[i]][0] * 15))
                x2, y2 = np.cos(np.radians(positions[planet_names[j]][0] * 15)), np.sin(np.radians(positions[planet_names[j]][0] * 15))
                ax.plot([x1, x2], [y1, y2], linestyle='--', color='gray', alpha=0.5)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, aspect, fontsize=8, ha='center', va='center')

# Plot elements with labels for air, water, fire, earth
def plot_elements_with_labels(ax):
    radius_golden = 2  # Radius of the golden circle
    ax.text(0, radius_golden + 0.2, 'Air', fontsize=12, ha='center')
    ax.text(radius_golden + 0.2, 0, 'Fire', fontsize=12, ha='center')
    ax.text(0, -radius_golden - 0.2, 'Earth', fontsize=12, ha='center')
    ax.text(-radius_golden - 0.2, 0, 'Water', fontsize=12, ha='center')

# New function to draw Gann's Double Mirrored Progression
def draw_ganns_progression(ax):
    radius_golden = 2  # Radius for Gann's Progression
    octaves = 8
    progression = []

    for octave in range(octaves):
        interval = octave * (1 / 9)  # Each octave represents a division of 9
        progression.append(interval)
        progression.append(-interval)  # Add mirrored intervals

    progression = np.array(progression)

    for i, interval in enumerate(progression):
        angle = np.pi * interval  # Convert to angle
        ax.plot([0, radius_golden * np.cos(angle)], [0, radius_golden * np.sin(angle)], linestyle='--', color='purple', lw=1)
        ax.text(radius_golden * np.cos(angle) * 1.1, radius_golden * np.sin(angle) * 1.1,
                f'Octave {i // 2 + 1}, Interval {interval:.2f}', fontsize=9, ha='center')

# Main execution flow
local_time = get_local_time()
observer = ephem.Observer()
observer.lat = '45.7489'  # Latitude for Timișoara, Romania
observer.lon = '21.2087'  # Longitude for Timișoara, Romania
observer.elevation = 0
today = local_time.replace(hour=0, minute=0, second=0, microsecond=0)

# Data storage for historical plotting
historical_positions = {planet: [] for planet in ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']}
sigils = {
    'Sun': '☀',
    'Moon': '☽',
    'Mercury': '☿',
    'Venus': '♀',
    'Mars': '♂',
    'Jupiter': '♃',
    'Saturn': '♄',
    'Uranus': '♅',
    'Neptune': '♆',
    'Pluto': '♇'
}

# Create the plot for astrophysical visualization
fig, ax = plt.subplots(figsize=(18, 18))  # Increased figure size for better visibility

# For storing positions of planets in the animation
orbit_positions = []

def update(frame):
    observer.date = today + timedelta(hours=frame)

    # Clear previous plot data
    ax.clear()

    # Set axes limits and title
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    current_time = get_local_time()
    ax.set_title(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14)

    # Draw the main circle
    draw_circle(ax, radius=2)

    # Draw inner symmetries
    draw_inner_symmetries(ax)

    # Draw Gann's Double Mirrored Progression
    draw_ganns_progression(ax)

    # Get planetary positions
    positions = get_planet_positions(observer)

    # Store current positions into the orbit_positions list
    orbit_positions.clear()  # Clear previous positions
    # Plot planets on the circumference
    for planet, (ra, dec) in positions.items():
        angle = ra * 15  # Convert hours to degrees
        x = 2 * np.cos(np.radians(angle))  # Positioned on the circle
        y = 2 * np.sin(np.radians(angle))  # Positioned on the circle
        
        orbit_positions.append((x, y))  # Store the position

        # Plot the current position
        ax.plot(x, y, 'o', markersize=10, label=planet)

        # Plot sigils near the planet
        ax.text(x * 1.1, y * 1.1, sigils[planet], fontsize=24, ha='center')

        # Calculate percentage completion of the current synodic cycle
        synodic_period = calculate_synodic_periods().get(planet, 365.0)
        completed_percentage = ((frame % synodic_period) / synodic_period) * 100
        ax.text(x * 1.2, y * 1.2, f'{planet} {completed_percentage:.2f}%', fontsize=10, ha='center', color='red')

        # Draw key points for the starting and ending of the current cycle
        if frame == 0:
            ax.text(x * 1.5, y * 1.5, f'Start {current_time.strftime("%Y-%m-%d %H:%M")}', fontsize=10, ha='center', color='black')
        elif frame == synodic_period - 1:
            ax.text(x * 1.5, (y * 1.5) - 0.3, f'End {current_time + timedelta(days=synodic_period):%Y-%m-%d %H:%M}', fontsize=10, ha='center', color='purple')

    # Draw aspects and relationships
    draw_aspects(ax, positions)

    # Print current positions of all planets and aspects for each iteration
    print(f"Iteration: {frame}")
    for planet, (ra, dec) in positions.items():
        print(f"{planet}: RA={ra:.2f}h, Dec={dec:.2f}°")

    # Make sure lines between planets intersect on the circle's curvature
    for i in range(len(orbit_positions)):
        for j in range(i + 1, len(orbit_positions)):
            x_start, y_start = orbit_positions[i]
            x_end, y_end = orbit_positions[j]

            # Normalize to the circle's radius
            norm_start = np.sqrt(x_start ** 2 + y_start ** 2)
            norm_end = np.sqrt(x_end ** 2 + y_end ** 2)

            ax.plot(
                [x_start * 2 / norm_start, x_end * 2 / norm_end],
                [y_start * 2 / norm_start, y_end * 2 / norm_end],
                linestyle='--', color='gray', alpha=0.5
            )
            
            # Calculate the angle for the text label
            angle_mid = (np.arctan2(y_end, x_end) + np.arctan2(y_start, x_start)) / 2
            
            ax.text(1.5 * np.cos(angle_mid), 1.5 * np.sin(angle_mid),
                    f'{round(np.degrees(abs(np.arctan2(y_end, x_end) - np.arctan2(y_start, x_start))), 2)}°',
                    fontsize=8, ha='center')  # Display angle at midpoint

# Start the animation
ani = FuncAnimation(fig, update, frames=range(0, 365), interval=1000, repeat=True)

# Show the animation
plt.show()

# Print synodic periods and planetary aspects
synodic = calculate_synodic_periods()
positions = get_planet_positions(observer)  # Get positions once for reuse
for planet, period in synodic.items():
    print(f'{planet} Synodic Period: {period} days')
    current_position = positions.get(planet, None)
    if current_position is not None:
        print(f" - Current Position: {current_position}")
    else:
        print(f" - Current Position: Not available ")
    print(" ----- ")

# Print the current local time at the start
print(f"Current Local Time: {get_local_time().strftime('%Y-%m-%d %H:%M:%S')}")