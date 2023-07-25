import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

def construct_shapes():
    # Define the 1/phi length
    phi = (1 + math.sqrt(5)) / 2
    length = 1 / phi
    
    # Find the radius of the inner circle
    radius_inner = length / (2 * math.sin(math.pi/3))
    
    # Construct the three circles, concentric to each other
    radius_middle = radius_inner + length
    radius_outer = radius_middle + length
    center = (0, 0)
    circle_inner = Circle(center, radius_inner, fill=False)
    circle_middle = Circle(center, radius_middle, fill=False)
    circle_outer = Circle(center, radius_outer, fill=False)
    
    # Construct the square containing all the shapes
    diagonal = (radius_outer + radius_inner) * math.sqrt(2)
    side_square = diagonal / math.sqrt(2)
    corner_square = (-side_square/2, -side_square/2)
    square = Polygon([(corner_square[0], corner_square[1]), 
                      (corner_square[0]+side_square, corner_square[1]), 
                      (corner_square[0]+side_square, corner_square[1]+side_square), 
                      (corner_square[0], corner_square[1]+side_square)], fill=False)
    
    # Plot the shapes
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.add_patch(circle_inner)
    ax.add_patch(circle_middle)
    ax.add_patch(circle_outer)
    ax.add_patch(square)
    ax.set_xlim(-radius_outer, radius_outer)
    ax.set_ylim(-radius_outer, radius_outer)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
    # Return all the shapes as a dictionary
    shapes = {'circle_inner': circle_inner, 'circle_middle': circle_middle, 'circle_outer': circle_outer, 'square': square}
    return shapes

shapes = construct_shapes()