import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

# Read the particle data from the CSV file generated earlier
particle_data = pd.read_csv('lagrangian_simulation_data (1).csv')

# Define 'lon' and 'lat' based on the data
lon = particle_data['Longitude']
lat = particle_data['Latitude']

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 10))
sc = ax.scatter([], [], c=[], cmap='viridis', edgecolors='k')

# Set axis labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Lagrangian Dispersion Model Simulation')

# Set axis limits
ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())

# Create color bar
cbar = plt.colorbar(sc)
cbar.set_label('Time (hours)')

def init():
    """Initialize the scatter plot."""
    sc.set_offsets(np.empty((0, 2)))  # Empty 2D array
    sc.set_array([])
    return sc,

def update(frame):
    """Update the scatter plot for each frame."""
    current_time = frame
    current_data = particle_data[particle_data['Time'] == current_time]
    coords = current_data[['Longitude', 'Latitude']].values
    sc.set_offsets(coords)
    sc.set_array(current_data['Time'].values)
    return sc,

ani = FuncAnimation(fig, update, frames=range(0, 24), init_func=init, blit=True)
plt.show()