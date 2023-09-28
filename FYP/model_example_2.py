import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from scipy.stats import kde

# Read the particle data from the CSV file generated earlier
particle_data = pd.read_csv('lagrangian_simulation_data.csv')

# Define 'lon' and 'lat' based on the data
lon = particle_data['Longitude']
lat = particle_data['Latitude']

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Set axis labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Lagrangian Dispersion Model Simulation Heatmap')

# Set axis limits
ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())

# Create an empty heatmap
heatmap, xedges, yedges = np.histogram2d([], [], bins=50, range=[[lon.min(), lon.max()], [lat.min(), lat.max()]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot the heatmap
img = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

# Create color bar
cbar = plt.colorbar(img)
cbar.set_label('Particle Density')

def update(frame):
    """Update the heatmap for each frame."""
    current_time = frame
    current_data = particle_data[particle_data['Time'] == current_time]
    coords = current_data[['Longitude', 'Latitude']].values
    heatmap, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=50, range=[[lon.min(), lon.max()], [lat.min(), lat.max()]])
    img.set_array(heatmap.T)
    img.autoscale()
    return img,

ani = FuncAnimation(fig, update, frames=range(0, 24), blit=True)
plt.show()
