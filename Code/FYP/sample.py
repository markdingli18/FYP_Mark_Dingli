import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

# Step 1: Load Data
ds = xr.open_dataset('med-cmcc-cur-rean-h_1694161199370.nc')
weather_df = pd.read_excel('August_2023.xlsx')

# Step 2: Initialize Variables
num_particles = 1000
initial_lons = np.random.uniform(float(ds['lon'].min()), float(ds['lon'].max()), num_particles)
initial_lats = np.random.uniform(float(ds['lat'].min()), float(ds['lat'].max()), num_particles)

particles = pd.DataFrame({
    'Longitude': initial_lons,
    'Latitude': initial_lats,
    'Time': 0
})

# Debugging: Check the shape of your data arrays
print("Shapes:", ds['uo'][0].shape, ds['vo'][0].shape, ds['lon'].shape, ds['lat'].shape)

def advect(particles, ds, weather_df, dt):
    # Convert min and max to float for scalar comparison
    lon_min = float(ds['lon'].min())
    lon_max = float(ds['lon'].max())
    lat_min = float(ds['lat'].min())
    lat_max = float(ds['lat'].max())

    # Prepare grid points and values
    lon, lat = np.meshgrid(ds['lon'].values, ds['lat'].values)
    points = np.array([lon.flatten(), lat.flatten()]).T
    uo_values = ds['uo'][0].values.flatten()
    vo_values = ds['vo'][0].values.flatten()

    # Interpolate uo and vo for each particle
    particle_points = np.array([particles['Longitude'].values, particles['Latitude'].values]).T
    uo = griddata(points, uo_values, particle_points, method='linear')
    vo = griddata(points, vo_values, particle_points, method='linear')

    # Interpolate weather data
    mean_wind = weather_df['Mean Wind (km/h)'].mean() / 3.6  # Convert to m/s

    # Update particle positions
    particles['Longitude'] += uo * dt + mean_wind * dt
    particles['Latitude'] += vo * dt

    # Adding random walk (eddy diffusivity)
    eddy_diffusivity = 0.1
    particles['Longitude'] += np.random.normal(0, eddy_diffusivity, len(particles))
    particles['Latitude'] += np.random.normal(0, eddy_diffusivity, len(particles))

    # Debugging: Check if any particle is out of bounds
    out_of_bounds = (
        (particles['Longitude'] < lon_min) | 
        (particles['Longitude'] > lon_max) |
        (particles['Latitude'] < lat_min) |
        (particles['Latitude'] > lat_max)
    )
    if np.any(out_of_bounds):
        print("Warning: Some particles are out of bounds!")
        
    # Handling out-of-bounds particles
    out_of_bounds = (
        (particles['Longitude'] < lon_min) | 
        (particles['Longitude'] > lon_max) |
        (particles['Latitude'] < lat_min) |
        (particles['Latitude'] > lat_max)
    )
    if np.any(out_of_bounds):
        print("Warning: Some particles are out of bounds!")
        particles.loc[out_of_bounds, 'Longitude'] = np.random.uniform(lon_min, lon_max, np.sum(out_of_bounds))
        particles.loc[out_of_bounds, 'Latitude'] = np.random.uniform(lat_min, lat_max, np.sum(out_of_bounds))

    return particles

# Step 4: Time Loop and Animation
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_xlim([float(ds['lon'].min()), float(ds['lon'].max())])
ax.set_ylim([float(ds['lat'].min()), float(ds['lat'].max())])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

def init():
    sc.set_offsets(np.empty((0, 2)))
    return (sc,)

def update(frame):
    global particles
    particles = advect(particles, ds, weather_df, dt=1)
    particles['Time'] = frame
    sc.set_offsets(particles[['Longitude', 'Latitude']].values)
    return (sc,)

ani = FuncAnimation(fig, update, frames=range(0, 100), init_func=init, blit=True)
plt.show()