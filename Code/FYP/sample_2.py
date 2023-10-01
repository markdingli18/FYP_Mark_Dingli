import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

# Step 1: Load Data
ds = xr.open_dataset('med-cmcc-cur-rean-h_1694161199370.nc')
weather_df = pd.read_excel('August_2023.xlsx')

# Print min and max values of lon and lat from the dataset
print(f"Dataset lon range: {float(ds['lon'].min())} to {float(ds['lon'].max())}")
print(f"Dataset lat range: {float(ds['lat'].min())} to {float(ds['lat'].max())}")

# Step 2: Initialize Variables
num_particles = 1000
initial_lons = np.random.uniform(float(ds['lon'].min()), float(ds['lon'].max()), num_particles)
initial_lats = np.random.uniform(float(ds['lat'].min()), float(ds['lat'].max()), num_particles)

# Print min and max initial positions
print(f"Initial particle lon range: {initial_lons.min()} to {initial_lons.max()}")
print(f"Initial particle lat range: {initial_lats.min()} to {initial_lats.max()}")

particles = pd.DataFrame({
    'Longitude': initial_lons,
    'Latitude': initial_lats,
    'Time': 0
})

print("Initial particles:", particles.head())  # Debugging line

def advect(particles, ds, weather_df, dt):
    print("Advecting...")  # Debugging line

    lon, lat = np.meshgrid(ds['lon'].values, ds['lat'].values)
    points = np.array([lon.flatten(), lat.flatten()]).T
    uo_values = ds['uo'][0].values.flatten()
    vo_values = ds['vo'][0].values.flatten()

    particle_points = np.array([particles['Longitude'].values, particles['Latitude'].values]).T
    uo = griddata(points, uo_values, particle_points, method='linear')
    vo = griddata(points, vo_values, particle_points, method='linear')

    # Update particle positions
    mean_wind = weather_df['Mean Wind (km/h)'].mean() / 3.6
    particles['Longitude'] += uo * dt + mean_wind * dt
    particles['Latitude'] += vo * dt

    # Handle NaN values by reinitializing those particles
    nan_idx = particles['Longitude'].isna() | particles['Latitude'].isna()
    if np.any(nan_idx):
        print(f"Reinitializing {np.sum(nan_idx)} particles.")
        particles.loc[nan_idx, 'Longitude'] = np.random.uniform(float(ds['lon'].min()), float(ds['lon'].max()), np.sum(nan_idx))
        particles.loc[nan_idx, 'Latitude'] = np.random.uniform(float(ds['lat'].min()), float(ds['lat'].max()), np.sum(nan_idx))

    return particles

# Step 4: Time Loop and Animation
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_xlim([float(ds['lon'].min()), float(ds['lon'].max())])
ax.set_ylim([float(ds['lat'].min()), float(ds['lat'].max())])

def init():
    print("Initializing...")  # Debugging line
    sc.set_offsets(np.empty((0, 2)))
    return (sc,)

def update(frame):
    print(f"Updating frame {frame}")  # Debugging line
    global particles
    particles = advect(particles, ds, weather_df, dt=1)
    particles['Time'] = frame

    # Print min and max positions after update
    print(f"Updated particle lon range: {particles['Longitude'].min()} to {particles['Longitude'].max()}")
    print(f"Updated particle lat range: {particles['Latitude'].min()} to {particles['Latitude'].max()}")

    sc.set_offsets(particles[['Longitude', 'Latitude']].values)
    print("Updated particles:", particles.head())  # Debugging line
    return (sc,)

ani = FuncAnimation(fig, update, frames=range(0, 100), init_func=init, blit=True)
plt.show()