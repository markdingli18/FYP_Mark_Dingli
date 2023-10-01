from keras.models import load_model
import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to shape data for LSTM
def shape_data(data, steps):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps])
        y.append(data[i + steps])
    return np.array(X), np.array(y)

# Load trained LSTM model
model = load_model('my_lstm_model.h5')

# Load NetCDF data
dataset = nc.Dataset('med-cmcc-cur-rean-h_1694161199370.nc')
uo_data = dataset.variables['uo'][0, :, :].flatten()
vo_data = dataset.variables['vo'][0, :, :].flatten()

# Load Excel data
df = pd.read_excel('August_2023.xlsx')
mean_wind_data = np.repeat(df['Mean Wind (km/h)'][0], len(uo_data))

# Stack them to create a 2D array
combined_data = np.column_stack((uo_data, vo_data, mean_wind_data))

# Apply MinMax Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(combined_data)

# Shape data for LSTM
steps = 3
X, _ = shape_data(scaled_data, steps)

# Make predictions
predictions = model.predict(X[-1:,:,:])  # Just using the last sample for demonstration

# Extract predicted uo, vo, and wind
uo_pred = predictions[0, 0]
vo_pred = predictions[0, 1]
wind_pred = predictions[0, 2]

# Malta domain
long_min, long_max = 13.916667, 14.791667
lat_min, lat_max = 35.604168, 36.3125

# Function to simulate Lagrangian model for multiple particles
def simulate_multiple_particles(uo, vo, wind, dt, steps, num_particles):
    # Initialize particles within the Malta boundary
    initial_positions = np.column_stack((np.random.uniform(long_min, long_max, num_particles),
                                         np.random.uniform(lat_min, lat_max, num_particles)))
    trajectories = []

    for i in range(num_particles):
        x, y = initial_positions[i]
        positions = [(x, y)]

        for t in range(steps):
            x += uo * dt + wind * dt * 0.1
            y += vo * dt
            positions.append((x, y))

        trajectories.append(np.array(positions))

    return trajectories

dt = 0.1  # Time step for the Lagrangian model
steps = 100  # Number of time steps
num_particles = 10  # Number of particles to simulate

# Run simulation for multiple particles
trajectories = simulate_multiple_particles(uo_pred, vo_pred, wind_pred, dt, steps, num_particles)

# Plotting
plt.figure(figsize=(10, 10))

for i, positions in enumerate(trajectories):
    plt.scatter(positions[:, 0], positions[:, 1], label=f'Particle {i+1}')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Particle Trajectories Around Malta')
plt.xlim([long_min, long_max])
plt.ylim([lat_min, lat_max])
plt.grid(True)
plt.legend()
plt.show()