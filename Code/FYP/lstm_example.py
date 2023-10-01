from keras.models import Sequential
from keras.layers import LSTM, Dense
from tqdm import tqdm
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Function to shape data for LSTM
def shape_data(data, steps):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps])
        y.append(data[i + steps])
    return np.array(X), np.array(y)

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
X, y = shape_data(scaled_data, steps)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')

# Train model with progress bar
for epoch in tqdm(range(100)):
    model.fit(X_train, y_train, epochs=1, batch_size=72, verbose=0)
    
model.save('my_lstm_model.h5')

# Evaluate model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Calculate RMSE and MAE
rmse = sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

# Plotting Actual vs Predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual", color='blue')
plt.plot(predictions, label="Predicted", color='red')
plt.xlabel('Sample Index')
plt.ylabel('Scaled Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()