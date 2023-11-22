import xarray as xr

# File paths
file_path_12hours = 'Data\Wind Model 12hours.nc'
file_path_monthly = 'Data\Wind Satellite Monthly.nc'

# Loading the datasets using xarray
data_12hours = xr.open_dataset(file_path_12hours)
data_monthly = xr.open_dataset(file_path_monthly)

# Converting the datasets to Pandas DataFrame for a tabular view
# For the 12-hour wind data
df_12hours = data_12hours.to_dataframe().reset_index()

# For the monthly satellite wind data
df_monthly = data_monthly.to_dataframe().reset_index()

# Print the first few rows of each DataFrame as a sample
print("12-Hour Wind Model Data (Sample):\n", df_12hours.head())
print("\nMonthly Satellite Wind Data (Sample):\n", df_monthly.head())