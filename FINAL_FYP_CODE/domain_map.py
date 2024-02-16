import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature


boundaries = {
    'min_lon': 14.15,  
    'max_lon': 14.81,  
    'min_lat': 35.79,  
    'max_lat': 36.3   
}

# Set up the map
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([boundaries['min_lon'] - 0.5, boundaries['max_lon'] + 0.5, 
               boundaries['min_lat'] - 0.5, boundaries['max_lat'] + 0.5])

# Add features to the map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)

# Plot the red square with the updated boundaries
ax.plot([boundaries['min_lon'], boundaries['max_lon'], boundaries['max_lon'], boundaries['min_lon'], boundaries['min_lon']],
        [boundaries['min_lat'], boundaries['min_lat'], boundaries['max_lat'], boundaries['max_lat'], boundaries['min_lat']],
        color='red', linewidth=2, transform=ccrs.PlateCarree())

plt.show()