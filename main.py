# Import necessary packages and libraries
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import os
from datetime import datetime
# save initial GLDAS data
import pickle

## Mendefinisikan konstanta geografis dan path data.
## Variabel
# Jari-jari bui
EARTH_RADIUS_KM = 6371
# Konversi Centimeters to kilometers
CM_TO_KM_RATIO = 1e-5
# Here, input the file path to the base data directory
BASE_DIR = ''
# Figure size (plot) yang akan dibuat, yaitu 10x10 inci.
FIG_SIZE = (10, 10)

# Common functions, setting sumbu x as long, dan sumbu y as Lat
def add_labels(ax, title):
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title);

## Load shapefile (border) Jawa Timur
# Read the shapefile
basin_shapefile = gpd.read_file(os.path.join(BASE_DIR, 'Jawa_Madura_1.shp'))
# Plot of entire East Java
# fig, ax = plt.subplots(figsize=FIG_SIZE)
# basin_shapefile.plot(ax=ax)
# add_labels(ax, 'provinsi Jawa Timur')

# plt.show()

# Creating rectangle of extent
import geopandas as gpd
# Deprecated code to produce a bounding rectangle:

    # Extract the coordinates from the shapefile
    # coords = basin_shapefile.get_coordinates()
    # Find the maximum and minimum lat/longs
    # blon_min = min(coords['x'])
    # blon_max = max(coords['x'])
    # blat_min = min(coords['y'])
    # blat_max = max(coords['y'])

# Instead, we just use geopandas.GeoSeries.total_bounds
#.total_bounds adalah metode GeoPandas yang secara otomatis mengembalikan kotak pembatas terkecil
#yang mencakup semua geometri dalam GeoDataFrame tersebut, dalam bentuk array:
[lon_min, lat_min, lon_max, lat_max] = basin_shapefile.total_bounds


# Here, we visualize the bounding box
# First, we create a GeoDataFrame with a single rectangle geometry
bounding_box = gpd.GeoDataFrame(geometry=[box(lon_min, lat_min, lon_max, lat_max)])
# Then, we plot both the basin shapefile and the bounding_box
# fig, ax = plt.subplots(figsize=FIG_SIZE)
# basin_shapefile.plot(ax=ax)
# bounding_box.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)

# add_labels(ax, 'Provinsi Jawa Timur')
# plt.show()

# Load GRACE data
grace = xr.open_dataset(os.path.join(BASE_DIR, "GRCTellus.JPL.200204_202507.GLO.RL06.3M.MSCNv04CRI.nc" ))
# print(grace)

# Selecting the varibles we need: lwe_thickness, uncertainties, and scale factor
# Only extracting the variables we need from the grace dataset, after which we convert everything into a dataframe.

# Let's use 2005-2025 data only
# Filter GRACE data before converting to DataFrame
grace_subset = grace.sel(time=slice('2005-01-01', '2025-06-30'))

# Then extract only the needed variables and convert to DataFrame
grace_df = grace_subset[['lon', 'lat', 'time', 'lwe_thickness', 'scale_factor']].to_dataframe().reset_index()

# Expanding the entries (i.e., every lon lat combo has multiple times, thicknesses, and uncertainties, which we would like to be unique rows)
grace_df.reset_index(inplace=True)

# Relabel the columns to include units
grace_df = grace_df.rename(columns={'lwe_thickness': 'lwe_cm'})

# Convert longitude values
grace_df['lon'] = grace_df['lon'].apply(convert_longitude)

# How many duplicates exist?
"There are %s duplicate rows" % grace_df.duplicated().sum()

# Next, we subset the data to only include pixels within the bounding rectangle
grace_df = grace_df[(grace_df.lon > lon_min) & (grace_df.lon < lon_max) & (grace_df.lat > lat_min) & (grace_df.lat < lat_max)]

# Create a copy of the DataFrame
grace_df = grace_df.copy()

# Finally, we sort using the time column
grace_df.sort_values(by='time', inplace=True)

grace_df["scaled_lwe_cm"] = grace_df['lwe_cm'] * grace_df['scale_factor']
