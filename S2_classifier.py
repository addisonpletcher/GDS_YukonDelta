# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:59:12 2023

@author: addyp
"""

#%% Import Libraries
import rasterio
from rasterio.plot import show
import numpy as np
import os
import geopandas as gpd

os.getcwd()


#%% Open Sentinel-2 image
src = rasterio.open("OneDrive - University Of Oregon\Winter '23\GDS\GDS\FinalProj\QGIS\S2_tile.tif")
show(src)

#Open classification shapefiles
land = gpd.read_file("OneDrive - University Of Oregon\Winter '23\GDS\GDS\FinalProj\QGIS\Land.shp")
water = gpd.read_file("OneDrive - University Of Oregon\Winter '23\GDS\GDS\FinalProj\QGIS\Water.shp")
print(land)
print(water)


#%% prep to sample Sentinel-2 
from rasterio import mask as msk
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import pandas as pd

land_proj = land.to_crs('EPSG:32603')
#land_proj.crs

water_proj = water.to_crs('EPSG:32603')
#water_proj.crs


#%% Sample Sentinel-2 data (land)
ca_l, ct_l = msk.mask(src, [mapping(land_proj.iloc[0].geometry)], crop=True)
sa_array_land, clipped_transform = msk.mask(src, [mapping(geom) for geom in land_proj.geometry], crop=True)

all_land = []
for b in range(src.count):
    #Drop zeros, mask to make one dimensional list (all bands)
    temp_list_L=sa_array_land[b][np.nonzero(sa_array_land[b])]
    all_land.append(temp_list_L)

#Check length, should be amount of bands we have
#len(all_land)

# Convert to df
land_df = pd.DataFrame(all_land).T

#%% Sample Sentinel-2 data (water)
ca_w, ct_w = msk.mask(src, [mapping(water_proj.iloc[0].geometry)], crop=True)
sa_array_water, clipped_transform = msk.mask(src, [mapping(geom) for geom in water_proj.geometry], crop=True)

all_water = []
for c in range(src.count):
    #Drop zeros, mask to make one dimensional list (all bands)
    temp_list_W = sa_array_water[c][np.nonzero(sa_array_water[c])]
    all_water.append(temp_list_W)
    
#Check length, should be amount of bands we have
#len(all_water)

# Convert to df
water_df = pd.DataFrame(all_water).T

# No longer needed because included in for loop
#sa_array_water[0][np.nonzero(sa_array_water[0])]

#%% Combine dataframes, add column 

land_df['label'] = 1
water_df['label'] = 2

final_df = pd.concat([land_df,water_df],ignore_index=True)
final_df