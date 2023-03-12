# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:06:45 2023

@author: addyp
"""

#%% Import Libraries
import rasterio
from rasterio.plot import show
import numpy as np
import os
import geopandas as gpd

os.getcwd()
# To go back one folder in cwd
os.chdir("..")

#%% Open summer Sentinel-2 image for training
src = rasterio.open("QGIS\S2_tile.tif")
show(src)

#Open classification shapefiles
land = gpd.read_file("QGIS\Land.shp")
water = gpd.read_file("QGIS\Water.shp")
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

#Rename Columns (Extra column?? No! band 8A is the last one, need to reconfig for this)
final_df.rename(columns = {0:'Band 1', 1:'Band 2', 2:'Band 3', 3:'Band 4', 4:'Band 5', 5:'Band 6', 6:'Band 7', 7:'Band 8', 8:'Band 8A', 9:'Band 9', 10:'Band 10', 11:'Band 11', 12:'Band 12'}, inplace = True)
final_df

#%% Train Machine Learning Model
from sklearn.preprocessing import StandardScaler

#Define Feature List (Green, Red, NIR)
feature_list = ['Band 3', 'Band 4', 'Band 8']

#Define features/labels
X = final_df[feature_list]
y = final_df['label']

#Standardize data
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=feature_list)
print(df_scaled)

#Split data into training/testing subsets
from sklearn.model_selection import train_test_split

# Split data 
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, random_state=42)

#%% RandomForests model
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators = 30) #Define
forest_reg.fit(X_train, y_train) #Fit

# Predict test labels predictions
predictions = forest_reg.predict(X_test)

#%% Compute Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions.astype(int)))

#%% Import Spring S2 imagery
spring = rasterio.open("S2_spring21.tif")

# Read, change to 3D array
band_list = []
spring_array = spring.read()

for d in range(spring_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(spring_array[d,:,:]))
    
# Reshape array
spring_array_rs = np.reshape(spring_array, (13, 80553312))

# Change to DataFrame
spring_df = pd.DataFrame(spring_array_rs, columns=band_list).T

#%% Apply over all pixels in new image 
ice_pred = forest_reg.predict(spring_df)
ice_pred