# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:59:12 2023


This script's purpose is to read and sample training data for a machine learning model and subsequent RandomForest Classification for summer/spring trained models (based on inputs)


@author: addyp
"""

#%% Import Libraries
import rasterio
from rasterio.plot import show
import numpy as np
import os
import geopandas as gpd

#os.getcwd()
# To go back one folder in cwd
#os.chdir("..")

#%% Open summer Sentinel-2 image for training
src = rasterio.open("S2_spring21_3band.tif")
show(src)

#Open classification shapefiles
land = gpd.read_file("QGIS\Land.shp")
water = gpd.read_file("water_new.shp")
ice = gpd.read_file("ice.shp")
print(land)
print(water)
print(ice)

#%% prep to sample Sentinel-2 
from rasterio import mask as msk
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import pandas as pd

land_proj = land.to_crs('EPSG:32603')
#land_proj.crs

water_proj = water.to_crs('EPSG:32603')
#water_proj.crs

ice_proj = ice.to_crs('EPSG:32603')

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

#%% Sample Sentinel-2 data (ice)
ca_i, ct_i = msk.mask(src, [mapping(ice_proj.iloc[0].geometry)], crop=True)
sa_array_ice, clipped_transform = msk.mask(src, [mapping(geom) for geom in ice_proj.geometry], crop=True)

all_ice = []
for d in range(src.count): 
    #Drop zeros, mask to make one dimensional list (all bands)
    temp_list_L=sa_array_ice[d][np.nonzero(sa_array_ice[d])]
    all_ice.append(temp_list_L)

#Check length, should be amount of bands we have
#len(all_land)

# Convert to df
ice_df = pd.DataFrame(all_ice).T

#%% Combine dataframes, add column 
land_df['label'] = 1
water_df['label'] = 2
ice_df['label'] = 3

final_df = pd.concat([land_df,water_df,ice_df],ignore_index=True)

#Rename Columns (Extra column?? No! band 8A is the last one, need to reconfig for this)
final_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)
final_df

#%% Train Machine Learning Model
from sklearn.preprocessing import StandardScaler

#Define Feature List 
feature_list = ['B3:Green', 'B4:Red', 'B8:NIR']

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
from sklearn.ensemble import RandomForestClassifier

forest_reg = RandomForestClassifier(n_estimators = 30) #Define
forest_reg.fit(X_train, y_train) #Fit

# Predict test labels predictions
predictions = forest_reg.predict(X_test)

#%% Compute Confusion Matrix
from sklearn.metrics import confusion_matrix
training_cm = (confusion_matrix(y_test, predictions.astype(int)))

import seaborn as sns
sns.heatmap(training_cm/np.sum(training_cm), annot=True, 
            fmt='.2%', cmap='Blues')

#%% Apply to Summer (training image) to compare with SAR image
summer = rasterio.open("S2_training_clip.tif")

# Read, change to 3D array
band_list = []
summer_array = summer.read()

for d in range(summer_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(summer_array[d,:,:]))
    
# Reshape array
summer_array_rs = np.reshape(summer_array, (3, 13973880))

# Change to DataFrame
summer_df = pd.DataFrame(summer_array_rs, columns=band_list).T
summer_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)

summer_scaler = StandardScaler()  
summer_finaldf = summer_scaler.fit_transform(summer_df)

#apply classification over all pixels
summer_pred = forest_reg.predict(summer_finaldf)

#Reshape to origial spring array
summer_pred_2d = np.reshape(summer_pred, (summer_array.shape[1], summer_array.shape[2]))

#Plot
plt.imshow(summer_pred_2d)
plt.colorbar()

#%% Apply to spring imagery | May 2016
sp16 = rasterio.open("S2_spring16.tif")

# Read, change to 3D array
band_list = []
sp16_array = sp16.read()

for d in range(sp16_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(sp16_array[d,:,:]))
    
# Reshape array
sp16_array_rs = np.reshape(sp16_array, (3, 85058094))

# Change to DataFrame
sp16_df = pd.DataFrame(sp16_array_rs, columns=band_list).T
sp16_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)

# Standardize Data
sp16_scaler = StandardScaler()  
sp16_finaldf = sp16_scaler.fit_transform(sp16_df)
print(sp16_finaldf)

#Apply over all pixels in new image 
sp16_pred = forest_reg.predict(sp16_finaldf)

#Reshape to origial spring array
sp16_pred_2d = np.reshape(sp16_pred, (sp16_array.shape[1], sp16_array.shape[2]))

#Plot
plt.imshow(sp16_pred_2d)
plt.colorbar()

# See how many pixel classified as water/land/ice
(sp16_pred == 1).sum()
(sp16_pred == 2).sum()
(sp16_pred == 3).sum()

#%% Apply to spring imagery | May 2018
sp18 = rasterio.open("S2_spring18.tif")

# Read, change to 3D array
band_list = []
sp18_array = sp18.read()

for d in range(sp18_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(sp18_array[d,:,:]))
    
# Reshape array
sp18_array_rs = np.reshape(sp18_array, (3, 87825934))

# Change to DataFrame
sp18_df = pd.DataFrame(sp18_array_rs, columns=band_list).T
sp18_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)

# Standardize Data
sp18_scaler = StandardScaler()  
sp18_finaldf = sp18_scaler.fit_transform(sp18_df)
print(sp18_finaldf)

#Apply over all pixels in new image 
sp18_pred = forest_reg.predict(sp18_finaldf)

#Reshape to origial spring array
sp18_pred_2d = np.reshape(sp18_pred, (sp18_array.shape[1], sp18_array.shape[2]))

#Plot
plt.imshow(sp18_pred_2d)
plt.colorbar()

# See how many pixel classified as water/land
(sp18_pred == 1).sum()
(sp18_pred == 2).sum()
(sp18_pred == 3).sum()
#%% Apply to spring imagery | May 2021
sp21 = rasterio.open("S2_spring21_3band.tif")

# Read, change to 3D array
band_list = []
sp21_array = sp21.read()

for d in range(sp21_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(sp21_array[d,:,:]))
    
# Reshape array
sp21_array_rs = np.reshape(sp21_array, (3, 80472968))

# Change to DataFrame
sp21_df = pd.DataFrame(sp21_array_rs, columns=band_list).T
sp21_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)

# Standardize Data
sp21_scaler = StandardScaler()  
sp21_finaldf = sp21_scaler.fit_transform(sp21_df)
print(sp21_finaldf)

#Apply over all pixels in new image 
sp21_pred = forest_reg.predict(sp21_finaldf)

#Reshape to origial spring array
sp21_pred_2d = np.reshape(sp21_pred, (sp21_array.shape[1], sp21_array.shape[2]))

#Plot
plt.imshow(sp21_pred_2d)
plt.colorbar()

# See how many pixel classified as water/land
(sp21_pred == 1).sum()
(sp21_pred == 2).sum()
(sp21_pred == 3).sum()

#%%
# Write to GeoTiff   
transform = sp21.transform

with rasterio.open(
        "S2_Spring21_Classification.tif",
        mode="w",
        driver="GTiff",
        height=sp21_pred_2d.shape[0],
        width=sp21_pred_2d.shape[1],
        count=1,
        dtype=sp21_pred_2d.dtype,
        crs="EPSG:32603",
        transform=transform,
) as new_dataset:
        new_dataset.write(sp21_pred_2d, 1)
    
    
    
    
    
    
    