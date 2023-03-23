# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:36:09 2023

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
src = rasterio.open("S2_winter23.tif")
show(src)

#Open classification shapefiles
ice = gpd.read_file("ice.shp")
not_ice = gpd.read_file("not_ice.shp")

#%% prep to sample Sentinel-2 
from rasterio import mask as msk
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import pandas as pd

ice_proj = ice.to_crs('EPSG:32603')
not_ice_proj = not_ice.to_crs('EPSG:32603')
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

#%% Sample Sentinel-2 data (not ice)
ca_n, ct_n = msk.mask(src, [mapping(not_ice_proj.iloc[0].geometry)], crop=True)
sa_array_not_ice, clipped_transform = msk.mask(src, [mapping(geom) for geom in not_ice_proj.geometry], crop=True)

all_not_ice = []
for d in range(src.count): 
    #Drop zeros, mask to make one dimensional list (all bands)
    temp_list_L=sa_array_not_ice[d][np.nonzero(sa_array_not_ice[d])]
    all_not_ice.append(temp_list_L)

#Check length, should be amount of bands we have
#len(all_land)

# Convert to df
not_ice_df = pd.DataFrame(all_not_ice).T

#%% Combine dataframes, add column 
ice_df['label'] = 3
not_ice_df['label'] = 2

final_df = pd.concat([ice_df, not_ice_df],ignore_index=True)

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

#%% Apply to February Image to compare with SAR image
winter = rasterio.open("S2_winter23.tif")

# Read, change to 3D array
band_list = []
winter_array = winter.read()

for d in range(winter_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(winter_array[d,:,:]))
    
# Reshape array
winter_array_rs = np.reshape(winter_array, (3, 85058094))

# Change to DataFrame
winter_df = pd.DataFrame(winter_array_rs, columns=band_list).T
winter_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)

winter_scaler = StandardScaler()  
winter_finaldf = winter_scaler.fit_transform(winter_df)

#apply classification over all pixels
winter_pred = forest_reg.predict(winter_finaldf)

#Reshape to origial spring array
winter_pred_2d = np.reshape(winter_pred, (winter_array.shape[1], winter_array.shape[2]))

#Plot
plt.imshow(winter_pred_2d)
plt.colorbar()

#%%
# Write to GeoTiff   
transform = winter.transform

with rasterio.open(
        "S2_Winter23_Classification.tif",
        mode="w",
        driver="GTiff",
        height=winter_pred_2d.shape[0],
        width=winter_pred_2d.shape[1],
        count=1,
        dtype=winter_pred_2d.dtype,
        crs="EPSG:32603",
        transform=transform,
) as new_dataset:
        new_dataset.write(winter_pred_2d, 1)