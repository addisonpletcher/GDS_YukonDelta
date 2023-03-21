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
<<<<<<< HEAD:S2_classifier_v2.py

os.getcwd()
# To go back one folder in cwd
os.chdir("..")

#%% Open summer Sentinel-2 image for training
src = rasterio.open("S2_training_clip.tif")
show(src)

#Open classification shapefiles
land = gpd.read_file("QGIS\Land.shp")
water = gpd.read_file("QGIS\Water.shp")
print(land)
print(water)


#%% prep to sample Sentinel-2 
=======
>>>>>>> main:Scripts/S2_classifier_cleaned.py
from rasterio import mask as msk
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%% Open summer Sentinel-2 image for training
src = rasterio.open('...')
show(src)

#Open classification shapefiles
land = gpd.read_file('...')
water = gpd.read_file('...')
print(land)
print(water)

#%% prep to sample Sentinel-2 
land_proj = land.to_crs('EPSG:32603')
#land_proj.crs

water_proj = water.to_crs('EPSG:32603')
#water_proj.crs


#%% Sample Sentinel-2 data (land)
ca_l, ct_l = msk.mask(src, [mapping(land_proj.iloc[0].geometry)], crop=True)
sa_array_land, clipped_transform = msk.mask(src, [mapping(geom) for geom in land_proj.geometry], crop=True)

all_land = []
for b in range(src.count): 
    temp_list_L=sa_array_land[b][np.nonzero(sa_array_land[b])]
    all_land.append(temp_list_L)
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
# Convert to df
water_df = pd.DataFrame(all_water).T

#%% Combine dataframes, add column 
land_df['label'] = 1
water_df['label'] = 2

final_df = pd.concat([land_df,water_df],ignore_index=True)

<<<<<<< HEAD:S2_classifier_v2.py
#Rename Columns (Extra column?? No! band 8A is the last one, need to reconfig for this)
final_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)
final_df

#%% Train Machine Learning Model
from sklearn.preprocessing import StandardScaler

#Define Feature List 
feature_list = ['B3:Green', 'B4:Red', 'B8:NIR']
=======
#Rename Columns
final_df.rename(columns = {0:'...', 1:'...', ... , 12:'...'}, inplace = True)
final_df

#%% Train Machine Learning Model
#Define Feature List
feature_list = ['...', '...', '...']
>>>>>>> main:Scripts/S2_classifier_cleaned.py

#Define features/labels
X = final_df[feature_list]
y = final_df['label']

#Standardize data
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=feature_list)
print(df_scaled)

#Split data into training/testing subsets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, random_state=42)

#%% RandomForests model
forest_reg = RandomForestRegressor(n_estimators = 30) #Define
forest_reg.fit(X_train, y_train) #Fit

# Predict test labels predictions
predictions = forest_reg.predict(X_test)

#%% Compute Confusion Matrix
<<<<<<< HEAD:S2_classifier_v2.py
from sklearn.metrics import confusion_matrix
cm = (confusion_matrix(y_test, predictions.astype(int)))

import seaborn as sns
=======
cm = (confusion_matrix(y_test, predictions.astype(int)))
>>>>>>> main:Scripts/S2_classifier_cleaned.py
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')

#%% Import Spring S2 imagery
<<<<<<< HEAD:S2_classifier_v2.py
spring = rasterio.open("S2_spring21_3band.tif")
=======
spring = rasterio.open('...')
>>>>>>> main:Scripts/S2_classifier_cleaned.py

# Read, change to 3D array
band_list = []
spring_array = spring.read()

for d in range(spring_array.shape[0]-1): # inside () is selecting # bands... 
    band_list.append(np.ravel(spring_array[d,:,:]))
    
# Reshape array
<<<<<<< HEAD:S2_classifier_v2.py
spring_array_rs = np.reshape(spring_array, (3, 80472968))

# Change to DataFrame
spring_df = pd.DataFrame(spring_array_rs, columns=band_list).T
spring_df.rename(columns = {0:'B3:Green', 1:'B4:Red', 2:'B8:NIR'}, inplace = True)
=======
spring_array_rs = np.reshape(spring_array, (..., ...))

# Change to DataFrame
spring_df = pd.DataFrame(spring_array_rs, columns=band_list).T
spring_df.rename(columns = {0:'...', 1:'...', 2:'...'}, inplace = True)
>>>>>>> main:Scripts/S2_classifier_cleaned.py

# Standardize Data
spring_scaler = StandardScaler()  
spring_finaldf = spring_scaler.fit_transform(spring_df)
print(spring_finaldf)

#%% Apply over all pixels in new image 
<<<<<<< HEAD:S2_classifier_v2.py
ice_pred = forest_reg.predict(spring_finaldf)
=======
ice_pred = forest_reg.predict(spring_df)
>>>>>>> main:Scripts/S2_classifier_cleaned.py

#Reshape to origial spring array
ice_pred_2d = np.reshape(ice_pred, (spring_array.shape[1], spring_array.shape[2]))

#Plot
plt.imshow(ice_pred_2d)
plt.colorbar()
<<<<<<< HEAD:S2_classifier_v2.py

# See how many pixel classified as water/land
(ice_pred == 1).sum()
(ice_pred == 2).sum()

#%% Apply to Summer image (training image) to compare with SAR image
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

=======
>>>>>>> main:Scripts/S2_classifier_cleaned.py
