# -*- coding: utf-8 -*-
"""
This script is to compare the classifications between the Sentinel-1 SAR methodology and the Sentinel-2 Optical methodology. 

Requirements for this script include:
    - Classified SAR image as GeoTiff file (product of SAR_Classification.py)
    - Classified Sentinel-2 image as GeoTiff file (product of S2_Classifier.py)

@author: Kelly Bonnville-Sexton
"""
#%%
# Import libraries
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt

#Bring in classified SAR
src = rasterio.open(...)
src

src.transform

SAR = src.read(1)
SAR

#Bring in classifed Sentinel 2
src2 = rasterio.open(...)
src2

src2.transform
S2 = src2.read(1)
S2

#%%
# Re-assign ice classes so only one class for all ice
SAR[SAR == 2] = 1
SAR[SAR == 4] = 1

#mask SAR with S2
mask = SAR[S2 == 3]
unique, counts = np.unique(mask, return_counts=True)

#%%
#compare number of pixels classified as ice 
icechange_df = pd.DataFrame(list(zip(unique, counts, (counts/mask.shape[0])*100)),
                 columns=['class', 'count', 'fraction'])
icechange_df

#%%
#plot S2-ice vs SAR- ice
ice2ice = (S2 == 3) & (SAR == 1)
fig, ax = plt.subplots(figsize=(16,8))
im = ax.imshow(ice2ice.astype(int), cmap='Blues')
ax.set_title("Ice to ice in Alaska between SAR and Sentinel-2", fontsize=14)
fig.colorbar(im, orientation='vertical')


#%%
#plot S2- non-ice vs SAR ice 
water2ice = (S2 == 2) & (SAR == 1)
fig, ax = plt.subplots(figsize=(16,8))
im2 = ax.imshow(water2ice.astype(int), cmap='Blues')
ax.set_title("Water to ice in Alaska between SAR and Sentinel-2", fontsize=14)
fig.colorbar(im2, orientation='vertical')

