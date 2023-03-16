# -*- coding: utf-8 -*-
"""
This script is adapted from the RemoteSensing-of-River-Ice tutorial (https://github.com/SdeRodaHusman/remotesensing-of-river-ice/blob/main/README.md).
This script is used to create a csv-file with all feature information of the preprocessed SAR
images included. Each row in the matrix describes a pixel in the SAR image with all feature values 
that are computed (e.g. intensity, polarimetric and texture features.)
Since these files are very large, this scripts produces seperate csv-files for each feature class,
(i.e. intensity, polarimetric and texture features). Therefore, three csv-files are produced for one SAR image.    
 
As an input product, the collocated SAR image is required in NetCDF-format, with in each band
a preprocced feature of interest. This is the output file of: Scripts/XMLgraphs/Combine
PreprocessedImages.xml
In this example, the following features were used:
    - Intenisty features: VV, VH
    - Texture features: GLCM mean, GLCM variance, GLCM correlation
    
@author: Kelly Bonnville-Sexton , created on  March 2, 2023
"""

#%%
# Import libraries

import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import numpy.ma as ma
#%% 

#give the locations of the NetCDF files with preprocessed images
data1 =  ...
data2 =  ...

#(Give your output path to save the feature matrix of the prepocessed SAR image including all features of interest)
output_path = os.path.abspath("...")  

# Read the NetCDF file
read_nc = Dataset(data2, 'r')
read_nc2 = Dataset(data1, 'r')

#%%
# Create array of lon and lat coordinates of each pixel
lon = read_nc.variables['lon'][:]
lat = read_nc.variables['lat'][:]

lon_unmasked = ma.getdata(lon)
lat_unmasked = ma.getdata(lat)

XX_lon,YY_lat= np.meshgrid(lon_unmasked,lat_unmasked,sparse=False)

#%%

# Reformat VV ;
VV = read_nc.variables['Gamma0_VV_db'][:]
VV_unmasked = ma.getdata(VV)
VV_unmasked[VV_unmasked == 0] = 'nan'
VV_unravel = np.ravel(VV_unmasked)
VV_final = VV_unravel[~np.isnan(VV_unravel)]

#%%

# Give location of nans (all pixels outside the shapefile of interest, used when preprocessing the SAR image in previous steps)
location_nans = np.isnan(VV_unravel)

#%%
# Mask lat and lon pixels at locations with NaN values (outside area of interest)
lon_unmasked = ma.getdata(XX_lon)
lon_unravel = np.ravel(lon_unmasked)
(lon_unravel[location_nans])=-9999
lon_final = lon_unravel[lon_unravel != -9999]

lat_unmasked = ma.getdata(YY_lat)
lat_unravel = np.ravel(lat_unmasked)
(lat_unravel[location_nans])=-9999
lat_final = lat_unravel[lat_unravel != -9999]

#%%
# Reformat VH    
VH = read_nc.variables['Gamma0_VH_db'][:]
VH_unmasked = ma.getdata(VH)
VH_unravel = np.ravel(VH_unmasked)
(VH_unravel[location_nans])=-9999
VH_final = VH_unravel[VH_unravel != -9999]

#%%
# Convert to dB values
depol_ratio = (VH_final / VV_final)

#%%

# Reformat texture features
GLCMmean = read_nc2.variables['Gamma0_VH_db_GLCMMean'][:]
GLCMmean_unmasked = ma.getdata(GLCMmean)
GLCMmean_unravel = np.ravel(GLCMmean_unmasked)
(GLCMmean_unravel[location_nans])=-9999
GLCMmean_final = GLCMmean_unravel[GLCMmean_unravel != -9999]

GLCMvariance = read_nc2.variables['Gamma0_VH_db_GLCMVariance'][:]
GLCMvariance_unmasked = ma.getdata(GLCMvariance)
GLCMvariance_unravel = np.ravel(GLCMvariance_unmasked)
(GLCMvariance_unravel[location_nans])=-9999
GLCMvariance_final = GLCMvariance_unravel[GLCMvariance_unravel != -9999]

GLCMcorrelation = read_nc2.variables['Gamma0_VH_db_GLCMCorrelation'][:]
GLCMcorrelation_unmasked = ma.getdata(GLCMcorrelation)
GLCMcorrelation_unravel = np.ravel(GLCMcorrelation_unmasked)
(GLCMcorrelation_unravel[location_nans])=-9999
GLCMcorrelation_final = GLCMcorrelation_unravel[GLCMcorrelation_unravel != -9999]


#%%
#Write data to CSV file
df_final = pd.DataFrame({'lon': lon_final, 'lat':lat_final, 'VV_instensity': VV_final, 
                         'VH_intensity': VH_final, 'GLCM_mean': GLCMmean_final})
df_final.to_csv(output_path +'\FeatureMatrix_subset.csv', index = False, header=True)

#%%
# Write data to CSV file 
df_final = pd.DataFrame({'lon': lon_final, 'lat':lat_final, 'VV_instensity': VV_final, 
                         'VH_intensity': VH_final, 'Depol_ratio':depol_ratio, 'GLCM_mean': GLCMmean_final, 'GLCM_variance': GLCMvariance_final, 'GLCM_correlation': GLCMcorrelation_final})
                         
my_file = 'FeatureMatrix_SARimage.csv'
df_final.to_csv(output_path, my_file, index = False, header=True)  
