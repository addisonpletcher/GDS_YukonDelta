# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:02:31 2023

@author: addyp

This script is used to create a confusion matrix between the SAR data and Sentinel-2 data. The following input files are required:
    - Classified SAR image (product of SAR_Classification.py)
    - Classified Sentinel-2 image (product of S2_Classifier.py) 
"""

#%%
#import library
from sklearn.metrics import confusion_matrix
import seaborn as sns

SAR = "Classification_clipped.tif"
S2 = "S2_training_clip.tif"

#Compute Confusion Matrix
cf_matrix = (confusion_matrix(SAR, S2))






'''
import sklearn
from matplotlib import seaborn as sns #Give locations of SAR and Sentinel-2 images
SAR = "Documents\Geospatial_Data\Final\Outputs\Classification_clipped.tif"
S2 = "Documents\Geospatial_Data\Final\S2_training_clip.tif"