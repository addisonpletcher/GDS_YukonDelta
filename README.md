# Methodological Comparison of Water Classification: Yukon River Delta, AK

Addison Pletcher | Kelly Bonnville-Sexton 

![alt text](https://justfunfacts.com/wp-content/uploads/2017/08/yukon-river.jpg)

### Summary 
This project seeks to understand the differences in using SAR vs Optical Imagery for water classification. Overarchingly, this will aid in future work regarding timing and extent of ice breakup as it relates to travel between local communities in the Yukon Delta. Understanding where travel corridors exist during periods of melting/freezing is important to the safety of those traveling. 

### Use
The following scripts are for Optical Imagery classification: S2_WinterClassification.py, S2_classifier_unedited.py
The following scripts are for SAR Imagery classification: CreateFeatureMatrix_SARimage.py, SAR_Classification.py

### Problem Statement
Seasonal variability of river ice breakup alters accessibility throughout the region, and climate change stands to exacerbate these timing changes in the future. Radar data has potential to offer unique results, as optical imagery of the area was often obscured by clouds.

### Datasets
Earth Observation Imagery: Optical Imagery via Sentinel-2 and SAR Imagery via Sentinel-1 

### Python packages and dependencies: 
spyder, matplotlib, geopandas, numpy, rasterio, pandas, shapely, scikit-learn, netCDF4

### Methods
SAR Methodology: Acquire imagery, preprocess using GPT Script, combine imagery using net CDF file, create csv, create RandomForest classification. 
Sentinel-2 Methodology: Acquire imagery, landcover shapefile creation, load imagery and shapefiles to code, sample S2 data, train ML model, apply model for land cover prediction.

### Outcomes
Methodology Comparison in classifying water between Sentinel-1 SAR and Sentinel-2 Optical Imagery

### References
de Roda Husman, S. et al. (2021) Integrating intensity and context for improved supervised river ice classification from dual-pol sentinel-1 sar data, International Journal of Applied Earth Observation and Geoinformation. Elsevier. Available at: https://www.sciencedirect.com/science/article/pii/S0303243421000660 (Accessed: February 24, 2023). 
