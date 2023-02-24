# GDS: Effects of Ice Breakup on the Yukon Delta, AK

Addison Pletcher | Kelly Bonnville-Sexton 

![alt text](https://justfunfacts.com/wp-content/uploads/2017/08/yukon-river.jpg)

An analysis on river ice breakup timing in the Yukon River Delta of Alaska. The goal of this study is to better understand the seasonal relationships between communities that utilize the waterways for travel. 

### Summary 
This project seeks to understand the timing and extent of ice breakup as it relates to travel between local communities in the Yukon Delta. Understanding where travel corridors exist during periods of melting/freezing is important to the safety of those traveling. 

### Problem Statement
Seasonal variability of river ice breakup alters accessibility throughout the region, and climate change stands to exacerbate these timing changes in the future. 

### Datasets
Earth Observation Imagery: Sentinel-2 and SAR (Sentinel-1) 
Open Street Map: travel routes 

### Python packages and dependencies: 
spyder, matplotlib, geopandas, numpy, rasterio, osmnx, networkx, pandas, jupyter, shapely, jupyterlab, rioxarray, scikit-learn, selenium, bs4, descartes, momepy, folium

### Methods
SAR Methodology: Acquire imagery, preprocess using GPT Script, combine imagery using net CDF file, create csv, create RandomForest classification. 
Sentinel-2 Methodology: Acquire imagery, landcover shapefile creation, load imagery and shapefiles to code, sample S2 data, train ML model, apply model for land cover prediction.

### Outcomes
Information regarding timing of safe travel, changes in timing of ice breakup, etc.

### References
de Roda Husman, S. et al. (2021) Integrating intensity and context for improved supervised river ice classification from dual-pol sentinel-1 sar data, International Journal of Applied Earth Observation and Geoinformation. Elsevier. Available at: https://www.sciencedirect.com/science/article/pii/S0303243421000660 (Accessed: February 24, 2023). 
