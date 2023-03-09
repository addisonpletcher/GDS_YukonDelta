# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:10:44 2023

@author: addyp
"""
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx

import os
os.environ['USE_PYGEOS'] = '0'
# from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from descartes import PolygonPatch

import matplotlib.pyplot as plt
import folium


#Specify type of data 
tags = {'trail': True}    #unsure if 'trail' is correct here

# Download building geometries from OSM
gdf = ox.geometries_from_place('Alakanuk, Alaska, USA', tags) #am I able to do this using a ROI polygon or just listing multiple towns?

# Filter winter trails
winter_trails = gdf[gdf['Highway']].reset_index()
print(winter_trails.shape) #tells us number of trails we have to work with
# https://wiki.openstreetmap.org/wiki/Key:winter_road 
# https://wiki.openstreetmap.org/wiki/Map_features#Paths  track (for forestry and ag) vs  winter_Rd vs winter trail?


#%% Plot network data

# Define coordinates of Alakanuk
lat_lon = (62.78139939633688, -164.5320824520905)

# Define drivable winter trail 20 km around Alakanuk
g = ox.graph_from_point(lat_lon, dist=3200, network_type='drive')

# Plot map
fig, ax = ox.plot_graph(g, bgcolor='white', node_color='black', edge_color='grey', node_size=5)