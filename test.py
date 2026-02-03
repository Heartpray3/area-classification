import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


train_df = gpd.read_file('subset_tests.geojson', index_col=0)
train_df.geometry  = train_df.geometry.buffer(0)

print(train_df.geometry.area.isna().sum())
print(train_df.crs)

gdf = train_df.to_crs(epsg=6933)
print(train_df.geometry[gdf.geometry.isna()])
print(gdf)
