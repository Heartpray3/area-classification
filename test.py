import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


train_df = gpd.read_file('subset_tests.geojson', index_col=0)
train_df.geometry.set_crs(epsg=4979, allow_override=True)
# train_df.geometry  = train_df.geometry.buffer(0)

print(train_df.geometry.crs)
# utm_epsg = estimate_utm_crs(train_df)
# gdf = train_df.to_crs(epsg=3857)
# print(gdf.geometry.isna().sum())
# print(gdf)
