import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import numpy as np

from sklearn.metrics import accuracy_score

## Read csvs

train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)



