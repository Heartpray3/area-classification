

"""
Skeleton + GridSearch on small subset -> train full -> predict
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
print("1. Chargement des données...")
train_df = gpd.read_file('subset_tests.geojson')

# gdf_proj = train_df.to_crs(train_df.estimate_utm_crs())

a = train_df.geometry.convex_hull.area