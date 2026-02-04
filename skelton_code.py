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

# from area_classif import best_model
from utils import *

change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
    'Mega Projects': 5
}

# --- Read geojsons
train_df = gpd.read_file('train.geojson')
# train_df = train_df.set_crs("OGC:CRS84", allow_override=True)

test_df  = gpd.read_file('test.geojson')
# test_df = test_df.set_crs("OGC:CRS84", allow_override=True)

feature_cols = set()

train_df = encode_one_hot_types(train_df, feature_cols)
test_df  = encode_one_hot_types(test_df, feature_cols)

train_df = add_geometry_features(train_df, feature_cols)
test_df  = add_geometry_features(test_df, feature_cols)

train_df = add_max_gap_between_sets(train_df, feature_cols)
test_df = add_max_gap_between_sets(test_df, feature_cols)

train_df = add_last_state(train_df, feature_cols)
test_df  = add_last_state(test_df, feature_cols)

train_df = add_regressed_state(train_df, feature_cols)
test_df  = add_regressed_state(test_df, feature_cols)

img_c = [c for c in train_df.columns if c.startswith("img")]
feature_cols = sorted(set(list(feature_cols)))

train_y = train_df["change_type"].map(change_type_map).to_numpy()

print("n features:", len(feature_cols))
print("has last_:", any(c.startswith("last_") for c in feature_cols))
print("has nb_changes:", "nb_changes" in feature_cols)
print("has img_ aggregates:", any(c.endswith("_range") for c in feature_cols))

train_df = train_df.reindex(columns=feature_cols, fill_value=0)
test_df  = test_df.reindex(columns=feature_cols, fill_value=0)

train_df = train_df.fillna(0)
test_df  = test_df.fillna(0)

# --- X / y
train_x = train_df.to_numpy(dtype=float)
test_x  = test_df.to_numpy(dtype=float)

print(train_x.shape, train_y.shape, test_x.shape)
# print(len(img_c), len(feature_cols), img_c)

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=800,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    class_weight="balanced",
    random_state=0
)


cross_validation(3, model, train_x, train_y)
scores = cross_val_score(
    model,
    train_x,
    train_y,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)

print(scores.mean(), scores.std())

model.fit(train_x, train_y)

pred_y = model.predict(test_x)

# --- 5) Save submission (index = Id = index du test)
pred_df = pd.DataFrame({"change_type": pred_y}, index=test_df.index)
pred_df.index.name = "Id"
pred_df.to_csv("submission.csv")

print("Saved: submission.csv")
