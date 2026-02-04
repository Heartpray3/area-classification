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

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

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

# train_df = add_geometry_features(train_df, feature_cols)
# test_df  = add_geometry_features(test_df, feature_cols)
train_df = add_max_gap_between_sets(train_df, feature_cols)
test_df = add_max_gap_between_sets(test_df, feature_cols)

train_df = add_last_state(train_df, feature_cols)
test_df  = add_last_state(test_df, feature_cols)

train_df = add_regressed_state(train_df, feature_cols)
test_df  = add_regressed_state(test_df, feature_cols)

img_c = [c for c in train_df.columns if c.startswith("img")]
feature_cols = sorted(set(list(feature_cols)))

train_y = train_df["change_type"].map(change_type_map).to_numpy()



train_df = train_df.reindex(columns=feature_cols, fill_value=0)
test_df  = test_df.reindex(columns=feature_cols, fill_value=0)

train_df = train_df.fillna(0)
test_df  = test_df.fillna(0)

# --- X / y
train_x = train_df.to_numpy(dtype=float)
test_x  = test_df.to_numpy(dtype=float)

print(train_x.shape, train_y.shape, test_x.shape)
print(len(img_c), len(feature_cols), img_c)

# # --- 1) Sous-échantillon stratifié (pour chercher vite)
# # Mets 20k-80k selon ton CPU; 50k est souvent un bon compromis.
# SUBSAMPLE_N = min(50000, len(train_x))
#
# # stratified split to get ~SUBSAMPLE_N points
# # (on fait un split "train/test" où train = subset)
# subset_frac = SUBSAMPLE_N / len(train_x)
# X_small, _, y_small, _ = train_test_split(
#     train_x, train_y,
#     train_size=subset_frac,
#     random_state=0,
#     stratify=train_y
# )
#
# print("Subset:", X_small.shape, y_small.shape)
#
# # --- 2) Pipeline + GridSearch
# pipe = Pipeline([
#     ("scaler", StandardScaler(with_mean=False)),
#     ("rbf", "passthrough"),
#     ("clf", LinearSVC())
# ])
#
# param_grid = [
#     # Modèle A: LinearSVC
#     {
#         "rbf": ["passthrough"],
#         "clf": [LinearSVC(dual="auto", max_iter=8000)],
#         "clf__C": [0.1, 1, 10, 100],
#     },
#     # Modèle B: RBF approx + SGD hinge
#     {
#         "rbf": [RBFSampler(random_state=0)],
#         "rbf__gamma": [1e-3, 1e-2, 1e-1],
#         "rbf__n_components": [500, 1000, 2000],
#         "clf": [SGDClassifier(loss="hinge", max_iter=8000, tol=1e-3, random_state=0)],
#         "clf__alpha": [1e-6, 1e-5, 1e-4],
#     }
# ]
#
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
#
# gs = GridSearchCV(
#     estimator=pipe,
#     param_grid=param_grid,
#     cv=cv,
#     scoring="f1_macro",
#     n_jobs=-1,
#     verbose=3,
#     refit=True
# )
#
# gs.fit(X_small, y_small)
# print("\n===== BEST MODEL =====")
# print(gs.best_estimator_)
#
# print("BEST PARAMS:", gs.best_params_)
# print("BEST CV F1:", gs.best_score_)
#
pipe = Pipeline([
    ("clf", BernoulliNB())  # placeholder
])

param_grid = [
    # ===== BernoulliNB =====
    {
        "clf": [BernoulliNB()],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0],
        "clf__binarize": [None, 0.0],   # None si déjà binaire, 0.0 sinon
        "clf__fit_prior": [True, False],
    },

    # ===== MultinomialNB =====
    {
        "clf": [MultinomialNB()],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0],
        "clf__fit_prior": [True, False],
    }
]


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

gs = GridSearchCV(
    pipe, param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=3,
    refit=True
)

gs.fit(train_x, train_y)

# # --- 3) Refit sur tout le train avec le meilleur modèle
best_model = gs.best_estimator_

print("\n===== BEST MODEL =====")
print(gs.best_estimator_)

print("BEST PARAMS:", gs.best_params_)
print("BEST CV F1:", gs.best_score_)

# best_model = Pipeline([
#     ("scaler", StandardScaler(with_mean=False)),
#     ("rbf", RBFSampler(
#         gamma=0.01,
#         n_components=2000,
#         random_state=0
#     )),
#     ("clf", SGDClassifier(
#         loss="hinge",
#         alpha=1e-5,
#         max_iter=8000,
#         tol=1e-3,
#         random_state=0
#     ))
# ])

cross_validation(3, best_model, train_x, train_y)

best_model.fit(train_x, train_y)


# --- 4) Predict test
pred_y = best_model.predict(test_x)
print("pred:", pred_y.shape)


# --- 5) Save submission (index = Id = index du test)
pred_df = pd.DataFrame({"change_type": pred_y}, index=test_df.index)
pred_df.index.name = "Id"
pred_df.to_csv("submission.csv")

print("Saved: submission.csv")
