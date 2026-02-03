"""
This script can be used as skelton code to read the challenge train and test
geojsons, to train a trivial model, and write data to the submission file.
"""
import geopandas as gpd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from utils import *

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
       'Mega Projects': 5}

## Read csvs

train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

## Filtering column "mail_type"
# train_x = np.asarray(train_df[['geometry']].area)
# train_x = train_x.reshape(-1, 1)
# train_y = train_df['change_type'].apply(lambda x: change_type_map[x])
#
# test_x = np.asarray(test_df[['geometry']].area)
# test_x = test_x.reshape(-1, 1)

# train_df = add_geometry_features(train_df)
# test_df = add_geometry_features(test_df)


train_df = add_max_gap_between_sets(train_df)
test_df  = add_max_gap_between_sets(test_df)

feature_cols = [
    "pre_post_construction_time_gap_days",
    "polygon_area_m2",
    "polygon_perimeter_m",
    "compactness",
]

# X (numpy arrays)
train_x = train_df[feature_cols].to_numpy(dtype=float)
test_x  = test_df[feature_cols].to_numpy(dtype=float)
train_x.reshape(-1, train_x.shape[-1])
test_x.reshape(-1, test_x.shape[-1])

# y (labels encodés)
train_y = train_df["change_type"].map(change_type_map).to_numpy()

print(train_x.shape, train_y.shape, test_x.shape)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf", "passthrough"),   # activé seulement pour RBF approx
    ("clf", LinearSVC())      # remplacé via param_grid
])

param_grid = [
    # --- Modèle 1: LinearSVC ---
    {
        "rbf": ["passthrough"],
        "clf": [LinearSVC(dual="auto", max_iter=5000)],
        "clf__C": [0.01, 0.1, 1, 10, 100],
    },
    # --- Modèle 2: RBF approximé + SGD (hinge) ---
    {
        "rbf": [RBFSampler(random_state=0)],
        "rbf__gamma": [1e-3, 1e-2, 1e-1],
        "rbf__n_components": [500, 1000, 2000],
        "clf": [SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3)],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
    }
]

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="f1_macro",
    verbose=1
)

# rng = np.random.default_rng(0)
# idx = rng.choice(len(train_x), size=3000, replace=False)

# X_small = train_x[idx]
# y_small = train_y[idx]

# gs.fit(X_small, y_small)   # sous-échantillon recommandé
# print("BEST:", gs.best_params_, f"{gs.best_score_=}")
# best_model = gs.best_estimator_
best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf", RBFSampler(
        gamma=0.1,
        n_components=500,
        random_state=0
    )),
    ("clf", SGDClassifier(
        loss="hinge",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3
    ))
])

# entraînement final sur tout le train
best_model.fit(train_x, train_y)

# entraînement final sur tout le train
# best_model.fit(train_x, train_y)

## Train a simple OnveVsRestClassifier using featurized data
# neigh = KNeighborsClassifier(n_neighbors=3)

cross_validation(5, best_model, train_x, train_y)
best_model.fit(train_x, train_y)
pred_y = best_model.predict(test_x)

print(pred_y.shape)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')