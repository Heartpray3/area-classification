"""
Remote Sensing Change Detection – Full ML pipeline.
Target: F1-score > 90% on 6-class classification.
"""
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import re
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHANGE_TYPE_MAP = {
    "Demolition": 0,
    "Road": 1,
    "Residential": 2,
    "Commercial": 3,
    "Industrial": 4,
    "Mega Projects": 5,
}
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 3
CACHE_FEATURES = True  # save/load X_train, X_test, y to avoid recomputing
CACHE_DIR = "feature_cache"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    train_df = gpd.read_file("train.geojson")
    test_df = gpd.read_file("test.geojson")
    return train_df, test_df


def _multi_hot_tokens(series, sep=","):
    """Collect all unique tokens from comma-separated values."""
    all_tokens = set()
    for v in series.dropna().astype(str):
        for t in re.split(r"\s*,\s*", v.strip()):
            if t and t not in ("N,A", "nan"):
                all_tokens.add(t.strip())
    return sorted(all_tokens)


def _multi_hot_encode(series, tokens, sep=","):
    """Encode series as multi-hot matrix (one column per token)."""
    n = len(series)
    out = np.zeros((n, len(tokens)), dtype=np.float32)
    for i, v in enumerate(series.astype(str)):
        if pd.isna(series.iloc[i]):
            continue
        for t in re.split(r"\s*,\s*", v.strip()):
            t = t.strip()
            if t in tokens:
                out[i, tokens.index(t)] = 1.0
    return out


def _status_features(df, status_cols, status_encoder=None, fit=True):
    """Last status (encoded), transitions count, and full sequence encoded (5 cols). Vectorized."""
    status_cols = [c for c in status_cols if c in df.columns]
    if fit:
        status_encoder = LabelEncoder()
        all_vals = np.concatenate([df[c].fillna("__MISSING__").values for c in status_cols])
        status_encoder.fit(np.unique(all_vals))
    encoder_dict = {c: i for i, c in enumerate(status_encoder.classes_)}
    seq_encoded = np.zeros((len(df), len(status_cols)), dtype=np.float64)
    for j, col in enumerate(status_cols):
        seq_encoded[:, j] = df[col].fillna("__MISSING__").map(encoder_dict).fillna(0).values
    transitions = (np.diff(seq_encoded, axis=1) != 0).sum(axis=1).astype(np.float64)
    last_encoded = seq_encoded[:, -1]
    out = np.column_stack([last_encoded, transitions, seq_encoded])
    return out, status_encoder


def build_geometry_features(gdf):
    """Area, perimeter, compactness, log-scale from geometry."""
    geoms = gdf.geometry
    area = np.asarray(geoms.area, dtype=np.float64)
    perimeter = np.asarray(geoms.length, dtype=np.float64)
    # compactness = 4*pi*area / perimeter^2 (1 = circle)
    compactness = np.ones_like(area, dtype=np.float64)
    np.place(compactness, perimeter > 1e-10, (4 * np.pi * area)[perimeter > 1e-10] / (perimeter[perimeter > 1e-10] ** 2))
    np.place(compactness, ~np.isfinite(compactness), 0)
    log_area = np.log1p(np.maximum(area, 0))
    log_perimeter = np.log1p(np.maximum(perimeter, 0))
    return np.column_stack([area, perimeter, compactness, log_area, log_perimeter])


def build_feature_matrix(df, urban_tokens=None, geo_tokens=None, status_encoder=None, fit=True):
    """
    Build feature matrix: geometry + multi-hot urban/geo + status features + img + dates.
    When fit=True, compute and return tokens/encoders; when fit=False, use provided ones.
    """
    status_cols = [f"change_status_date{i}" for i in range(5)]
    img_cols = [c for c in df.columns if c.startswith("img_")]
    date_cols = [f"date{i}" for i in range(5)]

    # 1) Geometry
    X_geom = build_geometry_features(df)

    # 2) Urban type multi-hot
    if fit:
        urban_tokens = _multi_hot_tokens(df["urban_type"])
    X_urban = _multi_hot_encode(df["urban_type"], urban_tokens)

    # 3) Geography type multi-hot
    if fit:
        geo_tokens = _multi_hot_tokens(df["geography_type"])
    X_geo = _multi_hot_encode(df["geography_type"], geo_tokens)

    # 4) Status sequence + transitions
    X_status, status_encoder = _status_features(df, status_cols, status_encoder=status_encoder, fit=fit)

    # 5) Image features: fill NaN with median (per column, computed on train)
    X_img = df[img_cols].values.astype(np.float64)
    if fit:
        img_medians = np.nanmedian(X_img, axis=0)
    else:
        img_medians = img_medians_global  # set by caller for test
    mask = ~np.isfinite(X_img)
    X_img = np.where(mask, np.broadcast_to(img_medians, X_img.shape), X_img)
    np.clip(X_img, -1e10, 1e10, out=X_img)
    # 5b) Temporal image stats: mean/std across 5 dates (6 channels per date -> 12 features)
    n_dates = 5
    n_ch = 6  # r_mean, g_mean, b_mean, r_std, g_std, b_std per date
    img_reshaped = X_img.reshape(-1, n_dates, n_ch)
    img_mean_over_time = np.nanmean(img_reshaped, axis=1)
    img_std_over_time = np.nanstd(img_reshaped, axis=1)
    img_std_over_time[~np.isfinite(img_std_over_time)] = 0
    X_img_extra = np.hstack([img_mean_over_time, img_std_over_time])
    X_img = np.hstack([X_img, X_img_extra])

    # 6) Optional: date ordinals (simple)
    X_dates = np.zeros((len(df), len(date_cols)), dtype=np.float64)
    for j, col in enumerate(date_cols):
        if col not in df.columns:
            continue
        vals = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")
        X_dates[:, j] = vals.astype("int64") // 10**9  # seconds for scale

    # Replace NaT-derived NaN with 0
    X_dates[~np.isfinite(X_dates)] = 0

    # Concatenate all
    X = np.hstack([X_geom, X_urban, X_geo, X_status, X_img, X_dates])
    return X, (urban_tokens, geo_tokens, status_encoder, img_medians if fit else None)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "X_train.npy")
    test_ids = None
    test_df = None
    if CACHE_FEATURES and os.path.isfile(cache_path):
        print("Loading cached features...")
        X_train = np.load(os.path.join(CACHE_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(CACHE_DIR, "X_test.npy"))
        y = np.load(os.path.join(CACHE_DIR, "y.npy"))
        tid_path = os.path.join(CACHE_DIR, "test_ids.npy")
        if os.path.isfile(tid_path):
            test_ids = np.load(tid_path)
        else:
            test_df = gpd.read_file("test.geojson")
        train_df = gpd.read_file("train.geojson")
    else:
        print("Loading data...")
        train_df, test_df = load_data()
        y = train_df["change_type"].map(CHANGE_TYPE_MAP).values

        print("Building train features...")
        X_train, aux = build_feature_matrix(train_df, fit=True)
        urban_tokens, geo_tokens, status_encoder, img_medians = aux
        global img_medians_global
        img_medians_global = img_medians

        # Build test with same token lists / encoders as train
        X_test, _ = build_feature_matrix(
            test_df,
            urban_tokens=urban_tokens,
            geo_tokens=geo_tokens,
            status_encoder=status_encoder,
            fit=False,
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Replace any remaining inf/nan
        X_train[~np.isfinite(X_train)] = 0
        X_test[~np.isfinite(X_test)] = 0

        if CACHE_FEATURES:
            np.save(os.path.join(CACHE_DIR, "X_train.npy"), X_train)
            np.save(os.path.join(CACHE_DIR, "X_test.npy"), X_test)
            np.save(os.path.join(CACHE_DIR, "y.npy"), y)
            np.save(os.path.join(CACHE_DIR, "test_ids.npy"), test_df.index.values)
            print("Cached features to", CACHE_DIR)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Model: RandomForest (fast, parallel) with balanced class weight for high F1
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=26,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Optional: quick CV (set RUN_CV=True for full CV)
    RUN_CV = False
    if RUN_CV:
        print("Cross-validating (F1 weighted, {} folds)...".format(CV_FOLDS))
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            clf, X_train, y, cv=cv, scoring="f1_weighted", n_jobs=1, return_train_score=True
        )
        print("CV F1 weighted: mean = {:.4f}, std = {:.4f}".format(scores["test_score"].mean(), scores["test_score"].std()))
        cv_f1 = scores["test_score"].mean()
    else:
        cv_f1 = None

    print("Training final model...")
    clf.fit(X_train, y)
    y_pred = clf.predict(X_train)
    print("\nTrain set classification report (F1):")
    print(classification_report(y, y_pred, target_names=list(CHANGE_TYPE_MAP.keys()), zero_division=0))

    # Predict test
    pred_test = clf.predict(X_test)
    ids = test_ids if test_ids is not None else (test_df.index if hasattr(test_df.index, "__len__") else np.arange(len(test_df)))
    out = pd.DataFrame({"Id": ids, "change_type": pred_test})
    out.to_csv("submission.csv", index=False)
    print("\nSaved submission.csv with", len(out), "predictions.")
    return cv_f1 if cv_f1 is not None else f1_score(y, y_pred, average="weighted")


if __name__ == "__main__":
    main()
