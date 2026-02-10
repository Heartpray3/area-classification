"""
Remote Sensing Change Detection – Full ML pipeline.
Target: F1-score >= 95% (weighted). Integrates status-order features (max_gap, regression, last_state one-hot).
"""
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score, classification_report
import re
import os

try:
    from features_extra import add_max_gap_features, add_last_state_onehot, add_regression_flags
except ImportError:
    add_max_gap_features = add_last_state_onehot = add_regression_flags = None
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
CV_FOLDS = 2  # 2 folds = plus rapide, on vise 95% F1
CACHE_FEATURES = True  # save/load X_train, X_test, y to avoid recomputing
# v3 = sans status_extra (meilleur score ~86%). v2 = avec status_extra (souvent plus bas).
USE_STATUS_EXTRA = False  # False = moins de features, score plus haut en général
CACHE_DIR = "feature_cache_v3" if not USE_STATUS_EXTRA else "feature_cache_v2"
USE_ENSEMBLE = True  # RF + HGB (moyenne des proba) = souvent +1–2% F1
TARGET_F1 = 0.95
SUBSAMPLE = None  # Full data pour viser 95% CV F1

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
    log_perimeter = np.maximum(perimeter, 1e-10)
    log_perimeter = np.log1p(log_perimeter)
    # ratio area/perimeter = forme (elongated vs compact)
    area_perim = np.where(perimeter > 1e-10, area / perimeter, 0)
    return np.column_stack([area, perimeter, compactness, log_area, log_perimeter, area_perim])


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

    # 4b) Optionnel: status-order (max_gap, last_state one-hot, regression). Désactivé = souvent meilleur F1.
    if USE_STATUS_EXTRA and add_max_gap_features is not None:
        X_max_gap = add_max_gap_features(df)
        X_last_oh = add_last_state_onehot(df)
        X_regression = add_regression_flags(df)
        X_status_extra = np.hstack([X_max_gap, X_last_oh, X_regression])
    else:
        X_status_extra = None

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
    if X_status_extra is not None:
        X = np.hstack([X_geom, X_urban, X_geo, X_status, X_status_extra, X_img, X_dates])
    else:
        X = np.hstack([X_geom, X_urban, X_geo, X_status, X_img, X_dates])
    return X, (urban_tokens, geo_tokens, status_encoder, img_medians if fit else None)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "X_train.npy")
    # Si on veut sans status_extra et v3 n'existe pas, charger v2 et retirer les cols
    if not USE_STATUS_EXTRA and not os.path.isfile(cache_path) and os.path.isfile("feature_cache_v2/X_train.npy"):
        cache_path = "feature_cache_v2/X_train.npy"
        CACHE_DIR_LOAD = "feature_cache_v2"
    else:
        CACHE_DIR_LOAD = CACHE_DIR
    test_ids = None
    test_df = None
    if CACHE_FEATURES and os.path.isfile(cache_path):
        print("Loading cached features from", CACHE_DIR_LOAD, "...")
        X_train = np.load(os.path.join(CACHE_DIR_LOAD, "X_train.npy"))
        X_test = np.load(os.path.join(CACHE_DIR_LOAD, "X_test.npy"))
        y = np.load(os.path.join(CACHE_DIR_LOAD, "y.npy"))
        # Si on a chargé v2 (96 cols) mais qu'on veut sans status_extra, retirer les 17 colonnes du milieu
        if not USE_STATUS_EXTRA and X_train.shape[1] == 96:
            idx_s, idx_e = 32, 49  # bloc status_extra dans v2
            X_train = np.hstack([X_train[:, :idx_s], X_train[:, idx_e:]])
            X_test = np.hstack([X_test[:, :idx_s], X_test[:, idx_e:]])
            print("Dropped status_extra cols (use v2 cache as base): now", X_train.shape[1], "cols")
        tid_path = os.path.join(CACHE_DIR_LOAD, "test_ids.npy")
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

    _use_light_model = False
    if SUBSAMPLE is not None and SUBSAMPLE < 1.0:
        from sklearn.model_selection import train_test_split
        X_train, _, y, _ = train_test_split(
            X_train, y, train_size=SUBSAMPLE, stratify=y, random_state=RANDOM_STATE
        )
        print("Subsampled train to {:.0%}: shape {}".format(SUBSAMPLE, X_train.shape))
        _use_light_model = True

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Poids des classes: balanced mais plafonné
    _, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = 6
    balanced_w = n_samples / (n_classes * counts)
    cap = 6.0  # plafond poids classes rares (évite sur-prédiction)
    class_weights = np.minimum(balanced_w, cap)
    class_weight_dict = {i: float(class_weights[i]) for i in range(6)}
    sw = class_weights[y]

    # Ensemble RF + HGB + ExtraTrees + XGBoost (si installé) pour viser F1 >= 95%
    if _use_light_model:
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=14, class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1)
        hgb = HistGradientBoostingClassifier(max_iter=120, max_depth=10, min_samples_leaf=30, l2_regularization=0.2, learning_rate=0.06, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.12, n_iter_no_change=10)
        et = ExtraTreesClassifier(n_estimators=180, max_depth=20, min_samples_leaf=14, class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1)
        xgb_clf = xgb.XGBClassifier(n_estimators=220, max_depth=10, learning_rate=0.06, use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1) if HAS_XGB else None
    else:
        rf = RandomForestClassifier(n_estimators=600, max_depth=28, min_samples_leaf=8, class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1)
        hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=14, min_samples_leaf=22, l2_regularization=0.15, learning_rate=0.05, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.08, n_iter_no_change=18)
        et = ExtraTreesClassifier(n_estimators=550, max_depth=28, min_samples_leaf=8, class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1)
        xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=11, learning_rate=0.055, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1) if HAS_XGB else None

    RUN_CV = True  # F1 CV ensemble sur le train (objectif >= 95%)
    cv_f1 = None
    if RUN_CV:
        print("Cross-validation ensemble ({} folds, F1 weighted)...".format(CV_FOLDS), flush=True)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        y_oof = np.zeros_like(y, dtype=np.float64)
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            sw_tr = sw[tr_idx]
            rf_f = clone(rf)
            hgb_f = clone(hgb)
            et_f = clone(et)
            rf_f.fit(X_tr, y_tr)
            hgb_f.fit(X_tr, y_tr, sample_weight=sw_tr)
            et_f.fit(X_tr, y_tr)
            probs = [rf_f.predict_proba(X_val), hgb_f.predict_proba(X_val), et_f.predict_proba(X_val)]
            if xgb_clf is not None:
                xgb_f = clone(xgb_clf)
                xgb_f.fit(X_tr, y_tr, sample_weight=sw_tr)
                probs.append(xgb_f.predict_proba(X_val))
            p = np.mean(probs, axis=0)
            y_oof[val_idx] = np.argmax(p, axis=1)
            f1_fold = f1_score(y_val, y_oof[val_idx], average="weighted")
            print("  Fold {} F1 weighted: {:.4f}".format(fold + 1, f1_fold), flush=True)
        cv_f1 = f1_score(y, y_oof, average="weighted")
        print("  Ensemble CV F1 weighted: {:.4f}".format(cv_f1), flush=True)
        with open("cv_result.txt", "w") as f:
            f.write("CV_F1={:.4f}\n".format(cv_f1))

    print("Training final models on full data...")
    rf.fit(X_train, y)
    print("Training HGB...")
    hgb.fit(X_train, y, sample_weight=sw)
    print("Training ExtraTrees...")
    et.fit(X_train, y)
    if xgb_clf is not None:
        print("Training XGBoost...")
        xgb_clf.fit(X_train, y, sample_weight=sw)

    def predict_ensemble(X):
        probs = [rf.predict_proba(X), hgb.predict_proba(X), et.predict_proba(X)]
        if xgb_clf is not None:
            probs.append(xgb_clf.predict_proba(X))
        return np.argmax(np.mean(probs, axis=0), axis=1)

    y_pred = predict_ensemble(X_train)
    print("\nTrain set classification report (F1):")
    print(classification_report(y, y_pred, target_names=list(CHANGE_TYPE_MAP.keys()), zero_division=0))

    # Predict test (toujours l'ensemble à 3 pour viser 95%)
    pred_test = predict_ensemble(X_test)
    ids = test_ids if test_ids is not None else (test_df.index if hasattr(test_df.index, "__len__") else np.arange(len(test_df)))
    out = pd.DataFrame({"Id": ids, "change_type": pred_test})
    out.to_csv("submission.csv", index=False)
    f1_w = f1_score(y, y_pred, average="weighted")
    f1_macro = f1_score(y, y_pred, average="macro")
    print("\nTrain F1 weighted: {:.4f}  F1 macro: {:.4f}".format(f1_w, f1_macro))
    print("Saved submission.csv ({} rows).".format(len(out)))
    if cv_f1 is not None and cv_f1 >= TARGET_F1:
        print("Target CV F1 >= {} reached.".format(TARGET_F1))
    elif cv_f1 is not None:
        print("CV F1 {:.4f} < target {}. Run with full data (SUBSAMPLE=None) and large models; 95% may need XGBoost/LightGBM or more features.".format(cv_f1, TARGET_F1))
    return cv_f1 if cv_f1 is not None else f1_w


if __name__ == "__main__":
    main()
