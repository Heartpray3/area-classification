"""
train.py - 3-fold CV + train final + submission
(compatible with older xgboost versions + avoids pandas/QuantileDMatrix crash)

- 3-fold Stratified CV
- No early stopping (your xgboost doesn't support early_stopping_rounds/callbacks in sklearn API)
- Avoids XGBoostError from QuantileDMatrix/columnar by converting X/X_test to numpy float32
Outputs:
- model.pkl
- submission.csv
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from xgboost import XGBClassifier


# ---------------------------
# Load
# ---------------------------
TRAIN_PATH = "train_features.parquet"
TEST_PATH  = "test_features.parquet"

train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)

TARGET_COL = "target_change_type"
if TARGET_COL not in train_df.columns:
    raise ValueError(f"Missing target column '{TARGET_COL}' in {TRAIN_PATH}")

y = train_df[TARGET_COL].to_numpy()

label_names = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']
num_classes = int(np.max(y)) + 1  # should be 6


# ---------------------------
# Feature selection (by groups; aligned with preprocess)
# ---------------------------

# 1) Geometry features
GEOM_COLS = [
    "area_sqm", "perimeter_m", "compactness",
    "length_m", "width_m", "aspect_ratio",
    "centroid_lon", "centroid_lat",
    "convex_area", "bbox_height", "bbox_width",
    "num_vertices", "max_radius",
]
GEOM_COLS = [c for c in GEOM_COLS if c in train_df.columns and c in test_df.columns]

# 2) One-hot tags (urban_/geo_)
ONEHOT_COLS = []
for c in train_df.columns:
    if (c.startswith("urban_") or c.startswith("geo_")) and not c.endswith("_tag_count"):
        if c in test_df.columns:
            ONEHOT_COLS.append(c)

# 2b) Urban/Geo derived features
URB_GEO_DERIVED = [
    "urban_density_score", "n_urban_tags", "n_geo_tags",
    "has_water", "has_dense_forest", "has_farms", "is_rural",
]
URB_GEO_COLS = [c for c in URB_GEO_DERIVED if c in train_df.columns and c in test_df.columns]

# 3) Image/timeline features from process_timeline (t1..t5 + deltas)
IMG_PATTERNS = (
    "img_red_mean_t", "img_green_mean_t", "img_blue_mean_t",
    "img_red_std_t", "img_green_std_t", "img_blue_std_t",
    "brightness_t", "texture_t", "exg_t", "saturation_t", "exr_t",
)
IMG_DELTA_PREFIXES = ("delta_brightness", "delta_texture", "delta_exg", "delta_saturation", "delta_exr")

IMG_COLS = []
for c in train_df.columns:
    if c.startswith(IMG_PATTERNS) or c.startswith(IMG_DELTA_PREFIXES):
        if c in test_df.columns:
            IMG_COLS.append(c)

# 4) Time features (process_time_between_statuses) + extras
TIME_PREFIXES = (
    "days_to_",
    "delta_days_",
    "phase_days_",
    "prior_construction_present",
    "prior_construction_before_cleared",
    "tr_",
    "days_if_change_",
)
TIME_EXACT = ("timeline_len", "total_duration_days", "change_status_frequency")
TIME_EXTRA = (
    "work_duration_days",
    "status_rank_max",
    "status_progression_delta",
    "unique_status_count",
    "avg_days_between_status",
)

TIME_COLS = []
for c in train_df.columns:
    ok = (c in TIME_EXACT) or (c in TIME_EXTRA) or c.startswith(TIME_PREFIXES)
    if ok and c in test_df.columns:
        TIME_COLS.append(c)

# 5) Status flags per date (status_<STATUS>_date0..4)
STATUS_FLAG_COLS = []
for c in train_df.columns:
    if c.startswith("status_") and "_date" in c:
        if c in test_df.columns:
            STATUS_FLAG_COLS.append(c)

# Final feature list (toutes les features du preprocess)
FEATURE_COLS = GEOM_COLS + ONEHOT_COLS + URB_GEO_COLS + IMG_COLS + TIME_COLS + STATUS_FLAG_COLS

# Rattrapage : toute colonne numérique (train+test) hors target, pas déjà incluse
numeric_candidates = [
    c for c in train_df.columns
    if c != TARGET_COL and c in test_df.columns
    and pd.api.types.is_numeric_dtype(train_df[c])
    and c not in FEATURE_COLS
]
if numeric_candidates:
    FEATURE_COLS = FEATURE_COLS + numeric_candidates
    print(f"[Info] Added {len(numeric_candidates)} numeric column(s) from preprocess not in groups: {numeric_candidates[:15]}{'...' if len(numeric_candidates) > 15 else ''}")

if not FEATURE_COLS:
    raise ValueError("No features selected — check your parquet columns.")

X = train_df[FEATURE_COLS].copy()
X_test = test_df[FEATURE_COLS].copy()

# Keep numeric only
X = X.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Align train/test columns (safety)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

print(f"Using {X.shape[1]} features:")
print(" - geom:", len(GEOM_COLS),
      "| one-hot:", len(ONEHOT_COLS),
      "| urb/geo derived:", len(URB_GEO_COLS),
      "| img:", len(IMG_COLS),
      "| time:", len(TIME_COLS),
      "| status_flags:", len(STATUS_FLAG_COLS))


# ---------------------------
# SANITIZE
# ---------------------------
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

CLIP = 1e12
X = X.clip(lower=-CLIP, upper=CLIP)
X_test = X_test.clip(lower=-CLIP, upper=CLIP)

X = X.fillna(0)
X_test = X_test.fillna(0)

# Safety check
bad_cols = [c for c in X.columns if not np.isfinite(X[c].to_numpy()).all()]
if bad_cols:
    print("Still non-finite columns:", bad_cols[:20])
    raise ValueError("Non-finite values remain in X.")

# IMPORTANT: Avoid pandas->QuantileDMatrix / columnar crash by using numpy float32
X_np = X.to_numpy(dtype=np.float32, copy=False)
X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)


# ---------------------------
# XGBoost config (no early stopping; stable settings)
# ---------------------------
# Notes:
# - We avoid tree_method="hist" because on some versions it triggers QuantileDMatrix path with pandas.
# - Using numpy float32 already avoids columnar issues, but keeping "approx" is extra safety.
model_params = dict(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    min_child_weight=2.0,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="approx",
    n_jobs=-1,
    random_state=42,
)

cv_params = dict(model_params)


# ---------------------------
# Cross-validation stratifiée (3 folds: chaque fold ~67% train / ~33% val)
# ---------------------------
N_FOLDS = 3
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_scores = []
fold_macro = []
fold_accuracies = []
fold_balanced_accuracies = []
per_class_f1 = []  # F1 par classe par fold pour le résumé final

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y), 1):
    X_tr, X_va = X_np[tr_idx], X_np[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = XGBClassifier(**cv_params)
    clf.fit(X_tr, y_tr)

    pred = np.argmax(clf.predict_proba(X_va), axis=1)

    acc = accuracy_score(y_va, pred)
    f1_w = f1_score(y_va, pred, average="weighted")
    f1_m = f1_score(y_va, pred, average="macro")
    bal = balanced_accuracy_score(y_va, pred)

    fold_accuracies.append(acc)
    fold_scores.append(f1_w)
    fold_macro.append(f1_m)
    fold_balanced_accuracies.append(bal)

    rep = classification_report(
        y_va, pred,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_va, pred, labels=list(range(num_classes)))

    # Suivi F1 par classe pour chaque fold (pour résumé final)
    per_class_f1.append([rep[name]["f1-score"] for name in label_names])

    print(f"\n========== Fold {fold}/{N_FOLDS} ==========")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal:.4f}")
    print(f"F1 Weighted       : {f1_w:.4f}")
    print(f"F1 Macro          : {f1_m:.4f}")
    print("\nPer-class F1:")
    for name in label_names:
        print(f"  {name:14s}: {rep[name]['f1-score']:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

print("\n" + "="*55)
print(f"RESULTATS CROSS-VALIDATION (Stratified {N_FOLDS}-fold: ~{100*(N_FOLDS-1)/N_FOLDS:.0f}% train / ~{100/N_FOLDS:.0f}% val par fold)")
print("="*55)
fold_accuracies = np.array(fold_accuracies)
fold_scores = np.array(fold_scores)
fold_macro = np.array(fold_macro)
fold_balanced_accuracies = np.array(fold_balanced_accuracies)
per_class_f1 = np.array(per_class_f1)  # shape (n_folds, n_classes)

print(f"Accuracy           : {fold_accuracies.mean():.4f} +/- {fold_accuracies.std():.4f}")
print(f"Balanced Accuracy  : {fold_balanced_accuracies.mean():.4f} +/- {fold_balanced_accuracies.std():.4f}")
print(f"F1 Weighted        : {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}")
print(f"F1 Macro           : {fold_macro.mean():.4f} +/- {fold_macro.std():.4f}")
print(f"\nF1 par classe (moyenne +/- std sur les {N_FOLDS} folds):")
for i, name in enumerate(label_names):
    mean_f1 = per_class_f1[:, i].mean()
    std_f1 = per_class_f1[:, i].std()
    print(f"  {name:14s}: {mean_f1:.4f} +/- {std_f1:.4f}")
print("="*55)


# ---------------------------
# Train FINAL model on full train + save
# ---------------------------
final_clf = XGBClassifier(**model_params)
final_clf.fit(X_np, y)

with open("model.pkl", "wb") as f:
    pickle.dump(final_clf, f)
print("[OK] Saved model.pkl")


# ---------------------------
# Predict + submission
# ---------------------------
pred_y = np.argmax(final_clf.predict_proba(X_test_np), axis=1)

sub = pd.DataFrame({
    "Id": np.arange(len(pred_y), dtype=int),
    "change_type": pred_y.astype(int),
})
sub.to_csv("submission.csv", index=False)
print("[OK] Saved submission.csv", sub.shape)
