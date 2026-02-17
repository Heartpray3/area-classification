"""
train.py - Test accuracy des features sur le TRAIN SET

Teste les nouvelles features en faisant une cross-validation sur le train set.
Objectif: valider que les features ameliorent l'accuracy avant de soumettre.

Outputs:
- model.pkl (modele final entraine)
- Resultats CV affiches en console (accuracy, F1 macro/weighted)
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score

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

# Label mapping (for reports)
label_names = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']
inv_target_mapping = {
    0: 'Demolition',
    1: 'Road',
    2: 'Residential',
    3: 'Commercial',
    4: 'Industrial',
    5: 'Mega Projects'
}


# ---------------------------
# Feature selection
# ---------------------------

# 1) Geometry features (from preprocessing)
GEOM_COLS = [
    "area_sqm", "perimeter_m", "compactness",
    "length_m", "width_m", "aspect_ratio",
    "centroid_lon", "centroid_lat",
    "convex_area", "bbox_height", "bbox_width",
    "num_vertices", "max_radius",
]
GEOM_COLS = [c for c in GEOM_COLS if c in train_df.columns and c in test_df.columns]

# 2) One-hot tags (urban_/geo_), exclude counts
ONEHOT_COLS = []
for c in train_df.columns:
    if (c.startswith("urban_") or c.startswith("geo_")) and not c.endswith("_tag_count"):
        if c in test_df.columns:
            ONEHOT_COLS.append(c)

# 3) Image features (from timeline: per-t + deltas)
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

# 4) Time features (from process_time_between_statuses)
TIME_PREFIXES = (
    "days_to_",
    "delta_days_",
    "phase_days_",
    "prior_construction_present",
    "prior_construction_before_cleared",
    "tr_",
)
TIME_EXACT = ("timeline_len", "total_duration_days")

TIME_COLS = []
for c in train_df.columns:
    ok = (c in TIME_EXACT) or c.startswith(TIME_PREFIXES)
    if ok and c in test_df.columns:
        TIME_COLS.append(c)

# 5) Status flags per date (50 cols): status_<STATUS>_date0..4
STATUS_FLAG_COLS = []
for c in train_df.columns:
    if c.startswith("status_") and "_date" in c:
        if c in test_df.columns:
            STATUS_FLAG_COLS.append(c)

# Final feature list
FEATURE_COLS = GEOM_COLS + ONEHOT_COLS + IMG_COLS + TIME_COLS + STATUS_FLAG_COLS
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
      "| img:", len(IMG_COLS),
      "| time:", len(TIME_COLS),
      "| status_flags:", len(STATUS_FLAG_COLS))

num_classes = int(np.max(y)) + 1  # should be 6


# ---------------------------
# SANITIZE: inf / NaN / huge values
# ---------------------------
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

CLIP = 1e12
X = X.clip(lower=-CLIP, upper=CLIP)
X_test = X_test.clip(lower=-CLIP, upper=CLIP)

X = X.fillna(0)
X_test = X_test.fillna(0)

bad_cols = [c for c in X.columns if not np.isfinite(X[c].to_numpy()).all()]
if bad_cols:
    print("Still non-finite columns:", bad_cols[:20])
    for c in bad_cols[:5]:
        arr = X[c].to_numpy()
        print(c, "min/max:", np.nanmin(arr), np.nanmax(arr))
    raise ValueError("Non-finite values remain in X.")


# ---------------------------
# XGBoost config
# ---------------------------
model_params = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.2,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    missing=np.nan,
)

# For CV: use fewer estimators for speed
cv_params = dict(model_params)
cv_params.update(n_estimators=150)


# ---------------------------
# 2-Fold Cross-validation (F1 WEIGHTED) - FAST!
# ---------------------------
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

fold_scores = []
fold_accuracies = []
fold_balanced_accuracies = []
per_class_f1 = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = XGBClassifier(**cv_params)
    clf.fit(X_tr, y_tr)

    pred = np.argmax(clf.predict_proba(X_va), axis=1)

    acc = accuracy_score(y_va, pred)
    f1_w = f1_score(y_va, pred, average="weighted")
    balanced_acc = balanced_accuracy_score(y_va, pred)
    
    fold_accuracies.append(acc)
    fold_scores.append(f1_w)
    fold_balanced_accuracies.append(balanced_acc)

    rep = classification_report(
        y_va, pred,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1.append([rep[name]["f1-score"] for name in label_names])

    cm = confusion_matrix(y_va, pred, labels=list(range(num_classes)))

    print(f"\n========== Fold {fold}/2 ==========")
    print(f"[OK] Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"[OK] Balanced Accuracy : {balanced_acc:.4f}")
    print(f"[OK] F1 Weighted       : {f1_w:.4f}")

    print("\nPer-class F1:")
    for name in label_names:
        print(f"  {name:14s}: {rep[name]['f1-score']:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

fold_accuracies = np.array(fold_accuracies, dtype=float)
fold_scores = np.array(fold_scores, dtype=float)
fold_balanced_accuracies = np.array(fold_balanced_accuracies, dtype=float)
per_class_f1 = np.array(per_class_f1, dtype=float)

print("\n" + "="*50)
print("RESULTATS CROSS-VALIDATION (Dataset desequilibre)")
print("="*50)
print(f"Accuracy            : {fold_accuracies.mean():.4f} +/- {fold_accuracies.std():.4f}")
print(f"Balanced Accuracy   : {fold_balanced_accuracies.mean():.4f} +/- {fold_balanced_accuracies.std():.4f}")
print(f"F1 Weighted         : {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}")
print("="*50)
print("\nMetriques pertinentes pour donnees desequilibrees:")
print("   - Accuracy: Performance globale")
print("   - Balanced Accuracy: Moyenne de recall par classe (immunise au desequilibre)")
print("   - F1 Weighted: F1 pondere par support (robuste au desequilibre)")


# ---------------------------
# Train FINAL model on full train
# ---------------------------
final_clf = XGBClassifier(**model_params)
final_clf.fit(X, y)

# ---------------------------
# Save model to pickle
# ---------------------------
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(final_clf, f)
print("[OK] Saved: model.pkl")

print("\n[DONE] Script termine!")
print("Le modele est entraine et pret pour calculate_accuracy.py")


