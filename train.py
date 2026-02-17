import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# ---------------------------
# Load
# ---------------------------
train_df = pd.read_parquet("train_features.parquet")
test_df  = pd.read_parquet("test_features.parquet")

TARGET_COL = "target_change_type"
y = train_df[TARGET_COL].to_numpy()

# ---------------------------
# Feature selection (ONLY what you asked)
# ---------------------------

# 1) Geometry features (from your preprocessing)
GEOM_COLS = [
    "area_sqm", "perimeter_m", "compactness",
    "length_m", "width_m", "aspect_ratio",
    "centroid_lon", "centroid_lat",
    "convex_area", "bbox_height", "bbox_width",
    "num_vertices", "max_radius",
]
GEOM_COLS = [c for c in GEOM_COLS if c in train_df.columns and c in test_df.columns]

# 2) One-hot only (urban_/geo_), exclude counts
ONEHOT_COLS = []
for c in train_df.columns:
    if (c.startswith("urban_") or c.startswith("geo_")) and not c.endswith("_tag_count"):
        # keep only if also in test
        if c in test_df.columns:
            ONEHOT_COLS.append(c)

# 3) Image features only (from timeline: per-t + deltas)
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

FEATURE_COLS = GEOM_COLS + ONEHOT_COLS + IMG_COLS

if not FEATURE_COLS:
    raise ValueError("No features selected — check your parquet columns.")

X = train_df[FEATURE_COLS].copy()
X_test = test_df[FEATURE_COLS].copy()

# Ensure numeric + NaNs handled
X = X.select_dtypes(include=[np.number]).fillna(0)
X_test = X_test.select_dtypes(include=[np.number]).fillna(0)

# Align (safety)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

print(f"Using {X.shape[1]} features:")
print(" - geom:", len(GEOM_COLS), "| one-hot:", len(ONEHOT_COLS), "| img:", len(IMG_COLS))

num_classes = int(np.max(y)) + 1  # should be 6

# --- SANITIZE: remove inf / huge values ---
# 1) Replace inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# 2) Optional: clip extreme values (prevents "too large" issues)
# (safe for geometry + img features)
CLIP = 1e12
X = X.clip(lower=-CLIP, upper=CLIP)
X_test = X_test.clip(lower=-CLIP, upper=CLIP)

# 3) Fill NaN OR let xgboost handle missing
X = X.fillna(0)
X_test = X_test.fillna(0)

# 4) Debug: locate bad columns (if it happens again)
bad_cols = [c for c in X.columns if not np.isfinite(X[c].to_numpy()).all()]
if bad_cols:
    print("Still non-finite columns:", bad_cols[:20])
    for c in bad_cols[:5]:
        arr = X[c].to_numpy()
        print(c, "min/max:", np.nanmin(arr), np.nanmax(arr))
    raise ValueError("Non-finite values remain in X.")


# ---------------------------
# XGBoost config (your params)
# ---------------------------
model_params = dict(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    missing=np.nan,
)


# ---------------------------
# Cross-validation
# ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = XGBClassifier(**model_params)
    clf.fit(X_tr, y_tr)

    pred = np.argmax(clf.predict_proba(X_va), axis=1)
    f1 = f1_score(y_va, pred, average="macro")
    f1s.append(f1)
    print(f"Fold {fold}: F1-macro = {f1:.5f}")

print(f"\nCV mean F1-macro = {np.mean(f1s):.5f} ± {np.std(f1s):.5f}")

# ---------------------------
# Fit full + predict test
# ---------------------------
final_clf = XGBClassifier(**model_params)
final_clf.fit(X, y)

pred_y_int = np.argmax(final_clf.predict_proba(X_test), axis=1)

# If submission expects STRING labels:
inv_target_mapping = {
    0: 'Demolition', 1: 'Road', 2: 'Residential',
    3: 'Commercial', 4: 'Industrial', 5: 'Mega Projects'
}
pred_y = pd.Series(pred_y_int).map(inv_target_mapping).to_numpy()

# --- 5) Save submission (index = Id = index du test)
pred_df = pd.DataFrame({"change_type": pred_y}, index=test_df.index)
pred_df.index.name = "Id"
pred_df.to_csv("submission.csv")

print("Saved: submission.csv")
