"""Charge le cache, fait 2-fold CV ensemble (RF+HGB+ET+XGB), écrit le F1 dans cv_result.txt."""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.base import clone

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

CACHE_DIR = "feature_cache_v3"
RANDOM_STATE = 42
CV_FOLDS = 2
SUBSAMPLE = 0.30  # 30% = ~89k, 2 folds pour finir plus vite

def main():
    X_train = np.load(os.path.join(CACHE_DIR, "X_train.npy"))
    y = np.load(os.path.join(CACHE_DIR, "y.npy"))
    if SUBSAMPLE is not None and SUBSAMPLE < 1.0:
        X_train, _, y, _ = train_test_split(X_train, y, train_size=SUBSAMPLE, stratify=y, random_state=RANDOM_STATE)
        print("Subsampled to {:.0%}: n={}".format(SUBSAMPLE, len(y)), flush=True)
    _, counts = np.unique(y, return_counts=True)
    n = len(y)
    balanced_w = n / (6 * counts)
    cap = 6.0
    class_weights = np.minimum(balanced_w, cap)
    class_weight_dict = {i: float(class_weights[i]) for i in range(6)}
    sw = class_weights[y]

    rf = RandomForestClassifier(n_estimators=80, max_depth=18, min_samples_leaf=22, class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=70, max_depth=9, min_samples_leaf=45, l2_regularization=0.3, learning_rate=0.08, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15, n_iter_no_change=8)
    et = ExtraTreesClassifier(n_estimators=80, max_depth=18, min_samples_leaf=22, class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.08, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1) if HAS_XGB else None
    if HAS_XGB:
        print("XGBoost enabled.", flush=True)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_oof = np.zeros_like(y, dtype=np.float64)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
        print("Fold", fold + 1, "...", flush=True)
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, sw_tr = y[tr_idx], sw[tr_idx]
        rf_f = clone(rf); hgb_f = clone(hgb); et_f = clone(et)
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
        f1_fold = f1_score(y[val_idx], y_oof[val_idx], average="weighted")
        print("  Fold F1:", round(f1_fold, 4), flush=True)
    cv_f1 = f1_score(y, y_oof, average="weighted")
    print("CV F1 weighted:", round(cv_f1, 4), flush=True)
    with open("cv_result.txt", "w") as f:
        f.write("CV_F1={:.4f}\n".format(cv_f1))
    return cv_f1

if __name__ == "__main__":
    main()
