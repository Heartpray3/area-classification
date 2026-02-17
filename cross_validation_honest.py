"""
Cross-validation honnête pour area_classification.

Règles :
- Imputer ajusté sur le train de chaque fold uniquement (pas de fuite).
- F1 macro pour une évaluation équitable entre classes.
- StratifiedKFold 5 folds pour une estimation stable.
- Les deux modèles (RF et XGBoost) évalués avec la même pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# FAST = True : 3 folds, moins d'arbres. SAMPLE_SIZE > 0 : limite les lignes (stratifié) pour test rapide.
FAST = True
SAMPLE_SIZE = 12000   # 0 = tout le dataset. 12k = run ~2-3 min en FAST.
N_SPLITS = 2 if FAST else 5   # 2 folds en FAST pour résultat en ~2-4 min
RANDOM_STATE = 42
TARGET_NAMES = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']

FEATURES_TO_USE = [
    'area_sqm', 'perimeter_m', 'compactness', 'length_m', 'width_m',
    'aspect_ratio', 'rectangularity', 'centroid_lon', 'centroid_lat',
    'work_duration_days', 'avg_days_between_status', 'unique_status_count', 'status_progression',
    'status_progression_delta', 'status_rank_max',
    'started_as_green', 'ended_as_built', 'ended_as_cleared',
    'has_demolition', 'has_excavation', 'is_operational_final', 'max_consecutive_stable_status',
    'img_red_mean_date1', 'img_green_mean_date1', 'img_blue_mean_date1',
    'img_red_std_date1', 'img_green_std_date1', 'img_blue_std_date1',
    'img_red_mean_date2', 'img_green_mean_date2', 'img_blue_mean_date2',
    'img_red_std_date2', 'img_green_std_date2', 'img_blue_std_date2',
    'img_red_mean_date3', 'img_green_mean_date3', 'img_blue_mean_date3',
    'img_red_std_date3', 'img_green_std_date3', 'img_blue_std_date3',
    'img_red_mean_date4', 'img_green_mean_date4', 'img_blue_mean_date4',
    'img_red_std_date4', 'img_green_std_date4', 'img_blue_std_date4',
    'img_red_mean_date5', 'img_green_mean_date5', 'img_blue_mean_date5',
    'img_red_std_date5', 'img_green_std_date5', 'img_blue_std_date5',
    'texture_t1', 'texture_t2', 'texture_t3', 'texture_t4', 'texture_t5',
    'brightness_t1', 'brightness_t2', 'brightness_t3', 'brightness_t4', 'brightness_t5',
    'exg_t1', 'exg_t2', 'exg_t3', 'exg_t4', 'exg_t5',
    'exr_t1', 'exr_t2', 'exr_t3', 'exr_t4', 'exr_t5',
    'saturation_t1', 'saturation_t2', 'saturation_t3', 'saturation_t4', 'saturation_t5',
    'delta_exg', 'delta_exr', 'delta_saturation', 'delta_brightness', 'delta_texture',
    'urban_Dense_Urban', 'urban_Sparse_Urban', 'urban_Industrial', 'urban_Rural', 'urban_Urban_Slum',
    'geo_Sparse_Forest', 'geo_Grass_Land', 'geo_Dense_Forest', 'geo_Farms', 'geo_Barren_Land',
    'geo_Lakes', 'geo_River', 'geo_Coastal', 'geo_Desert', 'geo_Hills', 'geo_Snow'
]


def load_and_prepare():
    import os
    path = 'train_features.csv'
    if not os.path.isfile(path):
        for root, dirs, files in os.walk('.'):
            if 'train_features.csv' in files:
                path = os.path.join(root, 'train_features.csv')
                break
    if not os.path.isfile(path):
        raise FileNotFoundError("train_features.csv introuvable. Lance d'abord Feature.py pour le générer.")
    nrows = (SAMPLE_SIZE + 2000) if SAMPLE_SIZE else None  # chargement plus rapide en mode sample
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df['target_change_type'], random_state=RANDOM_STATE)
        print(f"  [Échantillon stratifié : {len(df)} lignes]")
    available = [f for f in FEATURES_TO_USE if f in df.columns]
    X = df[available]
    y = df['target_change_type']
    return X, y, available


def get_imputer_fitted_on_train(X_train):
    imp = SimpleImputer(strategy='median')
    imp.fit(X_train)
    return imp


def run_cv_one_model(X, y, feature_names, model_name, get_model_fn, use_sample_weight=False):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores_macro = []
    fold_scores_weighted = []
    fold_acc = []
    fold_bal_acc = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Imputer : fit UNIQUEMENT sur le train du fold (évite la fuite)
        imputer = get_imputer_fitted_on_train(X_train_fold)
        X_train_imp = pd.DataFrame(imputer.transform(X_train_fold), columns=feature_names)
        X_val_imp = pd.DataFrame(imputer.transform(X_val_fold), columns=feature_names)

        model = get_model_fn()
        if use_sample_weight:
            w = compute_sample_weight(class_weight='balanced', y=y_train_fold)
            model.fit(X_train_imp, y_train_fold, sample_weight=w)
        else:
            model.fit(X_train_imp, y_train_fold)

        y_pred = model.predict(X_val_imp)
        all_y_true.extend(y_val_fold.tolist())
        all_y_pred.extend(y_pred.tolist())

        f1_macro = f1_score(y_val_fold, y_pred, average='macro')
        f1_weighted = f1_score(y_val_fold, y_pred, average='weighted')
        acc = accuracy_score(y_val_fold, y_pred)
        bal_acc = balanced_accuracy_score(y_val_fold, y_pred)
        fold_scores_macro.append(f1_macro)
        fold_scores_weighted.append(f1_weighted)
        fold_acc.append(acc)
        fold_bal_acc.append(bal_acc)
        print(f"  Fold {fold+1}/{N_SPLITS}  Acc = {acc:.4f}  F1 macro = {f1_macro:.4f}  F1 weighted = {f1_weighted:.4f}")

    return {
        'macro_mean': np.mean(fold_scores_macro),
        'macro_std': np.std(fold_scores_macro),
        'weighted_mean': np.mean(fold_scores_weighted),
        'weighted_std': np.std(fold_scores_weighted),
        'acc_mean': np.mean(fold_acc),
        'acc_std': np.std(fold_acc),
        'bal_acc_mean': np.mean(fold_bal_acc),
        'fold_macro': fold_scores_macro,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
    }


def get_rf():
    n_est, depth = (150, 30) if FAST else (1000, 50)
    return RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_xgb():
    n_est, depth = (300, 10) if FAST else (1500, 14)
    return XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=0.02 if FAST else 0.01,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.9,
        objective='multi:softmax',
        num_class=6,
        tree_method='hist',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def main():
    import sys
    print("Chargement des données...", flush=True)
    X, y, feature_names = load_and_prepare()
    if SAMPLE_SIZE:
        print(f"  Mode échantillon : {len(X)} lignes (stratifié par classe)")
    print(f"Features utilisées : {len(feature_names)}")
    print(f"Échantillons : {len(X)}, Classes : {y.nunique()}\n")

    print("=" * 60)
    print("RANDOM FOREST — Cross-Validation honnête (imputer par fold)")
    print("=" * 60)
    rf_results = run_cv_one_model(X, y, feature_names, 'RF', get_rf, use_sample_weight=False)

    print("\n" + "=" * 60)
    print("XGBOOST — Cross-Validation honnête (imputer par fold)")
    print("=" * 60)
    xgb_results = run_cv_one_model(X, y, feature_names, 'XGB', get_xgb, use_sample_weight=True)

    # Synthèse
    best = 'XGBoost' if xgb_results['acc_mean'] >= rf_results['acc_mean'] else 'Random Forest'
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"Random Forest :  Accuracy = {rf_results['acc_mean']:.4f} ± {rf_results['acc_std']:.4f}  |  F1 macro = {rf_results['macro_mean']:.4f} ± {rf_results['macro_std']:.4f}")
    print(f"XGBoost       :  Accuracy = {xgb_results['acc_mean']:.4f} ± {xgb_results['acc_std']:.4f}  |  F1 macro = {xgb_results['macro_mean']:.4f} ± {xgb_results['macro_std']:.4f}")
    print(f"\nMeilleur modèle (accuracy) : {best}")
    print()
    print("Rapport de classification (tous folds agrégés) — XGBoost:")
    print(classification_report(
        xgb_results['y_true'], xgb_results['y_pred'],
        target_names=TARGET_NAMES, digits=3,
    ))
    print("Matrice de confusion — XGBoost:")
    cm = confusion_matrix(xgb_results['y_true'], xgb_results['y_pred'])
    print(cm)
    acc_agg = accuracy_score(xgb_results['y_true'], xgb_results['y_pred'])
    print(f"\nAccuracy agrégée (tous folds) : {acc_agg:.4f}")
    if FAST:
        print("\n[Mode FAST actif. Pour viser >97%, passe FAST = False et relance une fois les réglages validés.]")
    print("\nTerminé.")


if __name__ == '__main__':
    main()
