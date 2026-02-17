import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier  # <--- Le changement est ici
from sklearn.utils.class_weight import compute_sample_weight

print("1. Chargement des données...")
df_train = pd.read_csv('train_features.csv')
df_test = pd.read_csv('test_features.csv') 

# --- 2. NETTOYAGE (Train et Test) ---
print("Nettoyage des valeurs infinies...")
# On force tous les infinis à devenir des NaN
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Suppression des colonnes textuelles inutiles
cols_text_train = [col for col in df_train.columns if 'status_' in col and df_train[col].dtype == 'object']
df_train = df_train.drop(columns=cols_text_train, errors='ignore')

cols_text_test = [col for col in df_test.columns if 'status_' in col and df_test[col].dtype == 'object']
df_test = df_test.drop(columns=cols_text_test, errors='ignore')

# --- 3. SÉLECTION DES FEATURES ---
print("Préparation des colonnes...")

features_to_use = [
    # --- Géométrie & Spatial ---
    'area_sqm',
    'perimeter_m',
    'compactness',
    'length_m', 
    'width_m',
    'aspect_ratio',
    'rectangularity',
    'centroid_lon',  
    'centroid_lat',
    
    # --- Temps & Comportement des Statuts (TES OUBLIS SONT ICI) ---
    'work_duration_days',
    # 'status_changes_count', # N'apparaît pas dans ton header CSV, remplace par status_progression ou unique_status_count
    'unique_status_count',       # <-- Présent dans ton header
    'status_progression',        # <-- Présent dans ton header
    'status_progression_delta',
    'status_rank_max',
    
    # --- Les Flags Importants (OUBLIÉS) ---
    'started_as_green',
    'ended_as_built',
    'ended_as_cleared',
    'has_demolition',
    'has_excavation',              # <-- MANQUAIT : Crucial pour début de chantier
    'is_operational_final',        # <-- MANQUAIT : Crucial pour confirmer la fin
    'max_consecutive_stable_status', # <-- MANQUAIT : Détecte les projets abandonnés/lents
    
    # --- Variables Images Brutes (Il vaut mieux tout donner à XGBoost) ---
    'img_red_mean_date1', 'img_green_mean_date1', 'img_blue_mean_date1', 'img_red_std_date1', 'img_green_std_date1', 'img_blue_std_date1',
    'img_red_mean_date2', 'img_green_mean_date2', 'img_blue_mean_date2', 'img_red_std_date2', 'img_green_std_date2', 'img_blue_std_date2',
    'img_red_mean_date3', 'img_green_mean_date3', 'img_blue_mean_date3', 'img_red_std_date3', 'img_green_std_date3', 'img_blue_std_date3',
    'img_red_mean_date4', 'img_green_mean_date4', 'img_blue_mean_date4', 'img_red_std_date4', 'img_green_std_date4', 'img_blue_std_date4',
    'img_red_mean_date5', 'img_green_mean_date5', 'img_blue_mean_date5', 'img_red_std_date5', 'img_green_std_date5', 'img_blue_std_date5',
    
    # --- Texture & Luminance calculée ---
    'texture_t1', 'texture_t2', 'texture_t3', 'texture_t4', 'texture_t5',
    'brightness_t1', 'brightness_t2', 'brightness_t3', 'brightness_t4', 'brightness_t5',
    'exg_t1', 'exg_t2', 'exg_t3', 'exg_t4', 'exg_t5',
    'exr_t1', 'exr_t2', 'exr_t3', 'exr_t4', 'exr_t5',
    'saturation_t1', 'saturation_t2', 'saturation_t3', 'saturation_t4', 'saturation_t5',
    'delta_exg',
    'delta_exr',
    'delta_saturation',
    'delta_brightness',
    'delta_texture',
    
    
    # --- One-Hot Encoding ---
    'urban_Dense_Urban', 'urban_Sparse_Urban', 'urban_Industrial', 'urban_Rural', 'urban_Urban_Slum',
    'geo_Sparse_Forest', 'geo_Grass_Land', 'geo_Dense_Forest', 'geo_Farms', 'geo_Barren_Land', 
    'geo_Lakes', 'geo_River', 'geo_Coastal', 'geo_Desert', 'geo_Hills', 'geo_Snow'
]


# Vérification que toutes les colonnes existent bien dans le train et le test
available_features = [f for f in features_to_use if f in df_train.columns and f in df_test.columns]
print(f"Features retenues : {len(available_features)}")

X = df_train[available_features]
y = df_train['target_change_type']

# Préparation du vrai jeu de test (celui pour la soumission)
X_submission = df_test[available_features]

# --- 4. IMPUTATION ---
print("Application de l'Imputer (stratégie : médiane)...")
imputer = SimpleImputer(strategy='median')  # <--- Changement de stratégie pour les variables catégorielles (One-Hot)

# On apprend la médiane sur TOUT le jeu d'entraînement (X)
imputer.fit(X)

# On applique la transformation
X_imputed = imputer.transform(X)
X_submission_imputed = imputer.transform(X_submission)

# On remet en DataFrame
X = pd.DataFrame(X_imputed, columns=available_features)
X_submission = pd.DataFrame(X_submission_imputed, columns=available_features)

# --- 5. VALIDATION INTERNE ---
print("\n--- Phase de Validation Interne (XGBoost) ---")
# On sépare X en train/val juste pour vérifier que le modèle est bon
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calcul des poids pour gérer le déséquilibre des classes (XGBoost n'a pas de "class_weight='balanced'" automatique comme RF)
sample_weights_val = compute_sample_weight(class_weight='balanced', y=y_train_split)

# Configuration XGBoost "Expert" pour Imbalanced Multi-class
xgb_val = XGBClassifier(
    # --- Moteur ---
    n_estimators=3000,      # Beaucoup d'arbres pour corriger les erreurs fines
    learning_rate=0.01,     # Apprentissage lent et précis
    
    # --- Structure ---
    max_depth=15,           # Assez profond pour capter la complexité, mais pas trop (vs RF 50)
    min_child_weight=3,     # Évite d'isoler des cas uniques (bruit)
    
    # --- Robustesse (Bruit) ---
    subsample=0.8,          # Utilise 80% des lignes par arbre
    colsample_bytree=0.9,   # Utilise 60% des features par arbre (force la diversité)
    
    # --- Technique ---
    objective='multi:softmax', # Obligatoire pour multi-classe
    num_class=6,               # Tes 6 classes
    n_jobs=-1,                 # Tout les cœurs CPU
    random_state=40,
    tree_method='hist'         # 'hist' est beaucoup plus rapide sur les gros datasets (>10k lignes)
)

# IMPORTANT : L'entraînement doit se faire avec les poids !
# model.fit(X_train, y_train, sample_weight=sample_weights)

print("Entraînement validation...")
xgb_val.fit(X_train_split, y_train_split, sample_weight=sample_weights_val)
y_val_pred = xgb_val.predict(X_val_split)

target_names = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']
print(classification_report(y_val_split, y_val_pred, target_names=target_names))
print(f"Score F1 Macro (Validation) : {f1_score(y_val_split, y_val_pred, average='macro'):.4f}")

# --- 6. ENTRAÎNEMENT FINAL ET PRÉDICTION SUR TEST ---
print("\n--- Entraînement Final sur 100% des données ---")

# Recalcul des poids sur TOUT le dataset
sample_weights_full = compute_sample_weight(class_weight='balanced', y=y)

# Modèle plus robuste pour le final (plus d'arbres)
xgb_final = XGBClassifier(
    n_estimators=1500,     # Plus d'arbres pour le modèle final
    max_depth=14,
    learning_rate=0.01,    # Learning rate plus fin
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=40,
    n_jobs=-1,
    objective='multi:softmax',
    num_class=6,
    tree_method='hist' 
)

xgb_final.fit(X, y, sample_weight=sample_weights_full)

print("Prédiction sur le fichier test_features.csv...")
y_submission_pred = xgb_final.predict(X_submission)

# --- 7. CRÉATION DU FICHIER SUBMISSION.CSV ---
# On crée le DataFrame de soumission
submission_df = pd.DataFrame({
    'Id': df_test.index, 
    'change_type': y_submission_pred
})

# Sauvegarde
submission_df.to_csv("submission_xgboost.csv", index=False)
print("✅ Fichier 'submission_xgboost.csv' généré avec succès !")

# --- 8. TOP FEATURES DU MODÈLE FINAL ---
importances = pd.Series(xgb_final.feature_importances_, index=available_features)
print("\n🔥 Top 15 des features utilisées par le modèle final :")
print(importances.sort_values(ascending=False))