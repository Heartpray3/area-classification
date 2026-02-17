import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

print("1. Chargement des données...")
df_train = pd.read_csv('train_features.csv')
df_test = pd.read_csv('test_features.csv') 

# --- 2. NETTOYAGE (Train et Test) ---
print("Nettoyage des valeurs infinies...")
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Suppression des colonnes textuelles inutiles (status, etc.)

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
# features_to_drop = ['img_green_mean_date1', 'img_blue_mean_date1', 'img_green_std_date1', 
#                     'img_blue_std_date1', 'img_green_mean_date2', 'img_blue_mean_date2', ''
#                     'img_green_std_date2', 'img_blue_std_date2', 'img_green_mean_date3', 
#                     'img_blue_mean_date3', 'img_green_std_date3', 'img_blue_std_date3', ''
#                     'img_green_mean_date4', 'img_blue_mean_date4', 'img_green_std_date4', 
#                     'img_blue_std_date4', 'img_green_mean_date5', 'img_blue_mean_date5', 
#                     'img_green_std_date5', 'img_blue_std_date5', 'perimeter_m', 'brightness_t1', 
#                     'brightness_t2', 'brightness_t3', 'brightness_t4', 'brightness_t5', 'img_green_mean_t1', 
#                     'img_green_mean_t2', 'img_green_mean_t3', 'img_green_mean_t4', 'img_green_mean_t5', 
#                     'img_green_std_t1', 'img_green_std_t2', 'img_green_std_t3', 'img_green_std_t4', 
#                     'img_green_std_t5', 'img_red_mean_t1', 'img_red_mean_t2', 'img_red_mean_t3', 
#                     'img_red_mean_t4', 'img_red_mean_t5', 'img_red_std_t1', 'img_red_std_t2', 
#                     'img_red_std_t3', 'img_red_std_t4', 'img_red_std_t5', 'status_rank_max', 
#                     'texture_t1', 'texture_t2', 'texture_t3', 'texture_t4', 'texture_t5', 'urban_A', 'geo_A']

# Vérification que toutes les colonnes existent bien dans le train et le test
available_features = [
    f for f in features_to_use if f in df_train.columns 
    # and f not in features_to_drop
]
print(f"Features retenues : {len(available_features)}")

X = df_train[available_features]
y = df_train['target_change_type']

# Préparation du vrai jeu de test (celui pour la soumission)
X_submission = df_test[available_features]

# --- 4. IMPUTATION ---
print("Application de l'Imputer (stratégie : médiane)...")
imputer = SimpleImputer(strategy='most_frequent')  # On peut aussi tester 'median' ou 'mean' selon la nature des données

# On apprend la médiane sur TOUT le jeu d'entraînement (X)
imputer.fit(X)

# On applique la transformation sur X (pour l'entrainement) et sur X_submission (pour le fichier final)
X_imputed = imputer.transform(X)
X_submission_imputed = imputer.transform(X_submission)

# On remet en DataFrame pour garder les noms de colonnes (utile pour feature importance)
X = pd.DataFrame(X_imputed, columns=available_features)
X_submission = pd.DataFrame(X_submission_imputed, columns=available_features)

# # --- 5. VALIDATION INTERNE (Pour vérifier ton score F1 avant de soumettre) ---
print("\n--- Phase de Validation Interne ---")
# On sépare X en train/val juste pour vérifier que le modèle est bon
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# rf_val = RandomForestClassifier(
#     n_estimators=1200, # Un peu moins d'arbres pour la validation rapide
#     # max_depth=50,
#     min_samples_split=2,
#     min_samples_leaf=2,
#     class_weight= 'balanced',  
#     random_state=42,
#     n_jobs=-1
# )
# rf_val.fit(X_train_split, y_train_split)
# y_val_pred = rf_val.predict(X_val_split)

# target_names = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']
# print(classification_report(y_val_split, y_val_pred, target_names=target_names))
# print(f"Score F1 Macro (Validation) : {f1_score(y_val_split, y_val_pred, average='macro'):.4f}")

# --- 6. ENTRAÎNEMENT FINAL ET PRÉDICTION SUR TEST ---
print("\n--- Entraînement Final sur 100% des données ---")
# On utilise maintenant TOUT X et TOUT y pour entraîner le modèle ultime
rf_final = RandomForestClassifier(
    n_estimators=1000,
    max_depth=50,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X, y)

print("Prédiction sur le fichier test_features.csv...")
y_submission_pred = rf_final.predict(X_submission)

# --- 7. CRÉATION DU FICHIER SUBMISSION.CSV ---
# On crée le DataFrame de soumission
# Assumant que l'index du test.geojson correspond à l'Id attendu (0, 1, 2...)
submission_df = pd.DataFrame({
    'Id': df_test.index,  # Ou df_test['index'] si tu as gardé une colonne index
    'change_type': y_submission_pred
})

# Sauvegarde
submission_df.to_csv("submission.csv", index=False)
print("✅ Fichier 'submission.csv' généré avec succès !")
