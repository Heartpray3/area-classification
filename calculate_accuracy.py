"""
Calcul de l'accuracy.

Comme train.geojson a des labels et test.geojson n'en a pas (set Kaggle),
on calcule l'accuracy en predisant sur le TRAIN set et en comparant avec ses labels.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import pickle
import geopandas as gpd

# ===== Extraire les labels depuis train.geojson =====
def get_train_labels_from_geojson():
    """Recupere les vrais labels depuis train.geojson"""
    print("[*] Lecture de train.geojson (peut prendre quelques secondes)...")
    gdf = gpd.read_file("train.geojson")
    
    if "change_type" not in gdf.columns:
        print("[!] ERREUR: train.geojson n'a pas de colonne 'change_type'")
        return None
    
    # Mapper les labels texte aux indices
    target_mapping = {
        "Demolition": 0, "Road": 1, "Residential": 2,
        "Commercial": 3, "Industrial": 4, "Mega Projects": 5,
    }
    y_true = gdf["change_type"].map(target_mapping).values
    print(f"[OK] Charge {len(y_true)} labels du train set")
    return y_true


# ===== Charger le modèle et faire des prédictions =====
def get_train_predictions():
    """Charge le modele et predit sur le train set"""
    print("\n[*] Chargement des features du train...")
    X_train = pd.read_parquet("train_features.parquet")
    print(f"[OK] Shape train features: {X_train.shape}")
    
    print("[*] Chargement du modele...")
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("[OK] Modele charge (model.pkl)")
        
    except FileNotFoundError:
        print("[!] Erreur: model.pkl non trouve")
        print("\nOptions alternatives:")
        print("  1. Entrainez/sauvegardez un modele dans model.pkl")
        print("  2. Utilisez un modele deja entraine et sauvegarde")
        return None
    
    print("[*] Calcul des predictions sur le train set...")
    y_pred = model.predict(X_train)
    print(f"[OK] {len(y_pred)} predictions generees")
    
    return y_pred


# ===== Calcul de l'accuracy =====
def calculate_accuracy():
    """Calcule et affiche l'accuracy sur le TRAIN set"""
    
    # Recuperer les vrais labels du train
    y_true = get_train_labels_from_geojson()
    if y_true is None:
        return None
    
    # Recuperer les predictions du modele
    y_pred = get_train_predictions()
    if y_pred is None:
        return None
    
    # Verifier les dimensions
    if len(y_true) != len(y_pred):
        print(f"\n[!] Erreur: Dimensions differentes!")
        print(f"   Labels: {len(y_true)}")
        print(f"   Predictions: {len(y_pred)}")
        return None
    
    # Calcul des metriques
    print("\n" + "=" * 70)
    print("RESULTATS - ACCURACY SUR LE TRAIN SET")
    print("=" * 70)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[OK] ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"[OK] F1 Macro:    {f1_macro:.4f}")
    print(f"[OK] F1 Weighted: {f1_weighted:.4f}")
    
    # Rapport detaille par classe
    class_names = ['Demolition', 'Road', 'Residential', 
                   'Commercial', 'Industrial', 'Mega Projects']
    print("\nRAPPORT PAR CLASSE:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print("\nMATRICE DE CONFUSION:")
    print(cm)
    print("\n(Lignes = vrais labels, Colonnes = predits)")
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'y_pred': y_pred,
        'y_true': y_true
    }


if __name__ == "__main__":
    print("""
========================================================
          CALCUL D'ACCURACY SUR LE TRAIN SET
========================================================
Train:  train.geojson [OK] (avec labels)
Test:   test.geojson (sans labels - set Kaggle)

On evalue le modele sur le TRAIN SET
(predictions vs vrais labels du train)
========================================================
""")
    results = calculate_accuracy()
