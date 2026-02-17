# Analyse des résultats de cross-validation

## Résultats obtenus (run rapide)

- **Configuration** : 2 folds, ~12k échantillons stratifiés, RF 150 arbres / XGB 200 arbres.
- **Random Forest**
  - Fold 1 : Accuracy **78.35%**, F1 macro 0.507, F1 weighted 0.771
  - Fold 2 : Accuracy **79.65%**, F1 macro 0.520, F1 weighted 0.784
  - **Moyenne** : ~**79% accuracy**, F1 macro ~0.51
- **XGBoost** : run interrompu (timeout) avant la fin du 2e fold.

## Écart par rapport à l’objectif 97%

On est autour de **79% accuracy** en rapide. Pour viser **>97%** :

1. **Utiliser tout le jeu d’entraînement**
   - Le run rapide ne voit qu’une partie des données (~12k sur ~296k).
   - Lancer la CV complète :
     ```bash
     python3 run_cv_full_97.py
     ```
   - Dans `cross_validation_honest.py`, pour un run “full” à la main : mettre `FAST = False` et `SAMPLE_SIZE = 0`.

2. **Renforcer les modèles**
   - RF : 1000 arbres, max_depth 50 (déjà prévus en mode non-FAST).
   - XGBoost : 1500 arbres, max_depth 14, learning_rate 0.01 (déjà en non-FAST).
   - Possibilité d’augmenter encore `n_estimators` (ex. 2000) et de faire un petit grid search sur `max_depth` / `learning_rate` si besoin.

3. **Déséquilibre des classes**
   - Sur un échantillon, “Mega Projects” a très peu d’exemples (ordre de 5 dans 20k). Une classe si rare plafonne le F1 macro et peut limiter l’accuracy sur les petites classes.
   - Pistes : suréchantillonnage (SMOTE ou oversample des classes rares), ou fusionner “Mega Projects” avec une classe proche si le cahier des charges le permet.

4. **Ensemble**
   - Combiner RF + XGBoost (vote majoritaire ou moyenne des probas puis argmax) peut gagner quelques points.

5. **Features**
   - Vérifier que `status_progression` et `avg_days_between_status` sont bien présents dans le CSV (régénérer avec `Feature.py` si besoin).
   - Tester d’autres variables dérivées (ex. ratios temporels, autres indices spectraux) si tu en ajoutes.

## Comment lancer la CV “honnête” complète

- **Run rapide (≈2–4 min)**  
  Dans `cross_validation_honest.py` : `FAST = True`, `SAMPLE_SIZE = 12000` (ou autre valeur), puis :
  ```bash
  python3 cross_validation_honest.py
  ```

- **Run full pour viser 97% (30 min – 2 h)**  
  Soit :
  ```bash
  python3 run_cv_full_97.py
  ```
  Soit dans `cross_validation_honest.py` : `FAST = False`, `SAMPLE_SIZE = 0`, puis :
  ```bash
  python3 cross_validation_honest.py
  ```

Les résultats impriment Accuracy et F1 macro par fold, puis le rapport de classification et la matrice de confusion agrégés.
