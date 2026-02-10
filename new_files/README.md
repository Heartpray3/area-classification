# Remote Sensing Change Detection – ML Challenge

## 🎯 Objectif

Classifier une zone géographique (polygone) en **6 classes** à partir de features issues d’images satellites multi-dates.

**Classes :**

```
0 – Demolition
1 – Road
2 – Residential
3 – Commercial
4 – Industrial
5 – Mega Projects

```

---

## 📦 Données

Chaque échantillon contient :

- **Un polygone irrégulier** (géométrie)
- **5 statuts catégoriels temporels** (évolution du site)
- **Features urbaines de voisinage** (catégorielles, multi-valeurs)
- **Features géographiques de voisinage** (catégorielles, multi-valeurs)

Les données sont déjà extraites (pas d’images brutes).

---

## 🔁 Pipeline ML (attendu)

1. **Prétraitement**
  - Encodage des variables catégorielles (one-hot / ordinal / binaire)
  - Gestion des NaN / inf
  - Normalisation si nécessaire
2. **Feature engineering**
  - Polygones → `area`, `perimeter`, `compactness`, etc.
  - Statuts temporels →
    - dernier statut
    - transitions
    - durée entre changements
  - Urban / Geo features → multi-hot encoding
3. **Réduction / sélection de features**
  - PCA / SVD (si utile)
  - Feature selection (variance, mutual info, modèles linéaires)
4. **Modélisation**
  - Baseline : k-NN (~40%)
  - Modèles testés / envisagés :
    - Logistic Regression
    - SVM (linéaire / RBF)
    - Random Forest / Gradient Boosting
    - Naive Bayes (Bernoulli / Multinomial)
    - Ensembles
5. **Évaluation**
  - Classification **multi-classe**
  - Métrique principale : **F1-score**
  - Cross-validation
  - Attention au déséquilibre de classes

---

## 🧪 Contraintes connues

- Beaucoup de features catégorielles (one-hot)
- Haute dimension possible
- Bruit + redondance
- Données géographiques parfois instables (NaN après projection)

---

## 🚀 Pipeline fourni (objectif F1 ≥ 95 %)

**Script principal :** `train_and_predict.py`

- **Features :**
  - Géométrie : area, perimeter, compactness, log(area), log(perimeter)
  - Multi-hot : `urban_type`, `geography_type`
  - Statuts : séquence encodée (5 dates) + nb transitions + **ordre sémantique** (max_gap, last_state one-hot, flags de régression) — voir `features_extra.py`
  - Image : 30 canaux (RGB mean/std × 5 dates) + 12 stats temporelles (moyenne/std dans le temps)
  - Dates en ordinal
- **Prétraitement :** médiane (train) pour NaN, `StandardScaler`, inf/nan → 0.
- **Modèle :** Ensemble RF + HistGradientBoosting + ExtraTrees + XGBoost (si installé). Poids de classe plafonnés. Moyenne des probas pour la prédiction.
- **Sortie :** `submission.csv` (colonnes `Id`, `change_type` 0–5).

**Lancer :**
```bash
pip install -r requirements.txt
python train_and_predict.py
```
- Première run : construction des features + cache dans `feature_cache_v2/` (long).
- Runs suivants : chargement du cache (plus rapide).
- **Pour monter le score (75% → 80%+)** : garder `USE_STATUS_EXTRA = False` et les poids de classe plafonnés (déjà activés). Avec `SUBSAMPLE = None` (tout le train), viser ~85%+ F1 (run long).
- **Run rapide** : `SUBSAMPLE = 0.2` ou `0.25` → ~77% F1 en quelques min.
- **Viser 95 % en CV sur le train** : `SUBSAMPLE = None`, lancer `python train_and_predict.py`. Le script affiche **Ensemble CV F1 weighted** après la 2-fold CV et l’écrit dans `cv_result.txt`. Run complet (données entières + gros modèles) : compter 1–2 h selon la machine. Pour viser 95 %, installer XGBoost/LightGBM et les intégrer à l’ensemble peut aider.

**Entraînement depuis le cache uniquement :**
```bash
python train_from_cache.py   # après au moins une run complète de train_and_predict.py
```

**Ton code (évaluation) :** ta CV en k folds et F1 macro est cohérente ; ici on utilise une CV stratifiée et F1 weighted pour coller à la métrique du README. Les features (ordre des statuts, max_gap, régressions, last_state one-hot) sont intégrées dans `features_extra.py` et utilisées dans le pipeline.

