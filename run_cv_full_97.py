"""
Run cross-validation sur TOUT le dataset pour viser >97%.
À lancer en ligne de commande (peut prendre 30 min à 2 h selon la machine).
Usage: python3 run_cv_full_97.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Forcer mode "full" avant que le module n'utilise ses constantes
import cross_validation_honest as cvmod
cvmod.FAST = False
cvmod.SAMPLE_SIZE = 0
cvmod.N_SPLITS = 5
# Réappliquer les configs dérivées (get_rf / get_xgb utilisent FAST)
cvmod.get_rf = lambda: cvmod.RandomForestClassifier(
    n_estimators=1000, max_depth=50, min_samples_split=3, min_samples_leaf=1,
    class_weight='balanced', random_state=cvmod.RANDOM_STATE, n_jobs=-1)
cvmod.get_xgb = lambda: cvmod.XGBClassifier(
    n_estimators=1500, max_depth=14, learning_rate=0.01, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.9, objective='multi:softmax', num_class=6,
    tree_method='hist', n_jobs=-1, random_state=cvmod.RANDOM_STATE)

if __name__ == '__main__':
    print("Mode FULL : tout le dataset, 5 folds, modèles complets (RF 1000, XGB 1500).")
    print("Cela peut prendre 30 min à 2 h...\n")
    cvmod.main()
