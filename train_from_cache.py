"""
Train from cached features only (no GeoJSON load). Run after train_and_predict.py
has created feature_cache/ at least once. Tune model here for F1 > 90%.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

CHANGE_TYPE_MAP = {
    "Demolition": 0, "Road": 1, "Residential": 2,
    "Commercial": 3, "Industrial": 4, "Mega Projects": 5,
}
CACHE_DIR = "feature_cache"
RANDOM_STATE = 42


def main():
    X_train = np.load(os.path.join(CACHE_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(CACHE_DIR, "X_test.npy"))
    y = np.load(os.path.join(CACHE_DIR, "y.npy"))
    print("Loaded cache. Train:", X_train.shape, "Test:", X_test.shape)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=28,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y)
    y_pred = clf.predict(X_train)
    f1 = f1_score(y, y_pred, average="weighted")
    print("Train F1 weighted: {:.4f}".format(f1))
    print(classification_report(y, y_pred, target_names=list(CHANGE_TYPE_MAP.keys()), zero_division=0))

    pred_test = clf.predict(X_test)
    tid_path = os.path.join(CACHE_DIR, "test_ids.npy")
    if os.path.isfile(tid_path):
        test_ids = np.load(tid_path)
    else:
        import geopandas as gpd
        test_ids = gpd.read_file("test.geojson").index.values
    out = pd.DataFrame({"Id": test_ids, "change_type": pred_test})
    out.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
