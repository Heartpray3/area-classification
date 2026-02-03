import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score


PRE_CONSTRUCTION_STATUSES = {
    "Greenland",
    "Land Cleared",
    "Prior Construction",
    "Excavation",
    "Materials Dumped",
    "Materials Introduced",
    "Construction Started",
}

POST_CONSTRUCTION_STATUSES = {
    "Construction Midway",
    "Construction Done",
    "Operational",
}


def add_geometry_features(gdf, metric_epsg=6933):
    """
    gdf: GeoDataFrame en EPSG:4326
    Ajoute:
      - polygon_area_m2
      - polygon_perimeter_m
      - compactness (sans unité)
    """
    gdf["area_orig"] = gdf.geometry.area
    gdf["perimeter_orig"] = gdf.geometry.length

    # 1) reprojection métrique
    gdf_m = gdf.to_crs(epsg=metric_epsg)

    # 2) calculs métriques
    gdf_m["polygon_area_m2"] = gdf_m.geometry.area
    gdf_m["polygon_perimeter_m"] = gdf_m.geometry.length

    # 3) fallback pour les NaN
    mask_nan = (
            gdf_m["polygon_area_m2"].isna()
            | gdf_m["polygon_perimeter_m"].isna()
    )

    gdf_m.loc[mask_nan, "polygon_area_m2"] = gdf.loc[mask_nan, "area_orig"]
    gdf_m.loc[mask_nan, "polygon_perimeter_m"] = gdf.loc[mask_nan, "perimeter_orig"]

    # 4) compacité
    gdf_m["compactness"] = (
            4 * np.pi * gdf_m["polygon_area_m2"]
            / (gdf_m["polygon_perimeter_m"] ** 2)
    )

    return gdf_m




def add_max_gap_between_sets(df):
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]
    status_cols = [f"change_status_date{i}" for i in range(5)]

    dates = out[date_cols].apply(pd.to_datetime, format="%d-%m-%Y", errors="coerce", utc=True)
    statuses = out[status_cols]

    def row_max_gap(row_dates, row_statuses):
        set1_dates = []
        set2_dates = []

        for d, s in zip(row_dates, row_statuses):
            if pd.isna(d) or pd.isna(s):
                continue
            if s in PRE_CONSTRUCTION_STATUSES:
                set1_dates.append(d)
            elif s in POST_CONSTRUCTION_STATUSES:
                set2_dates.append(d)

        if not set1_dates or not set2_dates:
            return 0.0

        # max |t2 - t1|
        max_gap = max(
            abs((t2 - t1).total_seconds())
            for t1 in set1_dates
            for t2 in set2_dates
        )

        return max_gap / 86400.0  # en jours

    out["pre_post_construction_time_gap_days"] = [
        row_max_gap(d, s)
        for d, s in zip(dates.to_numpy(), statuses.to_numpy())
    ]

    return out


def cross_validation(k, model, train_x, train_y):
    n = len(train_x)
    fold_size = n // k
    idx = np.arange(n)
    acc_sum = 0
    f1_sum = 0
    for f in range(k):
        start = f * fold_size
        end = (f + 1) * fold_size if f < k - 1 else n

        mask_train = (idx < start) | (idx >= end)

        mini_train_x = train_x[mask_train, :]
        mini_train_y = train_y[mask_train]

        val_x = train_x[start:end, :]
        val_y = train_y[start:end]

        m = clone(model)
        m.fit(mini_train_x, mini_train_y)
        y_pred = m.predict(val_x)
        f1 = f1_score(val_y, y_pred, average="macro")
        acc = accuracy_score(val_y, y_pred)
        f1_sum += f1
        acc_sum += acc
    print(f"Cross validation: {k} folds, mean f1 score = {f1_sum/k}")
    print(f"Cross validation: {k} folds, mean acc = {acc_sum/k}")
