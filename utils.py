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

STATUS_ORDER = {key: i for i, key in enumerate([
        "Greenland",
        "Land Cleared",
        "Prior Construction",
        "Excavation",
        "Materials Dumped",
        "Materials Introduced",
        "Construction Started",
        "Construction Midway",
        "Construction Done",
        "Operational"
        ]
    )
}

ALL_STATUSES = [
        "Greenland",
        "Land Cleared",
        "Prior Construction",
        "Excavation",
        "Materials Dumped",
        "Materials Introduced",
        "Construction Started",
        "Construction Midway",
        "Construction Done",
        "Operational"
        ]

ALL_GEO_TYPES = ['Coastal', 'Hills', 'A', 'Sparse Forest', 'Lakes', 'Desert', 'Grass Land', 'N', 'Snow', 'River', 'Barren Land', 'Dense Forest', 'Farms']
ALL_URBAN_TYPES = ['Sparse Urban', 'A', 'Rural', 'N', 'Dense Urban', 'Urban Slum', 'Industrial']


def encode_one_hot_types(df, feat_cols, prefix=None):
    """
    Encode les colonnes 'geography_type' et 'urban_type' en one-hot.

    Args:
        df: DataFrame
        feat_cols: set() dans lequel ajouter les noms des nouvelles colonnes features
        prefix: str ou dict avec préfixes pour chaque type (ex: {'geo': 'geo', 'urban': 'urban'})

    Returns:
        DataFrame modifié avec les nouvelles colonnes one-hot
    """
    # Définir les préfixes par défaut
    if prefix is None:
        prefix = {'geo': 'geo', 'urban': 'urban'}
    elif isinstance(prefix, str):
        prefix = {'geo': f"{prefix}_geo", 'urban': f"{prefix}_urban"}

    # ===== ENCODAGE DE GEOGRAPHY_TYPE =====
    if 'geography_type' in df.columns:
        # Créer les colonnes one-hot pour geography_type
        # On combine 'N' et 'A' en 'N,A' pour geography_type
        geo_categories = [cat for cat in ALL_GEO_TYPES if cat not in ['N', 'A']]
        if 'N,A' not in geo_categories:
            geo_categories.append('N,A')

        for category in geo_categories:
            col_name = f"{prefix['geo']}_{category}"
            df[col_name] = 0
            feat_cols.add(col_name)

        # Remplir les colonnes
        for idx, value in df['geography_type'].items():
            if pd.isna(value):
                continue

            types = [t.strip() for t in str(value).split(',')]

            # Vérifier si 'N' et 'A' sont présents ensemble pour geography_type
            has_n = 'N' in types
            has_a = 'A' in types

            if has_n and has_a:
                # Ajouter la colonne combinée 'N,A' pour geography
                col_name = f"{prefix['geo']}_N,A"
                df.at[idx, col_name] = 1
                # Retirer N et A de la liste pour éviter les doublons
                types = [t for t in types if t not in ['N', 'A']]

            # Ajouter les autres types de geography
            for type_name in types:
                if type_name in ALL_GEO_TYPES:
                    col_name = f"{prefix['geo']}_{type_name}"
                    df.at[idx, col_name] = 1

    # ===== ENCODAGE DE URBAN_TYPE =====
    if 'urban_type' in df.columns:
        # Créer les colonnes one-hot pour urban_type
        # On combine 'N' et 'A' en 'N,A' pour urban_type (colonne séparée de geography!)
        urban_categories = [cat for cat in ALL_URBAN_TYPES if cat not in ['N', 'A']]
        if 'N,A' not in urban_categories:
            urban_categories.append('N,A')

        for category in urban_categories:
            col_name = f"{prefix['urban']}_{category}"
            df[col_name] = 0
            feat_cols.add(col_name)

        # Remplir les colonnes
        for idx, value in df['urban_type'].items():
            if pd.isna(value):
                continue

            types = [t.strip() for t in str(value).split(',')]

            # Vérifier si 'N' et 'A' sont présents ensemble pour urban_type
            has_n = 'N' in types
            has_a = 'A' in types

            if has_n and has_a:
                # Ajouter la colonne combinée 'N,A' pour urban (colonne différente de geography!)
                col_name = f"{prefix['urban']}_N,A"
                df.at[idx, col_name] = 1
                # Retirer N et A de la liste pour éviter les doublons
                types = [t for t in types if t not in ['N', 'A']]

            # Ajouter les autres types de urban
            for type_name in types:
                if type_name in ALL_URBAN_TYPES:
                    col_name = f"{prefix['urban']}_{type_name}"
                    df.at[idx, col_name] = 1

    return df


def add_geometry_features(gdf, feature_cols, metric_epsg=6933):
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
    # gdf_m = gdf.to_crs(epsg=metric_epsg)
    gdf_m = gdf.copy()
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
    # gdf_m["compactness"] = (
    #         4 * np.pi * gdf_m["polygon_area_m2"]
    #         / (gdf_m["polygon_perimeter_m"] ** 2)
    # )

    feature_cols |= {"polygon_area_m2", "polygon_perimeter_m"}

    return gdf_m



def add_max_gap_between_sets(df, feat_cols):
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]
    status_cols = [f"change_status_date{i}" for i in range(5)]

    dates = out[date_cols].apply(pd.to_datetime, format="%d-%m-%Y", errors="raise", utc=True)
    statuses = out[status_cols]

    def row_max_gap(row_dates, row_statuses):

        # map_gap = { for d, s in zip(row_dates, row_statuses) if not pd.isna(d) and not pd.isna(s)}
        pair = [
            (d, s)
            for i, (d, s) in enumerate(zip(row_dates, row_statuses))
            if not pd.isna(d) and not pd.isna(s)
        ]
        sorted_dates = sorted(pair, key=lambda x: x[0])
        if len(sorted_dates) <= 1:
            return 0.0, 0.0, 0.0

        changes = []
        for i in range(len(sorted_dates)-1):
            d1, s1 = sorted_dates[i]
            d2, s2 = sorted_dates[i+1]
            if s1 != s2:
                changes.append(((d2 - d1).total_seconds() / 86400.0, STATUS_ORDER[s2]-STATUS_ORDER[s1]))
        if len(changes) == 0:
            return 0.0, 0.0, 0.0
        return len(changes), *max(changes, key=lambda x: x[1])

    date_changes = np.array([
        row_max_gap(d, s)
        for d, s in zip(dates.to_numpy(), statuses.to_numpy())
    ])

    cols = ["nb_changes", "nb_days_max_change", "change_gradiant"]
    for i, c in enumerate(cols):
        out[c] = date_changes[:, i]
    feat_cols |= set(cols)
    return out


def add_last_state(df, feat_cols):
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]
    status_cols = [f"change_status_date{i}" for i in range(5)]

    dates = out[date_cols].apply(pd.to_datetime, format="%d-%m-%Y", errors="raise", utc=True)
    statuses = out[status_cols]

    def encode_last(row_dates, row_statuses):
        pairs = [
            (d, s)
            for d, s in zip(row_dates, row_statuses)
            if not pd.isna(d) and not pd.isna(s)
        ]
        if not pairs:
            return None  # pas de last_state

        _, last_status = max(pairs, key=lambda x: x[0])
        return last_status

    last_labels = [
        encode_last(d, s)
        for d, s in zip(dates.to_numpy(), statuses.to_numpy())
    ]

    last_oh = pd.get_dummies(pd.Series(last_labels, index=out.index), prefix="last")

    expected = [f"last_{s}" for s in ALL_STATUSES]
    feat_cols |= set(expected)
    last_oh = last_oh.reindex(columns=expected, fill_value=0).astype(float)

    out = pd.concat([out, last_oh], axis=1)

    return out

def add_regressed_state(df, feat_cols):
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]
    status_cols = [f"change_status_date{i}" for i in range(5)]

    dates = out[date_cols].apply(pd.to_datetime, format="%d-%m-%Y", errors="raise", utc=True)
    statuses = out[status_cols]

    def encode_regression(row_dates, row_statuses):
        regression_flags = [False] * 4
        pairs = [
            (d, s, i)
            for i, (d, s) in enumerate(zip(row_dates, row_statuses))
            if not pd.isna(d) and not pd.isna(s)
        ]
        if not pairs:
            return None  # pas de last_state

        # 2. Trier par date (chronologique)
        pairs_sorted = sorted(pairs, key=lambda x: x[0])

        # 3. Vérifier les régressions dans l'ordre chronologique
        previous_valid_rank = None
        previous_valid_status = None

        for i, (date, status, original_idx) in enumerate(pairs_sorted):
            current_rank = STATUS_ORDER.get(status)

            # Si on a un précédent statut valide pour comparer
            if previous_valid_rank is not None:
                # Détecter régression: rang courant < rang précédent
                if current_rank < previous_valid_rank:
                    regression_flags[i-1] = True

                # Mettre à jour le précédent seulement si nouveau statut
                if status != previous_valid_status:
                    previous_valid_rank = current_rank
                    previous_valid_status = status
            else:
                # Premier statut valide
                previous_valid_rank = current_rank
                previous_valid_status = status

        return regression_flags

    # Appliquer à chaque ligne
    regression_results_list = []

    for idx in range(len(out)):
        row_dates = dates.iloc[idx].values
        row_statuses = statuses.iloc[idx].values
        flags = encode_regression(row_dates, row_statuses)

        # Si None, créer une liste de False (ou NaN si tu préfères)
        if flags is None:
            flags = [False] * 4

        regression_results_list.append(flags)

    # Convertir en DataFrame
    cols = [f"regression_date{i}" for i in range(4)]
    regression_results = pd.DataFrame(
        regression_results_list,
        columns=cols,
        index=out.index
    )

    feat_cols |= set(cols)

    # Concaténer
    out = pd.concat([out, regression_results], axis=1)

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
