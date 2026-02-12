"""
preprocess_features.py

But:
- Lire un GeoJSON (train / test)
- Nettoyer + normaliser certaines colonnes
- Réordonner dates + features liées aux dates (date0..date4, img_*_date0..date4, change_status_date*)
- Ajouter des features RGB dérivées (brightness/ratios/texture + stats temporelles)
- Ajouter des features géométriques métriques (area_m2, perimeter_m, compactness) via UTM local
- Imputer les NaN géométriques par kNN spatial (voisins)
- Sauvegarder un fichier préprocessé pour l'entraînement plus tard

⚠️ Hypothèses:
- Après process_columns(), on travaille avec date0..date4 et img_*_date0..date4
- Le dataset est majoritairement lon/lat; on force CRS=4326 pour les calculs métriques.
"""

import re
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.ops import transform
from pyproj import CRS, Transformer
from sklearn.neighbors import NearestNeighbors


# =========================
# Utils: Renommage colonnes
# =========================

def process_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Renomme img_*_date1..date5 -> img_*_date0..date4 (décalage -1).
    Ne touche pas aux colonnes date0..date4 déjà présentes.
    """
    df = gdf.copy()
    rename_map = {}

    for c in df.columns:
        m = re.match(r"^(img_.*_date)(\d+)$", c)
        if m:
            idx = int(m.group(2))
            # 1->0, 2->1, ..., 5->4
            rename_map[c] = f"{m.group(1)}{idx - 1}"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# =========================
# Normalisation catégories
# =========================

not_seen = "NAN"

def normalize_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise geography_type et urban_type:
    - NaN -> "NAN"
    - "N,A" -> "NAN"
    """
    out = df.copy()
    for col in ["geography_type", "urban_type"]:
        if col in out.columns:
            out[col] = out[col].fillna(not_seen).replace({"N,A": not_seen})
    return out


# =========================
# Réordonnancement temporel
# =========================

def reorder_dates(df: pd.DataFrame, date_format="%d-%m-%Y", errors="coerce") -> pd.DataFrame:
    """
    Trie date0..date4 (NaT à la fin) et réordonne toutes les colonnes finissant par date0..date4
    avec la même permutation ligne-par-ligne.
    Puis ffill horizontal sur change_status_date0..4 (si au moins une date existe).
    """
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]
    date_cols = [c for c in date_cols if c in out.columns]
    if len(date_cols) != 5:
        raise ValueError(f"Il manque des colonnes date0..date4, trouvé: {date_cols}")

    # colonnes qui finissent par date0..date4 (inclut img_*, change_status_*, etc.)
    related_cols = [c for c in out.columns if any(c.endswith(f"date{i}") for i in range(5))]

    # parse dates
    out[date_cols] = out[date_cols].apply(pd.to_datetime, format=date_format, errors=errors)

    d = out[date_cols].to_numpy("datetime64[ns]")
    di = d.astype("int64")
    di[np.isnat(d)] = np.iinfo(np.int64).max
    order = np.argsort(di, axis=1)

    rows = np.arange(len(out))[:, None]

    # regrouper par base "...date"
    bases = {}
    for c in related_cols:
        base = c[:-1]  # enlève le dernier char (0..4)
        bases.setdefault(base, []).append(c)

    for base, _cols in bases.items():
        cols = [f"{base}{i}" for i in range(5)]
        if not all(c in out.columns for c in cols):
            continue
        block = out[cols].to_numpy()
        out[cols] = block[rows, order]

    # ffill status si dates pas toutes NaN
    status_cols = [f"change_status_date{i}" for i in range(5)]
    if all(c in out.columns for c in status_cols):
        all_dates_nan = out[date_cols].isna().all(axis=1)
        out.loc[~all_dates_nan, status_cols] = out.loc[~all_dates_nan, status_cols].ffill(axis=1)

    return out


# =========================
# Features RGB dérivées
# =========================

def _detect_date_indices(df: pd.DataFrame, prefix: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    idx = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            idx.append(int(m.group(1)))
    return sorted(set(idx))


def add_rgb_derived_features(
    df: pd.DataFrame,
    date_prefix_mean: str = "img_{}_mean_date",   # img_red_mean_date0..4
    date_prefix_std: str  = "img_{}_std_date",
    channels: tuple[str, str, str] = ("red", "green", "blue"),
    add_per_date: bool = True,
    add_temporal: bool = True,
    eps: float = 1e-8,
) -> pd.DataFrame:
    out = df.copy()

    sample_prefix = date_prefix_mean.format(channels[0])
    idxs = _detect_date_indices(out, sample_prefix)
    if not idxs:
        raise ValueError(f"Aucune colonne trouvée du type '{sample_prefix}0'..'4'.")

    def col_mean(ch, i): return date_prefix_mean.format(ch) + str(i)
    def col_std(ch, i):  return date_prefix_std.format(ch) + str(i)

    # check cols
    missing = []
    for i in idxs:
        for ch in channels:
            if col_mean(ch, i) not in out.columns:
                missing.append(col_mean(ch, i))
            if col_std(ch, i) not in out.columns:
                missing.append(col_std(ch, i))
    if missing:
        raise ValueError(f"Colonnes manquantes ({len(missing)}). Ex: {missing[:8]}")

    T = len(idxs)
    n = len(out)

    Rm = np.column_stack([out[col_mean("red", i)].to_numpy(float) for i in idxs])
    Gm = np.column_stack([out[col_mean("green", i)].to_numpy(float) for i in idxs])
    Bm = np.column_stack([out[col_mean("blue", i)].to_numpy(float) for i in idxs])

    Rs = np.column_stack([out[col_std("red", i)].to_numpy(float) for i in idxs])
    Gs = np.column_stack([out[col_std("green", i)].to_numpy(float) for i in idxs])
    Bs = np.column_stack([out[col_std("blue", i)].to_numpy(float) for i in idxs])

    brightness = (Rm + Gm + Bm) / 3.0
    denom = (Rm + Gm + Bm) + eps
    red_ratio = Rm / denom
    green_ratio = Gm / denom
    blue_ratio = Bm / denom

    chroma = np.std(np.stack([Rm, Gm, Bm], axis=2), axis=2)
    texture = (Rs + Gs + Bs) / 3.0
    rel_texture = texture / (brightness + eps)
    texture_rgb_std = np.std(np.stack([Rs, Gs, Bs], axis=2), axis=2)

    if add_per_date:
        for j, i in enumerate(idxs):
            out[f"brightness_date{i}"] = brightness[:, j]
            out[f"red_ratio_date{i}"] = red_ratio[:, j]
            out[f"green_ratio_date{i}"] = green_ratio[:, j]
            out[f"blue_ratio_date{i}"] = blue_ratio[:, j]
            out[f"chroma_date{i}"] = chroma[:, j]
            out[f"texture_date{i}"] = texture[:, j]
            out[f"rel_texture_date{i}"] = rel_texture[:, j]
            out[f"texture_rgb_std_date{i}"] = texture_rgb_std[:, j]

    if add_temporal:
        t = np.arange(T, dtype=float)

        def add_time_stats(name: str, X: np.ndarray):
            # X shape (n, T)
            out[f"{name}_mean"] = np.nanmean(X, axis=1)
            out[f"{name}_std"] = np.nanstd(X, axis=1)
            out[f"{name}_min"] = np.nanmin(X, axis=1)
            out[f"{name}_max"] = np.nanmax(X, axis=1)
            out[f"{name}_range"] = out[f"{name}_max"] - out[f"{name}_min"]

            # slope vs time index 0..T-1
            t = np.arange(T, dtype=float)
            slopes = np.full(n, np.nan, dtype=float)
            for r in range(n):
                y = X[r, :]
                msk = np.isfinite(y)
                if msk.sum() >= 2:
                    tt = t[msk]
                    yy = y[msk]
                    tt0 = tt - tt.mean()
                    denom2 = (tt0 ** 2).sum()
                    slopes[r] = (tt0 * (yy - yy.mean())).sum() / (denom2 + eps)
            out[f"{name}_slope"] = slopes

            # deltas + jumps (SAFE)
            if T < 2:
                out[f"{name}_max_jump"] = np.nan
                out[f"{name}_argmax_jump_step"] = -1
                return

            d = np.diff(X, axis=1)  # (n, T-1)
            absd = np.abs(d)

            # max jump: nanmax mais safe (nanmax(all-NaN) => warning + NaN, c'est ok)
            max_jump = np.nanmax(absd, axis=1)
            out[f"{name}_max_jump"] = max_jump

            # argmax jump: éviter nanargmax sur lignes all-NaN
            all_nan_row = ~np.isfinite(absd).any(axis=1)  # True si aucun delta fini
            argmax = np.full(n, -1, dtype="int32")
            good = ~all_nan_row
            if good.any():
                # remplace NaN par -inf pour faire argmax normal
                absd2 = absd.copy()
                absd2[~np.isfinite(absd2)] = -np.inf
                argmax[good] = np.argmax(absd2[good], axis=1).astype("int32")
            out[f"{name}_argmax_jump_step"] = argmax

            # per-step deltas
            for k in range(T - 1):
                out[f"delta_{name}_step{k}"] = d[:, k]

        add_time_stats("brightness", brightness)
        add_time_stats("texture", texture)
        add_time_stats("green_ratio", green_ratio)

    return out


# =========================
# Géométrie métrique + imputation
# =========================

def lonlat_compatibility_report(gdf: gpd.GeoDataFrame):
    def is_lonlat_like(geom):
        if geom is None or geom.is_empty:
            return False
        p = geom.representative_point()
        x, y = p.x, p.y
        return (-180 <= x <= 180) and (-90 <= y <= 90)

    mask = gdf.geometry.apply(is_lonlat_like)

    report = {
        "total": len(gdf),
        "lonlat_like": int(mask.sum()),
        "not_lonlat_like": int((~mask).sum()),
        "pct_lonlat_like": float(mask.mean() * 100),
        "pct_not_lonlat_like": float((~mask).mean() * 100),
    }
    return report, mask


def add_metric_geometry_features(
    gdf: gpd.GeoDataFrame,
    area_col: str = "area_m2",
    perimeter_col: str = "perimeter_m",
    compactness_col: str = "compactness",
) -> gpd.GeoDataFrame:
    """
    Calcule area/perimeter/compactness en reprojetant chaque géométrie dans sa zone UTM locale.
    """
    gdf = gdf.copy()

    # IMPORTANT: make_valid() doit être assigné (sinon ça ne change rien)
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        # fallback (moins clean mais robuste)
        gdf["geometry"] = gdf.geometry.buffer(0)

    if gdf.crs is None:
        raise ValueError("GeoDataFrame sans CRS. Fais gdf.set_crs(4326, allow_override=True) avant.")
    if gdf.crs.to_epsg() != 4326:
        raise ValueError("Le GeoDataFrame doit être en EPSG:4326 pour le choix UTM par lon/lat.")

    def _compute(geom, src_crs):
        if geom is None or geom.is_empty:
            return np.nan, np.nan, np.nan

        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                return np.nan, np.nan, np.nan

        p = geom.representative_point()
        lon, lat = p.x, p.y

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            return np.nan, np.nan, np.nan

        utm_zone = int(np.floor((lon + 180) / 6) + 1)
        if not (1 <= utm_zone <= 60):
            return np.nan, np.nan, np.nan

        epsg = (32600 + utm_zone) if lat >= 0 else (32700 + utm_zone)
        utm_crs = CRS.from_epsg(epsg)

        try:
            transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True).transform
            geom_utm = transform(transformer, geom)

            area = float(geom_utm.area)
            per = float(geom_utm.length)
            if not np.isfinite(area) or not np.isfinite(per) or area <= 0 or per <= 0:
                return np.nan, np.nan, np.nan

            comp = 4 * np.pi * area / (per ** 2)
            return area, per, float(comp)
        except Exception:
            return np.nan, np.nan, np.nan

    res = gdf.geometry.apply(lambda geom: _compute(geom, gdf.crs))
    gdf[[area_col, perimeter_col, compactness_col]] = list(res)
    return gdf


def spatial_knn_impute(
    gdf: gpd.GeoDataFrame,
    cols=("area_m2", "perimeter_m", "compactness"),
    k=40,
    agg="mean",  # "mean" ou "median"
) -> gpd.GeoDataFrame:
    """
    Impute NaN values using k nearest spatial neighbors (by centroid),
    using only neighbors with valid values.
    """
    gdf = gdf.copy()

    centroids = gdf.geometry.centroid
    coords = np.vstack([centroids.x.values, centroids.y.values]).T

    nan_mask = gdf[list(cols)].isna().any(axis=1)
    valid_mask = ~nan_mask

    if nan_mask.sum() == 0:
        return gdf

    # Si jamais (très rare) tout est NaN, on fallback médiane globale
    if valid_mask.sum() == 0:
        for c in cols:
            gdf[c] = gdf[c].fillna(gdf[c].median())
        return gdf

    nbrs = NearestNeighbors(
        n_neighbors=min(k, int(valid_mask.sum())),
        algorithm="ball_tree",
    )
    nbrs.fit(coords[valid_mask])

    _, indices = nbrs.kneighbors(coords[nan_mask])

    valid_idx = np.where(valid_mask)[0]
    nan_idx = np.where(nan_mask)[0]

    for i, row_idx in enumerate(nan_idx):
        neighbor_rows = valid_idx[indices[i]]
        for col in cols:
            values = gdf.iloc[neighbor_rows][col].dropna()
            if len(values) == 0:
                continue
            gdf.at[row_idx, col] = values.median() if agg == "median" else values.mean()

    return gdf


# =========================
# Pipeline complet
# =========================

def preprocess_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Pipeline complet et propre:
    - rename img_* date1..5 -> date0..4
    - normalize types
    - reorder dates + columns
    - rgb derived features
    - geometry features + knn impute
    - log geometry
    """
    gdf = process_columns(gdf)
    gdf = normalize_type(gdf)
    gdf = reorder_dates(gdf, errors="coerce")
    gdf = add_rgb_derived_features(gdf)

    # geometry
    report, lonlat_mask = lonlat_compatibility_report(gdf)
    print("Lon/lat compatibility:", report)
    gdf["is_lonlat_valid"] = lonlat_mask.astype("int8")

    # On force CRS 4326 (tu as mesuré que ~99.986% est cohérent)
    gdf = gdf.set_crs(epsg=4326, allow_override=True)

    gdf = add_metric_geometry_features(gdf)
    gdf = spatial_knn_impute(gdf, k=40, agg="mean")

    # logs géométrie (souvent meilleur que brut)
    gdf["log_area_m2"] = np.log1p(gdf["area_m2"])
    gdf["log_perimeter_m"] = np.log1p(gdf["perimeter_m"])

    # sanity checks
    assert np.isfinite(gdf[["area_m2", "perimeter_m", "compactness"]]).all().all()

    return gdf


def save_preprocessed(gdf: gpd.GeoDataFrame, out_path: str):
    """
    Sauvegarde:
    - si out_path finit par .parquet => parquet (recommandé, rapide, compact)
    - sinon => GeoJSON (plus lourd)
    """
    if out_path.lower().endswith(".parquet"):
        gdf.to_parquet(out_path, index=False)
    else:
        gdf.to_file(out_path, driver="GeoJSON")


# =========================
# Main
# =========================

if __name__ == "__main__":
    train_path = "train.geojson"
    test_path = "test.geojson"

    train_out = "train_preprocessed.parquet"
    test_out = "test_preprocessed.parquet"

    # Train
    train_gdf = gpd.read_file(train_path)
    train_gdf = preprocess_gdf(train_gdf)
    save_preprocessed(train_gdf, train_out)
    print(f"Saved: {train_out}  (rows={len(train_gdf)}, cols={train_gdf.shape[1]})")


    try:
        test_gdf = gpd.read_file(test_path)
        test_gdf = preprocess_gdf(test_gdf)
        save_preprocessed(test_gdf, test_out)
        print(f"Saved: {test_out}  (rows={len(test_gdf)}, cols={test_gdf.shape[1]})")
    except Exception as e:
        print(f"(Info) test preprocess skipped: {e}")
