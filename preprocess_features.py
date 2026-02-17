"""
preprocess_train_test_parquet.py

Train + Test GeoJSON -> Parquet (ONLY parquet)
- Fix GEOSException convex_hull (NaN/Inf coords) via filtering + safe convex area
- process_columns(): img_*_date1..5 -> img_*_date0..4
- reorder_dates(): CLEAN (date0..4 + reorder all *date0..4 blocks)
- timeline features (uses img_*_date0..4)
- one-hot tags CLEAN: ignore N/A/N,A + strip spaces, no flags, no tag_count
- urban/geo derived features (density score, has_water, has_farms, etc.)
- spatial features in meters using estimate_utm_crs()

Outputs:
- train_features.parquet
- test_features.parquet
"""

import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.validation import make_valid

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
    "Operational",
]

STATUS_TO_IDX = {s: i for i, s in enumerate(ALL_STATUSES)}

PHASES = {
    "pre": {"Greenland", "Prior Construction"},
    "site_prep": {"Land Cleared", "Excavation"},
    "materials": {"Materials Dumped", "Materials Introduced"},
    "construction": {"Construction Started", "Construction Midway", "Construction Done"},
    "post": {"Operational"},
}


def process_time_between_statuses(row, max_points=5) -> pd.Series:
    """
    Encode time deltas between statuses.
    Uses date0..date4 + change_status_date0..4 (already cleaned/reordered).
    """
    timeline = []
    for i in range(max_points):
        dt = row.get(f"date{i}")
        st = row.get(f"change_status_date{i}")
        if pd.isna(dt):
            continue
        dt = dt if isinstance(dt, pd.Timestamp) else pd.to_datetime(dt, errors="coerce")
        if pd.isna(dt):
            continue
        st = "Unknown" if pd.isna(st) else str(st).strip()
        timeline.append((dt, st))

    timeline.sort(key=lambda x: x[0])

    out = {}
    out["timeline_len"] = int(len(timeline))
    out["total_duration_days"] = 0.0
    for s in ALL_STATUSES:
        out[f"days_to_{s.replace(' ', '_')}"] = -1.0
    for k in range(1, max_points):
        out[f"delta_days_{k}"] = 0.0
    for p in PHASES:
        out[f"phase_days_{p}"] = 0.0

    out["prior_construction_present"] = 0
    out["prior_construction_before_cleared"] = 0

    for a in ALL_STATUSES:
        for b in ALL_STATUSES:
            out[f"tr_{a.replace(' ', '_')}_to_{b.replace(' ', '_')}"] = 0

    if not timeline:
        return pd.Series(out)

    comp = [timeline[0]]
    for dt, st in timeline[1:]:
        if st == comp[-1][1]:
            continue
        comp.append((dt, st))

    out["timeline_len"] = int(len(comp))
    out["total_duration_days"] = float((comp[-1][0] - comp[0][0]).days)

    for i in range(min(len(comp) - 1, max_points - 1)):
        out[f"delta_days_{i+1}"] = float((comp[i+1][0] - comp[i][0]).days)

    first_date = comp[0][0]
    seen = set()
    for dt, st in comp:
        if st in STATUS_TO_IDX and st not in seen:
            out[f"days_to_{st.replace(' ', '_')}"] = float((dt - first_date).days)
            seen.add(st)

    out["prior_construction_present"] = int(any(st == "Prior Construction" for _, st in comp))

    idx_prior = next((i for i, (_, st) in enumerate(comp) if st == "Prior Construction"), None)
    idx_cleared = next((i for i, (_, st) in enumerate(comp) if st == "Land Cleared"), None)
    out["prior_construction_before_cleared"] = int(
        (idx_prior is not None) and (idx_cleared is not None) and (idx_prior < idx_cleared)
    )

    def phase_of(status: str) -> str | None:
        for ph, statuses in PHASES.items():
            if status in statuses:
                return ph
        return None

    for i in range(len(comp) - 1):
        a = comp[i][1]
        b = comp[i+1][1]
        days = float((comp[i+1][0] - comp[i][0]).days)

        if a in STATUS_TO_IDX and b in STATUS_TO_IDX:
            out[f"tr_{a.replace(' ', '_')}_to_{b.replace(' ', '_')}"] += 1

        ph = phase_of(a)
        if ph is not None:
            out[f"phase_days_{ph}"] += max(days, 0.0)

    return pd.Series(out)



# =========================
# GEOS safety for NaN/Inf coords
# =========================

def _geom_has_nonfinite_coords(geom) -> bool:
    """True if geometry contains NaN/Inf coordinates (or is empty/None)."""
    if geom is None or geom.is_empty:
        return True
    try:
        coords = np.asarray(geom.coords, dtype=float)
        return coords.size == 0 or (not np.isfinite(coords).all())
    except Exception:
        try:
            gi = geom.__geo_interface__
            coords = gi.get("coordinates", None)
            if coords is None:
                return True

            def walk(x):
                if isinstance(x, (list, tuple)):
                    for v in x:
                        yield from walk(v)
                else:
                    yield x

            nums = np.asarray(list(walk(coords)), dtype=float)
            return nums.size == 0 or (not np.isfinite(nums).all())
        except Exception:
            return True

def safe_convex_area(geom) -> float:
    """convex_hull.area but never crashes; returns 0.0 on failure."""
    if geom is None or geom.is_empty or _geom_has_nonfinite_coords(geom):
        return 0.0
    try:
        ch = geom.convex_hull
        a = float(ch.area)
        return a if np.isfinite(a) else 0.0
    except Exception:
        return 0.0


# =========================
# 0) Rename img_* date1..5 -> date0..4
# =========================

def process_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = gdf.copy()
    rename_map = {}

    for c in df.columns:
        m = re.match(r"^(img_.*_date)(\d+)$", c)
        if m:
            idx = int(m.group(2))
            rename_map[c] = f"{m.group(1)}{idx - 1}"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# =========================
# 1) Reorder dates (CLEAN)
# =========================

def reorder_dates(df: pd.DataFrame, date_format="%d-%m-%Y", errors="coerce") -> pd.DataFrame:
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]
    date_cols = [c for c in date_cols if c in out.columns]
    if len(date_cols) != 5:
        raise ValueError(f"Il manque des colonnes date0..date4, trouvé: {date_cols}")

    related_cols = [c for c in out.columns if any(c.endswith(f"date{i}") for i in range(5))]

    out[date_cols] = out[date_cols].apply(pd.to_datetime, format=date_format, errors=errors)

    d = out[date_cols].to_numpy("datetime64[ns]")
    di = d.astype("int64")
    di[np.isnat(d)] = np.iinfo(np.int64).max
    order = np.argsort(di, axis=1)

    rows = np.arange(len(out))[:, None]

    bases = {}
    for c in related_cols:
        base = c[:-1]
        bases.setdefault(base, []).append(c)

    for base in bases:
        cols = [f"{base}{i}" for i in range(5)]
        if not all(c in out.columns for c in cols):
            continue
        block = out[cols].to_numpy()
        out[cols] = block[rows, order]

    status_cols = [f"change_status_date{i}" for i in range(5)]
    if all(c in out.columns for c in status_cols):
        all_dates_nan = out[date_cols].isna().all(axis=1)
        out.loc[~all_dates_nan, status_cols] = out.loc[~all_dates_nan, status_cols].ffill(axis=1)

    return out


# =========================
# 2) Spatial features helpers
# =========================

def get_length_width(geom):
    if geom is None or geom.is_empty or (hasattr(geom, "is_valid") and not geom.is_valid):
        return 0.0, 0.0
    try:
        mrr = geom.minimum_rotated_rectangle
        if mrr.geom_type != "Polygon":
            return 0.0, 0.0
        coords = list(mrr.exterior.coords)
        edge1 = np.hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1])
        edge2 = np.hypot(coords[1][0] - coords[2][0], coords[1][1] - coords[2][1])
        return float(max(edge1, edge2)), float(min(edge1, edge2))
    except Exception:
        return 0.0, 0.0

def get_num_vertices(geom):
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon":
        return sum(len(part.exterior.coords) for part in geom.geoms)
    return 0

def get_max_radius(geom):
    if geom is None or geom.is_empty:
        return 0.0
    try:
        return float(geom.centroid.hausdorff_distance(geom.boundary))
    except Exception:
        return 0.0


# =========================
# 3) Timeline features (img_*_date0..4)
# =========================

STATUS_RANK_MAP = {
    "Greenland": 0, "Land Cleared": 10, "Demolition": 10,
    "Materials Introduced": 20, "Materials Dumped": 20,
    "Prior Construction": 30, "Excavation": 30,
    "Construction Started": 40,
    "Construction Midway": 60,
    "Construction Done": 90,
    "Operational": 100,
}

def process_timeline(row):
    timeline = []
    for i in range(5):
        dt = row.get(f"date{i}")
        if pd.isna(dt):
            continue

        status = row.get(f"change_status_date{i}")
        status = "Unknown" if pd.isna(status) else str(status)

        timeline.append({
            "date": dt if isinstance(dt, pd.Timestamp) else pd.to_datetime(dt, errors="coerce"),
            "status": status,
            "rank": STATUS_RANK_MAP.get(status, 0),
            "r_mean": row.get(f"img_red_mean_date{i}"),
            "g_mean": row.get(f"img_green_mean_date{i}"),
            "b_mean": row.get(f"img_blue_mean_date{i}"),
            "r_std":  row.get(f"img_red_std_date{i}"),
            "g_std":  row.get(f"img_green_std_date{i}"),
            "b_std":  row.get(f"img_blue_std_date{i}"),
        })

    timeline = [t for t in timeline if pd.notna(t["date"])]
    timeline.sort(key=lambda x: x["date"])

    res = {}
    metrics = [
        "img_red_mean", "img_green_mean", "img_blue_mean",
        "img_red_std", "img_green_std", "img_blue_std",
        "brightness", "texture", "exg", "saturation", "exr",
    ]
    for i in range(1, 6):
        for m in metrics:
            res[f"{m}_t{i}"] = 0.0

    if not timeline:
        res.update({
            "work_duration_days": 0,
            "status_rank_max": 0,
            "status_progression_delta": 0,
            "unique_status_count": 0,
            "avg_days_between_status": 0,
            "delta_brightness": 0,
            "delta_texture": 0,
            "delta_exg": 0,
            "delta_saturation": 0,
            "delta_exr": 0,
        })
        return pd.Series(res)

    ranks = [t["rank"] for t in timeline]
    statuses = [t["status"] for t in timeline]

    construction_dates = [t["date"] for t in timeline if 30 <= t["rank"] < 90]
    res["work_duration_days"] = (max(construction_dates) - min(construction_dates)).days if len(construction_dates) >= 2 else 0
    res["status_rank_max"] = int(max(ranks))
    res["status_progression_delta"] = int(ranks[-1] - ranks[0])
    res["unique_status_count"] = int(len(set(statuses)))

    if len(timeline) > 1:
        total_days = (timeline[-1]["date"] - timeline[0]["date"]).days
        res["avg_days_between_status"] = float(total_days / (len(timeline) - 1))
    else:
        res["avg_days_between_status"] = 0.0

    eps = 1e-6
    for idx, t in enumerate(timeline[:5]):
        suffix = f"_t{idx+1}"
        r = float(t["r_mean"]) if pd.notna(t["r_mean"]) else 0.0
        g = float(t["g_mean"]) if pd.notna(t["g_mean"]) else 0.0
        b = float(t["b_mean"]) if pd.notna(t["b_mean"]) else 0.0
        rs = float(t["r_std"]) if pd.notna(t["r_std"]) else 0.0
        gs = float(t["g_std"]) if pd.notna(t["g_std"]) else 0.0
        bs = float(t["b_std"]) if pd.notna(t["b_std"]) else 0.0

        res[f"img_red_mean{suffix}"] = r
        res[f"img_green_mean{suffix}"] = g
        res[f"img_blue_mean{suffix}"] = b
        res[f"img_red_std{suffix}"] = rs
        res[f"img_green_std{suffix}"] = gs
        res[f"img_blue_std{suffix}"] = bs

        res[f"brightness{suffix}"] = 0.299 * r + 0.587 * g + 0.114 * b
        res[f"texture{suffix}"] = float(np.mean([rs, gs, bs]))
        res[f"exg{suffix}"] = 2 * g - r - b
        res[f"exr{suffix}"] = 1.4 * r - g
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        res[f"saturation{suffix}"] = (max_val - min_val) / (max_val + eps)

    if len(timeline) >= 2:
        last = min(len(timeline), 5)
        res["delta_brightness"] = res[f"brightness_t{last}"] - res["brightness_t1"]
        res["delta_texture"] = res[f"texture_t{last}"] - res["texture_t1"]
        res["delta_exg"] = res[f"exg_t{last}"] - res["exg_t1"]
        res["delta_saturation"] = res[f"saturation_t{last}"] - res["saturation_t1"]
        res["delta_exr"] = res[f"exr_t{last}"] - res["exr_t1"]
    else:
        res["delta_brightness"] = 0.0
        res["delta_texture"] = 0.0
        res["delta_exg"] = 0.0
        res["delta_saturation"] = 0.0
        res["delta_exr"] = 0.0

    return pd.Series(res)

def add_color_transition_velocity_features(df):
    """
    Crée 48 colonnes : 24 de deltas et 24 de vélocité (delta / jours).
    Transitions entre les dates 1, 2, 3, 4 et 5.
    """
    channels = ['red', 'green', 'blue']
    metrics = ['mean', 'std']
    
    # Assurer que les dates sont au format datetime pour le calcul
    for d in range(5):
        df[f'date{d}'] = pd.to_datetime(df[f'date{d}'], errors='coerce')
    
    # Parcourir les 4 transitions (1->2, 2->3, 3->4, 4->5)
    for i in range(0, 4):
        j = i + 1
        
        # Calcul de l'intervalle en jours (Δt)
        # Rappel : img_..._date1 correspond à la colonne date0
        delta_days = (df[f'date{j}'] - df[f'date{i}']).dt.days
        # Remplacer 0 par 1 pour éviter la division par zéro si deux dates sont identiques
        delta_days = delta_days.replace(0, 1)
        
        for metric in metrics:
            for channel in channels:
                col_i = f'img_{channel}_{metric}_date{i}'
                col_j = f'img_{channel}_{metric}_date{j}'
                
                delta_name = f'delta_{metric}_{channel}_{i}_{j}'
                vel_name = f'vel_{metric}_{channel}_{i}_{j}'
                
                if col_i in df.columns and col_j in df.columns:
                    # 1. Calcul du Delta (Variation brute)
                    df[delta_name] = df[col_j] - df[col_i]
                    
                    # 2. Calcul de la Vélocité (Vitesse de changement par jour)
                    df[vel_name] = df[delta_name] / delta_days
                
    return df




# =========================
# 4) One-hot tags CLEAN (ignore N/A/N,A)
# =========================


def _split_tags_special(x):
    """
    - "N,A" reste un tag unique
    - sinon split par virgule
    """
    if not isinstance(x, str):
        return []

    x = x.strip()

    if x == "N,A":
        return ["N,A"]

    return [p.strip() for p in x.split(",") if p.strip() != ""]


def one_hot_tags(df: pd.DataFrame, col: str, tags: list[str], prefix: str) -> pd.DataFrame:
    out = df.copy()

    def has_tag(x, tag):
        return 1 if tag in _split_tags_special(x) else 0

    for tag in tags:
        safe = tag.replace(" ", "_")
        out[f"{prefix}_{safe}"] = out[col].apply(lambda x, t=tag: has_tag(x, t)).astype("int8")

    return out


# =========================
# 4b) Status one-hots per date
# =========================

def add_status_onehots_per_date(df: pd.DataFrame, statuses=ALL_STATUSES, n_dates=5) -> pd.DataFrame:
    out = df.copy()

    def norm_status(x):
        if pd.isna(x):
            return None
        return str(x).strip()

    for i in range(n_dates):
        col = f"change_status_date{i}"
        if col not in out.columns:
            continue

        s = out[col].map(norm_status)

        for st in statuses:
            st_safe = st.replace(" ", "_")
            out[f"status_{st_safe}_date{i}"] = (s == st).astype("int8")

    return out


# =========================
# 4c) Urban / Geo derived features
# =========================

URBAN_DENSITY = {
    "Rural": 0, "N,A": 1,
    "Urban Slum": 2, "Sparse Urban": 3,
    "Dense Urban": 4, "Industrial": 4,
}

def add_urban_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features dérivées de urban_type et geography_type :
    - urban_density_score (ordinal 0-4)
    - n_urban_tags, n_geo_tags
    - has_water, has_dense_forest, has_farms, is_rural
    """
    out = df.copy()

    def _urban_density(x):
        tags = _split_tags_special(x)
        if not tags:
            return 1
        return max(URBAN_DENSITY.get(t, 1) for t in tags)

    out["urban_density_score"] = out["urban_type"].apply(_urban_density).astype("int8")

    out["n_urban_tags"] = out["urban_type"].apply(
        lambda x: len(_split_tags_special(x)) if isinstance(x, str) and x.strip() != "N,A" else 0
    ).astype("int8")

    out["n_geo_tags"] = out["geography_type"].apply(
        lambda x: len(_split_tags_special(x)) if isinstance(x, str) and x.strip() != "N,A" else 0
    ).astype("int8")

    out["has_water"] = out["geography_type"].apply(
        lambda x: 1 if isinstance(x, str) and any(w in x for w in ["River", "Lakes", "Coastal"]) else 0
    ).astype("int8")

    out["has_dense_forest"] = out["geography_type"].apply(
        lambda x: 1 if isinstance(x, str) and "Dense Forest" in x else 0
    ).astype("int8")

    out["has_farms"] = out["geography_type"].apply(
        lambda x: 1 if isinstance(x, str) and "Farms" in x else 0
    ).astype("int8")

    out["is_rural"] = out["urban_type"].apply(
        lambda x: 1 if isinstance(x, str) and "Rural" in x else 0
    ).astype("int8")

    return out


# =========================
# 5) Preprocess one file
# =========================

def preprocess_geojson(path: str, is_train: bool) -> pd.DataFrame:
    print(f"\nLoading: {path}")
    gdf = gpd.read_file(path)

    # CRS
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    # Drop empty
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[~gdf.geometry.is_empty].copy()

    # Repair invalid
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: make_valid(geom) if (geom is not None and hasattr(geom, "is_valid") and not geom.is_valid) else geom
    )

    # Drop NaN/Inf coords (prevents GEOS crashes: convex_hull, to_crs, etc.)
    bad = gdf["geometry"].apply(_geom_has_nonfinite_coords)
    if bad.any():
        print(f"(Warn) Dropping {int(bad.sum())} geometries with NaN/Inf coords")
        gdf = gdf.loc[~bad].copy()

    # Target
    if is_train:
        if "change_type" not in gdf.columns:
            raise ValueError("Missing 'change_type' in train.")
        target_mapping = {
            "Demolition": 0, "Road": 1, "Residential": 2,
            "Commercial": 3, "Industrial": 4, "Mega Projects": 5,
        }
        gdf["target_change_type"] = gdf["change_type"].map(target_mapping).astype("Int64")

    # Spatial (meters)
    print("Spatial features...")
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    gdf["area_sqm"] = gdf_proj.area
    gdf["perimeter_m"] = gdf_proj.length
    gdf["compactness"] = np.where(
        gdf["perimeter_m"] > 0,
        (4 * np.pi * gdf["area_sqm"]) / (gdf["perimeter_m"] ** 2),
        0.0,
    )

    gdf[["length_m", "width_m"]] = gdf_proj.apply(lambda row: pd.Series(get_length_width(row.geometry)), axis=1)
    gdf["aspect_ratio"] = np.where(gdf["width_m"] > 0, gdf["length_m"] / gdf["width_m"], 0.0)

    gdf["centroid_lon"] = gdf.geometry.apply(lambda g: g.centroid.x if (g is not None and g.is_valid) else np.nan)
    gdf["centroid_lat"] = gdf.geometry.apply(lambda g: g.centroid.y if (g is not None and g.is_valid) else np.nan)

    # SAFE convex_area
    gdf["convex_area"] = gdf_proj["geometry"].apply(safe_convex_area)

    bounds = gdf_proj.bounds
    gdf["bbox_height"] = bounds["maxy"] - bounds["miny"]
    gdf["bbox_width"] = bounds["maxx"] - bounds["minx"]

    gdf["num_vertices"] = gdf_proj.geometry.apply(get_num_vertices)
    gdf["max_radius"] = gdf_proj.geometry.apply(get_max_radius)

    # Temporal: rename + reorder + timeline
    print("Temporal reorder + timeline...")
    gdf = process_columns(gdf)                 # img_*_date1..5 -> img_*_date0..4
    gdf = reorder_dates(gdf, errors="coerce")  # clean reorder blocks date0..4

    time_features = gdf.apply(process_time_between_statuses, axis=1)

    # Status flags (5 dates x 10 statuses)
    gdf = add_status_onehots_per_date(gdf)

    gdf = pd.concat([gdf, time_features], axis=1)

    timeline_features = gdf.apply(process_timeline, axis=1)
    gdf = pd.concat([gdf, timeline_features], axis=1)

    # Urban / Geo derived features
    print("Urban/Geo derived features...")
    gdf = add_urban_geo_features(gdf)

    # -- Ajout des COULEURS ET VÉLOCITÉ ---
    print('Ajout des features de couleur et vélocité...')
    gdf = add_color_transition_velocity_features(gdf)

    

    # One-hot tags clean (NO N/A/N,A cols)
    print("One-hot tags clean...")
    urban_tags = ["Dense Urban", "Sparse Urban", "Industrial", "Rural", "Urban Slum"]
    geo_tags = ["Sparse Forest", "Grass Land", "Dense Forest", "Farms", "Barren Land",
                "Lakes", "River", "Coastal", "Desert", "Hills", "Snow"]

    gdf = one_hot_tags(gdf, col="urban_type", tags=urban_tags, prefix="urban")
    gdf = one_hot_tags(gdf, col="geography_type", tags=geo_tags, prefix="geo")

    # Drop non-ML columns
    df = gdf.drop(columns=[c for c in ["index", "geometry"] if c in gdf.columns], errors="ignore")
    return df


# =========================
# Main
# =========================

if __name__ == "__main__":
    train_path = "train.geojson"
    test_path = "test.geojson"

    train_out = "train_features.parquet"
    test_out = "test_features.parquet"

    train_df = preprocess_geojson(train_path, is_train=True)
    train_df.to_parquet(train_out, index=False)
    print(f"Saved: {train_out}  shape={train_df.shape}")

    test_df = preprocess_geojson(test_path, is_train=False)
    test_df.to_parquet(test_out, index=False)
    print(f"Saved: {test_out}  shape={test_df.shape}")