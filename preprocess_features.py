import geopandas as gpd
import numpy as np
import pandas as pd
import re

# --- Read geojsons
# train_df = gpd.read_file('train.geojson')
test_df  = gpd.read_file('subset_tests.geojson')

def process_columns(df):
    df = test_df.copy()

    rename_map = {}
    for c in df.columns:
        m = re.match(r"^(img_.*_date)(\d+)$", c)
        if m:
            idx = int(m.group(2))
            rename_map[c] = f"{m.group(1)}{idx-1}"  # 1->0, 2->1, ..., 5->4

    df = df.rename(columns=rename_map)

    return df

def geometric_attributes(df):
    out = df.copy()

    out["area"] = out.geometry.area
    out["perimeter"] = out.geometry.length
    out["compactness"] = 4 * np.pi * out["area"] / (out["perimeter"] ** 2)
    return out

def reorder_dates(df, date_format="%d-%m-%Y", errors="raise"):
    out = df.copy()
    date_cols = [f"date{i}" for i in range(5)]

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

    for base, cols in bases.items():
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

df = process_columns(test_df)

reordered = reorder_dates(df)

row = 0

img_cols = [f"img_red_mean_date{i}" for i in range(5) if f"img_red_mean_date{i}" in df.columns]
date_cols = [f"date{i}" for i in range(5)]

print("DATES BEFORE:", df.loc[row, date_cols].tolist())
print("DATES AFTER :", reordered.loc[row, date_cols].tolist())

print("IMG BEFORE:", df.loc[row, img_cols].tolist())
print("IMG AFTER :", reordered.loc[row, img_cols].tolist())


# --- Save to geojson
reordered.to_file(
    "subset_tests_reordered.geojson",
    driver="GeoJSON"
)
