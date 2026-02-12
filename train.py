import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


TRAIN_PATH = "train_preprocessed.parquet"
TARGET_COL = "change_type"   # adapte si besoin
RANDOM_STATE = 42


def encode_urban_geography(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # urban one-hot
    if "urban_type" in out.columns:
        urban = pd.get_dummies(out["urban_type"].fillna("NAN"), prefix="urban")
    else:
        urban = pd.DataFrame(index=out.index)

    # geo multi-hot
    if "geography_type" in out.columns:
        geo = (
            out["geography_type"]
            .fillna("NAN")
            .astype(str)
            .str.get_dummies(sep=",")
            .add_prefix("geo_")
        )
    else:
        geo = pd.DataFrame(index=out.index)

    out = out.drop(columns=[c for c in ["urban_type", "geography_type"] if c in out.columns])
    out = pd.concat([out, urban, geo], axis=1)
    return out


def make_Xy(train_df: pd.DataFrame):
    df = encode_urban_geography(train_df)

    y = df[TARGET_COL].astype(str)
    X = df.drop(columns=[TARGET_COL])

    # drop non-num (geometry si jamais)
    X = X.select_dtypes(include=["number", "bool"]).copy()

    # NaN -> median (simple et stable)
    for c in X.columns:
        if X[c].dtype.kind in "fc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(0)

    return X, y


def spatial_bins(df: pd.DataFrame, n_bins=20):
    """
    Create a 'spatial group' label using quantile bins of centroid lon/lat.
    Requires centroid columns if you saved them; if not, compute earlier in preprocess.
    """
    if "centroid_x" not in df.columns or "centroid_y" not in df.columns:
        raise ValueError("Add centroid_x/centroid_y in preprocess (recommended) for spatial split.")

    xbin = pd.qcut(df["centroid_x"], q=n_bins, duplicates="drop")
    ybin = pd.qcut(df["centroid_y"], q=n_bins, duplicates="drop")
    return (xbin.astype(str) + "|" + ybin.astype(str))


def eval_one_split(name, X_train, X_val, y_train, y_val, models):
    print(f"\n=== {name} ===")
    for mname, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        print(f"{mname:20s} macroF1 = {f1:.5f}")


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not found in parquet.")

    X, y = make_Xy(train_df)

    models = {
        "LogReg(balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, n_jobs=-1, class_weight="balanced"))
        ]),
        "LinearSVC(balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced"))
        ]),
        "SGD(log_loss)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
        ])
    }

    # 1) Stratified random split (optimiste)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    eval_one_split("Stratified random 80/20", X_tr, X_va, y_tr, y_va, models)

    # 2) CV stratifié (moyenne + std)
    print("\n=== StratifiedKFold (5 folds) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for mname, model in models.items():
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            scores.append(f1_score(y_va, pred, average="macro"))
        scores = np.array(scores)
        print(f"{mname:20s} mean={scores.mean():.5f}  std={scores.std():.5f}")

    print("\nNext step: add spatial split (needs centroid_x/centroid_y in parquet).")


if __name__ == "__main__":
    main()

