"""
Extra features inspired by user's approach: status order, max gap, regression flags, last state one-hot.
"""
import numpy as np
import pandas as pd

# Ordre sémantique des statuts (avant -> après construction)
STATUS_ORDER = {
    "Greenland": 0,
    "Land Cleared": 1,
    "Prior Construction": 2,
    "Excavation": 3,
    "Materials Dumped": 4,
    "Materials Introduced": 5,
    "Construction Started": 6,
    "Construction Midway": 7,
    "Construction Done": 8,
    "Operational": 9,
}
ALL_STATUSES = list(STATUS_ORDER.keys())


def _parse_dates_statuses(df):
    """Return (dates array, statuses array) with NaT for invalid dates."""
    date_cols = [f"date{i}" for i in range(5)]
    status_cols = [f"change_status_date{i}" for i in range(5)]
    dates = df[date_cols].apply(
        lambda s: pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    ).values
    statuses = df[status_cols].fillna("__MISSING__").values
    return dates, statuses


def add_max_gap_features(df):
    """
    Pour chaque ligne: tri par date, puis pour chaque transition (s1 != s2):
    - nb_changes
    - nb_days du plus long écart entre deux changements
    - change_gradiant = différence d'ordre de statut du changement qui a la plus grande progression
    """
    dates, statuses = _parse_dates_statuses(df)
    n = len(df)
    nb_changes = np.zeros(n, dtype=np.float64)
    nb_days_max = np.zeros(n, dtype=np.float64)
    change_gradiant = np.zeros(n, dtype=np.float64)

    for i in range(n):
        row_dates = dates[i]
        row_statuses = statuses[i]
        pairs = [
            (pd.Timestamp(t).value if pd.notna(t) else None, s)
            for t, s in zip(row_dates, row_statuses)
        ]
        pairs = [(t, s) for t, s in pairs if t is not None and s != "__MISSING__"]
        if len(pairs) <= 1:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        changes = []
        for j in range(len(pairs_sorted) - 1):
            t1, s1 = pairs_sorted[j]
            t2, s2 = pairs_sorted[j + 1]
            if s1 != s2:
                days = (t2 - t1) / 1e9 / 86400.0  # nanoseconds -> days
                rank1 = STATUS_ORDER.get(s1, -1)
                rank2 = STATUS_ORDER.get(s2, -1)
                grad = rank2 - rank1 if rank2 >= 0 and rank1 >= 0 else 0
                changes.append((days, grad))
        if not changes:
            continue
        nb_changes[i] = len(changes)
        max_days = max(c[0] for c in changes)
        max_grad = max(c[1] for c in changes)
        nb_days_max[i] = max_days
        change_gradiant[i] = max_grad

    return np.column_stack([nb_changes, nb_days_max, change_gradiant])


def add_last_state_onehot(df):
    """Dernier statut (par date) encodé en one-hot sur ALL_STATUSES."""
    dates, statuses = _parse_dates_statuses(df)
    n = len(df)
    last_status_idx = np.zeros(n, dtype=np.int64)  # index in ALL_STATUSES, -1 = missing

    for i in range(n):
        row_dates = dates[i]
        row_statuses = list(statuses[i])
        pairs = []
        for j, t in enumerate(row_dates):
            if pd.notna(t) and row_statuses[j] != "__MISSING__":
                ts = pd.Timestamp(t).value if hasattr(t, "value") else t
                pairs.append((ts, row_statuses[j]))
        if not pairs:
            last_status_idx[i] = -1
            continue
        _, last_s = max(pairs, key=lambda x: x[0])
        last_status_idx[i] = ALL_STATUSES.index(last_s) if last_s in ALL_STATUSES else -1

    # one-hot: n x len(ALL_STATUSES)
    out = np.zeros((n, len(ALL_STATUSES)), dtype=np.float64)
    for i in range(n):
        if last_status_idx[i] >= 0:
            out[i, last_status_idx[i]] = 1.0
    return out


def add_regression_flags(df):
    """
    Pour chaque ligne: tri par date, puis pour chaque paire consécutive (4 gaps),
    flag = 1 si le rang du statut diminue (régression).
    """
    dates, statuses = _parse_dates_statuses(df)
    n = len(df)
    flags = np.zeros((n, 4), dtype=np.float64)

    for i in range(n):
        row_dates = dates[i]
        row_statuses = list(statuses[i])
        pairs = []
        for j, t in enumerate(row_dates):
            if pd.notna(t) and row_statuses[j] != "__MISSING__":
                ts = pd.Timestamp(t).value if hasattr(t, "value") else t
                pairs.append((ts, row_statuses[j]))
        if len(pairs) < 2:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        for gap_idx in range(min(4, len(pairs_sorted) - 1)):
            _, s1 = pairs_sorted[gap_idx]
            _, s2 = pairs_sorted[gap_idx + 1]
            r1 = STATUS_ORDER.get(s1)
            r2 = STATUS_ORDER.get(s2)
            if r1 is not None and r2 is not None and r2 < r1:
                flags[i, gap_idx] = 1.0
    return flags
