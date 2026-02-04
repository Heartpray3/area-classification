import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)
# Pour geography_type
print("=== Vérification pour geography_type ===")
for df, name in [(train_df, 'train'), (test_df, 'test')]:
    print(f"\nDans {name}_df:")
    t = df['geography_type'].to_numpy(dtype=str)

    # Compter les occurrences de 'N' et 'A' seuls et ensemble
    n_only = 0
    a_only = 0
    n_and_a = 0
    neither = 0

    for item in t:
        types = set(item.split(','))
        if 'N' in types and 'A' in types:
            n_and_a += 1
        elif 'N' in types:
            n_only += 1
        elif 'A' in types:
            a_only += 1
        else:
            neither += 1

    print(f"  'N' et 'A' ensemble: {n_and_a}")
    print(f"  'N' seul: {n_only}")
    print(f"  'A' seul: {a_only}")
    print(f"  Sans 'N' ni 'A': {neither}")

    # Vérifier spécifiquement si 'N' est toujours avec 'A'
    if n_only == 0:
        print(f"  ✓ 'N' est TOUJOURS accompagné de 'A' dans {name}_df")
    else:
        print(f"  ✗ 'N' apparaît seul {n_only} fois dans {name}_df")

# Pour urban_type
print("\n=== Vérification pour urban_type ===")
for df, name in [(train_df, 'train'), (test_df, 'test')]:
    print(f"\nDans {name}_df:")
    t = df['urban_type'].to_numpy(dtype=str)

    # Compter les occurrences de 'N' et 'A' seuls et ensemble
    n_only = 0
    a_only = 0
    n_and_a = 0
    neither = 0

    for item in t:
        types = set(item.split(','))
        if 'N' in types and 'A' in types:
            n_and_a += 1
        elif 'N' in types:
            n_only += 1
        elif 'A' in types:
            a_only += 1
        else:
            neither += 1

    print(f"  'N' et 'A' ensemble: {n_and_a}")
    print(f"  'N' seul: {n_only}")
    print(f"  'A' seul: {a_only}")
    print(f"  Sans 'N' ni 'A': {neither}")

    # Vérifier spécifiquement si 'N' est toujours avec 'A'
    if n_only == 0:
        print(f"  ✓ 'N' est TOUJOURS accompagné de 'A' dans {name}_df")
    else:
        print(f"  ✗ 'N' apparaît seul {n_only} fois dans {name}_df")