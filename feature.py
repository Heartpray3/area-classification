import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import numpy as np

## Read csvs
# 1. Lisez le fichier sans options supplémentaires
train_df = gpd.read_file('train.geojson')
# 2. Définissez l'index APRES le chargement si nécessaire
# (Vérifiez le nom de votre colonne d'index, souvent c'est 'id')
if 'id' in train_df.columns:
    train_df = train_df.set_index('id')

test_df = gpd.read_file('test.geojson')
if 'id' in test_df.columns:
    test_df = test_df.set_index('id')

train_df.head(1)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualisation de la distribution des classes
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, y='change_type', palette='viridis')
plt.title("Répartition des types de changement (Target)")
plt.show()

# Configuration conforme au papier QFabric (5 dates)
date_cols = ['date1', 'date2', 'date3', 'date4', 'date0']
status_cols = ['change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4', 'change_status_date0']
# Mapping : date0 est la 5ème image dans les colonnes img_...
img_idx_map = {'date1': 1, 'date2': 2, 'date3': 3, 'date4': 4, 'date0': 5}
img_channels = ['red', 'green', 'blue']
img_types = ['mean', 'std']

# 1. Conversion datetime
for col in date_cols:
    train_df[col] = pd.to_datetime(train_df[col], dayfirst=True)

def sort_temporal_data(row):
    # Étape A : Extraire proprement toutes les données originales sans les modifier
    temporal_data = []
    for d_col in date_cols:
        img_idx = img_idx_map[d_col]
        # On stocke les pixels dans un sous-dictionnaire simple
        pixels = {}
        for ch in img_channels:
            for t in img_types:
                pixels[f"{ch}_{t}"] = row[f"img_{ch}_{t}_date{img_idx}"]
        
        temporal_data.append({
            'date': row[d_col],
            'status': row[f'change_status_{d_col}'],
            'pixel_values': pixels
        })

    # Étape B : Trier par date chronologique
    temporal_data.sort(key=lambda x: x['date'])

    # Étape C : Créer une copie de la ligne pour la mise à jour
    new_row = row.copy()
    
    # Étape D : Ré-attribuer de 1 à 5
    for i, data in enumerate(temporal_data, 1):
        new_row[f'date{i}'] = data['date']
        new_row[f'change_status_date{i}'] = data['status']
        for ch in img_channels:
            for t in img_types:
                new_row[f'img_{ch}_{t}_date{i}'] = data['pixel_values'][f"{ch}_{t}"]
                
    return new_row

# 2. Application
train_df = train_df.apply(sort_temporal_data, axis=1)

# 3. Nettoyage final des colonnes orphelines
if 'date0' in train_df.columns:
    train_df = train_df.drop(columns=['date0', 'change_status_date0'])

train_df.head(1)


def add_advanced_temporal_features(df):
    # 1. Calcul des indices composites pour chaque date (1 à 5)
    for i in range(1, 6):
        R = df[f'img_red_mean_date{i}']
        G = df[f'img_green_mean_date{i}']
        B = df[f'img_blue_mean_date{i}']
        # Rouge (0.5) : Priorité à la complexité des matériaux de construction
        # Vert (0.3) : Observation de la perte d'homogénéité végétale
        # Bleu (0.2) : Réduction de l'influence des ombres et du bruit atmosphérique
        avg_std = (
            0.5 * df[f'img_red_std_date{i}'] + 
            0.3 * df[f'img_green_std_date{i}'] + 
            0.2 * df[f'img_blue_std_date{i}']
        )
        
        # Indices de base
        vari = (G - R) / (G + R - B + 1e-6)
        ngrdi = (G - R) / (G + R + 1e-6)
        gli = ((G - R) + (G - B)) / (2*G + R + B + 1e-6)
        bi = ((R**2 + G**2 + B**2) / 3)**0.5
        
        # Features Composites
        df[f'RG_date{i}'] = (ngrdi + gli) * vari # Reliable Greenness
        df[f'UTD_date{i}'] = (bi * avg_std) / (vari + 1.5) # Urban Texture Density

        

    # 2. Analyse Temporelle 
    for i in range(2, 6):
        # Intervalle de temps en jours
        delta_days = (df[f'date{i}'] - df[f'date{i-1}']).dt.days.replace(0, 1)
        
        # Deltas bruts
        df[f'delta_RG_{i}_{i-1}'] = df[f'RG_date{i}'] - df[f'RG_date{i-1}']
        df[f'delta_UTD_{i}_{i-1}'] = df[f'UTD_date{i}'] - df[f'UTD_date{i-1}']
        
        # Vitesses (Taux de changement par jour)
        df[f'vel_RG_{i}_{i-1}'] = df[f'delta_RG_{i}_{i-1}'] / delta_days
        df[f'vel_UTD_{i}_{i-1}'] = df[f'delta_UTD_{i}_{i-1}'] / delta_days
        
        # Accélération Urbaine (Changement de vitesse de construction)
        if i > 2:
            df[f'accel_UTD_{i}'] = df[f'vel_UTD_{i}_{i-1}'] - df[f'vel_UTD_{i-1}_{i-2}']

    # 3. Volatilité Globale sur les 5 dates
    df['volatility_RG'] = df[[f'RG_date{i}' for i in range(1, 6)]].std(axis=1)
    df['volatility_UTD'] = df[[f'UTD_date{i}' for i in range(1, 6)]].std(axis=1)

    return df

# Application du pipeline
train_df = add_advanced_temporal_features(train_df)


import numpy as np
import pandas as pd

def add_advanced_temporal_features_2(df):
    # --- 1. Calcul des indices composites pour chaque date (1 à 5) ---
    for i in range(1, 6):
        R = df[f'img_red_mean_date{i}']
        G = df[f'img_green_mean_date{i}']
        B = df[f'img_blue_mean_date{i}']
        
        # Capture du contraste maximal (Urban Edge) pour identifier les structures complexes
        max_std = df[[f'img_red_std_date{i}', f'img_green_std_date{i}', f'img_blue_std_date{i}']].max(axis=1)
        
        # Moyenne pondérée classique
        avg_std = (0.5 * df[f'img_red_std_date{i}'] + 0.3 * df[f'img_green_std_date{i}'] + 0.2 * df[f'img_blue_std_date{i}'])
        
        # Indices de base
        vari = (G - R) / (G + R - B + 1e-6) #Visible Atmospherically Resistant Index indice de végétation conçu pour être résistant aux effets de l'atmosphère en utilisant le canal bleu
        ngrdi = (G - R) / (G + R + 1e-6) #Normalized Green Red Difference Index
        #gli = ((G - R) + (G - B)) / (2*G + R + B + 1e-6) #Green Leaf Index compare le Vert à la somme du Rouge et du Bleu.
        bi = ((R**2 + G**2 + B**2) / 3)**0.5 # Brightness Index Il mesure l'intensité lumineuse totale (la brillance) du polygone.
        
        # Indice de Brillance Relative (RBI) 
        rbi = (R - B) / (R + B + 1e-6)
        df[f'RBI_date{i}'] = rbi

        # Features Composites enrichies
        df[f'RG_date{i}'] = (ngrdi) * vari 
        # UTD amélioré avec le contraste maximal et le ratio spectral
        df[f'UTD_date{i}'] = (bi * max_std * (1 + rbi)) / (vari + 1.5)

    # --- 2. Analyse Temporelle et Tendances ---
    for i in range(2, 6):
        delta_days = (df[f'date{i}'] - df[f'date{i-1}']).dt.days.replace(0, 1)
        
        # Vitesses
        df[f'vel_RG_{i}_{i-1}'] = (df[f'RG_date{i}'] - df[f'RG_date{i-1}']) / delta_days
        df[f'vel_UTD_{i}_{i-1}'] = (df[f'UTD_date{i}'] - df[f'UTD_date{i-1}']) / delta_days
        
        # Accélération Urbaine
        if i > 2:
            df[f'accel_UTD_{i}'] = df[f'vel_UTD_{i}_{i-1}'] - df[f'vel_UTD_{i-1}_{i-2}']

    # --- 3. Statistiques Globales et Volatilité ---
    df['volatility_RG'] = df[[f'RG_date{i}' for i in range(1, 6)]].std(axis=1)
    df['volatility_UTD'] = df[[f'UTD_date{i}' for i in range(1, 6)]].std(axis=1)

    # --- 4. Normalisation Contextuelle par Groupe (si la colonne existe) ---
    # On suppose que 'urban_type' ou 'city' est une colonne de regroupement
    group_col = 'urban_type' if 'urban_type' in df.columns else None
    
    if group_col:
        # Calcul des déviations par rapport à la moyenne du groupe géographique
        df['rel_volatility_UTD'] = df.groupby(group_col)['volatility_UTD'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        # Capture des anomalies relatives au contexte local
        df['rel_UTD_final'] = df.groupby(group_col)['UTD_date5'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

    return df

# Application du pipeline
train_df = add_advanced_temporal_features_2(train_df)


def add_raw_color_volatility(df):
    channels = ['red', 'green', 'blue']
    
    for ch in channels:
        # 1. Volatilité de la moyenne (Luminosité brute)
        # Permet de voir les sauts de brillance (ex: béton vs herbe)
        mean_cols = [f'img_{ch}_mean_date{i}' for i in range(1, 6)]
        df[f'volatility_{ch}_mean'] = df[mean_cols].std(axis=1)
        
        # 2. Volatilité de la texture (Écart-type brut)
        # Permet de voir si la complexité visuelle change brutalement
        std_cols = [f'img_{ch}_std_date{i}' for i in range(1, 6)]
        df[f'volatility_{ch}_std'] = df[std_cols].std(axis=1)
        
    return df

# Application
train_df = add_raw_color_volatility(train_df)


# Transformer les catégories en colonnes binaires (0 ou 1)
df_encoded = pd.get_dummies(train_df, columns=['change_type'], prefix='type')

# Calculer la corrélation uniquement pour les nouvelles colonnes types
target_cols = [c for c in df_encoded.columns if c.startswith('type_')]

raw_vol_cols = [f'volatility_{ch}_mean' for ch in ['red', 'green', 'blue']] + \
               [f'volatility_{ch}_std' for ch in ['red', 'green', 'blue']]
composite_cols = ['volatility_RG', 'volatility_UTD', 'accel_UTD_5', 'RG_date5']

all_features = composite_cols + raw_vol_cols

# 4. Calcul de la table de corrélation
corr_table = df_encoded[target_cols + all_features].corr().loc[all_features, target_cols]

# 5. Affichage de la Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr_table, annot=True, cmap='coolwarm', fmt=".4f")
plt.title("Corrélation : Indices Composites et Volatilités Brutes par Type")
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sélection de toutes vos features (Composites + Brutes)
# On inclut tout pour donner un maximum de "munitions" à la LDA
features_list = [
    'volatility_RG', 'volatility_UTD', 'accel_UTD_5', 'RG_date5',
    'volatility_red_mean', 'volatility_green_mean', 'volatility_blue_mean',
    'volatility_red_std', 'volatility_green_std', 'volatility_blue_std'
]

X = train_df[features_list].fillna(0)
y = train_df['change_type']

# 2. Standardisation (Crucial : la LDA est sensible aux échelles de valeurs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Application de la LDA
# On réduit à 2 dimensions pour pouvoir faire un graphique 2D
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 4. Visualisation de la séparabilité
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_lda[:, 0], 
    y=X_lda[:, 1], 
    hue=y, 
    palette='Set1', 
    alpha=0.7, 
    edgecolor='w', 
    s=60
)

plt.title("Séparabilité des Classes QFabric (Analyse Discriminante Linéaire - LDA)")
plt.xlabel("LD1 (Premier axe de séparation)")
plt.ylabel("LD2 (Second axe de séparation)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Types de Changement')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mutual_information(df, features, target_col):
    # 1. Préparation des données
    X = df[features].fillna(0)
    y = df[target_col]
    
    # 2. Calcul du score MI
    # discrete_features=False car nos mesures (couleurs, indices) sont continues
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # 3. Création du DataFrame de résultats
    mi_df = pd.DataFrame({'Feature': features, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
    
    # 4. Visualisation
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=mi_df, palette='viridis')
    plt.title("Importance des Features (Mutual Information)")
    plt.xlabel("Score d'Information Mutuelle (plus haut = plus prédictif)")
    plt.show()
    
    return mi_df

# Liste des features à tester
features_to_test = [
    'volatility_RG', 'volatility_UTD', 'accel_UTD_5', 'RG_date5',
    'volatility_red_mean', 'volatility_green_mean', 'volatility_blue_mean',
    'volatility_red_std', 'volatility_green_std', 'volatility_blue_std'
]

mi_results = plot_mutual_information(train_df, features_to_test, 'change_type')

from sklearn.feature_selection import mutual_info_classif
import numpy as np

def calculate_pairwise_mi(df, feature, categories):
    n = len(categories)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # On filtre le dataframe pour ne garder que deux classes
            sub_df = df[df['change_type'].isin([categories[i], categories[j]])]
            X = sub_df[[feature]].fillna(0)
            y = sub_df['change_type']
            
            # Calcul du score MI spécifique à ce duo
            score = mutual_info_classif(X, y, random_state=42)[0]
            matrix[i, j] = score
            matrix[j, i] = score
            
    return pd.DataFrame(matrix, index=categories, columns=categories)

# Test sur votre meilleure feature : volatility_blue_mean
cats = train_df['change_type'].unique()
pairwise_df = calculate_pairwise_mi(train_df, 'volatility_RG', cats)

plt.figure(figsize=(10, 8))
sns.heatmap(pairwise_df, annot=True, cmap='YlGnBu')
plt.title("Séparabilité par paire (MI) pour 'volatility_RG'")
plt.show()


def plot_pair_dist(df, feature, cat1, cat2):
    plt.figure(figsize=(10, 5))
    subset = df[df['change_type'].isin([cat1, cat2])]
    sns.kdeplot(data=subset, x=feature, hue='change_type', fill=True, common_norm=False)
    plt.title(f"Distinction entre {cat1} et {cat2} via {feature}")
    plt.show()

# Exemple : Vérifier si UTD sépare les routes des zones résidentielles
plot_pair_dist(train_df, 'volatility_green_mean', 'Road', 'Residential')


from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mutual_information(df, features, target_col):
    # 1. Préparation des données (on s'assure que les colonnes existent)
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)
    y = df[target_col]
    
    # 2. Calcul du score MI
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # 3. Création du DataFrame de résultats
    mi_df = pd.DataFrame({'Feature': available_features, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
    
    # 4. Visualisation
    plt.figure(figsize=(12, 10))
    # Palette différenciée pour repérer les nouvelles features
    sns.barplot(x='MI_Score', y='Feature', data=mi_df, palette='magma')
    plt.title("Impact des Nouvelles Features Contextuelles (Mutual Information)")
    plt.xlabel("Score d'Information Mutuelle (Pouvoir Prédictif)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return mi_df

# --- Liste mise à jour avec les nouveaux indices ---
features_to_test = [
    # Les classiques
    'volatility_RG', 'volatility_UTD', 'accel_UTD_5', 
    
    # Les nouveaux indices de séparation spectrale
    'RBI_date5', 'max_std_date5', 
    
    # Les features de normalisation contextuelle (si générées)
    'rel_volatility_UTD', 'rel_UTD_final',
    
    # Rappel des brutes pour comparaison
    'vol_raw_red_mean', 'vol_raw_green_mean', 'vol_raw_blue_mean'
]

mi_results = plot_mutual_information(train_df, features_to_test, 'change_type')

from sklearn.feature_selection import mutual_info_classif
import numpy as np

def calculate_pairwise_mi(df, feature, categories):
    n = len(categories)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # On filtre le dataframe pour ne garder que deux classes
            sub_df = df[df['change_type'].isin([categories[i], categories[j]])]
            X = sub_df[[feature]].fillna(0)
            y = sub_df['change_type']
            
            # Calcul du score MI spécifique à ce duo
            score = mutual_info_classif(X, y, random_state=42)[0]
            matrix[i, j] = score
            matrix[j, i] = score
            
    return pd.DataFrame(matrix, index=categories, columns=categories)

# Test sur votre meilleure feature : volatility_blue_mean
cats = train_df['change_type'].unique()
pairwise_df = calculate_pairwise_mi(train_df, 'RBI_date5', cats)

plt.figure(figsize=(10, 8))
sns.heatmap(pairwise_df, annot=True, cmap='YlGnBu')
plt.title("Séparabilité par paire (MI) pour 'RBI_date5'")
plt.show()


from shapely import wkt
from shapely.ops import transform
import pyproj
import numpy as np

def compute_geometry_with_centroid_reprojection(wkt_string):
    """
    Méthode conforme au plan : Reprojection UTM basée sur le centroïde.
    """
    # 1. Charger le polygone depuis le WKT
    try:
        poly_wgs84 = wkt.loads(wkt_string)
    except:
        return None

    # 2. Déterminer la zone UTM à partir du centroïde
    centroid = poly_wgs84.centroid
    lon, lat = centroid.x, centroid.y
    
    utm_zone = int((lon + 180) / 6) + 1
    # Déterminer si Nord (326xx) ou Sud (327xx)
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    
    # 3. Créer le transformateur (WGS84 -> UTM local)
    project = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True).transform
    
    # 4. Reprojeter le polygone
    poly_utm = transform(project, poly_wgs84)
    
    # 5. Calculer les caractéristiques en MÈTRES
    area = poly_utm.area
    perimeter = poly_utm.length
    
    # Calcul du Bounding Box area en UTM
    minx, miny, maxx, maxy = poly_utm.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    
    # Ratios de forme
    shape_ratio = area / bbox_area if bbox_area > 0 else 0
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    return {
        "area_m2": area,
        "perimeter_m": perimeter,
        "shape_ratio": shape_ratio,
        "compactness": compactness,
        "utm_zone": utm_zone
    }

# Test avec ton exemple
wkt_ex = "POLYGON ((112.16774 32.02198, 112.16845 32.02055, 112.16895 32.02077, 112.16831 32.02204, 112.16774 32.02198))"
features = compute_geometry_with_centroid_reprojection(wkt_ex)
print(features)

# Application de ta fonction validée sur tout le dataset
geom_features = train_df['geometry'].apply(compute_geometry_with_centroid_reprojection)

# On transforme la liste de dictionnaires en colonnes propres
geom_df = pd.DataFrame(geom_features.tolist(), index=train_df.index)

# Fusion avec tes indices spectraux (UTD, RG, RBI)
train_df = pd.concat([train_df, geom_df], axis=1)

# Mapping from fine-level labels to coarse categories.
FINE_TO_COARSE = {
    'Grass Land': 'Vegetation',
    'Lakes': 'Water',
    'Barren Land': 'Barren',
    'Desert': 'Barren',
    'Dense Forest': 'Vegetation',
    'Hills': 'Terrain',
    'Snow': 'Terrain',
    'Sparse Forest': 'Vegetation',
    'Coastal': 'Water',
    'River': 'Water'
}

# Fine-level categories (note: "Farms" has been removed).
FINE_CATEGORIES = [
    'Grass Land', 'Lakes', 'Barren Land', 'Desert', 'Dense Forest',
    'Hills', 'Snow', 'Sparse Forest', 'Coastal', 'River'
]

# Coarse-level categories.
# Added "Unknown" to handle cases when "N,A" is found.
COARSE_CATEGORIES = [
    'Water', 'Vegetation', 'Barren', 'Terrain', 'Agriculture', 'Unknown'
]

import numpy as np
import pandas as pd

def parse_geography_type(geo_str):
    """ Découpe la chaîne par virgules et retourne un ensemble de types distincts. """
    if not isinstance(geo_str, str):
        return set()
    return {x.strip() for x in geo_str.split(",")}

def compute_geography_indicators(geo_set):
    """ Calcule des indicateurs thématiques basés sur les mots-clés. """
    water_terms = {'River', 'Lakes', 'Coastal'}
    vegetation_terms = {'Sparse Forest', 'Dense Forest', 'Grass Land'}

    has_water = 1 if len(geo_set.intersection(water_terms)) > 0 else 0
    has_vegetation = 1 if len(geo_set.intersection(vegetation_terms)) > 0 else 0
    terrain_complexity = len(geo_set)

    return has_water, has_vegetation, terrain_complexity

def encode_geography_type(geo_str: str):
    """ Encode la chaîne en vecteurs multi-hot (Fine et Coarse). """
    fine_vector = np.zeros(len(FINE_CATEGORIES), dtype=np.float32)
    coarse_vector = np.zeros(len(COARSE_CATEGORIES), dtype=np.float32)

    if not isinstance(geo_str, str):
        return fine_vector, coarse_vector

    # Nettoyage et normalisation des "N,A"
    geo_str = geo_str.replace("N,A", "Unknown")
    tokens = [token.strip() for token in geo_str.split(",") if token.strip()]

    for token in tokens:
        if token == "Unknown":
            coarse_vector[COARSE_CATEGORIES.index("Unknown")] = 1
            continue

        if token in FINE_CATEGORIES:
            fine_vector[FINE_CATEGORIES.index(token)] = 1

        coarse_label = FINE_TO_COARSE.get(token, None)
        if coarse_label and coarse_label in COARSE_CATEGORIES:
            coarse_vector[COARSE_CATEGORIES.index(coarse_label)] = 1

    return fine_vector, coarse_vector

def parse_urban_tags(val):
    """ Analyse les étiquettes urbaines en gérant les valeurs manquantes. """
    if pd.isnull(val) or val == "N,A":
        return {"Unknown"}
    return {x.strip() for x in val.split(",")}

def encode_urban_types(df, col="urban_type"):
    """ Crée 7 colonnes binaires à partir de la colonne urban_type. """
    parsed = df[col].apply(parse_urban_tags)

    # Niveaux Coarse (Généraux)
    df["has_urban"] = parsed.apply(lambda tags: 1 if any(t in ["Sparse Urban", "Dense Urban", "Urban Slum"] for t in tags) else 0)
    df["has_rural"] = parsed.apply(lambda tags: 1 if "Rural" in tags else 0)
    df["has_industrial"] = parsed.apply(lambda tags: 1 if "Industrial" in tags else 0)
    df["has_unknown"] = parsed.apply(lambda tags: 1 if "Unknown" in tags else 0)

    # Niveaux Fine (Détails de densité)
    df["is_sparse"] = parsed.apply(lambda tags: 1 if "Sparse Urban" in tags else 0)
    df["is_dense"] = parsed.apply(lambda tags: 1 if "Dense Urban" in tags else 0)
    df["is_slum"] = parsed.apply(lambda tags: 1 if "Urban Slum" in tags else 0)

    return df

# 1. Traitement Urbain
train_df = encode_urban_types(train_df, col="urban_type")

# 2. Traitement Géographique (Vecteurs et Indicateurs)
def apply_geo_features(row):
    geo_str = row['geography_type']
    geo_set = parse_geography_type(geo_str)
    
    # Indicateurs simples
    h_water, h_veg, complexity = compute_geography_indicators(geo_set)
    
    # Vecteurs multi-hot
    fine_v, coarse_v = encode_geography_type(geo_str)
    
    # Création d'un dictionnaire de sortie pour fusion rapide
    res = {
        "has_water": h_water, 
        "has_vegetation": h_veg, 
        "terrain_complexity": complexity
    }
    # On ajoute les colonnes des vecteurs
    for i, cat in enumerate(FINE_CATEGORIES): res[f"geo_fine_{cat}"] = fine_v[i]
    for i, cat in enumerate(COARSE_CATEGORIES): res[f"geo_coarse_{cat}"] = coarse_v[i]
    
    return pd.Series(res)

# Application et fusion
geo_features = train_df.apply(apply_geo_features, axis=1)
train_df = pd.concat([train_df, geo_features], axis=1)