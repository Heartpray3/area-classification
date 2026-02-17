import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime
from shapely.validation import make_valid
import shapely

# 1. Chargement des données
file_path = 'train.geojson'
print("Chargement du GeoJSON...")
gdf = gpd.read_file(file_path)

# 2. Encodage de la variable cible
target_mapping = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}
gdf['target_change_type'] = gdf['change_type'].map(target_mapping)

# 3. Features Spatiales (Nécessite une reprojection en mètres)
print("Nettoyage et calcul des features spatiales...")

# a. Définir explicitement le CRS initial (WGS84 / Coordonnées GPS)
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)

# b. Retirer les géométries totalement vides ou nulles
gdf = gdf[~gdf.geometry.isna()]
gdf = gdf[~gdf.geometry.is_empty]

# c. Réparer les géométries invalides (ex: polygones qui s'auto-intersectent)
gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

# d. Reprojection en UTM (pour avoir des mètres au lieu de degrés)
gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

# Calculs de base
gdf['area_sqm'] = gdf_proj.area
gdf['perimeter_m'] = gdf_proj.length

# Compacité (Polsby-Popper) avec sécurité contre la division par zéro
gdf['compactness'] = np.where(
    gdf['perimeter_m'] > 0,
    (4 * np.pi * gdf['area_sqm']) / (gdf['perimeter_m'] ** 2),
    0
)

# e. Longueur et Largeur robustes
def get_length_width(geom):
    if geom is None or geom.is_empty or not geom.is_valid:
        return 0.0, 0.0
    
    try:
        # Trouver le rectangle englobant minimum
        mrr = geom.minimum_rotated_rectangle
        
        # Si la forme est une ligne ou un point (dégénéré), ce n'est pas un polygone
        if mrr.geom_type != 'Polygon':
            return 0.0, 0.0
            
        coords = list(mrr.exterior.coords)
        # Calcul de la longueur des deux côtés adjacents
        edge1 = np.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
        edge2 = np.sqrt((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)
        
        return max(edge1, edge2), min(edge1, edge2)
    
    except Exception as e:
        # Si GEOS plante sur des coordonnées NaN/Inf, on renvoie 0, 0 sans faire crasher le script
        return 0.0, 0.0

# Appliquer la fonction
gdf[['length_m', 'width_m']] = gdf_proj.apply(lambda row: pd.Series(get_length_width(row.geometry)), axis=1)

# 1. L'Allongement (Aspect Ratio) : 
# Sépare instantanément les 'Road' (ratio immense) des bâtiments (ratio proche de 1)
gdf['aspect_ratio'] = np.where(gdf['width_m'] > 0, gdf['length_m'] / gdf['width_m'], 0)

# 2. Rectangularité : 
# Aire du polygone divisée par l'aire de son rectangle englobant (length * width)
# Un 'Commercial' ou 'Industrial' sera très proche de 1.0 (rectangle parfait)
#gdf['rectangularity'] = np.where((gdf['length_m'] * gdf['width_m']) > 0,
                                 #gdf['area_sqm'] / (gdf['length_m'] * gdf['width_m']), 0)

                                 
# Position X, Y (Centroïde en degrés WGS84)
# On utilise un try/except rapide via une fonction lambda sécurisée pour les centroïdes problématiques
gdf['centroid_lon'] = gdf.geometry.apply(lambda g: g.centroid.x if g and g.is_valid else np.nan)
gdf['centroid_lat'] = gdf.geometry.apply(lambda g: g.centroid.y if g and g.is_valid else np.nan)

# Utile pour détecter les formes irrégulières vs formes simples
gdf['convex_area'] = gdf_proj.convex_hull.area

#  Bounding Box Height (Hauteur de la boîte englobante alignée sur les axes)
# bounds renvoie : minx, miny, maxx, maxy
bounds = gdf_proj.bounds
gdf['bbox_height'] = bounds['maxy'] - bounds['miny']
# J'ajoute bbox_width par sécurité car souvent utile avec la hauteur
gdf['bbox_width'] = bounds['maxx'] - bounds['minx']

#  Num Vertices (Nombre de sommets)
# Indique la complexité de la forme 
def get_num_vertices(geom):
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == 'Polygon':
        return len(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        # Somme des sommets de toutes les parties
        return sum(len(part.exterior.coords) for part in geom.geoms)
    return 0

gdf['num_vertices'] = gdf_proj.geometry.apply(get_num_vertices)

# Max Radius (Distance maximale du centroïde à un sommet)
# On utilise la distance de Hausdorff entre le centroïde et le contour extérieur.
# La distance de Hausdorff entre un point et une ligne est la distance max entre ce point et la ligne.
def get_max_radius(geom):
    if geom is None or geom.is_empty:
        return 0
    try:
        # Distance max entre le centroïde et l'extérieur (boundary)
        return geom.centroid.hausdorff_distance(geom.boundary)
    except:
        return 0

gdf['max_radius'] = gdf_proj.geometry.apply(get_max_radius)

# 4. Features Temporelles et Chromatiques
print("Tri temporel et ingénierie des couleurs...")

# --- DÉFINITION DE LA HIÉRARCHIE DES STATUTS ---
# On transforme les mots en échelle de 0 à 100 pour mesurer l'avancement
STATUS_RANK_MAP = {
    'Greenland': 0, 'Land Cleared': 10, 'Demolition': 10,
    'Materials Introduced': 20, 'Materials Dumped': 20,
    'Prior Construction': 30, 'Excavation': 30,
    'Construction Started': 40,
    'Construction Midway': 60,
    'Construction Done': 90,
    'Operational': 100
}

def process_timeline(row):
    timeline = []
    # Collecter les données pour les 5 dates
    for i in range(5):
        date_str = row.get(f'date{i}')
        if pd.isna(date_str): continue
        try:
            dt = datetime.strptime(date_str, '%d-%m-%Y')
            status = str(row.get(f'change_status_date{i}'))
            if status == 'nan': status = "Unknown"
            
            timeline.append({
                'date': dt,
                'status': status,
                'rank': STATUS_RANK_MAP.get(status, 0),
                # Couleurs brutes
                'r_mean': row.get(f'img_red_mean_date{i+1}'),
                'g_mean': row.get(f'img_green_mean_date{i+1}'),
                'b_mean': row.get(f'img_blue_mean_date{i+1}'),
                'r_std': row.get(f'img_red_std_date{i+1}'), 
                'g_std': row.get(f'img_green_std_date{i+1}'),
                'b_std': row.get(f'img_blue_std_date{i+1}')
            })
        except ValueError: pass

    # 1. TRI CHRONOLOGIQUE
    timeline.sort(key=lambda x: x['date'])
    
    res = {}

    # 2. INITIALISATION COMPLETE (Pour éviter les NaN si < 5 dates)
    # On définit tout ce qu'on va calculer par date
    metrics = [
        'img_red_mean', 'img_green_mean', 'img_blue_mean', 
        'img_red_std', 'img_green_std', 'img_blue_std',
        'brightness', 'texture', 'exg', 'saturation', 'exr' # <--- Ajout des nouveaux indices
    ]
    
    # On met tout à 0.0 par défaut pour t1 à t5
    for i in range(1, 6):
        for m in metrics:
            res[f'{m}_t{i}'] = 0.0

    # Si vide, on renvoie le dictionnaire rempli de zéros + les scalaires à 0
    if not timeline:
        res.update({
            'work_duration_days': 0, 'status_rank_max': 0, 'status_progression_delta': 0,
            'unique_status_count': 0, 'avg_days_between_status': 0,
            'delta_exg': 0, 'delta_saturation': 0
        })
        return pd.Series(res)

    # --- 3. CALCULS TEMPORELS ---
    ranks = [t['rank'] for t in timeline]
    statuses = [t['status'] for t in timeline]
    
    construction_dates = [t['date'] for t in timeline if t['rank'] >= 30 and t['rank'] < 90]
    res['work_duration_days'] = (max(construction_dates) - min(construction_dates)).days if len(construction_dates) >= 2 else 0
    
    res['status_rank_max'] = max(ranks)
    res['status_progression_delta'] = ranks[-1] - ranks[0]
    res['unique_status_count'] = len(set(statuses))
    
    # Vélocité (Moyenne de jours entre chaque étape)
    if len(timeline) > 1:
        total_days = (timeline[-1]['date'] - timeline[0]['date']).days
        res['avg_days_between_status'] = total_days / (len(timeline) - 1)
    else:
        res['avg_days_between_status'] = 0

    # Flags / Stagnation
    res['has_demolition'] = 1 if any('Demolition' in s for s in statuses) else 0
    res['has_excavation'] = 1 if any('Excavation' in s for s in statuses) else 0
    res['is_operational_final'] = 1 if ranks[-1] == 100 else 0
    res['started_as_green'] = 1 if len(statuses) > 0 and 'Greenland' in statuses[0] else 0
    res['ended_as_built'] = 1 if len(statuses) > 0 and statuses[-1] in ['Operational', 'Construction Done'] else 0
    cleared_terms = ['Land Cleared', 'Demolition', 'Materials Dumped']
    res['ended_as_cleared'] = 1 if len(statuses) > 0 and statuses[-1] in cleared_terms else 0
    
    max_stable = 1
    current_stable = 1
    for i in range(1, len(statuses)):
        if statuses[i] == statuses[i-1]:
            current_stable += 1
        else:
            max_stable = max(max_stable, current_stable)
            current_stable = 1
    res['max_consecutive_stable_status'] = max(max_stable, current_stable)

    # --- 4. CALCULS IMAGES & INDICES ---
    epsilon = 1e-6
    for idx, t in enumerate(timeline):
        if idx >= 5: break
        suffix = f"_t{idx+1}"
        r, g, b = t['r_mean'], t['g_mean'], t['b_mean']
        
        # Remplissage des valeurs brutes
        res[f'img_red_mean{suffix}'] = r
        res[f'img_green_mean{suffix}'] = g
        res[f'img_blue_mean{suffix}'] = b
        res[f'img_red_std{suffix}'] = t['r_std']
        res[f'img_green_std{suffix}'] = t['g_std']
        res[f'img_blue_std{suffix}'] = t['b_std']
        
        # Indices calculés
        res[f'brightness{suffix}'] = 0.299*r + 0.587*g + 0.114*b
        res[f'texture{suffix}'] = np.mean([t['r_std'], t['g_std'], t['b_std']])
        
        # Spectraux
        res[f'exg{suffix}'] = 2*g - r - b
        res[f'exr{suffix}'] = 1.4*r - g
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        res[f'saturation{suffix}'] = (max_val - min_val) / (max_val + epsilon)

    # --- 5. DELTAS (Dynamique) ---
    # C'est ce qui manquait dans ton code !
    if len(timeline) >= 2:
        last = len(timeline)
        # On compare la dernière date dispo (ex: t3) avec la première (t1)
        res['delta_brightness'] = res[f'brightness_t{last}'] - res['brightness_t1']
        res['delta_texture'] = res[f'texture_t{last}'] - res['texture_t1']
        
        res['delta_exg'] = res[f'exg_t{last}'] - res[f'exg_t1']
        res['delta_saturation'] = res[f'saturation_t{last}'] - res[f'saturation_t1']
        res['delta_exr'] = res[f'exr_t{last}'] - res[f'exr_t1']
    else:
        res['delta_brightness'] = 0
        res['delta_texture'] = 0
        res['delta_exg'] = 0
        res['delta_saturation'] = 0
        res['delta_exr'] = 0

    return pd.Series(res)



print("Génération des features temporelles avancées...")
timeline_features = gdf.apply(process_timeline, axis=1)
gdf = pd.concat([gdf, timeline_features], axis=1)

# 5. One-Hot Encoding pour Urban et Geography Types
print("One-Hot Encoding des tags...")

urban_tags = ['Dense Urban', 'Sparse Urban', 'Industrial', 'N', 'A', 'Rural', 'Urban Slum']
geo_tags = ['Sparse Forest', 'Grass Land', 'Dense Forest', 'Farms', 'Barren Land', 
            'Lakes', 'River', 'N', 'A', 'Coastal', 'Desert', 'Hills', 'Snow']

for tag in urban_tags:
    # On vérifie si le tag (entouré de limites de mots ou virgules) est dans la chaîne
    gdf[f'urban_{tag.replace(" ", "_")}'] = gdf['urban_type'].apply(
        lambda x: 1 if isinstance(x, str) and tag in x.split(',') else 0
    )

for tag in geo_tags:
    gdf[f'geo_{tag.replace(" ", "_")}'] = gdf['geography_type'].apply(
        lambda x: 1 if isinstance(x, str) and tag in x.split(',') else 0
    )

# 6. Nettoyage et Sauvegarde
print("Nettoyage et sauvegarde vers 'train_features.csv'...")

# Colonnes inutiles à supprimer pour le ML (on garde uniquement nos nouvelles features)
# cols_to_drop = [ 'change_type', 'geometry', 'index'] 
# cols_to_drop += [f'date{i}' for i in range(5)]
# cols_to_drop += [f'change_status_date{i}' for i in range(5)]
# cols_to_drop += [f'img_red_mean_date{i}' for i in range(1,6)]
# cols_to_drop += [f'img_green_mean_date{i}' for i in range(1,6)]
# cols_to_drop += [f'img_blue_mean_date{i}' for i in range(1,6)]
# cols_to_drop += [f'img_red_std_date{i}' for i in range(1,6)]
# cols_to_drop += [f'img_green_std_date{i}' for i in range(1,6)]
# cols_to_drop += [f'img_blue_std_date{i}' for i in range(1,6)]

gdf_ml = gdf.copy()

# On supprime UNIQUEMENT la colonne 'index' (ainsi que la 'geometry' qui n'est pas lisible par les algos ML)
cols_to_drop = ['index', 'geometry']
gdf_ml = gdf_ml.drop(columns=[col for col in cols_to_drop if col in gdf_ml.columns], errors='ignore')

# Sauvegarde en CSV (sans la géométrie qui n'est pas supportée par la plupart des algos ML)
gdf_ml.to_csv('train_features.csv', index=False)
print("Terminé ! Le fichier train_features.csv est prêt pour l'entraînement.")