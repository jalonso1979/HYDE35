"""Agricultural pathway clustering for Paper 1."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLUSTERING_FEATURES = [
    "peak_extensification_year", "intensification_onset_year",
    "urbanization_takeoff_year", "max_pop_density", "min_land_labor_ratio",
]

def prepare_clustering_features(
    trajectory_df: pd.DataFrame, entity_col: str = "country",
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.Series]:
    if feature_cols is None:
        feature_cols = CLUSTERING_FEATURES
    df = trajectory_df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].values
    entities = df[entity_col].reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, entities

def cluster_pathways(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> tuple[np.ndarray, dict]:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if n_clusters > 1 else 0.0
    return labels, {"silhouette": sil, "inertia": km.inertia_}

def label_clusters(entities: pd.Series, labels: np.ndarray, entity_col: str = "country") -> pd.DataFrame:
    return pd.DataFrame({entity_col: entities.values, "cluster": labels})
