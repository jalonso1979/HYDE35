"""Tests for agricultural pathway clustering."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper1_escape.clustering import (
    prepare_clustering_features,
    cluster_pathways,
    label_clusters,
)


@pytest.fixture
def trajectory_features():
    return pd.DataFrame({
        "country": ["FRA", "GBR", "CHN", "EGY", "BRA", "IND"],
        "peak_extensification_year": [1200, 1100, 800, 500, 1500, 600],
        "intensification_onset_year": [1400, 1300, 1000, 700, np.nan, 900],
        "urbanization_takeoff_year": [1500, 1400, 1200, 1000, np.nan, 1100],
        "max_pop_density": [50, 60, 120, 80, 10, 100],
        "min_land_labor_ratio": [0.005, 0.004, 0.002, 0.003, 0.02, 0.003],
    })


def test_prepare_clustering_features(trajectory_features):
    X, entities = prepare_clustering_features(trajectory_features, entity_col="country")
    assert len(entities) == 5
    assert "BRA" not in entities.values
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.1)


def test_cluster_pathways(trajectory_features):
    X, entities = prepare_clustering_features(trajectory_features, entity_col="country")
    labels, metrics = cluster_pathways(X, n_clusters=2)
    assert len(labels) == len(entities)
    assert set(labels).issubset({0, 1})
    assert "silhouette" in metrics
    assert "inertia" in metrics


def test_label_clusters(trajectory_features):
    X, entities = prepare_clustering_features(trajectory_features, entity_col="country")
    labels, _ = cluster_pathways(X, n_clusters=2)
    result = label_clusters(entities, labels, entity_col="country")
    assert "cluster" in result.columns
    assert len(result) == 5
