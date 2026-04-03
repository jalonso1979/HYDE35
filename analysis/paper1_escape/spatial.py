"""Spatial diffusion analysis for Paper 1.

Tests whether agricultural transitions spread geographically using
spatial autocorrelation (Moran's I) and spatial lag models.
"""
import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran


def build_spatial_weights(coords: np.ndarray, k: int = 5) -> KNN:
    """Build K-nearest-neighbor spatial weights from centroid coordinates."""
    return KNN.from_array(coords, k=k)


def compute_morans_i(values: np.ndarray, weights: KNN) -> dict:
    """Compute Moran's I for spatial autocorrelation."""
    mi = Moran(values, weights)
    return {"I": mi.I, "p_value": mi.p_sim, "z_score": mi.z_sim}


def spatial_lag(values: np.ndarray, weights: KNN) -> np.ndarray:
    """Compute spatial lag of a variable (weighted average of neighbors)."""
    from libpysal.weights import lag_spatial
    return lag_spatial(weights, values)
