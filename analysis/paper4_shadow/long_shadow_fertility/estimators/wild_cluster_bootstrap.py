"""Cameron-Gelbach-Miller wild cluster bootstrap with Rademacher weights.

For each bootstrap, multiply y in cluster c by w_c in {-1, +1} (drawn per cluster).
Preserves within-cluster dependence while randomizing sign at the cluster level.
Recommended for cluster inference when N_clusters < 30.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd


def wild_cluster_bootstrap(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], float],
    cluster_col: str = "iso3",
    y_col: str = "y",
    n_boot: int = 500,
    seed: int = 0,
) -> list[float]:
    clusters = df[cluster_col].unique()
    n_c = len(clusters)
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_boot):
        weights = rng.choice([-1.0, 1.0], size=n_c)
        weight_by_cluster = dict(zip(clusters, weights))
        boot = df.copy()
        boot[y_col] = boot[y_col] * boot[cluster_col].map(weight_by_cluster)
        try:
            v = fit_fn(boot)
            out.append(float(v))
        except Exception:
            continue
    return out
