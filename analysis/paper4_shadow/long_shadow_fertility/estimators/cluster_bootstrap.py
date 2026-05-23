"""Block bootstrap by cluster (e.g., country). Resample N clusters with
replacement, then take ALL rows from each sampled cluster. Preserves
within-cluster dependence.

Caveat: with small N (<30 clusters) cluster bootstrap is approximate;
wild cluster bootstrap is the refinement.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd


def block_bootstrap(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], float],
    cluster_col: str = "iso3",
    n_boot: int = 500,
    seed: int = 0,
) -> list[float]:
    clusters = df[cluster_col].unique()
    n_c = len(clusters)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=n_c, replace=True)
        parts = [df.loc[df[cluster_col] == c] for c in sampled]
        boot = pd.concat(parts, ignore_index=True)
        try:
            v = fit_fn(boot)
            out.append(float(v))
        except Exception:
            continue
    return out
