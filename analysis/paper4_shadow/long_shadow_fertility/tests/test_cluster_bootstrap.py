import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.cluster_bootstrap import (
    block_bootstrap,
)


def test_bootstrap_returns_estimates_per_draw():
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y": rng.normal()}
             for iso in ("A", "B", "C", "D") for y in range(1900, 1950)]
    df = pd.DataFrame(rows)

    def mean_y(d):
        return float(d["y"].mean())

    res = block_bootstrap(df, mean_y, cluster_col="iso3", n_boot=50, seed=0)
    assert len(res) == 50
    assert all(isinstance(v, float) for v in res)


def test_bootstrap_resamples_whole_clusters():
    rows = [{"iso3": iso, "year": y, "y": float(iso == "A")}
             for iso in ("A", "B") for y in range(1900, 1950)]
    df = pd.DataFrame(rows)

    def sum_y(d):
        return float(d["y"].sum())

    res = block_bootstrap(df, sum_y, cluster_col="iso3", n_boot=30, seed=0)
    seen = set(round(v) for v in res)
    assert seen.issubset({0, 50, 100})
