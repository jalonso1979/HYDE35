import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.wild_cluster_bootstrap import (
    wild_cluster_bootstrap,
)


def test_returns_bootstrap_statistic_list():
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y": 0.5 * rng.normal() + rng.normal(0, 0.1), "x": rng.normal()}
             for iso in ("A", "B", "C", "D") for y in range(1900, 1950)]
    df = pd.DataFrame(rows)

    def beta_x(d):
        import statsmodels.api as sm
        X = sm.add_constant(d["x"].astype(float))
        res = sm.OLS(d["y"].astype(float).to_numpy(), X.to_numpy()).fit()
        return float(res.params[1])

    boot = wild_cluster_bootstrap(df, beta_x, cluster_col="iso3", n_boot=100, seed=0)
    assert len(boot) == 100
    assert all(isinstance(v, float) for v in boot)


def test_rademacher_signs_used():
    """With y=0 everywhere, sign-flipping a cluster doesn't change anything."""
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y": 0.0, "x": rng.normal()}
             for iso in ("A", "B") for y in range(1900, 1950)]
    df = pd.DataFrame(rows)

    def beta_x(d):
        import statsmodels.api as sm
        X = sm.add_constant(d["x"].astype(float))
        res = sm.OLS(d["y"].astype(float).to_numpy(), X.to_numpy()).fit()
        return float(res.params[1])

    boot = wild_cluster_bootstrap(df, beta_x, cluster_col="iso3", n_boot=20, seed=0)
    assert max(boot) - min(boot) < 1e-9
