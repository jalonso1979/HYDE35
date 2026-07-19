import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.distributed_lag import (
    fit_distributed_lag,
)


def test_recovers_known_lag_structure():
    """DGP: y_t = 0.04 * x_t - 0.06 * x_{t-1} + noise (peaks at lag 1)."""
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        alpha = rng.normal(0, 0.5)
        x_prev = 0.0
        for year in range(1800, 2000):
            x = rng.normal(0, 1)
            y = alpha + 0.04 * x - 0.06 * x_prev + rng.normal(0, 0.02)
            rows.append({"iso3": iso, "year": year, "y": y, "x": x})
            x_prev = x
    df = pd.DataFrame(rows)
    out = fit_distributed_lag(df, y="y", x="x", lags=2)
    b0 = out.loc[out["lag"] == 0, "beta"].iloc[0]
    b1 = out.loc[out["lag"] == 1, "beta"].iloc[0]
    assert 0.02 < b0 < 0.06
    assert -0.08 < b1 < -0.04


def test_columns_and_cumulative():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB"):
        for year in range(1900, 2000):
            rows.append({"iso3": iso, "year": year,
                          "y": rng.normal(), "x": rng.normal()})
    df = pd.DataFrame(rows)
    out = fit_distributed_lag(df, y="y", x="x", lags=3)
    for c in ("lag", "beta", "se", "ci_low", "ci_high"):
        assert c in out.columns
    assert (out["lag"] == "cumulative").any()
