import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_distributed_lag import (
    fit_pooled_distributed_lag,
)


def test_recovers_panel_lag_structure():
    rng = np.random.default_rng(0)
    rows = []
    year_fe = {y: rng.normal(0, 0.1) for y in range(1800, 2000)}
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        alpha = rng.normal(0, 0.5)
        x_prev = 0.0
        for year in range(1800, 2000):
            x = rng.normal(0, 1)
            y = alpha + year_fe[year] + 0.05 * x + 0.03 * x_prev + rng.normal(0, 0.02)
            rows.append({"iso3": iso, "year": year, "y": y, "x": x})
            x_prev = x
    df = pd.DataFrame(rows)
    out = fit_pooled_distributed_lag(df, y="y", x="x", lags=2)
    b0 = out.loc[out["lag"] == 0, "beta"].iloc[0]
    b1 = out.loc[out["lag"] == 1, "beta"].iloc[0]
    assert 0.03 < b0 < 0.07
    assert 0.01 < b1 < 0.05


def test_columns():
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y": rng.normal(), "x": rng.normal()}
             for iso in ("AAA", "BBB") for y in range(1900, 2000)]
    df = pd.DataFrame(rows)
    out = fit_pooled_distributed_lag(df, y="y", x="x", lags=3)
    for c in ("lag", "beta", "se", "ci_low", "ci_high"):
        assert c in out.columns
    assert (out["lag"] == "cumulative").any()
