import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_rolling_window import (
    pooled_rolling_elasticity,
)


def test_recovers_panel_dgp():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        alpha = rng.normal(0, 0.5)
        for year in range(1800, 2000):
            x = rng.normal(0, 1)
            y = 0.05 * x + alpha + rng.normal(0, 0.02)
            rows.append({"iso3": iso, "year": year, "y": y, "x": x})
    df = pd.DataFrame(rows)
    out = pooled_rolling_elasticity(df, y="y", x="x", window=40)
    assert (out["beta"].between(0.03, 0.07)).mean() > 0.85


def test_columns():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB"):
        for year in range(1900, 2000):
            rows.append({"iso3": iso, "year": year,
                          "y": rng.normal(), "x": rng.normal()})
    df = pd.DataFrame(rows)
    out = pooled_rolling_elasticity(df, y="y", x="x", window=40)
    for c in ("center_year", "beta", "se", "ci_low", "ci_high", "n"):
        assert c in out.columns
