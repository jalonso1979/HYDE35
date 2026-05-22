import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.rolling_window import (
    rolling_elasticity,
)


def test_rolling_recovers_linear_dgp():
    """Synthetic DGP: log y = 0.05 * x + noise. Rolling estimate ~0.05."""
    rng = np.random.default_rng(0)
    years = np.arange(1700, 2000)
    x = rng.normal(0.0, 1.0, size=len(years))
    y = 0.05 * x + rng.normal(0.0, 0.02, size=len(years))
    df = pd.DataFrame({"year": years, "y": y, "x": x})
    out = rolling_elasticity(df, y="y", x="x", window=40)
    assert (out["beta"].between(0.03, 0.07)).mean() > 0.9


def test_rolling_columns_and_window():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "year": np.arange(1600, 2000),
        "y": rng.normal(0, 1, 400),
        "x": rng.normal(0, 1, 400),
    })
    out = rolling_elasticity(df, y="y", x="x", window=40)
    for col in ("center_year", "beta", "se", "ci_low", "ci_high", "n"):
        assert col in out.columns
    assert out["n"].max() == 40


def test_rolling_handles_nas_correctly():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "year": np.arange(1500, 2000),
        "y": rng.normal(0, 1, 500),
        "x": rng.normal(0, 1, 500),
    })
    df.loc[df["year"].between(1500, 1540), "x"] = np.nan
    out = rolling_elasticity(df, y="y", x="x", window=40)
    early = out.loc[out["center_year"] < 1540]
    assert early["n"].fillna(0).max() < 40
