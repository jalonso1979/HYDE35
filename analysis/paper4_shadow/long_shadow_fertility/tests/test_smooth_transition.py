import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.smooth_transition import (
    fit_smooth_transition,
)


def test_recovers_known_regime_change():
    """DGP: beta_M=0.04 (Malthusian), beta_T=-0.02 (modern); transition at z=0."""
    rng = np.random.default_rng(0)
    n = 300
    z = np.linspace(-3, 3, n)
    g = 1.0 / (1.0 + np.exp(-2.0 * (z - 0.0)))
    x = rng.normal(0, 1, n)
    beta_t = 0.04 * (1 - g) + (-0.02) * g
    y = beta_t * x + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"year": np.arange(n), "y": y, "x": x, "z": z})
    res = fit_smooth_transition(df, y="y", x="x", z="z")
    assert abs(res["beta_M"] - 0.04) < 0.015
    assert abs(res["beta_T"] - (-0.02)) < 0.015
    assert abs(res["c"]) < 0.6


def test_keys_present():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "year": np.arange(100),
        "y": rng.normal(0, 1, 100),
        "x": rng.normal(0, 1, 100),
        "z": rng.normal(0, 1, 100),
    })
    res = fit_smooth_transition(df, y="y", x="x", z="z")
    for k in ("beta_M", "beta_T", "c", "theta", "beta_M_se", "beta_T_se", "n"):
        assert k in res
