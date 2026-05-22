import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_smooth_transition import (
    fit_pooled_smooth_transition,
)


def test_recovers_panel_regime_change():
    rng = np.random.default_rng(0)
    rows = []
    for iso, alpha_c in zip(("AAA", "BBB", "CCC", "DDD"), (0.3, -0.4, 0.1, 0.0)):
        z = np.linspace(-3, 3, 200)
        for year, zv in zip(range(1800, 2000), z):
            g = 1.0 / (1.0 + np.exp(-2.0 * (zv - 0.0)))
            x = rng.normal(0, 1)
            beta_t = 0.04 * (1 - g) + (-0.02) * g
            y = alpha_c + beta_t * x + rng.normal(0, 0.02)
            rows.append({"iso3": iso, "year": year, "y": y, "x": x, "z": zv})
    df = pd.DataFrame(rows)
    res = fit_pooled_smooth_transition(df, y="y", x="x", z="z")
    assert abs(res["beta_M"] - 0.04) < 0.015
    assert abs(res["beta_T"] + 0.02) < 0.015


def test_keys_present():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB"):
        for year in range(1900, 2000):
            rows.append({"iso3": iso, "year": year, "y": rng.normal(),
                          "x": rng.normal(), "z": rng.normal()})
    df = pd.DataFrame(rows)
    res = fit_pooled_smooth_transition(df, y="y", x="x", z="z")
    for k in ("beta_M", "beta_T", "c", "theta", "beta_M_se", "beta_T_se", "n"):
        assert k in res
