import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.bivariate_sur import (
    fit_bivariate_sur,
)


def test_recovers_two_known_slopes():
    """DGP: y1 = -0.05 x + eps1; y2 = +0.10 x + eps2; correlated errors."""
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        a1 = rng.normal(0, 0.3); a2 = rng.normal(0, 0.3)
        for year in range(1800, 2000):
            x = rng.normal(0, 1)
            eps_common = rng.normal(0, 0.01)
            y1 = a1 - 0.05 * x + rng.normal(0, 0.02) + eps_common
            y2 = a2 + 0.10 * x + rng.normal(0, 0.02) - eps_common
            rows.append({"iso3": iso, "year": year, "y1": y1, "y2": y2, "x": x})
    df = pd.DataFrame(rows)
    res = fit_bivariate_sur(df, y1="y1", y2="y2", x="x")
    assert -0.07 < res["beta_y1"] < -0.03
    assert 0.08 < res["beta_y2"] < 0.12
    assert res["wald_eq_pvalue"] < 0.01


def test_keys_present():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB"):
        for year in range(1900, 2000):
            rows.append({"iso3": iso, "year": year,
                          "y1": rng.normal(), "y2": rng.normal(), "x": rng.normal()})
    df = pd.DataFrame(rows)
    res = fit_bivariate_sur(df, y1="y1", y2="y2", x="x")
    for k in ("beta_y1", "beta_y2", "se_y1", "se_y2", "wald_eq_stat", "wald_eq_pvalue", "n"):
        assert k in res
