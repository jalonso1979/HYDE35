import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.triple_sur import (
    fit_triple_sur,
)


def test_recovers_three_known_slopes():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        a1, a2, a3 = rng.normal(0, 0.3, 3)
        for year in range(1800, 2000):
            x = rng.normal(0, 1)
            ec = rng.normal(0, 0.01)
            y1 = a1 - 0.05 * x + rng.normal(0, 0.02) + ec
            y2 = a2 + 0.10 * x + rng.normal(0, 0.02) - ec
            y3 = a3 + 0.20 * x + rng.normal(0, 0.02) + ec
            rows.append({"iso3": iso, "year": year, "y1": y1, "y2": y2, "y3": y3, "x": x})
    df = pd.DataFrame(rows)
    res = fit_triple_sur(df, ys=["y1", "y2", "y3"], x="x")
    assert -0.07 < res["beta_y1"] < -0.03
    assert 0.08 < res["beta_y2"] < 0.12
    assert 0.18 < res["beta_y3"] < 0.22


def test_keys_present():
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y1": rng.normal(), "y2": rng.normal(),
              "y3": rng.normal(), "x": rng.normal()}
             for iso in ("AAA", "BBB") for y in range(1900, 2000)]
    df = pd.DataFrame(rows)
    res = fit_triple_sur(df, ys=["y1", "y2", "y3"], x="x")
    for k in ("beta_y1", "beta_y2", "beta_y3", "se_y1", "se_y2", "se_y3",
              "wald_eq12_pvalue", "wald_eq13_pvalue", "wald_eq23_pvalue", "n"):
        assert k in res
