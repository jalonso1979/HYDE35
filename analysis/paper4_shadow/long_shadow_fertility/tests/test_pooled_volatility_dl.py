import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_volatility_dl import (
    fit_pooled_volatility_dl,
)


def test_recovers_two_regressor_lag_structure():
    rng = np.random.default_rng(0)
    rows = []
    year_fe = {y: rng.normal(0, 0.05) for y in range(1900, 2000)}
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        alpha = rng.normal(0, 0.3)
        for year in range(1900, 2000):
            T = rng.normal(0, 1); V = rng.uniform(0.1, 1.0)
            y = alpha + year_fe[year] - 0.05 * T - 0.04 * V + rng.normal(0, 0.02)
            rows.append({"iso3": iso, "year": year, "y": y, "T": T, "V": V})
    df = pd.DataFrame(rows)
    res = fit_pooled_volatility_dl(df, y="y", x_level="T", x_vol="V", lags=0)
    bL = res.loc[(res["regressor"] == "level") & (res["lag"] == 0), "beta"].iloc[0]
    bV = res.loc[(res["regressor"] == "vol") & (res["lag"] == 0), "beta"].iloc[0]
    assert -0.07 < bL < -0.03
    assert -0.06 < bV < -0.02


def test_keys_present():
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y": rng.normal(), "T": rng.normal(), "V": rng.normal()}
             for iso in ("AAA", "BBB") for y in range(1900, 2000)]
    df = pd.DataFrame(rows)
    res = fit_pooled_volatility_dl(df, y="y", x_level="T", x_vol="V", lags=2)
    for c in ("regressor", "lag", "beta", "se", "ci_low", "ci_high"):
        assert c in res.columns
