import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.iv_2sls import (
    fit_iv_2sls,
)


def test_recovers_known_beta():
    """DGP: x = 0.7*z + e1; y = -0.1*x + e2; z is exogenous instrument."""
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        a = rng.normal(0, 0.3)
        for year in range(1900, 2000):
            z = rng.normal(0, 1)
            x = 0.7 * z + rng.normal(0, 0.5)
            y = a - 0.1 * x + rng.normal(0, 0.05)
            rows.append({"iso3": iso, "year": year, "y": y, "x": x, "z": z})
    df = pd.DataFrame(rows)
    res = fit_iv_2sls(df, y="y", x="x", instruments=["z"])
    assert -0.13 < res["beta"] < -0.07
    assert res["first_stage_f"] > 10


def test_keys_present():
    rng = np.random.default_rng(0)
    rows = [{"iso3": iso, "year": y, "y": rng.normal(), "x": rng.normal(), "z": rng.normal()}
             for iso in ("AAA", "BBB") for y in range(1900, 2000)]
    df = pd.DataFrame(rows)
    res = fit_iv_2sls(df, y="y", x="x", instruments=["z"])
    for k in ("beta", "se", "first_stage_f", "ar_pvalue", "n"):
        assert k in res
