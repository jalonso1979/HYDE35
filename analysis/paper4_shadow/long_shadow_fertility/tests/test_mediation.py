import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.mediation import (
    fit_mediation,
)


def test_recovers_known_mediation():
    """DGP: x -> m (phi=0.3); x -> y direct (beta=0.02); m -> y (delta=0.4).
    Indirect = phi * delta = 0.12; total = 0.14."""
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB", "CCC", "DDD"):
        a_m = rng.normal(0, 0.2); a_y = rng.normal(0, 0.2)
        for year in range(1800, 2000):
            x = rng.normal(0, 1)
            m = a_m + 0.3 * x + rng.normal(0, 0.05)
            y = a_y + 0.02 * x + 0.4 * m + rng.normal(0, 0.05)
            rows.append({"iso3": iso, "year": year, "y": y, "x": x, "m": m})
    df = pd.DataFrame(rows)
    res = fit_mediation(df, y="y", x="x", m="m", n_boot=100, seed=0)
    assert 0.10 < res["indirect"] < 0.14
    assert 0.0 < res["direct"] < 0.05
    assert 0.12 < res["total"] < 0.16


def test_keys_present():
    rng = np.random.default_rng(0)
    rows = []
    for iso in ("AAA", "BBB"):
        for year in range(1900, 2000):
            rows.append({"iso3": iso, "year": year, "y": rng.normal(),
                          "x": rng.normal(), "m": rng.normal()})
    df = pd.DataFrame(rows)
    res = fit_mediation(df, y="y", x="x", m="m", n_boot=50, seed=0)
    for k in ("direct", "indirect", "total", "direct_se", "indirect_se", "phi", "delta", "n"):
        assert k in res
