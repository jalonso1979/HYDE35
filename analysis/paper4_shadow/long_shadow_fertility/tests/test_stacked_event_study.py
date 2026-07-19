import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.stacked_event_study import (
    stacked_event_study,
)


def test_recovers_pooled_event_effect():
    rng = np.random.default_rng(0)
    rows = []
    for iso, alpha in zip(("AAA", "BBB", "CCC", "DDD"), (1.0, -0.5, 0.3, 0.1)):
        for year in range(1800, 1830):
            h = year - 1815
            y = alpha + (-0.10 if h == 0 else (-0.05 if h == 1 else 0)) + rng.normal(0, 0.01)
            rows.append({"iso3": iso, "year": year, "y": y})
    df = pd.DataFrame(rows)
    res = stacked_event_study(df, y="y", eruption_year=1815, pre=5, post=10)
    h0 = res.loc[res["h"] == 0, "delta"].iloc[0]
    h1 = res.loc[res["h"] == 1, "delta"].iloc[0]
    assert -0.12 < h0 < -0.08
    assert -0.07 < h1 < -0.03


def test_reference_excluded():
    rng = np.random.default_rng(0)
    df = pd.DataFrame([
        {"iso3": "AAA", "year": y, "y": rng.normal()} for y in range(1810, 1830)
    ])
    res = stacked_event_study(df, y="y", eruption_year=1815, pre=5, post=10)
    assert -1 not in res["h"].tolist()
