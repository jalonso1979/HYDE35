import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.volcanic_event_study import (
    event_study_single_event,
)


def test_recovers_known_event_effect():
    rng = np.random.default_rng(0)
    years = np.arange(1800, 1830)
    eruption_year = 1815
    # True effect: -0.10 in h=0, -0.05 in h=1, 0 elsewhere
    y = 5.0 * np.ones_like(years, dtype=float)
    h = years - eruption_year
    y[h == 0] += -0.10
    y[h == 1] += -0.05
    y += rng.normal(0, 0.01, size=len(years))
    df = pd.DataFrame({"year": years, "y": y})
    res = event_study_single_event(df, y="y", eruption_year=eruption_year,
                                     pre=5, post=10)
    coef_h0 = res.loc[res["h"] == 0, "delta"].iloc[0]
    coef_h1 = res.loc[res["h"] == 1, "delta"].iloc[0]
    assert -0.13 < coef_h0 < -0.07
    assert -0.08 < coef_h1 < -0.02


def test_reference_year_excluded():
    df = pd.DataFrame({"year": np.arange(1810, 1830), "y": np.random.normal(0, 1, 20)})
    res = event_study_single_event(df, y="y", eruption_year=1815, pre=5, post=10)
    assert -1 not in res["h"].tolist()


def test_recovers_event_effect_with_wider_window():
    rng = np.random.default_rng(0)
    years = np.arange(1790, 1840)
    eruption_year = 1815
    # True effect: -0.10 in h=0, -0.05 in h=1, 0 elsewhere
    y = 5.0 * np.ones_like(years, dtype=float)
    h = years - eruption_year
    y[h == 0] += -0.10
    y[h == 1] += -0.05
    y += rng.normal(0, 0.01, size=len(years))
    df = pd.DataFrame({"year": years, "y": y})
    res = event_study_single_event(df, y="y", eruption_year=eruption_year,
                                     pre=5, post=10)
    # Trend is retained -> HC1 SEs should be finite for all returned rows.
    assert np.isfinite(res["se"].to_numpy()).all()
    coef_h0 = res.loc[res["h"] == 0, "delta"].iloc[0]
    coef_h1 = res.loc[res["h"] == 1, "delta"].iloc[0]
    assert -0.13 < coef_h0 < -0.07
    assert -0.08 < coef_h1 < -0.02
