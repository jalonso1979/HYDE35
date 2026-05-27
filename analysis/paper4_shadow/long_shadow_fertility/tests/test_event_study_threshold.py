"""Tests for event_study_threshold helpers (D3a).

Run with: pytest long_shadow_fertility/tests/test_event_study_threshold.py
"""
import pandas as pd
import numpy as np
import pytest
from long_shadow_fertility.estimators.event_study_threshold import (
    crossing_year,
    rolling_elasticity,
)


def test_crossing_year_basic():
    df = pd.DataFrame({"year": [1800, 1850, 1900, 1950],
                       "log_real_wage": [9.0, 9.5, 9.98, 10.5]})
    assert crossing_year(df, c=9.97) == 1900


def test_crossing_year_never():
    df = pd.DataFrame({"year": [1800, 1900], "log_real_wage": [8.0, 9.5]})
    assert crossing_year(df, c=9.97) is None


def test_crossing_year_exact_boundary_not_crossed():
    """Exactly at the threshold is NOT a crossing (strictly greater than)."""
    df = pd.DataFrame({"year": [1800, 1900], "log_real_wage": [8.0, 9.97]})
    assert crossing_year(df, c=9.97) is None


def test_crossing_year_returns_first_year():
    """When multiple years are above threshold, return the earliest."""
    df = pd.DataFrame({"year": [1800, 1850, 1900],
                       "log_real_wage": [10.0, 10.1, 10.2]})
    assert crossing_year(df, c=9.97) == 1800


def test_crossing_year_custom_z_col():
    df = pd.DataFrame({"year": [1800, 1900], "wage_alt": [9.0, 10.0]})
    assert crossing_year(df, z="wage_alt", c=9.97) == 1900


def test_rolling_elasticity_returns_per_year_betas():
    rng = np.random.default_rng(0)
    n = 50
    years = np.arange(1850, 1900)
    x = rng.normal(size=n)
    y = 0.3 * x + rng.normal(size=n)
    df = pd.DataFrame({"year": years, "y": y, "x": x})
    out = rolling_elasticity(df, y="y", x="x", window=15)
    assert {"year", "beta", "se"} <= set(out.columns)
    assert len(out) <= n
    # The middle betas should be in the ballpark of 0.3
    middle = out[(out["year"] > 1860) & (out["year"] < 1890)]
    assert abs(middle["beta"].mean() - 0.3) < 0.2


def test_rolling_elasticity_se_positive():
    rng = np.random.default_rng(42)
    n = 40
    years = np.arange(1860, 1900)
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"year": years, "y": y, "x": x})
    out = rolling_elasticity(df, y="y", x="x", window=15)
    assert (out["se"] >= 0).all()


def test_rolling_elasticity_skips_sparse_windows():
    """Windows with fewer than 5 observations should be skipped."""
    df = pd.DataFrame({"year": [1800, 1810, 1820],
                       "y": [1.0, 2.0, 3.0],
                       "x": [0.1, 0.2, 0.3]})
    out = rolling_elasticity(df, y="y", x="x", window=15)
    # Only 3 obs total — all windows have < 5 obs, so output should be empty
    assert len(out) == 0


def test_rolling_elasticity_custom_year_col():
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame({"t": np.arange(1870, 1900),
                       "y": rng.normal(size=n),
                       "x": rng.normal(size=n)})
    out = rolling_elasticity(df, y="y", x="x", window=11, year_col="t")
    assert "t" in out.columns
    assert "year" not in out.columns
