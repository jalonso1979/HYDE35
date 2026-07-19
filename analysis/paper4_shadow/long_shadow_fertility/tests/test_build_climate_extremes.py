"""Tests for the climate extremes builder (Phase 2 Task 10)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_climate_extremes import (
    build_climate_extremes,
)


def test_columns():
    df = build_climate_extremes()
    for c in ("iso3", "year", "heat_extreme", "drought"):
        assert c in df.columns


def test_indicators_binary():
    df = build_climate_extremes()
    assert df["heat_extreme"].isin([0, 1]).all()
    assert df["drought"].isin([0, 1]).all()


def test_extreme_share_about_5_percent_pre1900():
    df = build_climate_extremes()
    pre = df.loc[df["year"] < 1900]
    share = pre["heat_extreme"].mean()
    assert 0.02 < share < 0.10
