"""Tests for the England ModE-RA growing-season climate builder (Task 5)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_england_climate_annual import (
    build_england_climate_annual,
)


def test_columns_and_coverage():
    df = build_england_climate_annual()
    for col in ("year", "t_growing", "t_winter", "p_growing", "p_winter", "vol_t_10y"):
        assert col in df.columns
    assert df["year"].min() == 1421
    assert df["year"].max() >= 2008


def test_growing_winter_uncorrelated_levels():
    """Sanity: growing-season and winter means are different (not duplicated)."""
    df = build_england_climate_annual()
    same_count = (df["t_growing"] == df["t_winter"]).sum()
    assert same_count < 5


def test_volatility_finite_post1430():
    df = build_england_climate_annual()
    assert df.loc[df["year"] > 1430, "vol_t_10y"].notna().all()
