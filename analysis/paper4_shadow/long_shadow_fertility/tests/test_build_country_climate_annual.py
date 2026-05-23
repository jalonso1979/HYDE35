"""Tests for the multi-country ModE-RA growing-season climate builder (Phase 2 Task 6)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_climate_annual import (
    build_country_climate_annual,
)


def test_seven_iso3():
    df = build_country_climate_annual()
    assert set(df["iso3"].unique()) == {"GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"}


def test_columns():
    df = build_country_climate_annual()
    for c in ("iso3", "year", "t_growing", "p_growing", "t_winter", "p_winter",
              "vol_t_10y", "vol_p_10y"):
        assert c in df.columns


def test_year_range_per_country():
    df = build_country_climate_annual()
    for iso in ("GBR", "FRA", "ITA", "SWE"):
        sub = df[df["iso3"] == iso]
        assert sub["year"].min() == 1421
        assert sub["year"].max() >= 2008
