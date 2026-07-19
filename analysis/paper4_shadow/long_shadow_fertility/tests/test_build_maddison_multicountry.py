"""Tests for the Maddison multi-country GDPpc builder (Phase 2 Task 1)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_maddison_multicountry import (
    build_maddison_multicountry,
)


def test_seven_countries_present():
    df = build_maddison_multicountry()
    countries = set(df["iso3"].unique())
    assert {"GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"}.issubset(countries)


def test_columns():
    df = build_maddison_multicountry()
    for c in ("iso3", "year", "gdppc", "log_gdppc"):
        assert c in df.columns


def test_log_gdppc_finite():
    df = build_maddison_multicountry()
    assert df["log_gdppc"].notna().sum() > 1000
