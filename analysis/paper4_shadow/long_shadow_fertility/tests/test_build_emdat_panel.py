"""Tests for the EMDAT country-year disaster panel builder (Phase 2 Task 9)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_emdat_panel import (
    build_emdat_panel,
)


def test_year_range():
    df = build_emdat_panel()
    assert df["year"].min() <= 1965
    assert df["year"].max() >= 2018


def test_columns():
    df = build_emdat_panel()
    for c in ("iso3", "year", "disaster_count", "log_disaster_deaths"):
        assert c in df.columns


def test_four_countries_present():
    df = build_emdat_panel()
    assert {"GBR", "FRA", "ITA", "SWE"}.issubset(set(df["iso3"].unique()))


def test_counts_nonneg():
    df = build_emdat_panel()
    assert (df["disaster_count"] >= 0).all()
