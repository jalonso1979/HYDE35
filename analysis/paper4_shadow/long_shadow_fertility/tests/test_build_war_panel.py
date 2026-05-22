"""Tests for the war-control country-year panel builder (Phase 2 Task 7)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_war_panel import (
    build_war_panel,
)


def test_year_range():
    df = build_war_panel()
    assert df["year"].min() <= 1500
    assert df["year"].max() >= 2000


def test_columns():
    df = build_war_panel()
    for c in ("iso3", "year", "war_active", "log_war_fatalities"):
        assert c in df.columns


def test_countries_covered():
    df = build_war_panel()
    covered = set(df["iso3"].unique())
    assert {"GBR", "FRA", "ITA", "SWE"}.issubset(covered)


def test_war_active_binary_or_zero():
    df = build_war_panel()
    assert df["war_active"].isin([0, 1]).all()


def test_wwi_wwii_active_france():
    """WWI 1914-18 and WWII 1939-45 must be war_active for France."""
    df = build_war_panel().set_index(["iso3", "year"])
    for y in (1914, 1915, 1916, 1917, 1918,
              1939, 1940, 1941, 1942, 1943, 1944, 1945):
        assert df.loc[("FRA", y), "war_active"] == 1, (
            f"FRA {y} should be war_active"
        )
