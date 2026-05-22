"""Tests for the pandemic-control country-year panel builder (Phase 2 Task 8)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_pandemic_panel import (
    build_pandemic_panel,
)


def test_year_range():
    df = build_pandemic_panel()
    assert df["year"].min() <= 1500
    assert df["year"].max() >= 2022


def test_columns():
    df = build_pandemic_panel()
    for c in ("iso3", "year", "pandemic_active"):
        assert c in df.columns


def test_1918_flu_pandemic_active_for_all():
    df = build_pandemic_panel().set_index(["iso3", "year"])
    for iso in ("GBR", "FRA", "ITA", "SWE"):
        assert df.loc[(iso, 1918), "pandemic_active"] == 1
        assert df.loc[(iso, 1919), "pandemic_active"] == 1


def test_covid():
    df = build_pandemic_panel().set_index(["iso3", "year"])
    for iso in ("GBR", "FRA", "ITA", "SWE"):
        assert df.loc[(iso, 2020), "pandemic_active"] == 1
        assert df.loc[(iso, 2021), "pandemic_active"] == 1


def test_black_death_active_for_eu_countries():
    df = build_pandemic_panel().set_index(["iso3", "year"])
    for iso in ("GBR", "FRA", "ITA", "SWE"):
        for y in (1348, 1349, 1350):
            assert df.loc[(iso, y), "pandemic_active"] == 1
