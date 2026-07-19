"""Tests for the Sweden country-year fertility builder (Phase 2 Task 4)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_sweden_annual import (
    build_sweden_annual,
)


def test_year_range_tabellverket_era():
    df = build_sweden_annual()
    # HMD Sweden starts at 1749 - Tabellverket era
    assert df["year"].min() <= 1750
    assert df["year"].max() >= 2018


def test_columns():
    df = build_sweden_annual()
    for c in ("year", "iso3", "births", "population", "cbr", "log_cbr", "source"):
        assert c in df.columns


def test_iso3_is_SWE():
    df = build_sweden_annual()
    assert (df["iso3"] == "SWE").all()


def test_cbr_in_plausible_range():
    df = build_sweden_annual()
    cbr = df["cbr"].dropna()
    assert (cbr > 5).all() and (cbr < 50).all()
