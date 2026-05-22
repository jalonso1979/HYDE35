"""Tests for the France country-year fertility builder (Phase 2 Task 2)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_france_annual import (
    build_france_annual,
)


def test_year_range():
    df = build_france_annual()
    assert df["year"].min() <= 1820
    assert df["year"].max() >= 2018


def test_columns():
    df = build_france_annual()
    for c in ("year", "iso3", "births", "population", "cbr", "log_cbr", "source"):
        assert c in df.columns


def test_iso3_is_FRA():
    df = build_france_annual()
    assert (df["iso3"] == "FRA").all()


def test_cbr_in_plausible_range():
    df = build_france_annual()
    cbr = df["cbr"].dropna()
    assert (cbr > 5).all() and (cbr < 50).all()
