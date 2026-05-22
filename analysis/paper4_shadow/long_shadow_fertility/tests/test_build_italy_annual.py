"""Tests for the Italy country-year fertility builder (Phase 2 Task 3)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_italy_annual import (
    build_italy_annual,
)


def test_year_range():
    df = build_italy_annual()
    assert df["year"].min() <= 1865  # HMD Italy starts 1862
    assert df["year"].max() >= 2018


def test_columns():
    df = build_italy_annual()
    for c in ("year", "iso3", "births", "population", "cbr", "log_cbr", "source"):
        assert c in df.columns


def test_iso3_is_ITA():
    df = build_italy_annual()
    assert (df["iso3"] == "ITA").all()


def test_cbr_in_plausible_range():
    df = build_italy_annual()
    cbr = df["cbr"].dropna()
    assert (cbr > 5).all() and (cbr < 50).all()
