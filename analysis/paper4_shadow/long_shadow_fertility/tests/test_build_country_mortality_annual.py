"""Tests for the HMD all-cause CDR builder (Phase 3 Task 1)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)


def test_twelve_iso3():
    df = build_country_mortality_annual()
    assert set(df["iso3"].unique()) == {
        "GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP",
        "NOR", "DNK", "FIN", "ISL", "CHE",
    }


def test_columns():
    df = build_country_mortality_annual()
    for c in ("iso3", "year", "deaths", "population", "cdr", "log_cdr", "source"):
        assert c in df.columns


def test_cdr_in_plausible_range():
    df = build_country_mortality_annual()
    cdr = df["cdr"].dropna()
    assert (cdr > 4).all() and (cdr < 60).all()


def test_year_coverage_swe_tabellverket():
    df = build_country_mortality_annual()
    swe = df.loc[df["iso3"] == "SWE"]
    assert swe["year"].min() <= 1755
