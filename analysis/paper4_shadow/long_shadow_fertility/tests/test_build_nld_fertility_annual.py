import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_nld_fertility_annual import (
    build_nld_fertility_annual,
)


def test_year_range():
    df = build_nld_fertility_annual()
    assert df["year"].min() <= 1855
    assert df["year"].max() >= 2018


def test_columns():
    df = build_nld_fertility_annual()
    for c in ("year", "iso3", "births", "population", "cbr", "log_cbr", "source"):
        assert c in df.columns


def test_iso3_is_NLD():
    df = build_nld_fertility_annual()
    assert (df["iso3"] == "NLD").all()


def test_cbr_in_plausible_range():
    df = build_nld_fertility_annual()
    cbr = df["cbr"].dropna()
    assert (cbr > 5).all() and (cbr < 50).all()
