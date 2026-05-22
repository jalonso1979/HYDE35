"""Tests for the England fertility 1541-2020 splice builder (Task 3)."""
from pathlib import Path
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_england_fertility_annual import (
    build_england_fertility_annual,
)

OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "england_fertility_annual_1541_2020.parquet"
)


def test_year_coverage():
    df = build_england_fertility_annual()
    assert df["year"].min() == 1541
    assert df["year"].max() >= 2018
    assert df["year"].is_monotonic_increasing
    assert df["year"].is_unique


def test_columns_present():
    df = build_england_fertility_annual()
    for col in ("year", "births", "population", "cbr", "log_cbr", "source"):
        assert col in df.columns


def test_campop_era_has_cbr():
    df = build_england_fertility_annual().set_index("year")
    # CamPOP era should have non-NaN CBR
    assert df.loc[1600:1800, "cbr"].notna().all()


def test_modern_era_has_cbr():
    df = build_england_fertility_annual().set_index("year")
    # HFD era (1938+) should have non-NaN CBR
    assert df.loc[1950:2010, "cbr"].notna().all()


def test_gap_years_are_nan_cbr():
    df = build_england_fertility_annual().set_index("year")
    # 1900 is in the documented gap — cbr should be NaN
    assert pd.isna(df.loc[1900, "cbr"])
    assert df.loc[1900, "source"] == "GAP_industrial"


def test_output_parquet_written():
    build_england_fertility_annual(write=True)
    assert OUT.exists()
    df = pd.read_parquet(OUT)
    assert "log_cbr" in df.columns


def test_cbr_magnitude_reasonable():
    """CBR per 1000 should be in 5-50 range across both eras."""
    df = build_england_fertility_annual()
    cbr_finite = df["cbr"].dropna()
    assert (cbr_finite > 5).all() and (cbr_finite < 60).all()
