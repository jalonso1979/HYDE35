from pathlib import Path
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_england_fertility_annual import (
    build_england_fertility_annual,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
            "england_fertility_annual_1541_2020.parquet")


def test_year_coverage():
    df = build_england_fertility_annual()
    assert df["year"].min() == 1541
    assert df["year"].max() >= 2018
    assert df["year"].is_monotonic_increasing
    assert df["year"].is_unique


def test_gap_now_only_three_years():
    """Phase 5 fix: HMD GBRTENW starts 1841, so gap is only 1838-1840."""
    df = build_england_fertility_annual().set_index("year")
    for y in (1838, 1839, 1840):
        assert pd.isna(df.loc[y, "cbr"])
    assert df.loc[1841, "cbr"] > 0
    assert df.loc[1900, "cbr"] > 0


def test_columns_present():
    df = build_england_fertility_annual()
    for col in ("year", "births", "population", "cbr", "log_cbr", "source"):
        assert col in df.columns


def test_modern_era_has_cbr():
    df = build_england_fertility_annual().set_index("year")
    assert df.loc[1950:2010, "cbr"].notna().all()


def test_industrial_era_has_cbr_now():
    """The whole point of Phase 5 Task 1 — 1850 to 1937 should now be populated."""
    df = build_england_fertility_annual().set_index("year")
    for y in (1850, 1880, 1900, 1925, 1937):
        assert df.loc[y, "cbr"] > 0, f"year {y} should now have CBR after HMD switch"


def test_cbr_magnitude_reasonable():
    df = build_england_fertility_annual()
    cbr = df["cbr"].dropna()
    assert (cbr > 5).all() and (cbr < 60).all()
