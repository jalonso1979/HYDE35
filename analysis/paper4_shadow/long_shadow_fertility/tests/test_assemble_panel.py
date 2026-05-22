"""Tests for the unified England panel assembler (Task 6)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel import (
    assemble_england_panel,
)

ERUPTION_YEARS = (1600, 1641, 1815, 1883, 1991)


def test_columns():
    df = assemble_england_panel()
    required = {"year", "log_cbr", "t_growing", "p_growing", "log_gdppc",
                "vol_t_10y", "is_eruption_year", "years_to_nearest_eruption"}
    assert required.issubset(df.columns)


def test_no_duplicate_years():
    df = assemble_england_panel()
    assert df["year"].is_unique


def test_eruption_flags_correct():
    df = assemble_england_panel().set_index("year")
    for y in ERUPTION_YEARS:
        if y in df.index:
            assert df.loc[y, "is_eruption_year"] == 1


def test_panel_complete_for_modeling():
    df = assemble_england_panel()
    # Modeling window: 1700-2008 should have non-null fertility, climate, GDP.
    # The known industrial gap (1838-1937) will have NaN log_cbr but that's
    # expected; the test only checks years OUTSIDE the gap.
    modeling = df.loc[df["year"].between(1700, 1837)]
    assert modeling[["log_cbr", "t_growing", "log_gdppc"]].notna().all().all()
    modeling2 = df.loc[df["year"].between(1938, 2008)]
    assert modeling2[["log_cbr", "t_growing", "log_gdppc"]].notna().all().all()
