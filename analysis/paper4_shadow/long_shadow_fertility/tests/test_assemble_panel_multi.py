"""Tests for the multi-country unified panel assembler (Task 12)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)


def test_four_iso3():
    df = assemble_panel_multi()
    assert set(df["iso3"].unique()) == {"GBR", "FRA", "ITA", "SWE"}


def test_columns():
    df = assemble_panel_multi()
    required = {"iso3", "year", "log_cbr", "t_growing", "p_growing",
                "log_gdppc", "war_active", "log_war_fatalities", "pandemic_active",
                "disaster_count", "heat_extreme", "drought",
                "vol_t_10y", "is_eruption_year"}
    assert required.issubset(df.columns)


def test_no_duplicate_keys():
    df = assemble_panel_multi()
    assert not df.duplicated(subset=["iso3", "year"]).any()


def test_england_logcbr_present():
    df = assemble_panel_multi()
    gbr = df.loc[df["iso3"] == "GBR"]
    assert gbr["log_cbr"].notna().sum() > 200
