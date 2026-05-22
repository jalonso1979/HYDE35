"""Tests for the unified controls panel merger (Phase 2 Task 11)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_controls_panel import (
    build_controls_panel,
)


def test_columns():
    df = build_controls_panel()
    expected = {"iso3", "year", "war_active", "log_war_fatalities",
                "pandemic_active", "disaster_count", "log_disaster_deaths",
                "heat_extreme", "drought", "vol_t_10y", "vol_p_10y"}
    assert expected.issubset(df.columns)


def test_four_countries():
    df = build_controls_panel()
    assert set(df["iso3"].unique()) == {"GBR", "FRA", "ITA", "SWE"}


def test_no_dup_keys():
    df = build_controls_panel()
    assert not df.duplicated(subset=["iso3", "year"]).any()
