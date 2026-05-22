"""Tests for the France dept-year fertility + climate builder (Phase 2 Task 5)."""
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_france_dept_annual import (
    build_france_dept_annual,
)


def test_dept_panel_shape():
    df = build_france_dept_annual()
    assert df["dep"].nunique() >= 80
    assert df["year"].min() <= 1810
    assert df["year"].max() >= 2000


def test_columns():
    df = build_france_dept_annual()
    for c in ("dep", "year", "iso3", "births", "log_cbr", "t_growing", "p_growing"):
        assert c in df.columns


def test_iso3_FRA():
    df = build_france_dept_annual()
    assert (df["iso3"] == "FRA").all()
