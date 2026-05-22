"""Tests for the Maddison England GDPpc builder (Task 4)."""
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_maddison_england import (
    build_maddison_england,
)


def test_columns():
    df = build_maddison_england()
    for col in ("year", "gdppc", "log_gdppc"):
        assert col in df.columns


def test_year_coverage():
    df = build_maddison_england()
    # Maddison has England GDPpc from ~1252 onward; require at least 1700-2020
    assert df["year"].min() <= 1700
    assert df["year"].max() >= 2020


def test_no_duplicates():
    df = build_maddison_england()
    assert df["year"].is_unique


def test_gdppc_increases_overall():
    df = build_maddison_england()
    early = df.loc[df["year"].between(1700, 1800), "gdppc"].mean()
    late = df.loc[df["year"].between(2000, 2020), "gdppc"].mean()
    assert late > 10 * early  # ~20x rise expected
