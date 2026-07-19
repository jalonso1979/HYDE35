import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel import (
    build_real_wage_panel,
)


def test_columns():
    df = build_real_wage_panel()
    for c in ("iso3", "year", "log_real_wage", "source"):
        assert c in df.columns


def test_four_iso3():
    df = build_real_wage_panel()
    assert {"GBR", "FRA", "ITA", "SWE"}.issubset(set(df["iso3"].unique()))


def test_allen_era_present():
    df = build_real_wage_panel()
    pre = df.loc[df["year"].between(1500, 1900)]
    assert pre["log_real_wage"].notna().sum() > 500


def test_modern_splice():
    df = build_real_wage_panel()
    post = df.loc[df["year"] > 1913]
    assert post["log_real_wage"].notna().sum() > 100
