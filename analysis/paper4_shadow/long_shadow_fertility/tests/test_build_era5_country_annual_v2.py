"""Phase 9 Pillar A1: ERA5 growing-season anomaly aggregator."""
import pytest
import pandas as pd


def test_build_era5_growing_season_runs():
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_era5_country_annual_v2 import (
        build_era5_country_annual_v2,
    )
    df = build_era5_country_annual_v2(write=False)
    assert isinstance(df, pd.DataFrame)
    assert {"iso3", "year", "t_growing_era5", "p_growing_era5"}.issubset(df.columns)
    long_shadow_countries = {"GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"}
    overlap = set(df["iso3"].unique()) & long_shadow_countries
    assert len(overlap) >= 4


def test_anomaly_baseline_centered_on_zero():
    """Per-country mean of t_growing_era5 over 1961-1990 must be ~0 by construction."""
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_era5_country_annual_v2 import (
        build_era5_country_annual_v2, BASELINE_WINDOW,
    )
    df = build_era5_country_annual_v2(write=False)
    base_lo, base_hi = BASELINE_WINDOW
    baseline = df[df["year"].between(base_lo, base_hi)]
    for iso, sub in baseline.groupby("iso3"):
        m = sub["t_growing_era5"].mean()
        assert abs(m) < 0.01, f"{iso} baseline t mean = {m:.4f}"


def test_t_growing_era5_celsius_anomaly_sensible_range():
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_era5_country_annual_v2 import (
        build_era5_country_annual_v2,
    )
    df = build_era5_country_annual_v2(write=False).dropna(subset=["t_growing_era5"])
    assert df["t_growing_era5"].min() > -5.0
    assert df["t_growing_era5"].max() < 8.0
