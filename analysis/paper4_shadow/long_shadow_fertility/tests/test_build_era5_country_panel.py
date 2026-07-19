"""Phase 8 Pillar B4b: ERA5 country-year panel."""
import pytest
import pandas as pd


def test_build_era5_country_panel_runs():
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_era5_country_panel import (
        build_era5_country_panel,
    )
    df = build_era5_country_panel(write=False)
    assert isinstance(df, pd.DataFrame)
    if len(df) == 0:
        pytest.skip("No ERA5 data extracted yet (download incomplete?)")
    assert "iso3" in df.columns and "year" in df.columns
    # At least some Long Shadow countries should appear
    long_shadow_countries = {"GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"}
    overlap = set(df["iso3"].unique()) & long_shadow_countries
    assert len(overlap) >= 4, f"Only {overlap} Long Shadow countries in ERA5 panel"


def test_era5_country_values_differ_across_countries():
    """If all countries had same value, the bbox extraction is broken."""
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_era5_country_panel import (
        build_era5_country_panel,
    )
    df = build_era5_country_panel(write=False)
    if len(df) == 0:
        pytest.skip("No ERA5 data extracted yet")
    # Within a given year, T should differ across countries (no two countries
    # in our panel are climatically identical)
    year_with_data = df.groupby("year")["iso3"].nunique().idxmax()
    sub = df[df["year"] == year_with_data]
    if len(sub) >= 2:
        assert sub["T"].std() > 0.01, (
            f"All countries have ~identical T in year {year_with_data} — "
            "bbox extraction broken"
        )
