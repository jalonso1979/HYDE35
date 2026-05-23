"""Phase 9 Pillar A3: ModE-RA + ERA5 spliced country-year panel."""
import pytest


def test_spliced_panel_runs():
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_country_climate_spliced import (
        build_country_climate_spliced,
    )
    df = build_country_climate_spliced(write=False)
    assert {"iso3", "year", "t_growing", "p_growing", "source"}.issubset(df.columns)
    assert (df["source"] == "ModE-RA").any()
    assert (df["source"].str.startswith("ERA5")).any()


def test_splice_boundary_continuous():
    """No huge jump in t_growing at the 1949/1950 boundary per country."""
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.build_country_climate_spliced import (
        build_country_climate_spliced,
    )
    df = build_country_climate_spliced(write=False)
    for iso, sub in df.groupby("iso3"):
        v1949 = sub.loc[sub["year"] == 1949, "t_growing"]
        v1950 = sub.loc[sub["year"] == 1950, "t_growing"]
        if not v1949.empty and not v1950.empty:
            jump = abs(float(v1950.iloc[0]) - float(v1949.iloc[0]))
            assert jump < 3.0, f"{iso} 1949->1950 jump = {jump:.2f}°C"
