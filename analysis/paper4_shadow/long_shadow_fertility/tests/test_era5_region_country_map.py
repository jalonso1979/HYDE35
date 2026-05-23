"""Phase 8 Pillar B4a: ERA5 region → country crosswalk."""
import pytest


def test_build_region_country_map_returns_dict():
    """Mapping should resolve at least 4 of 7 Long Shadow countries to valid regions."""
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.era5_region_country_map import (
        build_region_country_map,
        COUNTRY_CAPITAL_COORDS,
    )
    try:
        mapping = build_region_country_map()
    except FileNotFoundError:
        pytest.skip("ERA5 download incomplete")
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == set(COUNTRY_CAPITAL_COORDS.keys())
    valid = [iso for iso, r in mapping.items() if r > 0]
    assert len(valid) >= 4, f"Only {len(valid)} of 7 countries mapped: {mapping}"


def test_capital_centroid_resolves():
    """A single test coord (London) should map to some region or -1 (no crash)."""
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.era5_region_country_map import (
        build_region_country_map,
    )
    try:
        mapping = build_region_country_map(
            countries={"TEST_GBR": (51.5, -0.1)},
            n_regions=25,
        )
    except FileNotFoundError:
        pytest.skip("ERA5 download incomplete")
    assert "TEST_GBR" in mapping


def test_build_with_bboxes_returns_full_entries():
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.data.era5_region_country_map import (
        build_region_country_map_with_bboxes,
        COUNTRY_BBOXES,
    )
    try:
        full = build_region_country_map_with_bboxes()
    except FileNotFoundError:
        pytest.skip("ERA5 download incomplete")
    assert set(full.keys()) == set(COUNTRY_BBOXES.keys())
    for iso, entry in full.items():
        assert "region" in entry
        assert "bbox" in entry
        assert entry["bbox"] is not None
        lat_min, lat_max, lon_min, lon_max = entry["bbox"]
        assert lat_min < lat_max
        assert lon_min < lon_max
