"""Tests for region and country mask construction."""
import numpy as np
import xarray as xr
import pytest
from analysis.shared.masks import build_bbox_mask, build_country_mask


def test_build_bbox_mask(synthetic_grid, synthetic_land_mask):
    """Bounding box that covers the upper-right quadrant of our 4x4 grid."""
    bbox = {"lat_min": 0.0, "lat_max": 2.0, "lon_min": 2.0, "lon_max": 4.0}
    mask = build_bbox_mask(synthetic_grid, bbox, synthetic_land_mask)
    assert mask.sum().item() == 4.0
    assert mask.sel(lat=-1.5, lon=0.5).item() == 0.0


def test_build_bbox_mask_with_subtract(synthetic_grid, synthetic_land_mask):
    bbox = {"lat_min": -2.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 4.0}
    subtract = [{"lat_min": 0.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 4.0}]
    mask = build_bbox_mask(synthetic_grid, bbox, synthetic_land_mask, subtract_boxes=subtract)
    assert mask.sel(lat=1.5, lon=3.5).item() == 0.0
    assert mask.sel(lat=-0.5, lon=1.5).item() == 1.0


def test_build_country_mask(synthetic_grid, synthetic_land_mask):
    country_grid = xr.full_like(synthetic_grid, 1.0)
    country_grid[2:, :] = 2.0
    country_grid[0, 0] = np.nan
    mask_1 = build_country_mask(country_grid, country_id=1, land_mask=synthetic_land_mask)
    mask_2 = build_country_mask(country_grid, country_id=2, land_mask=synthetic_land_mask)
    assert mask_1.sum().item() == 7.0
    assert mask_2.sum().item() == 8.0
