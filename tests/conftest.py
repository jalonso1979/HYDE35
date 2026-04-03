"""Shared test fixtures — small synthetic data mimicking HYDE35 structure."""
import numpy as np
import pandas as pd
import xarray as xr
import pytest


@pytest.fixture
def synthetic_grid():
    """4x4 lat/lon grid with known values."""
    lat = np.array([-1.5, -0.5, 0.5, 1.5])
    lon = np.array([0.5, 1.5, 2.5, 3.5])
    data = np.arange(16, dtype="float64").reshape(4, 4)
    return xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


@pytest.fixture
def synthetic_land_mask(synthetic_grid):
    """Land mask: all land except one ocean cell."""
    mask = xr.ones_like(synthetic_grid)
    mask[0, 0] = 0.0
    return mask


@pytest.fixture
def synthetic_cell_area(synthetic_grid):
    """Cell area in km2: uniform 1000 km2 per cell."""
    return xr.full_like(synthetic_grid, 1000.0)


@pytest.fixture
def synthetic_time_grid():
    """4x4 grid with a time dimension (5 decadal steps)."""
    lat = np.array([-1.5, -0.5, 0.5, 1.5])
    lon = np.array([0.5, 1.5, 2.5, 3.5])
    years = [1000, 1200, 1400, 1600, 1800]
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 100, size=(5, 4, 4))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": years, "lat": lat, "lon": lon},
    )


@pytest.fixture
def synthetic_country_panel():
    """Country x year panel in the same format as hyde35_country_year_mean_std.csv."""
    rows = []
    for country in ["FRA", "GBR", "CHN"]:
        for year in range(1000, 1900, 100):
            for var, units, val_base in [
                ("pop_persons", "persons", 1e6 * (1 + year / 1000)),
                ("popdens_p_km2", "persons/km2_land", 10.0 * (1 + year / 2000)),
                ("cropland_mha", "Mha", 0.5 * (1 + year / 1500)),
                ("grazing_mha", "Mha", 0.3 * (1 + year / 2000)),
                ("irrigation_share", "share", 0.05 * (1 + year / 3000)),
                ("rice_mha", "Mha", 0.1 if country == "CHN" else 0.01),
                ("urban_share", "share", 0.02 * (1 + year / 5000)),
            ]:
                rows.append({
                    "year": year,
                    "country": country,
                    "var": var,
                    "units": units,
                    "mean": val_base,
                    "std": val_base * 0.1,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_scenario_panel():
    """Region x year x scenario panel matching hyde35_panel_region_year_by_scenario.csv."""
    rows = []
    for scenario in ["base", "lower", "upper"]:
        for region in ["Europe", "East Asia"]:
            for year in range(1000, 1900, 100):
                mult = {"base": 1.0, "lower": 0.85, "upper": 1.15}[scenario]
                for var, units, val in [
                    ("pop", "persons", 1e6 * (1 + year / 1000) * mult),
                    ("popdens_p_km2", "persons/km2_land", 10 * mult),
                    ("nonrice_cropland_mha", "Mha", 0.5 * mult),
                    ("grazing_mha", "Mha", 0.3 * mult),
                ]:
                    rows.append({
                        "scenario": scenario,
                        "year": year,
                        "region": region,
                        "var": var,
                        "units": units,
                        "value": val,
                    })
    return pd.DataFrame(rows)
