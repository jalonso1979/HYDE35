# UGT x HYDE 3.5 Research Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-paper research pipeline testing Unified Growth Theory using HYDE 3.5 land-use/population data and ERA5 climate data, progressing from descriptive typologies through Malthusian econometrics to causal climate identification.

**Architecture:** Shared Python module (`analysis/shared/`) provides data loading, variable construction, and utilities. Each paper gets its own directory with Jupyter notebooks for exploration and `.py` scripts for reproducible pipelines. All derived datasets land in `analysis/data/` as Parquet files. Existing notebook patterns (xarray + pandas + scenario iteration) are preserved and modularized.

**Tech Stack:** Python 3.10+, numpy, pandas, xarray, scipy, statsmodels, scikit-learn, lifelines (survival), pysal/libpysal (spatial), matplotlib/seaborn, pycountry, netCDF4

---

## File Structure

```
HYDE35/
├── analysis/
│   ├── shared/
│   │   ├── __init__.py              # Package init
│   │   ├── config.py                # Paths, constants, scenario labels
│   │   ├── loaders.py               # NetCDF, ASCII grid, CSV loaders
│   │   ├── masks.py                 # Region/country mask construction
│   │   ├── variables.py             # Derived variable construction
│   │   ├── uncertainty.py           # Scenario aggregation, Manski bounds
│   │   └── plotting.py              # Shared plotting utilities
│   ├── data/                        # Analysis-ready Parquet outputs
│   ├── paper1_escape/
│   │   ├── trajectories.py          # Transition trajectory extraction
│   │   ├── clustering.py            # Pathway clustering
│   │   ├── survival.py              # Time-to-exit survival analysis
│   │   ├── spatial.py               # Spatial diffusion tests
│   │   └── paper1_explore.ipynb     # Exploration notebook
│   ├── paper2_malthus/
│   │   ├── panels.py                # Malthusian panel construction
│   │   ├── regressions.py           # Core regressions + rolling windows
│   │   ├── breaks.py                # Structural break detection
│   │   ├── bounds.py                # Partial identification bounds
│   │   └── paper2_explore.ipynb     # Exploration notebook
│   └── paper3_climate/
│       ├── climate_shocks.py        # ERA5 anomaly construction
│       ├── local_projections.py     # Jordà IRFs
│       ├── regime_switching.py      # Threshold/STAR models
│       ├── counterfactuals.py       # Counterfactual simulations
│       └── paper3_explore.ipynb     # Exploration notebook
└── tests/
    ├── conftest.py                  # Shared fixtures (small synthetic data)
    ├── test_loaders.py
    ├── test_masks.py
    ├── test_variables.py
    ├── test_uncertainty.py
    ├── test_trajectories.py
    ├── test_clustering.py
    ├── test_survival.py
    ├── test_panels.py
    ├── test_regressions.py
    ├── test_climate_shocks.py
    └── test_local_projections.py
```

---

## Phase 1: Shared Data Infrastructure

### Task 1: Project Scaffolding and Configuration

**Files:**
- Create: `analysis/__init__.py`
- Create: `analysis/shared/__init__.py`
- Create: `analysis/shared/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml with dependencies**

```toml
[project]
name = "hyde35-ugt"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "xarray>=2023.1",
    "scipy>=1.10",
    "statsmodels>=0.14",
    "scikit-learn>=1.3",
    "lifelines>=0.27",
    "libpysal>=4.7",
    "esda>=2.5",
    "spreg>=1.3",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pycountry>=22.3",
    "netCDF4>=1.6",
    "pyarrow>=12.0",
    "pytest>=7.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create config.py**

```python
"""Paths, constants, and scenario definitions for HYDE35 analysis."""
from pathlib import Path

# Root data directory
DATA_ROOT = Path("/Volumes/BIGDATA/HYDE35")

# Scenario directories
SCENARIOS = {
    "base": DATA_ROOT / "gbc2025_7apr_base",
    "lower": DATA_ROOT / "gbc2025_7apr_lower",
    "upper": DATA_ROOT / "gbc2025_7apr_upper",
}
SCENARIO_LABELS = list(SCENARIOS.keys())

# NetCDF subdirectory
NC_SUBDIR = "NetCDF"

# NetCDF variable file stems (without .nc)
NC_VARIABLES = [
    "population", "population_density", "urban_population", "rural_population",
    "cropland", "rainfed_rice", "irrigated_rice", "rainfed_not_rice",
    "irrigated_not_rice", "total_rice", "total_rainfed", "total_irrigated",
    "grazing_land", "pasture", "rangeland", "urban_area",
]

# Reference grids
GENERAL_FILES = DATA_ROOT / "general_files" / "general_files"
GRID_AREA_PATH = GENERAL_FILES / "garea_cr.asc"
ISO_GRID_PATH = GENERAL_FILES / "iso_cr.asc"
LAND_MASK_PATH = GENERAL_FILES / "landlake.asc"
REGION_GRID_PATH = GENERAL_FILES / "im_reg_cr.asc"

# ERA5
ERA5_ROOT = DATA_ROOT / "ERA5"

# Existing processed CSVs
COUNTRY_YEAR_CSV = DATA_ROOT / "hyde35_country_year_mean_std.csv"
REGION_YEAR_CSV = DATA_ROOT / "hyde35_panel_region_year_by_scenario.csv"
ANTHRO_REGION_CSV = DATA_ROOT / "hyde35_anthropogenic_region_year_by_scenario.csv"
COUNTRY_ISO_MAP_CSV = DATA_ROOT / "hyde35_country_iso_mapping.csv"
REGION_DEFS_JSON = DATA_ROOT / "hyde35_region_defs.json"
VARIABLE_UNITS_JSON = DATA_ROOT / "hyde35_country_variable_units.json"

# Output directory for analysis-ready data
ANALYSIS_DATA = DATA_ROOT / "analysis" / "data"
ANALYSIS_DATA.mkdir(parents=True, exist_ok=True)

# Unit conversions
HUNDRED_HA_TO_MHA = 1.0 / 10.0  # HYDE native 100-ha grids to Mha

# Minimum land area threshold for country inclusion (km2)
MIN_COUNTRY_AREA_KM2 = 500

# Grazing caloric weight for agricultural output proxy
GRAZING_CALORIC_WEIGHT = 0.5
```

- [ ] **Step 3: Create package init files**

`analysis/__init__.py`:
```python
"""HYDE 3.5 UGT research analysis package."""
```

`analysis/shared/__init__.py`:
```python
"""Shared utilities for HYDE 3.5 analysis."""
from analysis.shared.config import DATA_ROOT, SCENARIOS, SCENARIO_LABELS
```

- [ ] **Step 4: Create test conftest with synthetic fixtures**

```python
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
```

- [ ] **Step 5: Create directory structure and empty init files**

```bash
mkdir -p /Volumes/BIGDATA/HYDE35/analysis/{shared,data,paper1_escape,paper2_malthus,paper3_climate}
mkdir -p /Volumes/BIGDATA/HYDE35/tests
touch /Volumes/BIGDATA/HYDE35/tests/__init__.py
touch /Volumes/BIGDATA/HYDE35/analysis/paper1_escape/__init__.py
touch /Volumes/BIGDATA/HYDE35/analysis/paper2_malthus/__init__.py
touch /Volumes/BIGDATA/HYDE35/analysis/paper3_climate/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git init  # if not already a repo
git add pyproject.toml analysis/ tests/
git commit -m "feat: scaffold project structure, config, and test fixtures"
```

---

### Task 2: Data Loaders

**Files:**
- Create: `analysis/shared/loaders.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests for loaders**

```python
"""Tests for data loading functions."""
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from unittest.mock import patch
from analysis.shared.loaders import (
    read_esri_ascii_grid,
    load_nc_variable,
    load_existing_country_panel,
    load_existing_scenario_panel,
    normalize_longitudes,
)


def test_normalize_longitudes_360_to_180():
    lon = np.array([0.5, 179.5, 180.5, 359.5])
    result = normalize_longitudes(lon)
    expected = np.array([-179.5, -0.5, 0.5, 179.5])
    # After sorting, should be -180 to 180 range
    assert result.min() >= -180.0
    assert result.max() <= 180.0


def test_normalize_longitudes_already_180():
    lon = np.array([-179.5, -0.5, 0.5, 179.5])
    result = normalize_longitudes(lon)
    np.testing.assert_array_equal(result, lon)


def test_load_existing_country_panel(tmp_path):
    df = pd.DataFrame({
        "year": [1000, 1000],
        "country": ["FRA", "FRA"],
        "var": ["pop_persons", "cropland_mha"],
        "units": ["persons", "Mha"],
        "mean": [1e6, 0.5],
        "std": [1e5, 0.05],
    })
    path = tmp_path / "country.csv"
    df.to_csv(path, index=False)
    result = load_existing_country_panel(path)
    assert list(result.columns) == ["year", "country", "var", "units", "mean", "std"]
    assert len(result) == 2


def test_load_existing_scenario_panel(tmp_path):
    df = pd.DataFrame({
        "scenario": ["base", "lower"],
        "year": [1000, 1000],
        "region": ["Europe", "Europe"],
        "var": ["pop", "pop"],
        "units": ["persons", "persons"],
        "value": [1e6, 0.85e6],
    })
    path = tmp_path / "scenario.csv"
    df.to_csv(path, index=False)
    result = load_existing_scenario_panel(path)
    assert "scenario" in result.columns
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_loaders.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.shared.loaders'`

- [ ] **Step 3: Implement loaders.py**

```python
"""Data loading functions for HYDE 3.5 NetCDF, ASCII grids, and CSVs."""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


def normalize_longitudes(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from 0..360 to -180..180 if needed, then sort."""
    lon = np.asarray(lon, dtype="float64")
    if lon.max() > 180.0:
        lon = np.where(lon > 180.0, lon - 360.0, lon)
    return np.sort(lon)


def read_esri_ascii_grid(path: Path) -> xr.DataArray:
    """Read an ESRI ASCII grid file into an xarray DataArray.

    Parses the 6-line header (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value)
    then reads the data block.
    """
    path = Path(path)
    header = {}
    with open(path) as f:
        for _ in range(6):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1])

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cellsize = header["cellsize"]
    nodata = header["nodata_value"]

    data = np.loadtxt(path, skiprows=6, dtype="float64")
    data[data == nodata] = np.nan

    lon = np.arange(xll + cellsize / 2, xll + ncols * cellsize, cellsize)
    lat = np.arange(yll + (nrows - 0.5) * cellsize, yll, -cellsize)

    return xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": lat[:nrows], "lon": lon[:ncols]},
    )


def align_grid(da: xr.DataArray, template: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Align a DataArray's lon/lat to match a template grid using nearest reindex."""
    t_lon = template["lon"].values if hasattr(template, "lon") else template.coords["lon"].values
    t_lat = template["lat"].values if hasattr(template, "lat") else template.coords["lat"].values

    # Normalize longitudes to match template convention
    da_lon = da["lon"].values.copy()
    if t_lon.min() < 0 and da_lon.min() >= 0:
        da_lon = normalize_longitudes(da_lon)
        da = da.assign_coords(lon=da_lon).sortby("lon")
    elif t_lon.min() >= 0 and da_lon.min() < 0:
        da_lon = np.where(da_lon < 0, da_lon + 360.0, da_lon)
        da = da.assign_coords(lon=da_lon).sortby("lon")

    return da.reindex(lat=t_lat, lon=t_lon, method="nearest")


def load_nc_variable(scenario_dir: Path, variable: str) -> xr.DataArray:
    """Load a single NetCDF variable from a scenario directory.

    Returns the first data variable found in the file as a DataArray
    with dimensions (time, lat, lon).
    """
    nc_path = scenario_dir / "NetCDF" / f"{variable}.nc"
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    # Get the first (and typically only) data variable
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    # Normalize longitudes to -180..180
    if da["lon"].values.max() > 180.0:
        new_lon = normalize_longitudes(da["lon"].values)
        da = da.assign_coords(lon=("lon", normalize_longitudes(da["lon"].values)))
        da = da.sortby("lon")

    return da


def load_existing_country_panel(path: Path) -> pd.DataFrame:
    """Load the pre-built country x year panel CSV."""
    df = pd.read_csv(path)
    expected_cols = ["year", "country", "var", "units", "mean", "std"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    return df


def load_existing_scenario_panel(path: Path) -> pd.DataFrame:
    """Load the pre-built scenario x region x year panel CSV."""
    df = pd.read_csv(path)
    expected_cols = ["scenario", "year", "region", "var", "units", "value"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_loaders.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/shared/loaders.py tests/test_loaders.py
git commit -m "feat: add data loaders for NetCDF, ASCII grids, and CSVs"
```

---

### Task 3: Mask Construction

**Files:**
- Create: `analysis/shared/masks.py`
- Create: `tests/test_masks.py`

- [ ] **Step 1: Write failing tests for masks**

```python
"""Tests for region and country mask construction."""
import numpy as np
import xarray as xr
import pytest
from analysis.shared.masks import build_bbox_mask, build_country_mask


def test_build_bbox_mask(synthetic_grid, synthetic_land_mask):
    """Bounding box that covers the upper-right quadrant of our 4x4 grid."""
    bbox = {"lat_min": 0.0, "lat_max": 2.0, "lon_min": 2.0, "lon_max": 4.0}
    mask = build_bbox_mask(synthetic_grid, bbox, synthetic_land_mask)
    # Should select cells at lat=[0.5,1.5], lon=[2.5,3.5] = 4 cells
    assert mask.sum().item() == 4.0
    # Cells outside bbox should be zero
    assert mask.sel(lat=-1.5, lon=0.5).item() == 0.0


def test_build_bbox_mask_with_subtract(synthetic_grid, synthetic_land_mask):
    """Subtract a sub-box from the main bbox."""
    bbox = {"lat_min": -2.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 4.0}
    subtract = [{"lat_min": 0.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 4.0}]
    mask = build_bbox_mask(synthetic_grid, bbox, synthetic_land_mask, subtract_boxes=subtract)
    # Full grid minus ocean cell = 15, upper half (4 cells) subtracted = 11
    # But upper half has 4 cells at lat=[0.5,1.5], so 15 - 4 = 11
    # Wait: lat[-1.5,-0.5] x lon[0.5..3.5] = 8 cells, minus 1 ocean = 7
    assert mask.sel(lat=1.5, lon=3.5).item() == 0.0  # subtracted
    assert mask.sel(lat=-0.5, lon=1.5).item() == 1.0  # kept


def test_build_country_mask(synthetic_grid, synthetic_land_mask):
    """Country grid with two countries (id=1 and id=2)."""
    country_grid = xr.full_like(synthetic_grid, 1.0)
    country_grid[2:, :] = 2.0
    country_grid[0, 0] = np.nan  # ocean
    mask_1 = build_country_mask(country_grid, country_id=1, land_mask=synthetic_land_mask)
    mask_2 = build_country_mask(country_grid, country_id=2, land_mask=synthetic_land_mask)
    # Country 1: rows 0-1, minus 1 ocean cell = 7
    assert mask_1.sum().item() == 7.0
    # Country 2: rows 2-3 = 8 cells
    assert mask_2.sum().item() == 8.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_masks.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement masks.py**

```python
"""Region and country mask construction for HYDE 3.5 grids."""
import numpy as np
import xarray as xr
from typing import Optional


def build_bbox_mask(
    template: xr.DataArray,
    bbox: dict,
    land_mask: xr.DataArray,
    subtract_boxes: Optional[list[dict]] = None,
) -> xr.DataArray:
    """Build a binary mask from a lat/lon bounding box.

    Args:
        template: DataArray whose lat/lon coordinates define the grid.
        bbox: Dict with keys lat_min, lat_max, lon_min, lon_max.
        land_mask: Binary land/ocean mask (1=land, 0=ocean).
        subtract_boxes: Optional list of bbox dicts to exclude from the mask.

    Returns:
        Binary DataArray (1=in region, 0=outside).
    """
    lat = template["lat"]
    lon = template["lon"]

    mask = (
        (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"])
        & (lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"])
    )
    mask = xr.where(mask, 1.0, 0.0) * land_mask

    if subtract_boxes:
        for sub in subtract_boxes:
            sub_mask = (
                (lat >= sub["lat_min"]) & (lat <= sub["lat_max"])
                & (lon >= sub["lon_min"]) & (lon <= sub["lon_max"])
            )
            mask = mask * xr.where(sub_mask, 0.0, 1.0)

    return mask


def build_country_mask(
    country_grid: xr.DataArray,
    country_id: int | float,
    land_mask: xr.DataArray,
) -> xr.DataArray:
    """Build a binary mask for a single country from the ISO numeric grid.

    Args:
        country_grid: DataArray with ISO numeric country codes per cell.
        country_id: The numeric country code to select.
        land_mask: Binary land/ocean mask.

    Returns:
        Binary DataArray (1=in country, 0=outside).
    """
    mask = xr.where(country_grid == country_id, 1.0, 0.0)
    return mask * land_mask


def build_region_masks_from_json(
    region_defs: dict,
    template: xr.DataArray,
    land_mask: xr.DataArray,
) -> xr.DataArray:
    """Build stacked region masks from the hyde35_region_defs.json structure.

    Args:
        region_defs: Dict mapping region names to bbox definitions.
        template: Grid template DataArray.
        land_mask: Binary land/ocean mask.

    Returns:
        DataArray with dims (region, lat, lon).
    """
    masks = []
    names = []
    for name, rdef in region_defs.items():
        bbox = {
            "lat_min": rdef["lat_min"],
            "lat_max": rdef["lat_max"],
            "lon_min": rdef["lon_min"],
            "lon_max": rdef["lon_max"],
        }
        subtract = rdef.get("subtract_boxes", None)
        m = build_bbox_mask(template, bbox, land_mask, subtract_boxes=subtract)
        masks.append(m)
        names.append(name)

    stacked = xr.concat(masks, dim="region")
    stacked["region"] = names
    return stacked
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_masks.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/shared/masks.py tests/test_masks.py
git commit -m "feat: add region and country mask construction"
```

---

### Task 4: Derived Variable Construction

**Files:**
- Create: `analysis/shared/variables.py`
- Create: `tests/test_variables.py`

- [ ] **Step 1: Write failing tests for derived variables**

```python
"""Tests for derived Malthusian and intensification variables."""
import numpy as np
import pandas as pd
import pytest
from analysis.shared.variables import (
    compute_land_labor_ratio,
    compute_ag_output_proxy,
    compute_intensification_index,
    compute_pop_growth_rate,
    pivot_country_panel_wide,
)


def test_compute_land_labor_ratio():
    cropland = np.array([10.0, 20.0, 5.0])
    grazing = np.array([5.0, 10.0, 15.0])
    population = np.array([1e6, 2e6, 0.5e6])
    result = compute_land_labor_ratio(cropland, grazing, population)
    expected = (cropland + grazing) / population
    np.testing.assert_allclose(result, expected)


def test_compute_land_labor_ratio_zero_pop():
    result = compute_land_labor_ratio(
        np.array([10.0]), np.array([5.0]), np.array([0.0])
    )
    assert np.isnan(result[0])


def test_compute_ag_output_proxy():
    cropland = np.array([10.0])
    grazing = np.array([6.0])
    result = compute_ag_output_proxy(cropland, grazing, grazing_weight=0.5)
    # 10 + 0.5 * 6 = 13
    np.testing.assert_allclose(result, [13.0])


def test_compute_intensification_index():
    irrigation_share = np.array([0.2, 0.0, 0.5])
    cropland = np.array([10.0, 0.0, 8.0])
    grazing = np.array([5.0, 10.0, 2.0])
    result = compute_intensification_index(irrigation_share, cropland, grazing)
    # irrigation_share * cropland / (cropland + grazing)
    expected = np.array([0.2 * 10 / 15, 0.0, 0.5 * 8 / 10])
    np.testing.assert_allclose(result, expected)


def test_compute_intensification_index_zero_land():
    result = compute_intensification_index(
        np.array([0.5]), np.array([0.0]), np.array([0.0])
    )
    assert np.isnan(result[0])


def test_compute_pop_growth_rate():
    pop = pd.Series([100.0, 110.0, 121.0, 133.1])
    years = pd.Series([1000, 1100, 1200, 1300])
    result = compute_pop_growth_rate(pop, years)
    # Annualized: (110/100)^(1/100) - 1 ≈ 0.000953
    assert len(result) == 4
    assert np.isnan(result.iloc[0])  # first value undefined
    np.testing.assert_allclose(result.iloc[1], (110 / 100) ** (1 / 100) - 1, rtol=1e-6)


def test_pivot_country_panel_wide(synthetic_country_panel):
    wide = pivot_country_panel_wide(synthetic_country_panel)
    assert "pop_persons_mean" in wide.columns
    assert "cropland_mha_mean" in wide.columns
    assert "year" in wide.columns
    assert "country" in wide.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_variables.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement variables.py**

```python
"""Derived variable construction for Malthusian and UGT analysis."""
import numpy as np
import pandas as pd


def compute_land_labor_ratio(
    cropland: np.ndarray, grazing: np.ndarray, population: np.ndarray
) -> np.ndarray:
    """Arable land per capita: (cropland + grazing) / population."""
    total_land = np.asarray(cropland) + np.asarray(grazing)
    pop = np.asarray(population, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pop > 0, total_land / pop, np.nan)
    return ratio


def compute_ag_output_proxy(
    cropland: np.ndarray,
    grazing: np.ndarray,
    grazing_weight: float = 0.5,
) -> np.ndarray:
    """Agricultural output proxy: cropland + grazing_weight * grazing.

    Grazing weight < 1 reflects lower caloric yield per hectare of grazing vs. cropland.
    """
    return np.asarray(cropland) + grazing_weight * np.asarray(grazing)


def compute_intensification_index(
    irrigation_share: np.ndarray,
    cropland: np.ndarray,
    grazing: np.ndarray,
) -> np.ndarray:
    """Intensification index: irrigation_share * (cropland / (cropland + grazing)).

    Captures both water management intensity and crop-vs-pastoral balance.
    Returns NaN where total agricultural land is zero.
    """
    crop = np.asarray(cropland, dtype="float64")
    graz = np.asarray(grazing, dtype="float64")
    total = crop + graz
    with np.errstate(divide="ignore", invalid="ignore"):
        crop_share = np.where(total > 0, crop / total, np.nan)
    return np.asarray(irrigation_share) * crop_share


def compute_pop_growth_rate(
    population: pd.Series, years: pd.Series
) -> pd.Series:
    """Annualized population growth rate between consecutive observations.

    Returns (pop_t / pop_{t-1})^(1 / dt) - 1, where dt is years between obs.
    First value is NaN.
    """
    pop = population.values.astype("float64")
    yrs = years.values.astype("float64")

    growth = np.full(len(pop), np.nan)
    for i in range(1, len(pop)):
        if pop[i - 1] > 0 and not np.isnan(pop[i - 1]):
            dt = yrs[i] - yrs[i - 1]
            if dt > 0:
                growth[i] = (pop[i] / pop[i - 1]) ** (1.0 / dt) - 1.0

    return pd.Series(growth, index=population.index)


def pivot_country_panel_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format country panel to wide format.

    Input columns: year, country, var, units, mean, std
    Output columns: year, country, {var}_mean, {var}_std for each variable.
    """
    mean_wide = df.pivot_table(
        index=["year", "country"], columns="var", values="mean"
    ).reset_index()
    mean_wide.columns = [
        f"{c}_mean" if c not in ("year", "country") else c
        for c in mean_wide.columns
    ]

    std_wide = df.pivot_table(
        index=["year", "country"], columns="var", values="std"
    ).reset_index()
    std_wide.columns = [
        f"{c}_std" if c not in ("year", "country") else c
        for c in std_wide.columns
    ]

    return mean_wide.merge(std_wide, on=["year", "country"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_variables.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/shared/variables.py tests/test_variables.py
git commit -m "feat: add derived Malthusian and intensification variable construction"
```

---

### Task 5: Uncertainty and Scenario Aggregation

**Files:**
- Create: `analysis/shared/uncertainty.py`
- Create: `tests/test_uncertainty.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for uncertainty quantification across HYDE scenarios."""
import numpy as np
import pandas as pd
import pytest
from analysis.shared.uncertainty import (
    compute_scenario_stats,
    compute_manski_bounds,
)


def test_compute_scenario_stats(synthetic_scenario_panel):
    result = compute_scenario_stats(synthetic_scenario_panel)
    assert "mean" in result.columns
    assert "std" in result.columns
    assert "se" in result.columns
    assert "scenario" not in result.columns
    # 2 regions x 9 years x 4 vars = 72 rows
    assert len(result) == len(synthetic_scenario_panel) // 3


def test_compute_scenario_stats_values():
    df = pd.DataFrame({
        "scenario": ["base", "lower", "upper"],
        "year": [1000, 1000, 1000],
        "region": ["Europe", "Europe", "Europe"],
        "var": ["pop", "pop", "pop"],
        "units": ["persons", "persons", "persons"],
        "value": [100.0, 80.0, 120.0],
    })
    result = compute_scenario_stats(df)
    row = result.iloc[0]
    np.testing.assert_allclose(row["mean"], 100.0)
    expected_std = np.std([100.0, 80.0, 120.0], ddof=0)
    np.testing.assert_allclose(row["std"], expected_std)
    np.testing.assert_allclose(row["se"], expected_std / np.sqrt(3))


def test_compute_manski_bounds():
    """Manski bounds from running a coefficient across 3 scenarios."""
    coefficients = {"base": -0.05, "lower": -0.08, "upper": -0.03}
    lb, ub = compute_manski_bounds(coefficients)
    assert lb == -0.08
    assert ub == -0.03
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_uncertainty.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement uncertainty.py**

```python
"""Scenario aggregation and uncertainty quantification."""
import numpy as np
import pandas as pd


def _std_ddof0(x: pd.Series) -> float:
    """Population standard deviation (ddof=0), matching HYDE convention."""
    return float(np.std(x.to_numpy(dtype="float64"), ddof=0))


def compute_scenario_stats(
    scenario_panel: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, std, and standard error across scenarios.

    Args:
        scenario_panel: DataFrame with columns [scenario, year, region/country, var, units, value].
        group_cols: Columns to group by (excluding 'scenario'). Defaults to
                    all columns except 'scenario' and 'value'.

    Returns:
        DataFrame with columns [group_cols..., mean, std, se].
    """
    if group_cols is None:
        group_cols = [c for c in scenario_panel.columns if c not in ("scenario", "value")]

    agg = scenario_panel.groupby(group_cols, as_index=False)["value"].agg(
        mean="mean",
        std=_std_ddof0,
    )
    # Standard error = std / sqrt(n_scenarios)
    n_scenarios = scenario_panel["scenario"].nunique()
    agg["se"] = agg["std"] / np.sqrt(n_scenarios)
    return agg


def compute_manski_bounds(
    coefficients: dict[str, float],
) -> tuple[float, float]:
    """Compute Manski-style bounds from coefficients estimated across scenarios.

    Args:
        coefficients: Dict mapping scenario label to estimated coefficient.

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    vals = list(coefficients.values())
    return min(vals), max(vals)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_uncertainty.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/shared/uncertainty.py tests/test_uncertainty.py
git commit -m "feat: add scenario aggregation and Manski bounds"
```

---

### Task 6: Build Analysis-Ready Panel

**Files:**
- Create: `analysis/shared/build_panels.py`

This task reads existing CSVs, constructs all derived variables, and writes the analysis-ready Parquet files that all three papers consume.

- [ ] **Step 1: Implement build_panels.py**

```python
"""Build analysis-ready panel datasets from existing HYDE35 CSVs.

Reads the pre-processed country and region panels, pivots to wide format,
computes derived Malthusian variables, and writes Parquet files.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.shared.config import (
    COUNTRY_YEAR_CSV,
    REGION_YEAR_CSV,
    ANTHRO_REGION_CSV,
    ANALYSIS_DATA,
    GRAZING_CALORIC_WEIGHT,
)
from analysis.shared.variables import (
    compute_land_labor_ratio,
    compute_ag_output_proxy,
    compute_intensification_index,
    compute_pop_growth_rate,
    pivot_country_panel_wide,
)
from analysis.shared.loaders import load_existing_country_panel, load_existing_scenario_panel
from analysis.shared.uncertainty import compute_scenario_stats


def build_country_analysis_panel() -> pd.DataFrame:
    """Build the country-level analysis panel with derived variables.

    Reads hyde35_country_year_mean_std.csv, pivots wide, and adds:
    - land_labor_ratio: (cropland + grazing) / population
    - ag_output_proxy: cropland + 0.5 * grazing
    - intensification_index: irrigation_share * cropland / (cropland + grazing)
    - pop_growth_rate: annualized growth between observations
    - scenario_spread: std column as uncertainty measure
    """
    df_long = load_existing_country_panel(COUNTRY_YEAR_CSV)
    df = pivot_country_panel_wide(df_long)

    # Rename columns to simpler forms for analysis
    # The pivot produces {var}_mean and {var}_std columns
    # Use _mean values for computation, keep _std for uncertainty

    cropland = df.get("nonrice_mha_mean", pd.Series(0.0, index=df.index))
    rice = df.get("rice_mha_mean", pd.Series(0.0, index=df.index))
    total_cropland = cropland + rice
    grazing = df.get("grazing_mha_mean", pd.Series(0.0, index=df.index))
    pop = df.get("pop_persons_mean", pd.Series(0.0, index=df.index))
    irr_share = df.get("irrigation_share_mean", pd.Series(0.0, index=df.index))

    df["land_labor_ratio"] = compute_land_labor_ratio(
        total_cropland.values, grazing.values, pop.values
    )
    df["ag_output_proxy_mha"] = compute_ag_output_proxy(
        total_cropland.values, grazing.values, grazing_weight=GRAZING_CALORIC_WEIGHT
    )
    df["intensification_index"] = compute_intensification_index(
        irr_share.values, total_cropland.values, grazing.values
    )

    # Population growth rate per country
    df = df.sort_values(["country", "year"])
    growth_parts = []
    for _, grp in df.groupby("country"):
        g = compute_pop_growth_rate(grp["pop_persons_mean"], grp["year"])
        growth_parts.append(g)
    df["pop_growth_rate"] = pd.concat(growth_parts)

    return df


def build_region_analysis_panel() -> pd.DataFrame:
    """Build the region-level analysis panel with scenario stats and derived vars.

    Reads the scenario panel, computes mean/std/SE across scenarios,
    pivots wide, and adds derived variables.
    """
    df_scen = load_existing_scenario_panel(REGION_YEAR_CSV)
    df_stats = compute_scenario_stats(df_scen)

    # Pivot to wide: one row per (year, region)
    mean_wide = df_stats.pivot_table(
        index=["year", "region"], columns="var", values="mean"
    ).reset_index()
    mean_wide.columns = [
        f"{c}_mean" if c not in ("year", "region") else c
        for c in mean_wide.columns
    ]

    std_wide = df_stats.pivot_table(
        index=["year", "region"], columns="var", values="std"
    ).reset_index()
    std_wide.columns = [
        f"{c}_std" if c not in ("year", "region") else c
        for c in std_wide.columns
    ]

    se_wide = df_stats.pivot_table(
        index=["year", "region"], columns="var", values="se"
    ).reset_index()
    se_wide.columns = [
        f"{c}_se" if c not in ("year", "region") else c
        for c in se_wide.columns
    ]

    df = mean_wide.merge(std_wide, on=["year", "region"]).merge(se_wide, on=["year", "region"])

    # Derived variables using _mean columns
    cropland = df.get("nonrice_cropland_mha_mean", pd.Series(0.0, index=df.index))
    rice = df.get("rice_mha_mean", pd.Series(0.0, index=df.index))
    total_cropland = cropland + rice
    grazing = df.get("grazing_mha_mean", pd.Series(0.0, index=df.index))
    pop = df.get("pop_mean", pd.Series(0.0, index=df.index))

    df["land_labor_ratio"] = compute_land_labor_ratio(
        total_cropland.values, grazing.values, pop.values
    )
    df["ag_output_proxy_mha"] = compute_ag_output_proxy(
        total_cropland.values, grazing.values, grazing_weight=GRAZING_CALORIC_WEIGHT
    )

    # Growth rate per region
    df = df.sort_values(["region", "year"])
    growth_parts = []
    for _, grp in df.groupby("region"):
        g = compute_pop_growth_rate(grp["pop_mean"], grp["year"])
        growth_parts.append(g)
    df["pop_growth_rate"] = pd.concat(growth_parts)

    return df


def save_panels():
    """Build and save all analysis-ready panels to Parquet."""
    print("Building country analysis panel...")
    country_df = build_country_analysis_panel()
    country_path = ANALYSIS_DATA / "country_analysis_panel.parquet"
    country_df.to_parquet(country_path, index=False)
    print(f"  Saved {len(country_df)} rows to {country_path}")

    print("Building region analysis panel...")
    region_df = build_region_analysis_panel()
    region_path = ANALYSIS_DATA / "region_analysis_panel.parquet"
    region_df.to_parquet(region_path, index=False)
    print(f"  Saved {len(region_df)} rows to {region_path}")

    print("Done.")


if __name__ == "__main__":
    save_panels()
```

- [ ] **Step 2: Run the panel builder against real data**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m analysis.shared.build_panels
```

Expected: Prints row counts, creates two Parquet files in `analysis/data/`.

- [ ] **Step 3: Verify output files**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -c "
import pandas as pd
for f in ['analysis/data/country_analysis_panel.parquet', 'analysis/data/region_analysis_panel.parquet']:
    df = pd.read_parquet(f)
    print(f'{f}: {df.shape}, cols={list(df.columns)[:10]}...')
    print(f'  year range: {df.year.min()} - {df.year.max()}')
    print(f'  nulls: {df.isnull().sum().sum()}')
"
```

- [ ] **Step 4: Commit**

```bash
git add analysis/shared/build_panels.py
git commit -m "feat: build analysis-ready country and region panels with derived vars"
```

---

## Phase 2: Paper 1 — The Geography of the Great Escape

### Task 7: Transition Trajectory Extraction

**Files:**
- Create: `analysis/paper1_escape/trajectories.py`
- Create: `tests/test_trajectories.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for transition trajectory feature extraction."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper1_escape.trajectories import (
    detect_peak_extensification,
    detect_intensification_onset,
    detect_urbanization_takeoff,
    extract_trajectory_features,
)


def test_detect_peak_extensification():
    """Peak extensification = year when land area growth rate is highest."""
    years = np.array([1000, 1100, 1200, 1300, 1400, 1500])
    # Land area grows fastest between 1200-1300
    land_area = np.array([10.0, 12.0, 15.0, 25.0, 28.0, 29.0])
    peak_year = detect_peak_extensification(years, land_area)
    assert peak_year == 1300


def test_detect_peak_extensification_monotonic_decline():
    """If land area always shrinks, return NaN."""
    years = np.array([1000, 1100, 1200])
    land_area = np.array([30.0, 20.0, 10.0])
    # Still has a "peak" growth rate (least negative)
    peak_year = detect_peak_extensification(years, land_area)
    assert peak_year == 1100  # -10 vs -10, picks first


def test_detect_intensification_onset():
    """Onset = first year when intensification index exceeds threshold while land area stabilizes."""
    years = np.array([1000, 1100, 1200, 1300, 1400])
    intens_index = np.array([0.01, 0.02, 0.05, 0.12, 0.18])
    land_growth = np.array([np.nan, 0.05, 0.03, 0.005, 0.002])
    onset = detect_intensification_onset(
        years, intens_index, land_growth,
        intens_threshold=0.05, land_growth_threshold=0.01,
    )
    assert onset == 1300  # first year with intens >= 0.05 AND land_growth < 0.01


def test_detect_intensification_onset_never():
    years = np.array([1000, 1100, 1200])
    intens_index = np.array([0.01, 0.02, 0.03])
    land_growth = np.array([np.nan, 0.05, 0.04])
    onset = detect_intensification_onset(years, intens_index, land_growth)
    assert np.isnan(onset)


def test_detect_urbanization_takeoff():
    years = np.array([1000, 1100, 1200, 1300, 1400])
    urban_share = np.array([0.01, 0.02, 0.03, 0.06, 0.12])
    takeoff = detect_urbanization_takeoff(years, urban_share, threshold=0.05)
    assert takeoff == 1300


def test_extract_trajectory_features():
    """Integration test: extract all features for a single country."""
    df = pd.DataFrame({
        "year": [1000, 1100, 1200, 1300, 1400, 1500],
        "country": "FRA",
        "pop_persons_mean": [1e6, 1.2e6, 1.5e6, 2.5e6, 3.0e6, 3.2e6],
        "popdens_p_km2_mean": [10, 12, 15, 25, 30, 32],
        "land_labor_ratio": [0.015, 0.013, 0.010, 0.006, 0.005, 0.005],
        "ag_output_proxy_mha": [10, 12, 15, 20, 21, 21.5],
        "intensification_index": [0.01, 0.02, 0.05, 0.12, 0.18, 0.22],
        "urban_share_mean": [0.01, 0.02, 0.03, 0.06, 0.12, 0.18],
    })
    features = extract_trajectory_features(df, entity_col="country")
    assert len(features) == 1
    row = features.iloc[0]
    assert row["country"] == "FRA"
    assert "peak_extensification_year" in features.columns
    assert "intensification_onset_year" in features.columns
    assert "urbanization_takeoff_year" in features.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_trajectories.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement trajectories.py**

```python
"""Transition trajectory feature extraction for Paper 1."""
import numpy as np
import pandas as pd


def detect_peak_extensification(
    years: np.ndarray, land_area: np.ndarray
) -> float:
    """Find year when land area growth rate was highest.

    Returns the year corresponding to the maximum absolute increase
    in total agricultural land area. Returns NaN if fewer than 2 observations.
    """
    if len(years) < 2:
        return np.nan
    growth = np.diff(land_area) / np.diff(years)
    idx = np.argmax(growth)
    return float(years[idx + 1])


def detect_intensification_onset(
    years: np.ndarray,
    intensification_index: np.ndarray,
    land_growth_rate: np.ndarray,
    intens_threshold: float = 0.05,
    land_growth_threshold: float = 0.01,
) -> float:
    """Find first year when intensification exceeds threshold while land area stabilizes.

    Args:
        years: Year values.
        intensification_index: Intensification index values.
        land_growth_rate: Annual land area growth rate (can contain leading NaN).
        intens_threshold: Minimum intensification index to qualify.
        land_growth_threshold: Maximum land growth rate to qualify as "stabilized."

    Returns:
        Year of onset, or NaN if never reached.
    """
    for i in range(len(years)):
        if np.isnan(land_growth_rate[i]):
            continue
        if (intensification_index[i] >= intens_threshold
                and land_growth_rate[i] < land_growth_threshold):
            return float(years[i])
    return np.nan


def detect_urbanization_takeoff(
    years: np.ndarray,
    urban_share: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Find first year when urban share exceeds threshold.

    Returns NaN if never reached.
    """
    for i in range(len(years)):
        if urban_share[i] >= threshold:
            return float(years[i])
    return np.nan


def _compute_land_growth_rate(years: np.ndarray, land_area: np.ndarray) -> np.ndarray:
    """Annualized land area growth rate."""
    growth = np.full(len(years), np.nan)
    for i in range(1, len(years)):
        dt = years[i] - years[i - 1]
        if dt > 0 and land_area[i - 1] > 0:
            growth[i] = (land_area[i] - land_area[i - 1]) / (land_area[i - 1] * dt)
    return growth


def extract_trajectory_features(
    df: pd.DataFrame,
    entity_col: str = "country",
) -> pd.DataFrame:
    """Extract trajectory features for each entity (country or region).

    Args:
        df: Wide-format panel with columns: year, entity_col, pop_persons_mean,
            popdens_p_km2_mean, land_labor_ratio, ag_output_proxy_mha,
            intensification_index, urban_share_mean.
        entity_col: Column identifying the entity (country or region).

    Returns:
        DataFrame with one row per entity and columns:
        entity_col, peak_extensification_year, intensification_onset_year,
        urbanization_takeoff_year, max_pop_density, min_land_labor_ratio.
    """
    results = []
    for entity, grp in df.sort_values("year").groupby(entity_col):
        years = grp["year"].values
        land_area = grp["ag_output_proxy_mha"].values
        intens = grp["intensification_index"].values
        urban = grp["urban_share_mean"].values
        popdens = grp["popdens_p_km2_mean"].values
        llr = grp["land_labor_ratio"].values

        land_growth = _compute_land_growth_rate(years, land_area)

        results.append({
            entity_col: entity,
            "peak_extensification_year": detect_peak_extensification(years, land_area),
            "intensification_onset_year": detect_intensification_onset(
                years, intens, land_growth
            ),
            "urbanization_takeoff_year": detect_urbanization_takeoff(years, urban),
            "max_pop_density": float(np.nanmax(popdens)),
            "min_land_labor_ratio": float(np.nanmin(llr)),
        })

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_trajectories.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper1_escape/trajectories.py tests/test_trajectories.py
git commit -m "feat(paper1): add transition trajectory feature extraction"
```

---

### Task 8: Pathway Clustering

**Files:**
- Create: `analysis/paper1_escape/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for agricultural pathway clustering."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper1_escape.clustering import (
    prepare_clustering_features,
    cluster_pathways,
    label_clusters,
)


@pytest.fixture
def trajectory_features():
    return pd.DataFrame({
        "country": ["FRA", "GBR", "CHN", "EGY", "BRA", "IND"],
        "peak_extensification_year": [1200, 1100, 800, 500, 1500, 600],
        "intensification_onset_year": [1400, 1300, 1000, 700, np.nan, 900],
        "urbanization_takeoff_year": [1500, 1400, 1200, 1000, np.nan, 1100],
        "max_pop_density": [50, 60, 120, 80, 10, 100],
        "min_land_labor_ratio": [0.005, 0.004, 0.002, 0.003, 0.02, 0.003],
    })


def test_prepare_clustering_features(trajectory_features):
    X, entities = prepare_clustering_features(trajectory_features, entity_col="country")
    # Should drop rows with NaN (BRA)
    assert len(entities) == 5
    assert "BRA" not in entities.values
    # Features should be standardized (mean ~0, std ~1)
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.1)


def test_cluster_pathways(trajectory_features):
    X, entities = prepare_clustering_features(trajectory_features, entity_col="country")
    labels, metrics = cluster_pathways(X, n_clusters=2)
    assert len(labels) == len(entities)
    assert set(labels).issubset({0, 1})
    assert "silhouette" in metrics
    assert "inertia" in metrics


def test_label_clusters(trajectory_features):
    X, entities = prepare_clustering_features(trajectory_features, entity_col="country")
    labels, _ = cluster_pathways(X, n_clusters=2)
    result = label_clusters(entities, labels, entity_col="country")
    assert "cluster" in result.columns
    assert len(result) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_clustering.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement clustering.py**

```python
"""Agricultural pathway clustering for Paper 1."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


CLUSTERING_FEATURES = [
    "peak_extensification_year",
    "intensification_onset_year",
    "urbanization_takeoff_year",
    "max_pop_density",
    "min_land_labor_ratio",
]


def prepare_clustering_features(
    trajectory_df: pd.DataFrame,
    entity_col: str = "country",
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Prepare and standardize features for clustering.

    Drops rows with NaN in any feature column, then standardizes.

    Args:
        trajectory_df: Output of extract_trajectory_features.
        entity_col: Column identifying the entity.
        feature_cols: Columns to use as features. Defaults to CLUSTERING_FEATURES.

    Returns:
        (X_scaled, entities): Standardized feature matrix and corresponding entity labels.
    """
    if feature_cols is None:
        feature_cols = CLUSTERING_FEATURES

    df = trajectory_df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].values
    entities = df[entity_col].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, entities


def cluster_pathways(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Run K-means clustering on prepared features.

    Args:
        X: Standardized feature matrix (n_samples, n_features).
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        (labels, metrics): Cluster labels and dict with silhouette score and inertia.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels) if n_clusters > 1 else 0.0

    return labels, {"silhouette": sil, "inertia": km.inertia_}


def label_clusters(
    entities: pd.Series,
    labels: np.ndarray,
    entity_col: str = "country",
) -> pd.DataFrame:
    """Combine entity identifiers with cluster labels.

    Returns:
        DataFrame with columns [entity_col, cluster].
    """
    return pd.DataFrame({entity_col: entities.values, "cluster": labels})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_clustering.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper1_escape/clustering.py tests/test_clustering.py
git commit -m "feat(paper1): add pathway clustering with K-means"
```

---

### Task 9: Survival Analysis

**Files:**
- Create: `analysis/paper1_escape/survival.py`
- Create: `tests/test_survival.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for time-to-Malthusian-exit survival analysis."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper1_escape.survival import (
    build_survival_dataset,
    fit_cox_model,
)


@pytest.fixture
def trajectory_features():
    return pd.DataFrame({
        "country": ["FRA", "GBR", "CHN", "BRA"],
        "peak_extensification_year": [1200, 1100, 800, 1500],
        "intensification_onset_year": [1400, 1300, 1000, np.nan],
        "urbanization_takeoff_year": [1500, 1400, 1200, np.nan],
        "max_pop_density": [50, 60, 120, 10],
        "min_land_labor_ratio": [0.005, 0.004, 0.002, 0.02],
    })


def test_build_survival_dataset(trajectory_features):
    """Build dataset with duration and event indicator."""
    result = build_survival_dataset(
        trajectory_features,
        event_col="intensification_onset_year",
        origin_col="peak_extensification_year",
        entity_col="country",
    )
    assert "duration" in result.columns
    assert "event" in result.columns
    # FRA: 1400 - 1200 = 200
    fra = result[result["country"] == "FRA"].iloc[0]
    assert fra["duration"] == 200
    assert fra["event"] == 1
    # BRA: no onset, censored
    bra = result[result["country"] == "BRA"].iloc[0]
    assert bra["event"] == 0


def test_fit_cox_model(trajectory_features):
    """Fit Cox model — just check it returns without error on small data."""
    surv_df = build_survival_dataset(
        trajectory_features,
        event_col="intensification_onset_year",
        origin_col="peak_extensification_year",
        entity_col="country",
    )
    surv_df["max_pop_density"] = trajectory_features.set_index("country").loc[
        surv_df["country"].values, "max_pop_density"
    ].values
    summary = fit_cox_model(surv_df, covariates=["max_pop_density"])
    assert summary is not None
    assert "coef" in summary.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_survival.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement survival.py**

```python
"""Survival analysis for time-to-Malthusian-exit (Paper 1)."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def build_survival_dataset(
    trajectory_df: pd.DataFrame,
    event_col: str = "intensification_onset_year",
    origin_col: str = "peak_extensification_year",
    entity_col: str = "country",
    max_duration: float = 2000.0,
) -> pd.DataFrame:
    """Build a survival dataset from trajectory features.

    Duration = event_year - origin_year (time from extensification peak to intensification onset).
    Censored if event_col is NaN (entity never reached onset).

    Args:
        trajectory_df: Output of extract_trajectory_features.
        event_col: Column with the event year (NaN = censored).
        origin_col: Column with the origin year (start of risk).
        entity_col: Entity identifier column.
        max_duration: Maximum duration for censored observations.

    Returns:
        DataFrame with columns: entity_col, duration, event (1=observed, 0=censored).
    """
    df = trajectory_df[[entity_col, origin_col, event_col]].copy()
    df["event"] = (~df[event_col].isna()).astype(int)

    # Duration: event_year - origin_year for observed, max_duration for censored
    df["duration"] = np.where(
        df["event"] == 1,
        df[event_col] - df[origin_col],
        max_duration,
    )

    # Drop if origin is NaN or duration <= 0
    df = df[df[origin_col].notna() & (df["duration"] > 0)]

    return df[[entity_col, "duration", "event"]].reset_index(drop=True)


def fit_cox_model(
    surv_df: pd.DataFrame,
    covariates: list[str],
    duration_col: str = "duration",
    event_col: str = "event",
) -> pd.DataFrame:
    """Fit a Cox proportional hazards model.

    Args:
        surv_df: Survival dataset with duration, event, and covariate columns.
        covariates: List of covariate column names.
        duration_col: Name of the duration column.
        event_col: Name of the event indicator column.

    Returns:
        Summary DataFrame from lifelines CoxPHFitter.
    """
    cph = CoxPHFitter()
    fit_cols = [duration_col, event_col] + covariates
    cph.fit(surv_df[fit_cols], duration_col=duration_col, event_col=event_col)
    return cph.summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_survival.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper1_escape/survival.py tests/test_survival.py
git commit -m "feat(paper1): add survival analysis for time-to-Malthusian-exit"
```

---

### Task 10: Spatial Diffusion Analysis

**Files:**
- Create: `analysis/paper1_escape/spatial.py`
- Create: `tests/test_spatial.py` (placeholder — spatial tests need coordinate data)

- [ ] **Step 1: Implement spatial.py**

```python
"""Spatial diffusion analysis for Paper 1.

Tests whether agricultural transitions spread geographically using
spatial autocorrelation (Moran's I) and spatial lag models.
"""
import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran


def build_spatial_weights(
    coords: np.ndarray, k: int = 5
) -> KNN:
    """Build K-nearest-neighbor spatial weights from centroid coordinates.

    Args:
        coords: Array of shape (n, 2) with [lat, lon] per entity.
        k: Number of nearest neighbors.

    Returns:
        libpysal KNN weights object.
    """
    return KNN.from_array(coords, k=k)


def compute_morans_i(
    values: np.ndarray, weights: KNN
) -> dict:
    """Compute Moran's I for spatial autocorrelation.

    Args:
        values: Array of values (e.g., transition dates) per entity.
        weights: Spatial weights object.

    Returns:
        Dict with keys: I, p_value, z_score.
    """
    mi = Moran(values, weights)
    return {"I": mi.I, "p_value": mi.p_sim, "z_score": mi.z_sim}


def spatial_lag(
    values: np.ndarray, weights: KNN
) -> np.ndarray:
    """Compute spatial lag of a variable (weighted average of neighbors).

    Args:
        values: Array of values per entity.
        weights: Spatial weights object.

    Returns:
        Array of spatial lag values.
    """
    from libpysal.weights import lag_spatial
    return lag_spatial(weights, values)
```

- [ ] **Step 2: Commit**

```bash
git add analysis/paper1_escape/spatial.py
git commit -m "feat(paper1): add spatial diffusion analysis (Moran's I, spatial lag)"
```

---

### Task 11: Paper 1 Exploration Notebook

**Files:**
- Create: `analysis/paper1_escape/paper1_explore.ipynb`

- [ ] **Step 1: Create the exploration notebook**

Create a Jupyter notebook with cells structured as:

1. **Setup**: imports and panel loading
2. **Trajectory extraction**: run on real data, inspect features
3. **Clustering**: try k=3,4,5, compare silhouette scores, visualize
4. **Survival analysis**: build dataset, fit Cox model, plot survival curves
5. **Spatial analysis**: Moran's I on transition dates
6. **Uncertainty mapping**: scenario spread by region

```python
# Cell 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.shared.config import ANALYSIS_DATA
from analysis.paper1_escape.trajectories import extract_trajectory_features
from analysis.paper1_escape.clustering import (
    prepare_clustering_features, cluster_pathways, label_clusters
)
from analysis.paper1_escape.survival import build_survival_dataset, fit_cox_model

country_panel = pd.read_parquet(ANALYSIS_DATA / "country_analysis_panel.parquet")
print(f"Country panel: {country_panel.shape}")
print(f"Year range: {country_panel.year.min()} - {country_panel.year.max()}")
print(f"Countries: {country_panel.country.nunique()}")
```

```python
# Cell 2: Extract trajectory features
features = extract_trajectory_features(country_panel, entity_col="country")
print(f"Features extracted for {len(features)} countries")
print(features.describe())
features.head(20)
```

```python
# Cell 3: Clustering — scan k values
X, entities = prepare_clustering_features(features, entity_col="country")
print(f"Clustering on {len(entities)} countries with {X.shape[1]} features")

results = []
for k in range(2, 7):
    labels, metrics = cluster_pathways(X, n_clusters=k)
    results.append({"k": k, **metrics})
    print(f"k={k}: silhouette={metrics['silhouette']:.3f}, inertia={metrics['inertia']:.0f}")

# Pick best k by silhouette
best_k = max(results, key=lambda r: r["silhouette"])["k"]
print(f"\nBest k by silhouette: {best_k}")
```

```python
# Cell 4: Final clustering with best k
labels, metrics = cluster_pathways(X, n_clusters=best_k)
clustered = label_clusters(entities, labels, entity_col="country")
clustered = clustered.merge(features, on="country")
print(clustered.groupby("cluster").agg(
    n=("country", "count"),
    mean_peak_ext=("peak_extensification_year", "mean"),
    mean_intens_onset=("intensification_onset_year", "mean"),
))
```

```python
# Cell 5: Survival analysis
surv_df = build_survival_dataset(features, entity_col="country")
surv_df = surv_df.merge(
    features[["country", "max_pop_density"]], on="country"
)
print(f"Survival dataset: {len(surv_df)} countries, {surv_df.event.sum()} events")
summary = fit_cox_model(surv_df, covariates=["max_pop_density"])
print(summary)
```

- [ ] **Step 2: Commit**

```bash
git add analysis/paper1_escape/paper1_explore.ipynb
git commit -m "feat(paper1): add exploration notebook"
```

---

## Phase 3a: Paper 2 — Malthusian Mechanics

### Task 12: Malthusian Panel Construction

**Files:**
- Create: `analysis/paper2_malthus/panels.py`
- Create: `tests/test_panels.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for Malthusian panel construction."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper2_malthus.panels import build_malthusian_panel


@pytest.fixture
def wide_panel():
    rows = []
    for country in ["FRA", "GBR"]:
        for year in range(1000, 1600, 100):
            rows.append({
                "year": year,
                "country": country,
                "pop_persons_mean": 1e6 * (1 + year / 1000),
                "popdens_p_km2_mean": 10 * (1 + year / 2000),
                "land_labor_ratio": 0.01 * (2 - year / 2000),
                "ag_output_proxy_mha": 5 * (1 + year / 1500),
                "intensification_index": 0.01 * (1 + year / 500),
                "pop_growth_rate": 0.001 * (1 + year / 5000),
                "urban_share_mean": 0.02,
            })
    return pd.DataFrame(rows)


def test_build_malthusian_panel(wide_panel):
    result = build_malthusian_panel(wide_panel, entity_col="country")
    # Should have lagged variables
    assert "popdens_lag" in result.columns
    assert "land_labor_ratio_lag" in result.columns
    assert "intens_x_density" in result.columns
    # First observation per country should be dropped (no lag)
    assert len(result) < len(wide_panel)


def test_malthusian_panel_no_future_leakage(wide_panel):
    result = build_malthusian_panel(wide_panel, entity_col="country")
    # For each row, lagged density should be from the previous year
    fra = result[result["country"] == "FRA"].sort_values("year")
    for i in range(1, len(fra)):
        # Lag should equal previous row's current value
        assert fra.iloc[i]["popdens_lag"] == pytest.approx(
            fra.iloc[i - 1]["popdens_p_km2_mean"]
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_panels.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement panels.py**

```python
"""Malthusian panel construction for Paper 2."""
import numpy as np
import pandas as pd


def build_malthusian_panel(
    wide_panel: pd.DataFrame,
    entity_col: str = "country",
) -> pd.DataFrame:
    """Construct the panel for Malthusian regressions.

    Adds lagged variables and interaction terms.

    Columns added:
    - popdens_lag: lagged population density
    - land_labor_ratio_lag: lagged land-labor ratio
    - intens_x_density: interaction of intensification index with lagged density
    - log_popdens_lag: log of lagged density
    - log_land_labor_lag: log of lagged land-labor ratio

    Drops the first observation per entity (no lag available).
    """
    df = wide_panel.sort_values([entity_col, "year"]).copy()

    # Create lags within each entity
    df["popdens_lag"] = df.groupby(entity_col)["popdens_p_km2_mean"].shift(1)
    df["land_labor_ratio_lag"] = df.groupby(entity_col)["land_labor_ratio"].shift(1)
    df["intens_lag"] = df.groupby(entity_col)["intensification_index"].shift(1)

    # Drop rows without lag
    df = df.dropna(subset=["popdens_lag"]).copy()

    # Interaction terms
    df["intens_x_density"] = df["intens_lag"] * df["popdens_lag"]

    # Log transforms (with floor to avoid log(0))
    df["log_popdens_lag"] = np.log(np.maximum(df["popdens_lag"], 1e-10))
    df["log_land_labor_lag"] = np.log(np.maximum(df["land_labor_ratio_lag"], 1e-10))

    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_panels.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper2_malthus/panels.py tests/test_panels.py
git commit -m "feat(paper2): add Malthusian panel construction with lags and interactions"
```

---

### Task 13: Core Regressions and Rolling Windows

**Files:**
- Create: `analysis/paper2_malthus/regressions.py`
- Create: `tests/test_regressions.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for Malthusian regressions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper2_malthus.regressions import (
    run_fe_regression,
    run_rolling_window,
)


@pytest.fixture
def malthus_panel():
    """Synthetic panel with known Malthusian dynamics."""
    rng = np.random.default_rng(42)
    rows = []
    for country in ["FRA", "GBR", "CHN", "IND", "EGY"]:
        for year in range(1000, 1800, 100):
            popdens = 10 + year / 100 + rng.normal(0, 2)
            rows.append({
                "year": year,
                "country": country,
                "pop_growth_rate": 0.01 - 0.0005 * popdens + rng.normal(0, 0.001),
                "popdens_lag": popdens,
                "land_labor_ratio_lag": 0.02 - popdens * 0.0001,
                "intens_x_density": 0.01 * popdens,
                "log_popdens_lag": np.log(max(popdens, 0.1)),
                "log_land_labor_lag": np.log(max(0.02 - popdens * 0.0001, 1e-10)),
                "intensification_index": 0.01 + year / 10000,
            })
    return pd.DataFrame(rows)


def test_run_fe_regression(malthus_panel):
    result = run_fe_regression(
        malthus_panel,
        dep_var="pop_growth_rate",
        indep_vars=["popdens_lag", "land_labor_ratio_lag"],
        entity_col="country",
    )
    assert "params" in result
    assert "pvalues" in result
    assert "popdens_lag" in result["params"]
    # Coefficient on density should be negative (Malthusian)
    assert result["params"]["popdens_lag"] < 0


def test_run_rolling_window(malthus_panel):
    results = run_rolling_window(
        malthus_panel,
        dep_var="pop_growth_rate",
        key_var="popdens_lag",
        control_vars=["land_labor_ratio_lag"],
        entity_col="country",
        window_years=400,
        step_years=100,
    )
    assert len(results) > 0
    assert "center_year" in results.columns
    assert "coefficient" in results.columns
    assert "pvalue" in results.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_regressions.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement regressions.py**

```python
"""Core Malthusian regressions and rolling window estimation for Paper 2."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def run_fe_regression(
    panel: pd.DataFrame,
    dep_var: str,
    indep_vars: list[str],
    entity_col: str = "country",
) -> dict:
    """Run a fixed-effects (entity-demeaned) OLS regression.

    Uses within-entity demeaning to absorb entity fixed effects.

    Returns:
        Dict with keys: params, pvalues, rsquared, nobs, summary.
    """
    df = panel.dropna(subset=[dep_var] + indep_vars).copy()

    # Demean within entity
    for col in [dep_var] + indep_vars:
        entity_means = df.groupby(entity_col)[col].transform("mean")
        df[f"{col}_dm"] = df[col] - entity_means

    y = df[f"{dep_var}_dm"]
    X = df[[f"{v}_dm" for v in indep_vars]]
    X = sm.add_constant(X)

    model = OLS(y, X).fit()

    # Map demeaned names back to original
    params = {}
    pvalues = {}
    for orig, dm in zip(indep_vars, [f"{v}_dm" for v in indep_vars]):
        params[orig] = model.params[dm]
        pvalues[orig] = model.pvalues[dm]

    return {
        "params": params,
        "pvalues": pvalues,
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "summary": model.summary(),
    }


def run_rolling_window(
    panel: pd.DataFrame,
    dep_var: str,
    key_var: str,
    control_vars: list[str],
    entity_col: str = "country",
    window_years: int = 400,
    step_years: int = 100,
) -> pd.DataFrame:
    """Estimate the key_var coefficient in rolling time windows.

    For each window, runs a fixed-effects regression and records the coefficient
    on key_var.

    Returns:
        DataFrame with columns: center_year, coefficient, std_err, pvalue, nobs.
    """
    years = sorted(panel["year"].unique())
    min_year, max_year = min(years), max(years)

    results = []
    start = min_year
    while start + window_years <= max_year + step_years:
        end = start + window_years
        center = start + window_years // 2

        window_df = panel[(panel["year"] >= start) & (panel["year"] < end)]
        if len(window_df) < 10:  # minimum observations
            start += step_years
            continue

        all_vars = [key_var] + control_vars
        try:
            reg = run_fe_regression(window_df, dep_var, all_vars, entity_col)
            results.append({
                "center_year": center,
                "coefficient": reg["params"][key_var],
                "pvalue": reg["pvalues"][key_var],
                "nobs": reg["nobs"],
            })
        except Exception:
            pass  # skip windows where regression fails

        start += step_years

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_regressions.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper2_malthus/regressions.py tests/test_regressions.py
git commit -m "feat(paper2): add FE regressions and rolling window estimation"
```

---

### Task 14: Structural Break Detection

**Files:**
- Create: `analysis/paper2_malthus/breaks.py`

- [ ] **Step 1: Implement breaks.py**

```python
"""Structural break detection in the Malthusian coefficient (Paper 2).

Uses sequential Chow tests to detect when the population-density coefficient
changes significantly — indicating a regime shift from Malthusian to post-Malthusian.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def chow_test(
    y: np.ndarray, X: np.ndarray, break_idx: int
) -> tuple[float, float]:
    """Perform a Chow test for structural break at break_idx.

    Returns:
        (F_statistic, p_value).
    """
    n = len(y)
    k = X.shape[1]

    # Full model
    ols_full = sm.OLS(y, X).fit()
    rss_full = ols_full.ssr

    # Sub-samples
    ols_1 = sm.OLS(y[:break_idx], X[:break_idx]).fit()
    ols_2 = sm.OLS(y[break_idx:], X[break_idx:]).fit()
    rss_sub = ols_1.ssr + ols_2.ssr

    # F-statistic
    f_stat = ((rss_full - rss_sub) / k) / (rss_sub / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)

    return f_stat, p_value


def detect_structural_breaks(
    panel: pd.DataFrame,
    entity: str,
    dep_var: str = "pop_growth_rate",
    key_var: str = "popdens_lag",
    entity_col: str = "country",
    min_segment: int = 3,
    significance: float = 0.05,
) -> list[dict]:
    """Detect structural breaks in the Malthusian coefficient for one entity.

    Scans all possible break points and returns those where the Chow test
    rejects the null of no break.

    Returns:
        List of dicts with keys: break_year, f_stat, p_value.
    """
    df = panel[panel[entity_col] == entity].sort_values("year").dropna(
        subset=[dep_var, key_var]
    )
    if len(df) < 2 * min_segment:
        return []

    y = df[dep_var].values
    X = sm.add_constant(df[key_var].values)
    years = df["year"].values

    breaks = []
    for i in range(min_segment, len(df) - min_segment):
        f_stat, p_val = chow_test(y, X, i)
        if p_val < significance:
            breaks.append({
                "break_year": int(years[i]),
                "f_stat": f_stat,
                "p_value": p_val,
            })

    return breaks


def detect_breaks_all_entities(
    panel: pd.DataFrame,
    entity_col: str = "country",
    **kwargs,
) -> pd.DataFrame:
    """Run structural break detection for all entities.

    Returns:
        DataFrame with columns: entity_col, break_year, f_stat, p_value.
    """
    all_breaks = []
    for entity in panel[entity_col].unique():
        breaks = detect_structural_breaks(panel, entity, entity_col=entity_col, **kwargs)
        for b in breaks:
            b[entity_col] = entity
            all_breaks.append(b)

    if not all_breaks:
        return pd.DataFrame(columns=[entity_col, "break_year", "f_stat", "p_value"])
    return pd.DataFrame(all_breaks)
```

- [ ] **Step 2: Commit**

```bash
git add analysis/paper2_malthus/breaks.py
git commit -m "feat(paper2): add structural break detection via Chow tests"
```

---

### Task 15: Partial Identification Bounds

**Files:**
- Create: `analysis/paper2_malthus/bounds.py`

- [ ] **Step 1: Implement bounds.py**

```python
"""Partial identification bounds using HYDE scenario variation (Paper 2).

Runs the core Malthusian regression on each scenario separately,
then constructs Manski-style bounds on the key coefficient.
"""
import pandas as pd
from analysis.shared.config import SCENARIO_LABELS
from analysis.shared.uncertainty import compute_manski_bounds
from analysis.paper2_malthus.regressions import run_fe_regression


def estimate_bounds(
    scenario_panels: dict[str, pd.DataFrame],
    dep_var: str = "pop_growth_rate",
    indep_vars: list[str] | None = None,
    key_var: str = "popdens_lag",
    entity_col: str = "country",
) -> dict:
    """Estimate Manski bounds on the key coefficient across scenarios.

    Args:
        scenario_panels: Dict mapping scenario label to its Malthusian panel.
        dep_var: Dependent variable.
        indep_vars: Independent variables. Defaults to [key_var, "land_labor_ratio_lag"].
        key_var: The variable whose coefficient bounds are of interest.
        entity_col: Entity identifier column.

    Returns:
        Dict with keys:
        - bounds: (lower, upper) tuple for the key coefficient
        - scenario_results: dict mapping scenario -> {coef, pvalue, nobs}
    """
    if indep_vars is None:
        indep_vars = [key_var, "land_labor_ratio_lag"]

    scenario_coefs = {}
    scenario_results = {}

    for label, panel in scenario_panels.items():
        reg = run_fe_regression(panel, dep_var, indep_vars, entity_col)
        scenario_coefs[label] = reg["params"][key_var]
        scenario_results[label] = {
            "coef": reg["params"][key_var],
            "pvalue": reg["pvalues"][key_var],
            "nobs": reg["nobs"],
        }

    lb, ub = compute_manski_bounds(scenario_coefs)

    return {
        "bounds": (lb, ub),
        "scenario_results": scenario_results,
    }


def bounds_summary_table(bounds_result: dict) -> pd.DataFrame:
    """Format bounds results as a summary table.

    Returns:
        DataFrame with one row per scenario plus a bounds row.
    """
    rows = []
    for label, res in bounds_result["scenario_results"].items():
        rows.append({"scenario": label, **res})

    lb, ub = bounds_result["bounds"]
    rows.append({
        "scenario": "BOUNDS",
        "coef": f"[{lb:.6f}, {ub:.6f}]",
        "pvalue": None,
        "nobs": None,
    })

    return pd.DataFrame(rows)
```

- [ ] **Step 2: Commit**

```bash
git add analysis/paper2_malthus/bounds.py
git commit -m "feat(paper2): add Manski-style partial identification bounds"
```

---

## Phase 3b: Paper 3 — Climate Shocks and Agricultural Regime Shifts

### Task 16: Climate Shock Construction

**Files:**
- Create: `analysis/paper3_climate/climate_shocks.py`
- Create: `tests/test_climate_shocks.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for ERA5 climate shock construction."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper3_climate.climate_shocks import (
    compute_anomalies,
    compute_volatility,
)


def test_compute_anomalies():
    """Anomaly = deviation from rolling mean."""
    years = np.arange(1950, 2020)
    values = np.sin(np.arange(70) * 0.3) * 2 + 15  # oscillating temp
    df = pd.DataFrame({"year": years, "temperature": values})
    result = compute_anomalies(df, "temperature", rolling_window=30)
    # Anomalies should have mean close to 0 (over the valid window)
    valid = result["temperature_anomaly"].dropna()
    assert abs(valid.mean()) < 1.0
    assert "temperature_anomaly" in result.columns


def test_compute_volatility():
    """Volatility = rolling std of the variable."""
    years = np.arange(1950, 2020)
    rng = np.random.default_rng(42)
    values = rng.normal(15, 2, size=70)
    df = pd.DataFrame({"year": years, "temperature": values})
    result = compute_volatility(df, "temperature", rolling_window=10)
    assert "temperature_volatility" in result.columns
    valid = result["temperature_volatility"].dropna()
    # Volatility should be roughly 2 (the std we used to generate)
    assert 0.5 < valid.mean() < 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_climate_shocks.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement climate_shocks.py**

```python
"""ERA5 climate shock construction for Paper 3."""
import numpy as np
import pandas as pd


def compute_anomalies(
    df: pd.DataFrame,
    variable: str,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Compute anomalies as deviations from a rolling mean.

    Args:
        df: DataFrame with 'year' and the variable column.
        variable: Column name of the climate variable.
        rolling_window: Window size in years for the rolling mean.

    Returns:
        DataFrame with original columns plus {variable}_anomaly.
    """
    df = df.sort_values("year").copy()
    rolling_mean = df[variable].rolling(window=rolling_window, center=True, min_periods=rolling_window // 2).mean()
    df[f"{variable}_anomaly"] = df[variable] - rolling_mean
    return df


def compute_volatility(
    df: pd.DataFrame,
    variable: str,
    rolling_window: int = 10,
) -> pd.DataFrame:
    """Compute rolling volatility (standard deviation) of a climate variable.

    Args:
        df: DataFrame with 'year' and the variable column.
        variable: Column name of the climate variable.
        rolling_window: Window size in years.

    Returns:
        DataFrame with original columns plus {variable}_volatility.
    """
    df = df.sort_values("year").copy()
    df[f"{variable}_volatility"] = df[variable].rolling(
        window=rolling_window, center=True, min_periods=rolling_window // 2
    ).std()
    return df


def build_climate_shock_panel(
    era5_panel: pd.DataFrame,
    climate_vars: list[str],
    entity_col: str = "region",
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Build a panel of climate shocks (anomalies + volatility) per entity.

    Args:
        era5_panel: Panel with columns [year, entity_col, ...climate_vars].
        climate_vars: List of climate variable column names.
        entity_col: Entity identifier.
        rolling_window: Rolling window for anomaly computation.

    Returns:
        Panel with anomaly and volatility columns added for each climate var.
    """
    parts = []
    for entity, grp in era5_panel.groupby(entity_col):
        df = grp.copy()
        for var in climate_vars:
            df = compute_anomalies(df, var, rolling_window=rolling_window)
            df = compute_volatility(df, var, rolling_window=min(10, rolling_window // 3))
        parts.append(df)

    return pd.concat(parts, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_climate_shocks.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper3_climate/climate_shocks.py tests/test_climate_shocks.py
git commit -m "feat(paper3): add ERA5 climate shock construction"
```

---

### Task 17: Local Projections (Jordà IRFs)

**Files:**
- Create: `analysis/paper3_climate/local_projections.py`
- Create: `tests/test_local_projections.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for local projection impulse response functions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper3_climate.local_projections import run_local_projection


@pytest.fixture
def climate_land_panel():
    """Synthetic panel: climate shock affects cropland share with lag."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["Europe", "East Asia", "South Asia"]:
        for year in range(1950, 2020):
            shock = rng.normal(0, 1)
            rows.append({
                "year": year,
                "region": region,
                "temp_anomaly": shock,
                "cropland_share": 0.3 + 0.02 * shock + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_run_local_projection(climate_land_panel):
    irf = run_local_projection(
        climate_land_panel,
        shock_var="temp_anomaly",
        response_var="cropland_share",
        entity_col="region",
        max_horizon=5,
    )
    assert len(irf) == 6  # horizons 0..5
    assert "horizon" in irf.columns
    assert "coefficient" in irf.columns
    assert "ci_lower" in irf.columns
    assert "ci_upper" in irf.columns


def test_local_projection_horizon_zero(climate_land_panel):
    """At horizon 0, coefficient should be close to 0.02 (our DGP)."""
    irf = run_local_projection(
        climate_land_panel,
        shock_var="temp_anomaly",
        response_var="cropland_share",
        entity_col="region",
        max_horizon=0,
    )
    assert abs(irf.iloc[0]["coefficient"] - 0.02) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_local_projections.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement local_projections.py**

```python
"""Local projection (Jordà) impulse response functions for Paper 3.

Estimates the dynamic response of land-use variables to climate shocks
at multiple horizons using Jordà's (2005) local projection method.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_local_projection(
    panel: pd.DataFrame,
    shock_var: str,
    response_var: str,
    entity_col: str = "region",
    max_horizon: int = 10,
    control_vars: list[str] | None = None,
    n_lags: int = 2,
) -> pd.DataFrame:
    """Estimate local projection IRF of response_var to shock_var.

    For each horizon h (0, 1, ..., max_horizon), estimates:
        y_{t+h} - y_{t-1} = alpha + beta_h * shock_t + controls + entity_FE + e

    Args:
        panel: Panel with [year, entity_col, shock_var, response_var, ...].
        shock_var: The climate shock variable.
        response_var: The land-use response variable.
        entity_col: Entity identifier.
        max_horizon: Maximum horizon (years ahead).
        control_vars: Additional control variable columns.
        n_lags: Number of lags of the response variable to include as controls.

    Returns:
        DataFrame with columns: horizon, coefficient, std_err, ci_lower, ci_upper, pvalue.
    """
    if control_vars is None:
        control_vars = []

    panel = panel.sort_values([entity_col, "year"]).copy()

    results = []
    for h in range(max_horizon + 1):
        # Build the h-horizon dataset
        rows = []
        for entity, grp in panel.groupby(entity_col):
            grp = grp.sort_values("year").reset_index(drop=True)
            for i in range(n_lags, len(grp) - h):
                t_row = grp.iloc[i]
                th_row = grp.iloc[i + h]

                # Dependent variable: change from t-1 to t+h
                if i == 0:
                    continue
                y_prev = grp.iloc[i - 1][response_var]
                y_future = th_row[response_var]
                dy = y_future - y_prev

                row = {
                    "dy": dy,
                    "shock": t_row[shock_var],
                    "entity": entity,
                }

                # Lags of response as controls
                for lag in range(1, n_lags + 1):
                    if i - lag >= 0:
                        row[f"y_lag{lag}"] = grp.iloc[i - lag][response_var]
                    else:
                        row[f"y_lag{lag}"] = np.nan

                for cv in control_vars:
                    row[cv] = t_row[cv]

                rows.append(row)

        hdf = pd.DataFrame(rows).dropna()
        if len(hdf) < 10:
            continue

        # Entity-demean for FE
        for col in ["dy", "shock"] + [f"y_lag{l}" for l in range(1, n_lags + 1)] + control_vars:
            if col in hdf.columns:
                hdf[col] = hdf[col] - hdf.groupby("entity")[col].transform("mean")

        y = hdf["dy"]
        X_cols = ["shock"] + [f"y_lag{l}" for l in range(1, n_lags + 1)] + control_vars
        X_cols = [c for c in X_cols if c in hdf.columns]
        X = sm.add_constant(hdf[X_cols])

        model = sm.OLS(y, X).fit(cov_type="HC1")

        coef = model.params["shock"]
        se = model.bse["shock"]
        results.append({
            "horizon": h,
            "coefficient": coef,
            "std_err": se,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "pvalue": model.pvalues["shock"],
        })

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/test_local_projections.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/paper3_climate/local_projections.py tests/test_local_projections.py
git commit -m "feat(paper3): add Jordà local projection IRFs for climate shocks"
```

---

### Task 18: Regime-Switching Models

**Files:**
- Create: `analysis/paper3_climate/regime_switching.py`

- [ ] **Step 1: Implement regime_switching.py**

```python
"""Regime-switching and threshold models for Paper 3.

Tests whether the response to climate shocks differs depending on where
a region sits in the Malthusian-to-modern transition.
"""
import numpy as np
import pandas as pd
from analysis.paper3_climate.local_projections import run_local_projection


def split_sample_by_regime(
    panel: pd.DataFrame,
    regime_var: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel into pre- and post-threshold subsamples.

    Args:
        panel: Full panel.
        regime_var: Variable to split on (e.g., intensification_index).
        threshold: Cutoff value.

    Returns:
        (pre_df, post_df) tuple.
    """
    pre = panel[panel[regime_var] < threshold].copy()
    post = panel[panel[regime_var] >= threshold].copy()
    return pre, post


def interaction_local_projection(
    panel: pd.DataFrame,
    shock_var: str,
    response_var: str,
    regime_var: str,
    entity_col: str = "region",
    max_horizon: int = 10,
) -> pd.DataFrame:
    """Local projection with shock x regime interaction.

    Estimates:
        dy_{t+h} = alpha + beta1*shock + beta2*shock*regime + beta3*regime + controls + FE

    Returns:
        DataFrame with columns: horizon, coef_shock, coef_interaction, pval_shock, pval_interaction.
    """
    import statsmodels.api as sm

    panel = panel.sort_values([entity_col, "year"]).copy()
    panel["shock_x_regime"] = panel[shock_var] * panel[regime_var]

    results = []
    for h in range(max_horizon + 1):
        rows = []
        for entity, grp in panel.groupby(entity_col):
            grp = grp.sort_values("year").reset_index(drop=True)
            for i in range(1, len(grp) - h):
                y_prev = grp.iloc[i - 1][response_var]
                y_future = grp.iloc[i + h][response_var]
                t_row = grp.iloc[i]
                rows.append({
                    "dy": y_future - y_prev,
                    "shock": t_row[shock_var],
                    "regime": t_row[regime_var],
                    "shock_x_regime": t_row["shock_x_regime"],
                    "entity": entity,
                })

        hdf = pd.DataFrame(rows).dropna()
        if len(hdf) < 10:
            continue

        # Entity-demean
        for col in ["dy", "shock", "regime", "shock_x_regime"]:
            hdf[col] = hdf[col] - hdf.groupby("entity")[col].transform("mean")

        y = hdf["dy"]
        X = sm.add_constant(hdf[["shock", "regime", "shock_x_regime"]])
        model = sm.OLS(y, X).fit(cov_type="HC1")

        results.append({
            "horizon": h,
            "coef_shock": model.params.get("shock", np.nan),
            "coef_interaction": model.params.get("shock_x_regime", np.nan),
            "pval_shock": model.pvalues.get("shock", np.nan),
            "pval_interaction": model.pvalues.get("shock_x_regime", np.nan),
        })

    return pd.DataFrame(results)
```

- [ ] **Step 2: Commit**

```bash
git add analysis/paper3_climate/regime_switching.py
git commit -m "feat(paper3): add regime-switching models with interaction IRFs"
```

---

### Task 19: Counterfactual Simulations

**Files:**
- Create: `analysis/paper3_climate/counterfactuals.py`

- [ ] **Step 1: Implement counterfactuals.py**

```python
"""Counterfactual simulations for Paper 3.

Given estimated IRFs, simulate: what would have happened to region X's
agricultural transition if it had experienced region Y's climate history?
"""
import numpy as np
import pandas as pd


def simulate_counterfactual(
    irf_coefficients: np.ndarray,
    actual_shocks: np.ndarray,
    counterfactual_shocks: np.ndarray,
    baseline_response: np.ndarray,
) -> np.ndarray:
    """Simulate counterfactual response path.

    Uses the estimated IRF to compute the cumulative effect of replacing
    actual climate shocks with counterfactual shocks.

    Args:
        irf_coefficients: Array of IRF coefficients at horizons 0, 1, ..., H.
        actual_shocks: Time series of actual climate shocks.
        counterfactual_shocks: Time series of counterfactual climate shocks.
        baseline_response: Actual observed response variable path.

    Returns:
        Counterfactual response path (same length as baseline_response).
    """
    T = len(baseline_response)
    H = len(irf_coefficients)
    cf_response = baseline_response.copy()

    for t in range(T):
        cumulative_diff = 0.0
        for h in range(min(H, t + 1)):
            shock_diff = counterfactual_shocks[t - h] - actual_shocks[t - h]
            cumulative_diff += irf_coefficients[h] * shock_diff
        cf_response[t] = baseline_response[t] + cumulative_diff

    return cf_response


def run_counterfactual_experiment(
    irf_df: pd.DataFrame,
    climate_panel: pd.DataFrame,
    response_panel: pd.DataFrame,
    source_entity: str,
    target_entity: str,
    shock_var: str,
    response_var: str,
    entity_col: str = "region",
) -> pd.DataFrame:
    """Run a counterfactual: what if target had source's climate?

    Args:
        irf_df: IRF results from run_local_projection (with 'horizon' and 'coefficient').
        climate_panel: Panel with [year, entity_col, shock_var].
        response_panel: Panel with [year, entity_col, response_var].
        source_entity: Entity whose climate shocks to transplant.
        target_entity: Entity to simulate the counterfactual for.
        shock_var: Climate shock variable name.
        response_var: Response variable name.
        entity_col: Entity identifier.

    Returns:
        DataFrame with columns: year, actual, counterfactual.
    """
    irf_coefs = irf_df.sort_values("horizon")["coefficient"].values

    target_climate = climate_panel[climate_panel[entity_col] == target_entity].sort_values("year")
    source_climate = climate_panel[climate_panel[entity_col] == source_entity].sort_values("year")
    target_response = response_panel[response_panel[entity_col] == target_entity].sort_values("year")

    # Align on common years
    common_years = sorted(
        set(target_climate["year"]) & set(source_climate["year"]) & set(target_response["year"])
    )

    actual_shocks = target_climate.set_index("year").loc[common_years, shock_var].values
    cf_shocks = source_climate.set_index("year").loc[common_years, shock_var].values
    baseline = target_response.set_index("year").loc[common_years, response_var].values

    cf_path = simulate_counterfactual(irf_coefs, actual_shocks, cf_shocks, baseline)

    return pd.DataFrame({
        "year": common_years,
        "actual": baseline,
        "counterfactual": cf_path,
    })
```

- [ ] **Step 2: Commit**

```bash
git add analysis/paper3_climate/counterfactuals.py
git commit -m "feat(paper3): add counterfactual simulation engine"
```

---

### Task 20: Paper 2 and Paper 3 Exploration Notebooks

**Files:**
- Create: `analysis/paper2_malthus/paper2_explore.ipynb`
- Create: `analysis/paper3_climate/paper3_explore.ipynb`

- [ ] **Step 1: Create Paper 2 exploration notebook**

```python
# Cell 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared.config import ANALYSIS_DATA
from analysis.paper2_malthus.panels import build_malthusian_panel
from analysis.paper2_malthus.regressions import run_fe_regression, run_rolling_window
from analysis.paper2_malthus.breaks import detect_breaks_all_entities

country_panel = pd.read_parquet(ANALYSIS_DATA / "country_analysis_panel.parquet")
malthus = build_malthusian_panel(country_panel)
print(f"Malthusian panel: {malthus.shape}")
```

```python
# Cell 2: Core Malthusian regression
result = run_fe_regression(
    malthus,
    dep_var="pop_growth_rate",
    indep_vars=["popdens_lag", "land_labor_ratio_lag", "intens_x_density"],
)
print(result["summary"])
print(f"\nDensity coefficient: {result['params']['popdens_lag']:.6f} (p={result['pvalues']['popdens_lag']:.4f})")
print(f"Interaction coefficient: {result['params']['intens_x_density']:.6f} (p={result['pvalues']['intens_x_density']:.4f})")
```

```python
# Cell 3: Rolling window estimation
rolling = run_rolling_window(
    malthus,
    dep_var="pop_growth_rate",
    key_var="popdens_lag",
    control_vars=["land_labor_ratio_lag"],
    window_years=400,
    step_years=100,
)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rolling["center_year"], rolling["coefficient"], "b-o")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Center year of window")
ax.set_ylabel("Malthusian coefficient (density → growth)")
ax.set_title("Time-varying Malthusian coefficient")
plt.tight_layout()
plt.show()
```

```python
# Cell 4: Structural breaks
breaks = detect_breaks_all_entities(malthus, entity_col="country")
print(f"Detected {len(breaks)} breaks across {breaks['country'].nunique()} countries")
print(breaks.groupby("country")["break_year"].min().sort_values().head(20))
```

- [ ] **Step 2: Create Paper 3 exploration notebook**

```python
# Cell 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared.config import ANALYSIS_DATA, ERA5_ROOT
from analysis.paper3_climate.climate_shocks import build_climate_shock_panel
from analysis.paper3_climate.local_projections import run_local_projection
from analysis.paper3_climate.counterfactuals import run_counterfactual_experiment

# Load ERA5 + HYDE merged panel
# (Adjust path based on existing ERA5 integration output)
country_panel = pd.read_parquet(ANALYSIS_DATA / "country_analysis_panel.parquet")
print(f"Country panel: {country_panel.shape}")
```

```python
# Cell 2: Build climate shock panel
# This cell needs the ERA5-integrated panel from HYDE_and_ERA notebooks
# climate_panel = build_climate_shock_panel(era5_panel, climate_vars=["temperature", "precipitation"])
print("TODO: Load ERA5-merged panel and construct shocks")
print("Run HYDE_and_ERA.ipynb first if not already done")
```

```python
# Cell 3: Local projections (placeholder until ERA5 panel is ready)
# irf = run_local_projection(
#     merged_panel,
#     shock_var="temperature_anomaly",
#     response_var="cropland_share",
#     entity_col="region",
#     max_horizon=10,
# )
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(irf["horizon"], irf["coefficient"], "b-o")
# ax.fill_between(irf["horizon"], irf["ci_lower"], irf["ci_upper"], alpha=0.2)
# ax.axhline(0, color="gray", linestyle="--")
# ax.set_xlabel("Horizon (years)")
# ax.set_ylabel("Response of cropland share")
# ax.set_title("IRF: Temperature shock → Cropland share")
# plt.tight_layout()
# plt.show()
print("Uncomment after ERA5 panel is built")
```

- [ ] **Step 3: Commit**

```bash
git add analysis/paper2_malthus/paper2_explore.ipynb analysis/paper3_climate/paper3_explore.ipynb
git commit -m "feat: add exploration notebooks for Papers 2 and 3"
```

---

### Task 21: Shared Plotting Utilities

**Files:**
- Create: `analysis/shared/plotting.py`

- [ ] **Step 1: Implement plotting.py**

```python
"""Shared plotting utilities for all three papers."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_rolling_coefficient(
    rolling_df: pd.DataFrame,
    title: str = "Time-varying coefficient",
    ylabel: str = "Coefficient",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot rolling window coefficient with significance shading.

    Args:
        rolling_df: Output of run_rolling_window with columns:
                    center_year, coefficient, pvalue.
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sig = rolling_df["pvalue"] < 0.05

    ax.plot(rolling_df["center_year"], rolling_df["coefficient"], "b-o", markersize=4)
    ax.scatter(
        rolling_df.loc[sig, "center_year"],
        rolling_df.loc[sig, "coefficient"],
        color="red", s=30, zorder=5, label="p < 0.05",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Center year of window")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_irf(
    irf_df: pd.DataFrame,
    title: str = "Impulse Response Function",
    ylabel: str = "Response",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot local projection IRF with confidence intervals.

    Args:
        irf_df: Output of run_local_projection with columns:
                horizon, coefficient, ci_lower, ci_upper.
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(irf_df["horizon"], irf_df["coefficient"], "b-o", markersize=5)
    ax.fill_between(
        irf_df["horizon"], irf_df["ci_lower"], irf_df["ci_upper"],
        alpha=0.2, color="blue",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_survival_curves(
    surv_df: pd.DataFrame,
    group_col: str = "cluster",
    duration_col: str = "duration",
    event_col: str = "event",
    title: str = "Kaplan-Meier survival curves",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot Kaplan-Meier survival curves by group.

    Args:
        surv_df: Survival dataset with duration, event, and group columns.
        group_col: Column to stratify by.
        duration_col: Duration column.
        event_col: Event indicator column.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=figsize)
    kmf = KaplanMeierFitter()

    for group, grp in surv_df.groupby(group_col):
        kmf.fit(grp[duration_col], grp[event_col], label=str(group))
        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel("Duration (years from extensification peak)")
    ax.set_ylabel("Probability of remaining in Malthusian regime")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_counterfactual(
    cf_df: pd.DataFrame,
    title: str = "Counterfactual simulation",
    ylabel: str = "Response variable",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot actual vs. counterfactual paths.

    Args:
        cf_df: Output of run_counterfactual_experiment with columns:
               year, actual, counterfactual.
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cf_df["year"], cf_df["actual"], "b-", label="Actual", linewidth=2)
    ax.plot(cf_df["year"], cf_df["counterfactual"], "r--", label="Counterfactual", linewidth=2)
    ax.fill_between(
        cf_df["year"], cf_df["actual"], cf_df["counterfactual"],
        alpha=0.15, color="red",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add analysis/shared/plotting.py
git commit -m "feat: add shared plotting utilities for all papers"
```

---

### Task 22: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS

- [ ] **Step 2: Run panel builder on real data**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -m analysis.shared.build_panels
```

Expected: creates `analysis/data/country_analysis_panel.parquet` and `analysis/data/region_analysis_panel.parquet`

- [ ] **Step 3: Verify Parquet files are loadable**

```bash
cd /Volumes/BIGDATA/HYDE35 && python -c "
import pandas as pd
cp = pd.read_parquet('analysis/data/country_analysis_panel.parquet')
rp = pd.read_parquet('analysis/data/region_analysis_panel.parquet')
print('Country panel:', cp.shape, 'columns:', list(cp.columns))
print('Region panel:', rp.shape, 'columns:', list(rp.columns))
print('Derived vars present:', all(c in cp.columns for c in ['land_labor_ratio', 'ag_output_proxy_mha', 'intensification_index', 'pop_growth_rate']))
"
```

- [ ] **Step 4: Final commit**

```bash
cd /Volumes/BIGDATA/HYDE35 && git add -A && git status
git commit -m "chore: verify full test suite and panel build"
```
