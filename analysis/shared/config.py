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
