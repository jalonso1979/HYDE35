"""Build a comprehensive HYDE+ERA5 merged panel for all countries, 1950-1967.

Steps:
1. Read ISO country grid and area grid (reference grids).
2. Load 6 HYDE NetCDF variables for time slices 1950-1967 only.
3. Aggregate each variable to country × year level using the ISO grid.
4. Assign every country to the closest ERA5 region via its grid centroid.
5. Merge with ERA5 climate data and compute derived variables.
6. Save to analysis/data/hyde_era5_full_panel.parquet.
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/BIGDATA/HYDE35")
sys.path.insert(0, str(ROOT))

NC_DIR = ROOT / "gbc2025_7apr_base" / "NetCDF"
GENERAL = ROOT / "general_files" / "general_files"
ISO_GRID_PATH = GENERAL / "iso_cr.asc"
AREA_GRID_PATH = GENERAL / "garea_cr.asc"
REGION_GRID_PATH = GENERAL / "im_reg_cr.asc"
ISO_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"
ERA5_PANEL = ROOT / "analysis" / "data" / "era5_full_panel.parquet"
OUT_PATH = ROOT / "analysis" / "data" / "hyde_era5_full_panel.parquet"

YEARS = list(range(1950, 1968))          # 18 years matching ERA5
# NetCDF time slices 52–69 correspond to 1950–1967 (confirmed by inspection)
TIME_SLICE = slice(52, 70)

# Variables and their native units  → output column names
NC_VARIABLES = {
    "population":    "pop",           # persons
    "cropland":      "cropland",      # 100-ha cells → km²  (1 cell = 1/12 ° ≈ ~86 km²; we keep km²)
    "grazing_land":  "grazing",       # 100-ha cells
    "total_rice":    "rice",          # 100-ha cells
    "total_irrigated": "irrigated",   # 100-ha cells
    "urban_area":    "urban",         # 100-ha cells
}

# ── loaders ───────────────────────────────────────────────────────────────────
from analysis.shared.loaders import read_esri_ascii_grid, align_grid


def read_nc_sliced(nc_path: Path) -> xr.DataArray:
    """Open a NetCDF file and return only the 1950-1967 time slices."""
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    var_name = list(ds.data_vars)[0]
    da = ds[var_name].isel(time=TIME_SLICE)
    # Normalise longitudes to -180..180
    if da["lon"].values.max() > 180.0:
        new_lon = np.where(da["lon"].values > 180.0,
                           da["lon"].values - 360.0,
                           da["lon"].values)
        da = da.assign_coords(lon=("lon", new_lon)).sortby("lon")
    return da


# ── Step 1: reference grids ───────────────────────────────────────────────────
print("Loading reference grids …")
iso_da = read_esri_ascii_grid(ISO_GRID_PATH)    # (2160, 4320) – ISO numeric codes
area_da = read_esri_ascii_grid(AREA_GRID_PATH)  # (2160, 4320) – grid-cell areas in km²
reg_da = read_esri_ascii_grid(REGION_GRID_PATH) # (2160, 4320) – ERA5 region codes 1-25

iso_arr = iso_da.values        # float64 with NaN for ocean
area_arr = area_da.values
reg_arr = reg_da.values

lats = iso_da.coords["lat"].values   # 2160 values, descending
lons = iso_da.coords["lon"].values   # 4320 values

# Build 2D lat/lon arrays for centroid computation
LAT2D, LON2D = np.meshgrid(lats, lons, indexing="ij")   # (2160, 4320)

# ── Step 2: ISO country mapping ────────────────────────────────────────────────
print("Building country code → ISO3 lookup …")
iso_map = pd.read_csv(ISO_MAP_CSV)
code_to_iso3 = dict(zip(iso_map["iso_num"].astype(int), iso_map["iso3"]))
code_to_name = dict(zip(iso_map["iso_num"].astype(int), iso_map["name"]))

# All country codes present in the ISO grid
all_codes = np.unique(iso_arr[~np.isnan(iso_arr)]).astype(int)
all_codes = all_codes[all_codes > 0]   # drop 0 if it appears
print(f"  {len(all_codes)} country codes found in ISO grid")

# ── Step 3: country → ERA5 region assignment via centroid ─────────────────────
print("Computing country centroids and assigning ERA5 regions …")

country_region_rows = []
for code in all_codes:
    mask = (iso_arr == code)
    if mask.sum() == 0:
        continue
    # Weighted centroid (area-weighted mean lat/lon)
    cell_areas = area_arr[mask]
    cell_areas = np.where(np.isnan(cell_areas), 0.0, cell_areas)
    total_area = cell_areas.sum()
    if total_area == 0:
        mean_lat = float(LAT2D[mask].mean())
        mean_lon = float(LON2D[mask].mean())
    else:
        mean_lat = float((LAT2D[mask] * cell_areas).sum() / total_area)
        mean_lon = float((LON2D[mask] * cell_areas).sum() / total_area)

    # Which ERA5 region appears most often in this country's cells?
    reg_codes = reg_arr[mask]
    reg_codes = reg_codes[~np.isnan(reg_codes)]
    if len(reg_codes) == 0:
        era5_region = np.nan
    else:
        values, counts = np.unique(reg_codes, return_counts=True)
        era5_region = int(values[np.argmax(counts)])

    country_region_rows.append({
        "country_id": code,
        "iso3": code_to_iso3.get(code, f"C{code:03d}"),
        "name": code_to_name.get(code, f"Unknown_{code}"),
        "centroid_lat": mean_lat,
        "centroid_lon": mean_lon,
        "era5_region": era5_region,
    })

country_meta = pd.DataFrame(country_region_rows)
print(f"  {len(country_meta)} countries assigned to ERA5 regions")
print(f"  Countries with no ERA5 region: {country_meta['era5_region'].isna().sum()}")

# ── Step 4: aggregate HYDE variables to country × year ────────────────────────
print("Extracting HYDE NetCDF variables …")

# Pre-flatten the ISO array for fast groupby
iso_flat = iso_arr.ravel()                # (9,331,200,)
area_flat = area_arr.ravel()

# Build a lookup from country_id → flat indices (done once)
country_indices = {}
for code in all_codes:
    idx = np.where(iso_flat == code)[0]
    if len(idx) > 0:
        country_indices[code] = idx

print(f"  Index map built for {len(country_indices)} countries")

all_rows = []

for nc_stem, col_name in NC_VARIABLES.items():
    nc_path = NC_DIR / f"{nc_stem}.nc"
    print(f"  Loading {nc_stem}.nc …", flush=True)
    da = read_nc_sliced(nc_path)   # (18, 2160, 4320)

    # Extract year labels from cftime coords
    times = da["time"].values
    year_labels = np.array([t.year for t in times])

    # Align grid to iso_da (should already match but confirm)
    da_aligned = align_grid(da.isel(time=0), iso_da)   # test first slice
    # If shapes match, proceed without re-aligning every slice
    shapes_match = (da.values.shape[1:] == iso_arr.shape)

    for ti, yr in enumerate(year_labels):
        if yr not in YEARS:
            continue

        if shapes_match:
            slice_2d = da.values[ti]   # (2160, 4320)
        else:
            slice_aligned = align_grid(da.isel(time=ti), iso_da)
            slice_2d = slice_aligned.values

        # Replace fill/nan with 0 for summation (land variables are 0 in ocean)
        slice_flat = slice_2d.ravel()
        nan_mask = np.isnan(slice_flat)
        slice_flat = np.where(nan_mask, 0.0, slice_flat)

        for code, idx in country_indices.items():
            val = slice_flat[idx].sum()
            all_rows.append({
                "year": yr,
                "country_id": code,
                col_name: val,
            })

print(f"  Collected {len(all_rows)} raw rows across all variables")

# Pivot from long to wide: one row per (year, country_id, variable) → wide
raw_df = pd.DataFrame(all_rows)
# There's one row per (year, country_id) per variable; pivot to wide
wide = raw_df.groupby(["year", "country_id"]).first().reset_index()

# Actually the rows were appended per variable separately – merge properly
parts = {}
for nc_stem, col_name in NC_VARIABLES.items():
    sub = raw_df[raw_df[col_name].notna()][["year", "country_id", col_name]]
    parts[col_name] = sub

# Start with first variable then merge the rest
col_names = list(NC_VARIABLES.values())
hyde_panel = parts[col_names[0]].copy()
for col in col_names[1:]:
    hyde_panel = hyde_panel.merge(parts[col], on=["year", "country_id"], how="outer")

print(f"HYDE panel shape after pivot: {hyde_panel.shape}")

# ── Step 5: attach country metadata ──────────────────────────────────────────
hyde_panel = hyde_panel.merge(country_meta, on="country_id", how="left")

# ── Step 6: compute area for each country (constant across years) ─────────────
print("Computing country land areas …")
area_by_country = {}
for code, idx in country_indices.items():
    a = area_flat[idx]
    a = np.where(np.isnan(a), 0.0, a)
    area_by_country[code] = a.sum()

hyde_panel["area_km2"] = hyde_panel["country_id"].map(area_by_country)

# ── Step 7: derived HYDE variables ────────────────────────────────────────────
print("Computing derived variables …")

# HYDE land variables are stored as km² per cell (garea_cr.asc is in km²)
# but the NetCDF stores actual land-use in 100-ha units.
# Convert cropland/grazing/etc. from 100-ha to km²:  1 km² = 100 ha, 1 unit = 100 ha = 1 km²
# So 1 HYDE unit = 100 ha = 1 km²
HYDE_TO_KM2 = 1.0   # already in km² (100-ha cells, 1 unit = 1 km²)

# Population density (persons/km²)
hyde_panel["density"] = hyde_panel["pop"] / hyde_panel["area_km2"].replace(0, np.nan)

# Ag land = cropland + grazing (in km²)
hyde_panel["ag_land_km2"] = (hyde_panel["cropland"] + hyde_panel["grazing"]) * HYDE_TO_KM2

# Ag per capita (km² per person)
hyde_panel["ag_per_capita"] = (
    hyde_panel["ag_land_km2"] / hyde_panel["pop"].replace(0, np.nan)
)

# Crop share of land
hyde_panel["crop_share"] = (
    hyde_panel["cropland"] * HYDE_TO_KM2 / hyde_panel["area_km2"].replace(0, np.nan)
).clip(0, 1)

# Irrigation share (irrigated / total cropland)
total_crop = hyde_panel["cropland"] + hyde_panel["rice"]
hyde_panel["irrigation_share"] = (
    hyde_panel["irrigated"] / total_crop.replace(0, np.nan)
).clip(0, 1)

# Urban share
hyde_panel["urban_share"] = (
    hyde_panel["urban"] * HYDE_TO_KM2 / hyde_panel["area_km2"].replace(0, np.nan)
).clip(0, 1)

# Population growth rate (year-over-year log difference)
hyde_panel = hyde_panel.sort_values(["country_id", "year"])
hyde_panel["pop_growth"] = hyde_panel.groupby("country_id")["pop"].transform(
    lambda s: np.log(s.replace(0, np.nan)).diff()
)

# Ag land growth rate
hyde_panel["ag_growth"] = hyde_panel.groupby("country_id")["ag_land_km2"].transform(
    lambda s: np.log(s.replace(0, np.nan)).diff()
)

# ── Step 8: merge ERA5 climate data ──────────────────────────────────────────
print("Merging ERA5 climate data …")
era5 = pd.read_parquet(ERA5_PANEL)
# era5 columns: region, year, temperature_c, precipitation_mm (and _k, _m raw)

# Keep only the clean units
era5_clean = era5[["region", "year", "temperature_c", "precipitation_mm"]].copy()
era5_clean = era5_clean.rename(columns={"region": "era5_region"})

merged = hyde_panel.merge(era5_clean, on=["era5_region", "year"], how="left")
print(f"  After ERA5 merge: {merged.shape}")
print(f"  Rows with temperature data: {merged['temperature_c'].notna().sum()}")

# ── Step 9: climate anomalies and volatility ──────────────────────────────────
print("Computing climate anomalies …")
# Compute anomalies relative to the 1950-1967 mean for each ERA5 region
clim_means = era5_clean.groupby("era5_region")[["temperature_c", "precipitation_mm"]].mean()
clim_stds  = era5_clean.groupby("era5_region")[["temperature_c", "precipitation_mm"]].std()

merged = merged.merge(
    clim_means.rename(columns={"temperature_c": "temp_mean", "precipitation_mm": "precip_mean"}),
    on="era5_region", how="left"
)
merged = merged.merge(
    clim_stds.rename(columns={"temperature_c": "temp_std", "precipitation_mm": "precip_std"}),
    on="era5_region", how="left"
)

merged["temp_anomaly"]   = merged["temperature_c"]    - merged["temp_mean"]
merged["precip_anomaly"] = merged["precipitation_mm"] - merged["precip_mean"]

# Standardised anomalies (z-scores)
merged["temp_z"]   = merged["temp_anomaly"]   / merged["temp_std"].replace(0, np.nan)
merged["precip_z"] = merged["precip_anomaly"] / merged["precip_std"].replace(0, np.nan)

# Rolling volatility (std over the panel window – very short, so just use full-period std)
# For a full 18-year window we already computed std above; volatility = std by region
merged["temp_volatility"]   = merged["temp_std"]
merged["precip_volatility"] = merged["precip_std"]

# ── Step 10: interaction terms ─────────────────────────────────────────────────
merged["temp_x_irrigation"] = merged["temperature_c"] * merged["irrigation_share"]
merged["temp_x_crop_share"] = merged["temperature_c"] * merged["crop_share"]
merged["temp_x_urban"]      = merged["temperature_c"] * merged["urban_share"]

# ── Final cleanup ─────────────────────────────────────────────────────────────
# Drop helper columns used only for anomaly computation
merged = merged.drop(columns=["temp_mean", "precip_mean", "temp_std", "precip_std"],
                     errors="ignore")

# Sort panel
merged = merged.sort_values(["country_id", "year"]).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving panel to {OUT_PATH} …")
merged.to_parquet(OUT_PATH, index=False)

# ── Summary statistics ─────────────────────────────────────────────────────────
n_countries = merged["country_id"].nunique()
n_years     = merged["year"].nunique()
n_rows      = len(merged)
n_with_era5 = merged["temperature_c"].notna().sum()
coverage    = n_with_era5 / n_rows * 100

print("\n" + "=" * 60)
print("HYDE+ERA5 full panel summary")
print("=" * 60)
print(f"  Total rows          : {n_rows:,}")
print(f"  Countries           : {n_countries}")
print(f"  Years               : {n_years}  ({merged['year'].min()} – {merged['year'].max()})")
print(f"  Rows with ERA5 data : {n_with_era5:,}  ({coverage:.1f}%)")
print(f"  Columns             : {merged.columns.tolist()}")
print()
print("ERA5 region coverage:")
print(merged.groupby("era5_region")["country_id"].nunique()
      .rename("n_countries")
      .sort_index()
      .to_string())
print()
print("Population summary (1960):")
yr60 = merged[merged["year"] == 1960]
print(yr60[["pop", "density", "crop_share", "irrigation_share", "temperature_c"]]
      .describe()
      .round(4)
      .to_string())
print()
print(f"Saved: {OUT_PATH}")
