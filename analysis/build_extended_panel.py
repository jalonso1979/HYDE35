"""Build extended HYDE+ERA5 country panel for 1950-2025.

Extends build_modern_panel.py to cover the full annual HYDE range (1950-2025),
merging with ERA5 climate data where available (regions 1-8 have 1950-2025).
Saves to analysis/data/hyde_era5_extended_panel.parquet.
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

ROOT = Path("/Volumes/BIGDATA/HYDE35")
sys.path.insert(0, str(ROOT))

NC_DIR = ROOT / "gbc2025_7apr_base" / "NetCDF"
GENERAL = ROOT / "general_files" / "general_files"
ISO_GRID_PATH = GENERAL / "iso_cr.asc"
AREA_GRID_PATH = GENERAL / "garea_cr.asc"
REGION_GRID_PATH = GENERAL / "im_reg_cr.asc"
ISO_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"
CLIMATE_PANEL = ROOT / "analysis" / "data" / "climate_panel_0_2025.parquet"
OUT_PATH = ROOT / "analysis" / "data" / "hyde_era5_extended_panel.parquet"

# 1950-2025 only: annual HYDE + annual ERA5, clean identification
YEARS = list(range(1950, 2026))
TIME_SLICE = slice(52, 128)  # indices 52-127 = 1950-2025

NC_VARIABLES = {
    "population":      "pop",
    "cropland":        "cropland",
    "grazing_land":    "grazing",
    "total_rice":      "rice",
    "total_irrigated": "irrigated",
    "urban_area":      "urban",
}

from analysis.shared.loaders import read_esri_ascii_grid, align_grid


def read_nc_sliced(nc_path: Path) -> xr.DataArray:
    """Open a NetCDF file and return 1950-2025 time slices."""
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    var_name = list(ds.data_vars)[0]
    da = ds[var_name].isel(time=TIME_SLICE)
    if da["lon"].values.max() > 180.0:
        new_lon = np.where(da["lon"].values > 180.0,
                           da["lon"].values - 360.0, da["lon"].values)
        da = da.assign_coords(lon=("lon", new_lon)).sortby("lon")
    return da


# ── Step 1: reference grids ─────────────────────────────────────────────────
print("Loading reference grids …")
iso_da = read_esri_ascii_grid(ISO_GRID_PATH)
area_da = read_esri_ascii_grid(AREA_GRID_PATH)
reg_da = read_esri_ascii_grid(REGION_GRID_PATH)

iso_arr = iso_da.values
area_arr = area_da.values
reg_arr = reg_da.values

lats = iso_da.coords["lat"].values
lons = iso_da.coords["lon"].values
LAT2D, LON2D = np.meshgrid(lats, lons, indexing="ij")

# ── Step 2: ISO country mapping ─────────────────────────────────────────────
print("Building country code → ISO3 lookup …")
iso_map = pd.read_csv(ISO_MAP_CSV)
code_to_iso3 = dict(zip(iso_map["iso_num"].astype(int), iso_map["iso3"]))
code_to_name = dict(zip(iso_map["iso_num"].astype(int), iso_map["name"]))

all_codes = np.unique(iso_arr[~np.isnan(iso_arr)]).astype(int)
all_codes = all_codes[all_codes > 0]
print(f"  {len(all_codes)} country codes found in ISO grid")

# ── Step 3: country → ERA5 region assignment ────────────────────────────────
print("Computing country centroids and assigning ERA5 regions …")
country_region_rows = []
for code in all_codes:
    mask = (iso_arr == code)
    if mask.sum() == 0:
        continue
    cell_areas = area_arr[mask]
    cell_areas = np.where(np.isnan(cell_areas), 0.0, cell_areas)
    total_area = cell_areas.sum()
    if total_area == 0:
        mean_lat = float(LAT2D[mask].mean())
        mean_lon = float(LON2D[mask].mean())
    else:
        mean_lat = float((LAT2D[mask] * cell_areas).sum() / total_area)
        mean_lon = float((LON2D[mask] * cell_areas).sum() / total_area)

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

# ── Step 4: aggregate HYDE variables to country × year ──────────────────────
print("Extracting HYDE NetCDF variables (1950-2025) …")

iso_flat = iso_arr.ravel()
area_flat = area_arr.ravel()

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
    da = read_nc_sliced(nc_path)

    times = da["time"].values
    year_labels = np.array([t.year for t in times])

    shapes_match = (da.values.shape[1:] == iso_arr.shape)

    for ti, yr in enumerate(year_labels):
        if yr not in YEARS:
            continue

        if shapes_match:
            slice_2d = da.values[ti]
        else:
            slice_aligned = align_grid(da.isel(time=ti), iso_da)
            slice_2d = slice_aligned.values

        slice_flat = slice_2d.ravel()
        slice_flat = np.where(np.isnan(slice_flat), 0.0, slice_flat)

        for code, idx in country_indices.items():
            val = slice_flat[idx].sum()
            all_rows.append({
                "year": yr,
                "country_id": code,
                col_name: val,
            })

    da.close()

print(f"  Collected {len(all_rows)} raw rows across all variables")

# Pivot to wide
raw_df = pd.DataFrame(all_rows)
parts = {}
for nc_stem, col_name in NC_VARIABLES.items():
    sub = raw_df[raw_df[col_name].notna()][["year", "country_id", col_name]]
    parts[col_name] = sub

col_names = list(NC_VARIABLES.values())
hyde_panel = parts[col_names[0]].copy()
for col in col_names[1:]:
    hyde_panel = hyde_panel.merge(parts[col], on=["year", "country_id"], how="outer")

print(f"HYDE panel shape: {hyde_panel.shape}")

# ── Step 5: attach metadata and compute areas ──────────────────────────────
hyde_panel = hyde_panel.merge(country_meta, on="country_id", how="left")

area_by_country = {}
for code, idx in country_indices.items():
    a = area_flat[idx]
    a = np.where(np.isnan(a), 0.0, a)
    area_by_country[code] = a.sum()

hyde_panel["area_km2"] = hyde_panel["country_id"].map(area_by_country)

# ── Step 6: derived HYDE variables ──────────────────────────────────────────
print("Computing derived variables …")

hyde_panel["density"] = hyde_panel["pop"] / hyde_panel["area_km2"].replace(0, np.nan)
hyde_panel["ag_land_km2"] = hyde_panel["cropland"] + hyde_panel["grazing"]
hyde_panel["ag_per_capita"] = (
    hyde_panel["ag_land_km2"] / hyde_panel["pop"].replace(0, np.nan)
)
hyde_panel["crop_share"] = (
    hyde_panel["cropland"] / hyde_panel["area_km2"].replace(0, np.nan)
).clip(0, 1)
total_crop = hyde_panel["cropland"] + hyde_panel["rice"]
hyde_panel["irrigation_share"] = (
    hyde_panel["irrigated"] / total_crop.replace(0, np.nan)
).clip(0, 1)
hyde_panel["urban_share"] = (
    hyde_panel["urban"] / hyde_panel["area_km2"].replace(0, np.nan)
).clip(0, 1)

hyde_panel = hyde_panel.sort_values(["country_id", "year"])
hyde_panel["pop_growth"] = hyde_panel.groupby("country_id")["pop"].transform(
    lambda s: np.log(s.replace(0, np.nan)).diff()
)
hyde_panel["ag_growth"] = hyde_panel.groupby("country_id")["ag_land_km2"].transform(
    lambda s: np.log(s.replace(0, np.nan)).diff()
)

# Log levels for regressions
hyde_panel["log_pop"] = np.log(hyde_panel["pop"].replace(0, np.nan))
hyde_panel["log_density"] = np.log(hyde_panel["density"].replace(0, np.nan))
hyde_panel["log_cropland"] = np.log(hyde_panel["cropland"].replace(0, np.nan))
hyde_panel["log_ag_land"] = np.log(hyde_panel["ag_land_km2"].replace(0, np.nan))
hyde_panel["log_ag_per_capita"] = np.log(hyde_panel["ag_per_capita"].replace(0, np.nan))

# ── Step 7: merge ERA5 climate data (1950-2025) ──────────────────────────
print("Merging ERA5 climate data …")
era5 = pd.read_parquet(ROOT / "analysis" / "data" / "era5_full_panel.parquet")
era5_clean = era5[["region", "year", "temperature_c", "precipitation_mm"]].copy()
era5_clean = era5_clean.rename(columns={"region": "era5_region"})

merged = hyde_panel.merge(era5_clean, on=["era5_region", "year"], how="left")
print(f"  After ERA5 merge: {merged.shape}")
print(f"  Rows with temperature data: {merged['temperature_c'].notna().sum()}")

# ── Step 8: climate anomalies ──────────────────────────────────────────────
print("Computing climate anomalies …")
# Compute anomalies relative to region-specific 30-year rolling baseline
# For shorter series, use full-period mean
for var, anom_col in [("temperature_c", "temp_anomaly"), ("precipitation_mm", "precip_anomaly")]:
    merged[anom_col] = np.nan
    for r in merged["era5_region"].dropna().unique():
        mask = merged["era5_region"] == r
        sub = merged.loc[mask, var].copy()
        rolling_mean = sub.rolling(window=30, center=True, min_periods=10).mean()
        # Where rolling is unavailable, use full-period mean
        full_mean = sub.mean()
        baseline = rolling_mean.fillna(full_mean)
        merged.loc[mask, anom_col] = sub - baseline

# Z-scores
clim_stats = era5_clean.groupby("era5_region").agg(
    temp_std=("temperature_c", "std"),
    precip_std=("precipitation_mm", "std"),
).reset_index()
merged = merged.merge(clim_stats, on="era5_region", how="left")
merged["temp_z"] = merged["temp_anomaly"] / merged["temp_std"].replace(0, np.nan)
merged["precip_z"] = merged["precip_anomaly"] / merged["precip_std"].replace(0, np.nan)
merged = merged.drop(columns=["temp_std", "precip_std"], errors="ignore")

# ── Step 9: interaction terms ──────────────────────────────────────────────
merged["temp_x_irrigation"] = merged["temperature_c"] * merged["irrigation_share"]
merged["temp_x_crop_share"] = merged["temperature_c"] * merged["crop_share"]
merged["temp_x_urban"] = merged["temperature_c"] * merged["urban_share"]

# ── Final cleanup ────────────────────────────────────────────────────────────
merged = merged.sort_values(["country_id", "year"]).reset_index(drop=True)

print(f"\nSaving panel to {OUT_PATH} …")
merged.to_parquet(OUT_PATH, index=False)

# ── Summary ─────────────────────────────────────────────────────────────────
n_countries = merged["country_id"].nunique()
n_years = merged["year"].nunique()
n_rows = len(merged)
n_with_era5 = merged["temperature_c"].notna().sum()

# Countries in regions 1-8 (extended ERA5 coverage)
ext_countries = merged[merged["era5_region"].isin(range(1, 9))]["country_id"].nunique()
ext_rows = merged[merged["era5_region"].isin(range(1, 9)) & merged["temperature_c"].notna()]

print("\n" + "=" * 60)
print("EXTENDED HYDE+ERA5 PANEL SUMMARY")
print("=" * 60)
print(f"  Total rows           : {n_rows:,}")
print(f"  Countries            : {n_countries}")
print(f"  Years                : {n_years}  ({merged['year'].min()} – {merged['year'].max()})")
print(f"  Rows with ERA5 data  : {n_with_era5:,}  ({n_with_era5/n_rows*100:.1f}%)")
print(f"  Extended-coverage countries (R1-8): {ext_countries}")
print(f"  Extended rows with ERA5: {len(ext_rows):,}")
print()
print("Columns:", merged.columns.tolist())
print()
print("ERA5 region coverage:")
coverage = merged.groupby("era5_region").agg(
    n_countries=("country_id", "nunique"),
    years_with_era5=("temperature_c", lambda x: x.notna().sum()),
    year_min=("year", "min"),
    year_max=("year", "max"),
)
print(coverage.to_string())
print()
print(f"Saved: {OUT_PATH}")
