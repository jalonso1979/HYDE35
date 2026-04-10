"""Build unified climate panel 0–2025 CE from four data sources.

Sources (in chronological order):
  1. PAGES 2k (0–1849 CE):  7 continental regions → mapped to 25 ERA5 regions
  2. Berkeley Earth (1850–1900): 1° gridded → aggregated to 25 ERA5 regions
  3. CRU TS 4.09 (1901–1949): 0.5° gridded → aggregated to 25 ERA5 regions
  4. ERA5 (1950–2025):  Already in panel

Calibration chain: ERA5 ← CRU TS ← Berkeley Earth ← PAGES 2k
Output: analysis/data/climate_panel_0_2025.parquet
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/BIGDATA/HYDE35")
sys.path.insert(0, str(ROOT))

RECON_DIR = ROOT / "climate_reconstructions"
ERA5_PANEL = ROOT / "analysis" / "data" / "era5_full_panel.parquet"
OUT_PATH = ROOT / "analysis" / "data" / "climate_panel_0_2025.parquet"

# ERA5 region bounding boxes (from im_reg_cr.asc grid)
# We'll compute these from the reference grid
from analysis.shared.loaders import read_esri_ascii_grid

GENERAL = ROOT / "general_files" / "general_files"
REGION_GRID_PATH = GENERAL / "im_reg_cr.asc"

# PAGES 2k continental regions → ERA5 region mapping
PAGES2K_TO_ERA5 = {
    "Arctic":       [1],
    "NAm_Trees":    [2, 3, 4],    # North America
    "SAmerica":     [5, 6],
    "Europe":       [12, 13, 14, 15],
    "Asia":         [9, 10, 11, 16, 17, 18, 19],
    "Australasia":  [20, 21, 22],
    # Africa has no PAGES 2k region → use global mean
}
# ERA5 regions that get Africa (no PAGES 2k coverage)
AFRICA_ERA5_REGIONS = [7, 8, 23, 24, 25]

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load ERA5 panel (anchor)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BUILDING UNIFIED CLIMATE PANEL (0–2025 CE)")
print("=" * 70)

print("\n1. Loading ERA5 panel (anchor) …")
era5 = pd.read_parquet(ERA5_PANEL)
era5 = era5[["region", "year", "temperature_c", "precipitation_mm"]].copy()
era5["source"] = "era5"
era5["source_resolution"] = "regional"
# Keep only 1950+
era5 = era5[era5["year"] >= 1950].copy()
print(f"   ERA5: {len(era5)} rows, {era5['year'].min()}–{era5['year'].max()}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Build ERA5 region bounding boxes for spatial aggregation
# ═══════════════════════════════════════════════════════════════════════════
print("\n2. Computing ERA5 region bounding boxes …")
reg_da = read_esri_ascii_grid(REGION_GRID_PATH)
reg_arr = reg_da.values
reg_lats = reg_da.coords["lat"].values
reg_lons = reg_da.coords["lon"].values

region_bounds = {}
for r in range(1, 26):
    mask = (reg_arr == r)
    if mask.sum() == 0:
        continue
    lat_idx, lon_idx = np.where(mask)
    region_bounds[r] = {
        "lat_min": float(reg_lats[lat_idx].min()),
        "lat_max": float(reg_lats[lat_idx].max()),
        "lon_min": float(reg_lons[lon_idx].min()),
        "lon_max": float(reg_lons[lon_idx].max()),
    }
    print(f"   Region {r:2d}: lat [{region_bounds[r]['lat_min']:.1f}, {region_bounds[r]['lat_max']:.1f}], "
          f"lon [{region_bounds[r]['lon_min']:.1f}, {region_bounds[r]['lon_max']:.1f}]")


def aggregate_gridded_to_regions(da_2d, lats, lons, region_bounds, land_mask=None):
    """Compute area-weighted mean of a 2D field over each ERA5 region."""
    # Cosine-latitude weights for area
    cos_lat = np.cos(np.deg2rad(lats))
    cos_2d = np.broadcast_to(cos_lat[:, None], da_2d.shape)

    results = {}
    for r, bounds in region_bounds.items():
        lat_mask = (lats >= bounds["lat_min"]) & (lats <= bounds["lat_max"])
        lon_mask = (lons >= bounds["lon_min"]) & (lons <= bounds["lon_max"])
        spatial_mask = np.outer(lat_mask, lon_mask)

        if land_mask is not None:
            spatial_mask = spatial_mask & land_mask

        vals = da_2d[spatial_mask]
        weights = cos_2d[spatial_mask]

        # Ignore NaN
        valid = ~np.isnan(vals)
        if valid.sum() < 5:
            results[r] = np.nan
            continue

        results[r] = float(np.average(vals[valid], weights=weights[valid]))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Process CRU TS 4.09 (1901–1949) — decade files
# ═══════════════════════════════════════════════════════════════════════════
print("\n3. Processing CRU TS 4.09 (1901–1949) …")

CRU_DIR = RECON_DIR / "cru_ts"
CRU_DECADES = ["1901.1910", "1911.1920", "1921.1930", "1931.1940", "1941.1950"]

cru_rows = []
cru_land = None
cru_lats = None
cru_lons = None

for decade in CRU_DECADES:
    tmp_path = CRU_DIR / f"cru_ts4.09.{decade}.tmp.dat.nc"
    pre_path = CRU_DIR / f"cru_ts4.09.{decade}.pre.dat.nc"

    if not tmp_path.exists():
        print(f"   WARNING: {tmp_path.name} not found, skipping")
        continue

    ds_tmp = xr.open_dataset(tmp_path, engine="netcdf4")
    if cru_lats is None:
        cru_lats = ds_tmp["lat"].values
        cru_lons = ds_tmp["lon"].values
        first_slice = ds_tmp["tmp"].isel(time=0).values
        cru_land = ~np.isnan(first_slice)

    times = pd.to_datetime(ds_tmp["time"].values)
    decade_start = int(decade.split(".")[0])
    decade_end = int(decade.split(".")[1])

    for year in range(decade_start, min(decade_end + 1, 1950)):
        year_mask = times.year == year
        if year_mask.sum() == 0:
            continue
        annual_mean = ds_tmp["tmp"].isel(time=year_mask).mean(dim="time").values

        region_temps = aggregate_gridded_to_regions(
            annual_mean, cru_lats, cru_lons, region_bounds, cru_land
        )
        for r, temp in region_temps.items():
            cru_rows.append({
                "region": r, "year": year,
                "temperature_c": temp,
                "source": "cru_ts",
            })

    ds_tmp.close()

    # Precipitation
    if pre_path.exists():
        ds_pre = xr.open_dataset(pre_path, engine="netcdf4")
        times_pre = pd.to_datetime(ds_pre["time"].values)

        for year in range(decade_start, min(decade_end + 1, 1950)):
            year_mask = times_pre.year == year
            if year_mask.sum() == 0:
                continue
            annual_mean = ds_pre["pre"].isel(time=year_mask).mean(dim="time").values

            region_precip = aggregate_gridded_to_regions(
                annual_mean, ds_pre["lat"].values, ds_pre["lon"].values,
                region_bounds, cru_land
            )
            for row in cru_rows:
                if row["year"] == year and row["region"] in region_precip:
                    if "precipitation_mm" not in row:
                        row["precipitation_mm"] = region_precip[row["region"]]

        ds_pre.close()

    print(f"   CRU {decade} done")

cru_df = pd.DataFrame(cru_rows)
cru_df["source_resolution"] = "0.5deg"
if "precipitation_mm" not in cru_df.columns:
    cru_df["precipitation_mm"] = np.nan
print(f"   CRU TS: {len(cru_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Process Berkeley Earth (1850–1900)
# ═══════════════════════════════════════════════════════════════════════════
print("\n4. Processing Berkeley Earth (1850–1900) …")
be_path = RECON_DIR / "berkeley_earth" / "Complete_TAVG_LatLong1.nc"

be_rows = []
if be_path.exists():
    ds_be = xr.open_dataset(be_path, engine="netcdf4")
    be_lats = ds_be["latitude"].values
    be_lons = ds_be["longitude"].values
    be_land = ds_be["land_mask"].values  # 1=land, NaN=ocean

    # Berkeley Earth stores anomalies relative to 1951-1980 climatology
    # climatology shape: (12, 180, 360) — monthly absolute temps
    climatology = ds_be["climatology"].values  # (12, 180, 360)
    annual_clim = np.nanmean(climatology, axis=0)  # annual mean climatology

    # Temperature anomalies: (time, 180, 360)
    # time is in fractional years (e.g., 1850.0417 = Jan 1850)
    be_times = ds_be["time"].values
    be_temp = ds_be["temperature"].values  # (ntime, 180, 360)

    be_land_mask = ~np.isnan(be_land)

    for year in range(1850, 1901):
        year_mask = (be_times >= year) & (be_times < year + 1)
        if year_mask.sum() == 0:
            continue

        # Annual mean anomaly
        annual_anom = np.nanmean(be_temp[year_mask], axis=0)  # (180, 360)
        # Convert to absolute temperature
        annual_abs = annual_anom + annual_clim

        region_temps = aggregate_gridded_to_regions(
            annual_abs, be_lats, be_lons, region_bounds, be_land_mask
        )
        for r, temp in region_temps.items():
            be_rows.append({
                "region": r, "year": year,
                "temperature_c": temp,
                "source": "berkeley",
            })

        if year % 10 == 0:
            print(f"   Berkeley {year} done")

    ds_be.close()
else:
    print("   WARNING: Berkeley Earth file not found")

be_df = pd.DataFrame(be_rows)
be_df["precipitation_mm"] = np.nan  # Berkeley Earth doesn't have precip
be_df["source_resolution"] = "1deg"
print(f"   Berkeley Earth: {len(be_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Process PAGES 2k (0–1849 CE)
# ═══════════════════════════════════════════════════════════════════════════
print("\n5. Processing PAGES 2k regional reconstructions (0–1849 CE) …")
pages2k_path = RECON_DIR / "pages2k" / "DatabaseS2-Regional-Temperature-Reconstructions.xlsx"

pages2k_rows = []
if pages2k_path.exists():
    df_pages = pd.read_excel(pages2k_path,
                             sheet_name="PAGES2k recon's - annual",
                             header=None, skiprows=2)
    # Columns: 8 regions × 4 cols each (year, temp, min, max)
    region_names = ["Antarctica", "Arctic", "Asia", "Australasia",
                    "Europe", "NAm_Pollen", "NAm_Trees", "SAmerica"]
    cols = []
    for r in region_names:
        cols.extend([f"{r}_year", f"{r}_temp", f"{r}_min", f"{r}_max"])
    df_pages.columns = cols

    # For North America, prefer Trees if available, fall back to Pollen
    # Build a combined NAm series
    nam_trees = df_pages[["NAm_Trees_year", "NAm_Trees_temp"]].dropna()
    nam_trees = nam_trees.rename(columns={"NAm_Trees_year": "year", "NAm_Trees_temp": "temp"})
    nam_pollen = df_pages[["NAm_Pollen_year", "NAm_Pollen_temp"]].dropna()
    nam_pollen = nam_pollen.rename(columns={"NAm_Pollen_year": "year", "NAm_Pollen_temp": "temp"})
    # Merge, preferring trees
    nam_combined = pd.concat([nam_pollen.set_index("year"),
                              nam_trees.set_index("year")]).groupby(level=0).last()

    # Also load global mean from PAGES 2k 2019 for Africa
    global_nc = RECON_DIR / "pages2k" / "pages2k_ngeo19_recons.nc"
    ds_global = xr.open_dataset(global_nc)
    # Use BHM method (Bayesian Hierarchical Model) ensemble median
    global_temp = ds_global["BHM"].median(dim="ens").values  # (2000,)
    global_years = ds_global["year"].values  # 1–2000
    global_series = pd.Series(global_temp, index=global_years, name="global_temp_anom")
    ds_global.close()

    # Process each PAGES 2k region
    pages2k_regions = {
        "Arctic":      df_pages[["Arctic_year", "Arctic_temp"]].dropna(),
        "Europe":      df_pages[["Europe_year", "Europe_temp"]].dropna(),
        "Asia":        df_pages[["Asia_year", "Asia_temp"]].dropna(),
        "Australasia": df_pages[["Australasia_year", "Australasia_temp"]].dropna(),
        "SAmerica":    df_pages[["SAmerica_year", "SAmerica_temp"]].dropna(),
    }
    # Rename columns
    for k, v in pages2k_regions.items():
        v.columns = ["year", "temp"]
        pages2k_regions[k] = v.set_index("year")["temp"]

    # Add combined North America
    pages2k_regions["NAm"] = nam_combined["temp"]

    # Map to ERA5 regions
    pages2k_mapping = {
        "Arctic":      [1],
        "NAm":         [2, 3, 4],
        "SAmerica":    [5, 6],
        "Europe":      [12, 13, 14, 15],
        "Asia":        [9, 10, 11, 16, 17, 18, 19],
        "Australasia": [20, 21, 22],
    }

    for pages_name, era5_regs in pages2k_mapping.items():
        series = pages2k_regions[pages_name]
        for year, temp in series.items():
            year = int(year)
            if year >= 1850:  # Berkeley Earth takes over
                continue
            for r in era5_regs:
                pages2k_rows.append({
                    "region": r,
                    "year": year,
                    "temperature_c": temp,  # These are anomalies — will calibrate later
                    "source": "pages2k",
                    "_is_anomaly": True,
                })

    # Africa: use global mean
    for year in range(1, 1850):
        if year in global_series.index:
            temp = float(global_series[year])
            for r in AFRICA_ERA5_REGIONS:
                pages2k_rows.append({
                    "region": r,
                    "year": year,
                    "temperature_c": temp,
                    "source": "pages2k",
                    "_is_anomaly": True,
                })

    print(f"   PAGES 2k: {len(pages2k_rows)} raw rows")
else:
    print("   WARNING: PAGES 2k file not found")

pages2k_df = pd.DataFrame(pages2k_rows)
pages2k_df["precipitation_mm"] = np.nan
pages2k_df["source_resolution"] = "continental"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Calibration chain
# ═══════════════════════════════════════════════════════════════════════════
print("\n6. Running calibration chain …")

# 6a. CRU TS → ERA5 calibration (overlap: 1950–1967)
# Compute per-region offset over overlap period
print("   6a. CRU TS → ERA5 (overlap 1950–1967) …")

# Get CRU TS for 1950-1967 (extend CRU processing)
cru_overlap_rows = []
# Use the 1941-1950 decade file which contains 1950
cru_overlap_path = CRU_DIR / "cru_ts4.09.1941.1950.tmp.dat.nc"
if cru_overlap_path.exists() and cru_lats is not None:
    ds_tmp = xr.open_dataset(cru_overlap_path, engine="netcdf4")
    times = pd.to_datetime(ds_tmp["time"].values)

    for year in range(1950, 1951):  # Only 1950 is in this file
        year_mask = times.year == year
        if year_mask.sum() == 0:
            continue
        annual_mean = ds_tmp["tmp"].isel(time=year_mask).mean(dim="time").values
        region_temps = aggregate_gridded_to_regions(
            annual_mean, cru_lats, cru_lons, region_bounds, cru_land
        )
        for r, temp in region_temps.items():
            cru_overlap_rows.append({"region": r, "year": year, "cru_temp": temp})

    ds_tmp.close()

# Also use CRU data from 1941-1949 already processed to extend overlap
# Use the already-processed cru_rows for 1941-1949
for row in cru_rows:
    if 1941 <= row["year"] <= 1949:
        cru_overlap_rows.append({
            "region": row["region"], "year": row["year"],
            "cru_temp": row["temperature_c"],
        })

cru_overlap = pd.DataFrame(cru_overlap_rows)
era5_overlap = era5[era5["year"].between(1950, 1967)][["region", "year", "temperature_c"]].copy()
era5_overlap = era5_overlap.rename(columns={"temperature_c": "era5_temp"})

overlap = cru_overlap.merge(era5_overlap, on=["region", "year"])
overlap = overlap.dropna()
overlap["_diff"] = overlap["era5_temp"] - overlap["cru_temp"]
cru_era5_offsets = overlap.groupby("region")["_diff"].mean().to_dict()

print("   CRU→ERA5 offsets (°C):")
for r in sorted(cru_era5_offsets):
    print(f"     Region {r:2d}: {cru_era5_offsets[r]:+.2f}")

# Apply offset to CRU TS
cru_df["temperature_c"] = cru_df.apply(
    lambda row: row["temperature_c"] + cru_era5_offsets.get(row["region"], 0),
    axis=1
)

# 6b. Berkeley Earth → CRU TS calibration (overlap: 1901–1930)
print("\n   6b. Berkeley Earth → CRU TS (overlap 1901–1930) …")
be_overlap = be_df[be_df["year"].between(1901, 1930)][["region", "year", "temperature_c"]].copy()
be_overlap = be_overlap.rename(columns={"temperature_c": "be_temp"})
cru_overlap2 = cru_df[cru_df["year"].between(1901, 1930)][["region", "year", "temperature_c"]].copy()
cru_overlap2 = cru_overlap2.rename(columns={"temperature_c": "cru_temp_cal"})

overlap2 = be_overlap.merge(cru_overlap2, on=["region", "year"])
overlap2 = overlap2.dropna()
overlap2["_diff"] = overlap2["cru_temp_cal"] - overlap2["be_temp"]
be_cru_offsets = overlap2.groupby("region")["_diff"].mean().to_dict()

print("   BE→CRU offsets (°C):")
for r in sorted(be_cru_offsets):
    print(f"     Region {int(r):2d}: {be_cru_offsets[r]:+.2f}")

be_df["temperature_c"] = be_df.apply(
    lambda row: row["temperature_c"] + be_cru_offsets.get(row["region"], 0),
    axis=1
)

# 6c. PAGES 2k → Berkeley Earth calibration (overlap: 1850–1900)
print("\n   6c. PAGES 2k → Berkeley Earth (overlap 1850–1900) …")
# PAGES 2k temperatures are anomalies — need to convert to absolute
# Use Berkeley Earth's regional 1850-1900 mean as anchor

if len(pages2k_df) > 0:
    be_means_1850_1900 = be_df[be_df["year"].between(1850, 1900)].groupby("region")["temperature_c"].mean()

    # For PAGES 2k, the values are anomalies relative to some baseline
    # We need to find the offset that makes PAGES 2k match Berkeley Earth
    # First, get PAGES 2k values for 1850-1849 (they stop at 1849)
    # Actually PAGES 2k goes up to ~2000, but we cut at 1850
    # We need some overlap — let's use PAGES 2k for 1850-1900 too (temporary)
    # and calibrate against Berkeley Earth

    # Re-extract PAGES 2k for 1850-1900 for calibration
    pages2k_cal_rows = []
    for pages_name, era5_regs in pages2k_mapping.items():
        if pages_name not in pages2k_regions:
            continue
        series = pages2k_regions[pages_name]
        for year, temp in series.items():
            year = int(year)
            if 1850 <= year <= 1900:
                for r in era5_regs:
                    pages2k_cal_rows.append({
                        "region": r, "year": year, "pages_temp": temp
                    })
    # Africa global
    for year in range(1850, 1901):
        if year in global_series.index:
            for r in AFRICA_ERA5_REGIONS:
                pages2k_cal_rows.append({
                    "region": r, "year": year,
                    "pages_temp": float(global_series[year])
                })

    pages2k_cal = pd.DataFrame(pages2k_cal_rows)
    be_cal = be_df[be_df["year"].between(1850, 1900)][["region", "year", "temperature_c"]].copy()
    be_cal = be_cal.rename(columns={"temperature_c": "be_temp_cal"})

    overlap3 = pages2k_cal.merge(be_cal, on=["region", "year"]).dropna()

    if len(overlap3) > 0:
        overlap3["_diff"] = overlap3["be_temp_cal"] - overlap3["pages_temp"]
        pages_be_offsets = overlap3.groupby("region")["_diff"].mean().to_dict()

        print("   PAGES→BE offsets (°C) — converting anomalies to absolute:")
        for r in sorted(pages_be_offsets):
            print(f"     Region {int(r):2d}: {pages_be_offsets[r]:+.2f}")

        # Apply offset to PAGES 2k
        pages2k_df["temperature_c"] = pages2k_df.apply(
            lambda row: row["temperature_c"] + pages_be_offsets.get(row["region"], np.nan),
            axis=1
        )
    else:
        print("   WARNING: No overlap found for PAGES 2k calibration")

# Drop calibration helper column
if "_is_anomaly" in pages2k_df.columns:
    pages2k_df = pages2k_df.drop(columns=["_is_anomaly"])

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Combine all sources
# ═══════════════════════════════════════════════════════════════════════════
print("\n7. Combining all sources …")

cols = ["region", "year", "temperature_c", "precipitation_mm", "source", "source_resolution"]

parts = []
if len(pages2k_df) > 0:
    parts.append(pages2k_df[cols])
if len(be_df) > 0:
    parts.append(be_df[cols])
if len(cru_df) > 0:
    parts.append(cru_df[cols])
parts.append(era5[cols])

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values(["region", "year"]).reset_index(drop=True)

# Remove duplicates (prefer higher-resolution source)
source_priority = {"era5": 0, "cru_ts": 1, "berkeley": 2, "pages2k": 3}
combined["_priority"] = combined["source"].map(source_priority)
combined = combined.sort_values(["region", "year", "_priority"])
combined = combined.drop_duplicates(subset=["region", "year"], keep="first")
combined = combined.drop(columns=["_priority"])
combined = combined.sort_values(["region", "year"]).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Save
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n8. Saving to {OUT_PATH} …")
combined.to_parquet(OUT_PATH, index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("UNIFIED CLIMATE PANEL SUMMARY")
print("=" * 70)
print(f"  Total rows:    {len(combined):,}")
print(f"  Year range:    {combined['year'].min()} – {combined['year'].max()}")
print(f"  Regions:       {combined['region'].nunique()}")
print(f"  With temp:     {combined['temperature_c'].notna().sum():,}")
print(f"  With precip:   {combined['precipitation_mm'].notna().sum():,}")
print()
print("  By source:")
for src in ["pages2k", "berkeley", "cru_ts", "era5"]:
    sub = combined[combined["source"] == src]
    if len(sub) > 0:
        yr_range = f"{sub['year'].min()}–{sub['year'].max()}"
        print(f"    {src:12s}: {len(sub):6,} rows  ({yr_range})")
print()
print("  Coverage by region:")
for r in sorted(combined["region"].unique()):
    sub = combined[combined["region"] == r]
    print(f"    Region {int(r):2d}: {sub['year'].min():5d}–{sub['year'].max():4d}  "
          f"({len(sub):,} years, {sub['temperature_c'].notna().sum()} with temp)")
print()
print(f"  Saved: {OUT_PATH}")
