"""Rebuild ERA5 regional panel from all available extracted NetCDFs.

Reads monthly t2m (temperature) and tp (precipitation) from extracted files,
computes annual region-level means, and saves to analysis/data/era5_full_panel.parquet.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
ERA5_ROOT = ROOT / "ERA5"
OUT_PATH = ROOT / "analysis" / "data" / "era5_full_panel.parquet"

# ---------------------------------------------------------------------------
# 1. Discover all extracted files
# ---------------------------------------------------------------------------
print("Scanning extracted ERA5 files …")

records = []
for reg_dir in sorted(ERA5_ROOT.glob("region=*")):
    region = int(reg_dir.name.split("=")[1])
    for yr_dir in sorted(reg_dir.glob("year=*")):
        year = int(yr_dir.name.split("=")[1])
        ext_dir = yr_dir / "_extracted"
        if not ext_dir.exists():
            continue

        # Find temperature and precipitation files
        # Old format: era5_R_YYYYMM.m0.nc (t2m), era5_R_YYYYMM.m1.nc (tp)
        # New format: data_stream-oper_stepType-instant.nc (t2m),
        #             data_stream-oper_stepType-accum.nc (tp)
        t2m_files = sorted(ext_dir.glob("*.m0.nc"))
        tp_files = sorted(ext_dir.glob("*.m1.nc"))
        instant_file = ext_dir / "data_stream-oper_stepType-instant.nc"
        accum_file = ext_dir / "data_stream-oper_stepType-accum.nc"

        if instant_file.exists():
            t2m_files = [instant_file]
            tp_files = [accum_file] if accum_file.exists() else []
        elif not t2m_files:
            continue

        records.append({
            "region": region,
            "year": year,
            "t2m_files": t2m_files,
            "tp_files": tp_files,
        })

print(f"  Found {len(records)} region-year combinations with extracted data")

# ---------------------------------------------------------------------------
# 2. Compute annual means
# ---------------------------------------------------------------------------
print("Computing annual means …")

rows = []
for rec in records:
    region = rec["region"]
    year = rec["year"]

    # Temperature (t2m in Kelvin → Celsius)
    try:
        t2m_vals = []
        for f in rec["t2m_files"]:
            ds = xr.open_dataset(f, engine="netcdf4")
            t2m_vals.append(float(ds["t2m"].mean().values))
            ds.close()
        t2m_k = float(np.mean(t2m_vals))
        t2m_c = t2m_k - 273.15
    except Exception as e:
        print(f"  WARN: region={region} year={year} t2m failed: {e}")
        t2m_k = np.nan
        t2m_c = np.nan

    # Precipitation (tp in metres → mm)
    try:
        if rec["tp_files"]:
            tp_vals = []
            for f in rec["tp_files"]:
                ds = xr.open_dataset(f, engine="netcdf4")
                tp_vals.append(float(ds["tp"].mean().values))
                ds.close()
            tp_m = float(np.mean(tp_vals))
            tp_mm = tp_m * 1000.0
        else:
            tp_m = np.nan
            tp_mm = np.nan
    except Exception as e:
        print(f"  WARN: region={region} year={year} tp failed: {e}")
        tp_m = np.nan
        tp_mm = np.nan

    rows.append({
        "region": region,
        "year": year,
        "temperature_k": t2m_k,
        "precipitation_m": tp_m,
        "temperature_c": t2m_c,
        "precipitation_mm": tp_mm,
    })

    if len(rows) % 100 == 0:
        print(f"  Processed {len(rows)}/{len(records)} …")

era5_panel = pd.DataFrame(rows).sort_values(["region", "year"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
print(f"\nSaving to {OUT_PATH} …")
era5_panel.to_parquet(OUT_PATH, index=False)

# Summary
print(f"\nERA5 panel: {era5_panel.shape}")
print(f"Regions: {sorted(era5_panel['region'].unique())}")
print(f"Years: {era5_panel['year'].min()} – {era5_panel['year'].max()}")
print(f"\nCoverage by region:")
for r in sorted(era5_panel["region"].unique()):
    sub = era5_panel[era5_panel["region"] == r]
    print(f"  Region {r:2d}: {sub['year'].min()}-{sub['year'].max()} ({len(sub)} years)")

print(f"\nTemperature (°C) summary:")
print(era5_panel["temperature_c"].describe().round(2).to_string())
print(f"\nPrecipitation (mm) summary:")
print(era5_panel["precipitation_mm"].describe().round(4).to_string())
