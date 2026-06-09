"""Build a country-decade KK10 anthropogenic-land-fraction panel.

KK10 (Kaplan et al. 2011, doi:10.1594/PANGAEA.871369) is an independent
reconstruction of Holocene anthropogenic land use that, unlike HYDE 3.5,
does NOT use population as an input variable.  This makes it the natural
cross-validation for the area-orthogonality finding of Section 4.5:
HYDE's cropland-area response to volcanic forcing collapses to zero when
we control for population growth.  If the same orthogonality holds on
KK10, the "land follows people" interpretation is mechanically robust.
If KK10 differs, the divergence is diagnostic of HYDE's
population-as-input back-projection mechanics.

KK10 raw is a 17.3 GB NetCDF at
  /Users/jalonso/Library/CloudStorage/.../Pandemics/Data/KK10/KK10.nc
with dims (time, lat, lon) = (7901, 2160, 4320) on a 5-arcmin grid
covering 6051 BCE to 1850 CE.  The single variable `land_use` is the
anthropogenic fraction (cropland + pasture combined; not separable).

We aggregate to country level at HYDE's decadal years 1500, 1600,
1700, 1710, ..., 1850 (matching the joint-VAR panel years) using the
HYDE 5-arcmin country raster `general_files/general_files/iso_cr.asc`.
Cell areas are cos(lat)-weighted: cell_area_km2 = (5/60)^2 * (Earth
radius)^2 * cos(lat) * (pi/180), then absolute anthropogenic area =
sum over cells of (anthropogenic_fraction * cell_area_km2).

Output: analysis/data/kk10_country_panel.parquet
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
KK10_NC = Path("/Users/jalonso/Library/CloudStorage/"
                "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Pandemics/"
                "Data/KK10/KK10.nc")
ISO_CR  = ROOT / "general_files" / "general_files" / "iso_cr.asc"

# HYDE decadal years that the joint-VAR panel uses (post-1700 decadal,
# plus century-spaced 1500 and 1600).  KK10 ends at 1850; we lose 1860-1890.
YEARS = [1500, 1600, 1700, 1710, 1720, 1730, 1740, 1750, 1760, 1770,
         1780, 1790, 1800, 1810, 1820, 1830, 1840, 1850]


def _load_iso_cr() -> tuple[np.ndarray, np.ndarray]:
    """Returns (iso_codes 2D int array, area_km2 2D float array)
    on the 5-arcmin grid (2160 rows × 4320 cols).  Row 0 = north pole."""
    print(f"Loading iso_cr raster: {ISO_CR}")
    iso = np.loadtxt(ISO_CR, skiprows=6, dtype=np.int32)
    assert iso.shape == (2160, 4320), f"Unexpected iso_cr shape: {iso.shape}"
    # iso_cr.asc convention: row 0 is north (lat = 90 - cellsize/2),
    # row 2159 is south.  Compute cell area at each latitude.
    EARTH_R_KM = 6371.0
    deg = np.pi / 180.0
    cell_deg = 5.0 / 60.0   # 0.0833333
    lats = 90.0 - (np.arange(2160) + 0.5) * cell_deg   # row centre lat
    area_per_row = (cell_deg * deg * EARTH_R_KM) ** 2 * np.cos(lats * deg)
    area = np.broadcast_to(area_per_row[:, None], (2160, 4320)).copy()
    return iso, area


def _kk10_year_slice(ds: xr.Dataset, year: int) -> np.ndarray:
    """Extract one year's anthropogenic-fraction grid as a numpy array
    oriented row 0 = north pole, matching iso_cr.asc convention."""
    idx = int(np.argmin(np.abs(ds["year"].values - year)))
    actual = int(ds["year"].values[idx])
    if actual != year:
        raise ValueError(f"Year {year} not found; nearest = {actual}")
    arr = ds["land_use"].isel(time=idx).values.astype(np.float32)
    # KK10 lat coord runs from south (-89.96) to north (+89.96) — flip
    # vertically so row 0 = north pole (matches iso_cr.asc).
    if float(ds["lat"].values[0]) < float(ds["lat"].values[-1]):
        arr = arr[::-1, :]
    return arr


def main() -> None:
    print(f"Opening KK10 NetCDF: {KK10_NC}")
    ds = xr.open_dataset(KK10_NC)
    print(f"  KK10 dims: time={ds.sizes['time']}, "
          f"lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")

    iso, area = _load_iso_cr()
    unique_iso = sorted([int(c) for c in np.unique(iso) if c > 0])
    print(f"  iso_cr unique countries: {len(unique_iso)}")

    # Load ISO numeric → 3-letter mapping (same one the ModE-RA panel uses)
    cmap_csv = ROOT / "hyde35_country_iso_mapping.csv"
    m = pd.read_csv(cmap_csv).dropna(subset=["iso_num", "iso3"])
    m["iso_num"] = m["iso_num"].astype(int)
    iso_num_to_iso3 = dict(zip(m["iso_num"], m["iso3"]))
    print(f"  Loaded {len(iso_num_to_iso3)} ISO mappings")

    # Aggregate each year
    rows = []
    for y in YEARS:
        try:
            lu = _kk10_year_slice(ds, y)
        except ValueError as e:
            print(f"  Year {y}: {e}, skipping")
            continue
        # Compute anthropogenic area per cell = fraction * cell area km^2
        anthro_area = lu * area
        # Aggregate by iso code
        for iso_num in unique_iso:
            mask = (iso == iso_num)
            if mask.sum() == 0:
                continue
            iso3 = iso_num_to_iso3.get(iso_num)
            if iso3 is None:
                continue
            # KK10 has NaN over ocean and some land cells; use nansum.
            total_anthro_km2 = float(np.nansum(anthro_area[mask]))
            total_land_km2   = float(np.nansum(area[mask]))
            # Count valid (non-NaN) KK10 cells to track coverage
            valid_cells = int(np.isfinite(lu[mask]).sum())
            total_cells = int(mask.sum())
            rows.append({
                "iso_num": iso_num, "iso3": iso3, "year": y,
                "kk10_anthro_km2": total_anthro_km2,
                "country_area_km2": total_land_km2,
                "kk10_anthro_frac": total_anthro_km2 / max(total_land_km2, 1e-9),
                "kk10_valid_cells": valid_cells,
                "kk10_total_cells": total_cells,
            })
        print(f"  year={y}: extracted {len([r for r in rows if r['year']==y])} countries")

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "kk10_country_panel.parquet", index=False)
    print(f"\nSaved {DATA/'kk10_country_panel.parquet'}: "
          f"{len(out):,} rows, {out['iso_num'].nunique()} countries, "
          f"{out['year'].nunique()} years")


if __name__ == "__main__":
    main()
