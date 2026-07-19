"""Build population-weighted versions of the ModE-RA country and sub-national
climate panels.

Motivation. The default aggregation uses cosine-latitude land-area weights:
each 5-arcmin cell contributes proportional to its physical surface area on
Earth. This gives huge weight to Siberia, the Sahara, and the Australian
outback, which never had significant population. The resulting country
"climate" is therefore the average over the country's territory, not the
average over where people actually lived. For a paper about climate's
effect on human outcomes (Malthusian density, agricultural pathways,
modern demographic outcomes), the right weight is population density.

We use HYDE 3.5 gridded population at 5-arcmin resolution. To avoid
endogeneity to climate-driven modern development, we use pre-industrial
mean population (mean over 1500-1750 timesteps) as the weight.

Outputs:
    analysis/data/modera_country_monthly_popw.parquet
    analysis/data/modera_subnational_monthly_popw.parquet
    analysis/data/popweight_diagnostic.parquet
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
MODERA = Path("/Volumes/BIGDATA/MODERA/extracted")
HYDE_POP_NC = ROOT / "gbc2025_7apr_base" / "NetCDF" / "population.nc"
COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
SUB_RASTER = ROOT / "general_files" / "general_files" / "sub_iso_cr.asc"
TEMP_NC = MODERA / "ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
PRECIP_NC = MODERA / "ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc"

REFERENCE_YEARS = [1500, 1600, 1700]  # average over pre-industrial reference timesteps


def _read_raster(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path) as f:
        h = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            h[k.lower()] = float(v)
    ncols = int(h["ncols"]); nrows = int(h["nrows"])
    xll = h["xllcorner"]; yll = h["yllcorner"]; cs = h["cellsize"]; nodata = h["nodata_value"]
    data = np.loadtxt(path, skiprows=6, dtype=np.float64).astype(np.int64)
    data[data == int(nodata)] = -1
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows)[::-1] + 0.5) * cs
    return data, lat, lon


def _load_reference_population() -> np.ndarray:
    """Mean HYDE population over REFERENCE_YEARS at 5-arcmin resolution."""
    print(f"Loading HYDE gridded population, mean over {REFERENCE_YEARS}...",
          flush=True)
    ds = xr.open_dataset(HYDE_POP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    years_in_ds = np.array([t.year for t in ds["time"].values])
    idx = [i for i, y in enumerate(years_in_ds) if y in REFERENCE_YEARS]
    print(f"  Found timesteps for years {[years_in_ds[i] for i in idx]}")
    pop = ds["population"].isel(time=idx).mean("time").values
    pop = np.nan_to_num(pop, nan=0.0)
    pop = np.clip(pop, 0, None)
    print(f"  pop grid shape: {pop.shape}, total: {pop.sum():.0f}, "
          f"non-zero cells: {(pop > 0).sum():,}/{pop.size:,}")
    return pop


def _build_weight_matrix(codes_raster: np.ndarray, raster_lat: np.ndarray,
                         raster_lon: np.ndarray, target_lat: np.ndarray,
                         target_lon: np.ndarray, weight_grid: np.ndarray,
                         id_to_iso3: dict) -> tuple[sparse.csr_matrix, list]:
    """Build sparse (n_unit × n_target_cell) weight matrix using a per-cell
    weight grid (instead of cos-lat). The weight_grid has same shape as the
    5-arcmin raster."""
    assert codes_raster.shape == weight_grid.shape, \
        f"shapes differ: codes {codes_raster.shape} vs weights {weight_grid.shape}"
    iso3_list = sorted(set(id_to_iso3.values()))
    iso3_to_idx = {c: i for i, c in enumerate(iso3_list)}
    n_lat_t = target_lat.size
    n_lon_t = target_lon.size

    lat_idx = np.argmin(np.abs(raster_lat[:, None] - target_lat[None, :]),
                         axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(raster_lon[:, None] - target_lon[None, :]),
                         axis=1).astype(np.int32)

    rows, cols, vals = [], [], []
    for i in range(codes_raster.shape[0]):
        cr = codes_raster[i]
        valid = cr > 0
        if not valid.any(): continue
        li = lat_idx[i]
        for j in np.flatnonzero(valid):
            w = float(weight_grid[i, j])
            if w <= 0: continue
            unit_id = int(cr[j])
            iso3 = id_to_iso3.get(unit_id)
            if iso3 is None: continue
            rows.append(iso3_to_idx[iso3])
            cols.append(li * n_lon_t + lon_idx[j])
            vals.append(w)
        if (i + 1) % 400 == 0:
            print(f"  raster row {i+1}/{codes_raster.shape[0]}", flush=True)
    W = sparse.coo_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(iso3_list), n_lat_t * n_lon_t),
    ).tocsr()
    return W, iso3_list


def _aggregate_with_weights(W: sparse.csr_matrix, var_nc: Path, var: str,
                             scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    print(f"  Loading {var_nc.name}...", flush=True)
    with xr.open_dataset(var_nc,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        arr = ds[var].values.astype(np.float64) * scale
        time = ds["time"].values
    flat = arr.reshape(arr.shape[0], -1)
    print(f"  Aggregating...", flush=True)
    out = (W @ flat.T) * inv[:, None]
    years = np.array([t.year for t in time], dtype=np.int32)
    months = np.array([t.month for t in time], dtype=np.int8)
    return out, years, months


def _build_iso_map() -> dict:
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    return dict(zip(iso_map["iso_num"], iso_map["iso3"]))


def build_country_panel(pop: np.ndarray) -> None:
    print("\n=== Country-level pop-weighted panel ===")
    codes, lat_c, lon_c = _read_raster(COUNTRY_RASTER)
    iso_num_to_iso3 = _build_iso_map()

    with xr.open_dataset(TEMP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        m_lat = ds["latitude"].values
        m_lon = ds["longitude"].values

    print("Building country weight matrix (pop-weighted)...", flush=True)
    W, iso3_list = _build_weight_matrix(codes, lat_c, lon_c, m_lat, m_lon,
                                          pop, iso_num_to_iso3)
    rs = np.asarray(W.sum(axis=1)).ravel()
    print(f"  W: shape {W.shape}, nnz {W.nnz:,}, "
          f"countries with positive weight: {(rs > 0).sum()}/{len(iso3_list)}")

    t_anom, years, months = _aggregate_with_weights(W, TEMP_NC, "temp2")
    p_flux, _, _ = _aggregate_with_weights(W, PRECIP_NC, "totprec")
    days = np.array([pd.Period(year=int(y), month=int(m), freq="M").days_in_month
                       for y, m in zip(years, months)], dtype=np.float64)
    p_mm = p_flux * 86400.0 * days

    n_c, n_t = t_anom.shape
    df = pd.DataFrame({
        "iso3": np.repeat(np.asarray(iso3_list), n_t),
        "year": np.tile(years, n_c),
        "month": np.tile(months, n_c),
        "t_anom_c": t_anom.ravel().astype(np.float32),
        "p_anom_mm": p_mm.ravel().astype(np.float32),
    })
    # Keep countries with positive population weight
    keep = pd.DataFrame({"iso3": iso3_list, "popw_sum": rs})
    keep = keep[keep["popw_sum"] > 0]
    df = df[df["iso3"].isin(keep["iso3"])]
    out = DATA / "modera_country_monthly_popw.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, {df['iso3'].nunique()} countries)")
    keep.to_parquet(DATA / "popweight_diagnostic.parquet", index=False)


def build_subnational_panel(pop: np.ndarray) -> None:
    print("\n=== Sub-national pop-weighted panel ===")
    codes, lat_c, lon_c = _read_raster(SUB_RASTER)
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    # Map sub_id (int) -> iso3 via leading-3-digit numeric
    unique_sub = np.unique(codes); unique_sub = unique_sub[unique_sub > 0]
    sub_to_id = {int(s): int(s) for s in unique_sub}  # identity (we keep sub-IDs)

    with xr.open_dataset(TEMP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        m_lat = ds["latitude"].values
        m_lon = ds["longitude"].values

    # Build matrix using sub_id as the row key
    print("Building sub-national weight matrix (pop-weighted)...", flush=True)
    sub_id_list = sorted([int(s) for s in unique_sub])
    sub_to_idx = {s: i for i, s in enumerate(sub_id_list)}
    n_lat_t = m_lat.size; n_lon_t = m_lon.size
    lat_idx = np.argmin(np.abs(lat_c[:, None] - m_lat[None, :]), axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(lon_c[:, None] - m_lon[None, :]), axis=1).astype(np.int32)
    rows, cols, vals = [], [], []
    for i in range(codes.shape[0]):
        cr = codes[i]
        valid = cr > 0
        if not valid.any(): continue
        li = lat_idx[i]
        for j in np.flatnonzero(valid):
            w = float(pop[i, j])
            if w <= 0: continue
            s = int(cr[j])
            if s not in sub_to_idx: continue
            rows.append(sub_to_idx[s])
            cols.append(li * n_lon_t + lon_idx[j])
            vals.append(w)
        if (i+1) % 400 == 0:
            print(f"  raster row {i+1}/{codes.shape[0]}", flush=True)
    W = sparse.coo_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(sub_id_list), n_lat_t * n_lon_t),
    ).tocsr()
    rs = np.asarray(W.sum(axis=1)).ravel()
    print(f"  W nnz {W.nnz:,}, sub-units with positive weight: "
          f"{(rs > 0).sum()}/{len(sub_id_list)}")

    t_anom, years, months = _aggregate_with_weights(W, TEMP_NC, "temp2")
    p_flux, _, _ = _aggregate_with_weights(W, PRECIP_NC, "totprec")
    days = np.array([pd.Period(year=int(y), month=int(m), freq="M").days_in_month
                       for y, m in zip(years, months)], dtype=np.float64)
    p_mm = p_flux * 86400.0 * days

    n_s, n_t = t_anom.shape
    df = pd.DataFrame({
        "sub_id": np.repeat(np.asarray(sub_id_list), n_t),
        "year": np.tile(years, n_s),
        "month": np.tile(months, n_s),
        "t_anom_c": t_anom.ravel().astype(np.float32),
        "p_anom_mm": p_mm.ravel().astype(np.float32),
    })
    df["iso_num"] = (df["sub_id"] // 1000).astype(np.int32)
    df["iso3"] = df["iso_num"].map(num_to_iso3)
    df = df.dropna(subset=["iso3"])
    # Keep sub-units with positive weight
    keep_subs = {sub_id_list[i] for i in np.flatnonzero(rs > 0)}
    df = df[df["sub_id"].isin(keep_subs)]
    out = DATA / "modera_subnational_monthly_popw.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, "
          f"{df['sub_id'].nunique()} sub-units in {df['iso3'].nunique()} countries)")


def main() -> None:
    pop = _load_reference_population()
    build_country_panel(pop)
    build_subnational_panel(pop)


if __name__ == "__main__":
    main()
