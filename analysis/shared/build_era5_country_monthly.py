"""Aggregate raw ERA5 hourly data to country-monthly means.

The existing extended panel maps 25 macro-region averages to countries
(all countries in the same region get identical T, P). This script builds
a TRUE country-monthly ERA5 panel from the raw 0.25-degree grid that we
already downloaded into /Volumes/BIGDATA/HYDE35/ERA5/region=R/year=Y/.

Raw files come in two container formats: the May-2026 CDS deliveries are zip
archives of two netCDFs (instant t2m, accum tp), and the July-2026 bulk
completion wrote plain merged netCDF4 files with both variables in one
dataset. Zip-era months through 1966/67 are 1.0-degree grids; everything
later is 0.25-degree (the per-grid-signature weight cache handles both). We
aggregate hourly -> monthly:
    t2m: mean of hourly K, converted to deg C
    tp:  sum of hourly accumulated m -> total monthly mm

Then we area-weight ERA5 grid cells to HYDE countries using the 5-arcmin
iso_cr raster, the same approach used for ModE-RA.

Output: analysis/data/era5_country_monthly.parquet
    iso3, year, month, t2m_c, tp_mm
"""

from __future__ import annotations

from pathlib import Path
import io
import zipfile
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
ERA5_DIR = ROOT / "ERA5"
COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
COUNTRY_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"
OUT = ROOT / "analysis" / "data" / "era5_country_monthly.parquet"


def _read_country_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(COUNTRY_RASTER) as f:
        header = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            header[k.lower()] = float(v)
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cs = header["cellsize"]
    nodata = header["nodata_value"]
    data = np.loadtxt(COUNTRY_RASTER, skiprows=6, dtype=np.int32)
    data[data == int(nodata)] = -1
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows)[::-1] + 0.5) * cs
    return data, lat, lon


def _iso_map() -> dict:
    mp = pd.read_csv(COUNTRY_MAP_CSV)
    mp = mp.dropna(subset=["iso_num", "iso3"]).copy()
    mp["iso_num"] = mp["iso_num"].astype(int)
    return dict(zip(mp["iso_num"], mp["iso3"]))


def _build_country_weights_for_grid(target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[sparse.csr_matrix, list[str]]:
    """Compute sparse (n_country, n_target_cells) weight matrix.

    Only includes 5-arcmin country cells that fall WITHIN the target grid's
    bbox plus a small buffer (1 cell width). Cells outside the bbox would
    otherwise get argmin'd to edge cells, producing spurious biases.
    """
    codes, lat_c, lon_c = _read_country_raster()
    num_to_iso3 = _iso_map()
    iso3_list = sorted(set(num_to_iso3.values()))
    iso3_to_idx = {c: i for i, c in enumerate(iso3_list)}
    n_lat = target_lat.size
    n_lon = target_lon.size

    # Target-grid bbox + one-cell buffer (use grid spacing)
    if n_lat > 1:
        d_lat = abs(target_lat[1] - target_lat[0])
    else:
        d_lat = 1.0
    if n_lon > 1:
        d_lon = abs(target_lon[1] - target_lon[0])
    else:
        d_lon = 1.0
    lat_min = target_lat.min() - d_lat
    lat_max = target_lat.max() + d_lat
    lon_min = target_lon.min() - d_lon
    lon_max = target_lon.max() + d_lon

    lat_in = (lat_c >= lat_min) & (lat_c <= lat_max)
    lon_in = (lon_c >= lon_min) & (lon_c <= lon_max)

    lat_idx_full = np.argmin(np.abs(lat_c[:, None] - target_lat[None, :]), axis=1).astype(np.int32)
    lon_idx_full = np.argmin(np.abs(lon_c[:, None] - target_lon[None, :]), axis=1).astype(np.int32)
    lat_w = np.cos(np.deg2rad(lat_c))

    rows, cols, weights = [], [], []
    lat_in_rows = np.flatnonzero(lat_in)
    for i in lat_in_rows:
        cr = codes[i]
        valid = (cr > 0) & lon_in
        if not valid.any():
            continue
        li = lat_idx_full[i]
        wgt = lat_w[i]
        for j in np.flatnonzero(valid):
            iso3 = num_to_iso3.get(int(cr[j]))
            if iso3 is None:
                continue
            rows.append(iso3_to_idx[iso3])
            cols.append(li * n_lon + lon_idx_full[j])
            weights.append(wgt)
    W = sparse.coo_matrix(
        (np.asarray(weights, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(iso3_list), n_lat * n_lon),
    ).tocsr()
    return W, iso3_list


def _open_month_datasets(path: Path) -> tuple[xr.Dataset | None, xr.Dataset | None]:
    """Return (t2m_ds, tp_ds) for a monthly ERA5 file.

    Zip containers hold separate instant (t2m) and accum (tp) members; plain
    merged netCDF4 files carry both variables, so the same dataset is
    returned twice.
    """
    # Leading-magic check: zipfile.is_zipfile() false-positives on some HDF5
    # files (it scans the file tail for the EOCD signature).
    with open(path, "rb") as fh:
        is_zip = fh.read(4) == b"PK\x03\x04"
    if not is_zip:
        try:
            ds = xr.open_dataset(path, engine="h5netcdf")
        except Exception:
            return None, None
        if "t2m" not in ds.data_vars or "tp" not in ds.data_vars:
            ds.close()
            return None, None
        return ds, ds
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            inst_name = next((n for n in names if "instant" in n), None)
            accm_name = next((n for n in names if "accum" in n), None)
            inst_ds = xr.open_dataset(io.BytesIO(zf.read(inst_name)), engine="h5netcdf") if inst_name else None
            accm_ds = xr.open_dataset(io.BytesIO(zf.read(accm_name)), engine="h5netcdf") if accm_name else None
            return inst_ds, accm_ds
    except (zipfile.BadZipFile, KeyError):
        return None, None


def main() -> None:
    rows_out = []
    # Restrict to the 25 real regions; region={99..102} are single-month
    # resolution-test tiles from the May-2026 debugging session.
    region_dirs = sorted(
        (d for d in ERA5_DIR.glob("region=*")
         if 1 <= int(d.name.split("=")[1]) <= 25),
        key=lambda d: int(d.name.split("=")[1]),
    )
    print(f"Found {len(region_dirs)} regions.", flush=True)

    # Cache weights per (lat, lon) signature (regions with same bbox share weights)
    weight_cache: dict[tuple, tuple[sparse.csr_matrix, list[str], np.ndarray]] = {}

    n_files_total = 0
    n_files_ok = 0
    for rdir in region_dirs:
        region = int(rdir.name.split("=")[1])
        year_dirs = sorted(rdir.glob("year=*"))
        for ydir in year_dirs:
            year = int(ydir.name.split("=")[1])
            ncs = sorted(ydir.glob("era5_*.nc"))
            for nc_path in ncs:
                n_files_total += 1
                month = int(nc_path.stem.split("_")[-1][-2:])
                inst_ds, accm_ds = _open_month_datasets(nc_path)
                if inst_ds is None or accm_ds is None:
                    continue

                lat = inst_ds["latitude"].values.astype(np.float64)
                lon = inst_ds["longitude"].values.astype(np.float64)
                sig = (region, len(lat), len(lon), float(lat[0]), float(lon[0]),
                       float(lat[-1]), float(lon[-1]))
                if sig not in weight_cache:
                    W, iso3_list = _build_country_weights_for_grid(lat, lon)
                    row_sums = np.asarray(W.sum(axis=1)).ravel()
                    weight_cache[sig] = (W, iso3_list, row_sums)
                W, iso3_list, row_sums = weight_cache[sig]
                inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)

                # t2m hourly mean -> monthly mean in Celsius
                t2m_hr = inst_ds["t2m"].values.astype(np.float32)  # (time, lat, lon)
                t2m_mn = t2m_hr.mean(axis=0).astype(np.float64) - 273.15
                # tp hourly accum -> monthly total in mm (m -> mm = *1000)
                tp_hr = accm_ds["tp"].values.astype(np.float32)
                tp_mn = tp_hr.sum(axis=0).astype(np.float64) * 1000.0

                t_flat = t2m_mn.ravel()
                p_flat = tp_mn.ravel()

                t_country = (W @ t_flat) * inv
                p_country = (W @ p_flat) * inv

                df = pd.DataFrame({
                    "iso3": iso3_list,
                    "year": year,
                    "month": month,
                    "t2m_c": t_country.astype(np.float32),
                    "tp_mm": p_country.astype(np.float32),
                    "region": region,
                    "weight_sum": row_sums.astype(np.float32),
                })
                df = df[df["weight_sum"] > 0]
                rows_out.append(df)
                n_files_ok += 1
                inst_ds.close(); accm_ds.close()
        if rows_out and (len(rows_out) % 200 == 0 or region <= 5):
            print(f"  processed {n_files_ok} files (region {region} done)", flush=True)

    print(f"Files OK: {n_files_ok}/{n_files_total}", flush=True)
    if not rows_out:
        print("No data produced.")
        return
    full = pd.concat(rows_out, ignore_index=True)

    # Multiple regions may produce values for the same country-year-month
    # (regions overlap geographically). Take the mean across them, weighted
    # by each region's coverage of the country (weight_sum). This avoids the
    # double-counting that comes with simple averaging.
    full["t_w"] = full["t2m_c"] * full["weight_sum"]
    full["p_w"] = full["tp_mm"] * full["weight_sum"]
    agg = full.groupby(["iso3", "year", "month"], as_index=False).agg(
        t_w=("t_w", "sum"),
        p_w=("p_w", "sum"),
        weight_sum=("weight_sum", "sum"),
    )
    agg["t2m_c"] = agg["t_w"] / agg["weight_sum"]
    agg["tp_mm"] = agg["p_w"] / agg["weight_sum"]
    agg = agg[["iso3", "year", "month", "t2m_c", "tp_mm"]]
    agg.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(agg):,} country-month rows, "
          f"{agg['iso3'].nunique()} countries, "
          f"{agg['year'].min()}-{agg['year'].max()})", flush=True)


if __name__ == "__main__":
    main()
