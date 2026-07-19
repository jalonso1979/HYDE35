"""One-sweep ERA5 compaction: derived products that retire the 1.5 TB raw archive.

Reads every raw monthly file (25 regions x 1950-2025, hourly t2m + tp, both
container formats: CDS zip at 1.0 deg for <=1966/67, plain merged netCDF4 at
0.25 deg after) exactly once and writes:

BIGDATA-resident (ERA5_derived/, gitignored):
  cell_daily/region=R/YYYYMM.parquet   quantized per-cell daily stats incl.
                                       hourly-irrecoverable true Tmin/Tmax,
                                       rolling 1/3/6h precip maxima, wet/heat/
                                       frost hour counts  (~150-200 GB total)
  cell_monthly/region=R/YYYYMM.parquet per-cell monthly stats incl. consecutive
                                       heat/frost hour spells and wet-hour
                                       intensity (p99, SDII)  (~10-15 GB)
  _country_parts/, _tbin_parts/        per region-month staging for finalize
  ref/                                 cells registry, sparse weights, manifest

Laptop bundle (analysis/data/era5_derived/, ~1-1.5 GB total):
  era5_country_daily.parquet           iso3 x day 1950-2025; area-, pop2000-
                                       and crop2000-weighted T/Tmin/Tmax/P,
                                       exposure hours, spatial dispersion,
                                       max-hourly-deluge indicator
  era5_country_day_tbins.parquet       iso3 x day x {pop,crop}: hours of
                                       exposure in 3 degC bins (-15..+39),
                                       the Deschenes-Greenstone object
  era5_country_monthly_v2.parquet      superset of the era5_country_monthly
                                       schema (iso3, year, month, t2m_c,
                                       tp_mm + extensions)
  ref/era5_cells.parquet, ref/era5_cell_country_weights.parquet

Aggregation conventions
-----------------------
- Days are UTC days (valid_time is UTC). Rolling precip windows are assigned
  to the day in which the window ENDS.
- Country values are weighted means over grid cells, weights from the HYDE
  5-arcmin rasters mapped to the nearest ERA5 cell: w_area = cos(lat),
  w_pop = HYDE population 2000 AD, w_crop = HYDE cropland km^2 2000 AD
  (fixed epoch so the climate signal is not confounded by population
  movement). Overlapping region tiles carry identical fields; cross-region
  combination sums weighted values and weights (identical to the
  weight_sum-blend convention in build_era5_country_monthly.py).
- Cell-level products are deduplicated at write time: each canonical cell is
  emitted only by its owner region (lowest region id containing it).
- Deferred (recomputable later; raw archive is retained): per-cell
  climatology normals/percentiles and the diurnal-cycle climatology.

Usage
-----
  python -m analysis.shared.build_era5_compact prepass
  python -m analysis.shared.build_era5_compact sweep    [--workers 6]
                                                        [--regions 11,12]
                                                        [--years 2010-2012]
  python -m analysis.shared.build_era5_compact finalize
The sweep is resumable: region-months whose outputs already exist are skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
ERA5_DIR = ROOT / "ERA5"
DERIVED = ROOT / "ERA5_derived"
REF_DIR = DERIVED / "ref"
LAPTOP_DIR = ROOT / "analysis" / "data" / "era5_derived"

COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
COUNTRY_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"
HYDE_POP_NC = ROOT / "gbc2025_7apr_base" / "NetCDF" / "population.nc"
HYDE_CROP_NC = ROOT / "gbc2025_7apr_base" / "NetCDF" / "cropland.nc"
EXISTING_MONTHLY = ROOT / "analysis" / "data" / "era5_country_monthly.parquet"

REGIONS = range(1, 26)
_MONTH_RE = re.compile(r"era5_(\d+)_(\d{4})(\d{2})\.nc$")

# 3 degC exposure-bin lattice: (-inf,-15], (-15,-12], ..., (36,39], (39,inf)
TBIN_EDGES = np.arange(-15.0, 40.0, 3.0)  # 19 edges -> 20 bins
N_TBINS = len(TBIN_EDGES) + 1
WET_HOUR_MM = 0.1  # tp >= 0.1 mm counts as a wet hour
RES_FLAG_1DEG = np.uint32(1 << 31)


# ───────────────────────── raw-file access ─────────────────────────

def _is_zip(path: Path) -> bool:
    """Container check via leading magic bytes.

    zipfile.is_zipfile() scans for the end-of-central-directory signature
    anywhere near the file tail and yields false positives on HDF5 files
    (observed: region=1 1979-06), so only trust the leading local-file magic.
    """
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"


def open_month(path: Path) -> xr.Dataset | None:
    """Open a raw monthly file (zip container or plain merged netCDF4)."""
    try:
        if _is_zip(path):
            with zipfile.ZipFile(path) as zf:
                parts = []
                for name in zf.namelist():
                    buf = io.BytesIO(zf.read(name))
                    parts.append(xr.open_dataset(buf).load())
                if not parts:
                    return None
                return parts[0] if len(parts) == 1 else xr.merge(
                    parts, compat="override", join="outer")
        return xr.open_dataset(path)
    except Exception:
        return None


def iter_raw_files(regions=None, years=None):
    """Yield (region, year, month, path) over the raw archive."""
    for r in (regions or REGIONS):
        reg_dir = ERA5_DIR / f"region={r}"
        if not reg_dir.exists():
            continue
        for yr_dir in sorted(reg_dir.glob("year=*")):
            y = int(yr_dir.name.split("=")[1])
            if years and y not in years:
                continue
            for nc in sorted(yr_dir.glob("era5_*.nc")):
                m = _MONTH_RE.search(nc.name)
                if m:
                    yield r, y, int(m.group(3)), nc


# ───────────────────────── prepass: cells + weights ─────────────────────────

def _detect_grid(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, float]:
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    res = round(abs(float(lat[1] - lat[0])), 4)
    return lat, lon, res


def _cell_ids(lat: np.ndarray, lon: np.ndarray, res: float) -> np.ndarray:
    """Canonical uint32 ids on the global lattice at this resolution.

    floor(+eps) rather than round: the 1.0-deg tiles sit on half-degree
    offsets (41.5, 42.5, ...), where round() would collide adjacent cells
    via banker's rounding. Index units are multiples of 0.5, so eps=0.01 is
    safe against float error in both the on-lattice and half-offset cases.
    """
    nlon_glob = int(round(360.0 / res))
    lat_idx = np.floor((90.0 - lat) / res + 0.01).astype(np.int64)
    lon_idx = np.floor((lon % 360.0) / res + 0.01).astype(np.int64) % nlon_glob
    ids = (lat_idx[:, None] * nlon_glob + lon_idx[None, :]).astype(np.uint32)
    if res >= 0.5:
        ids = ids | RES_FLAG_1DEG
    return ids  # (nlat, nlon)


def _read_iso_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-arcmin country-code raster (cached as .npy after first parse)."""
    cache = REF_DIR / "iso_cr.npy"
    with open(COUNTRY_RASTER) as f:
        h = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            h[k.lower()] = float(v)
    ncols, nrows = int(h["ncols"]), int(h["nrows"])
    lon = h["xllcorner"] + (np.arange(ncols) + 0.5) * h["cellsize"]
    lat = h["yllcorner"] + (np.arange(nrows)[::-1] + 0.5) * h["cellsize"]
    if cache.exists():
        codes = np.load(cache)
    else:
        codes = np.loadtxt(COUNTRY_RASTER, skiprows=6, dtype=np.int32)
        codes[codes == int(h["nodata_value"])] = -1
        REF_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, codes)
    return codes, lat, lon  # lat descending (index 0 = north)


def _hyde_2000(nc_path: Path, var: str) -> np.ndarray:
    ds = xr.open_dataset(nc_path)
    t2000 = [t for t in ds["time"].values if t.year == 2000][0]
    arr = ds[var].sel(time=t2000).values.astype(np.float64)
    ds.close()
    return np.nan_to_num(arr, nan=0.0)  # (2160, 4320), lat descending


def _nearest_idx(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of nearest grid point for each val; grid may be asc or desc."""
    asc = grid if grid[0] <= grid[-1] else grid[::-1]
    j = np.searchsorted(asc, vals)
    j = np.clip(j, 1, len(asc) - 1)
    left = asc[j - 1]
    right = asc[j]
    idx_asc = np.where(vals - left <= right - vals, j - 1, j)
    if grid[0] <= grid[-1]:
        return idx_asc
    return len(grid) - 1 - idx_asc


def build_signature(region: int, ds: xr.Dataset,
                    codes, lat5, lon5, pop5, crop5, num_to_iso3, iso3_list):
    """Sparse country-weight matrices + cell ids for one (region, res) grid."""
    lat, lon, res = _detect_grid(ds)
    nlat, nlon = len(lat), len(lon)
    ids = _cell_ids(lat, lon, res)

    # Restrict 5-arcmin cells to the tile bbox (+1 cell buffer), matching the
    # convention in build_era5_country_monthly.py.
    lon_t = lon.copy()
    lat_in = (lat5 >= lat.min() - res) & (lat5 <= lat.max() + res)
    lon5n = lon5.copy()
    if lon_t.min() >= 0:  # tile on 0..360 convention
        lon5n = lon5 % 360.0
    lon_in = (lon5n >= lon_t.min() - res) & (lon5n <= lon_t.max() + res)

    ri = np.flatnonzero(lat_in)
    ci = np.flatnonzero(lon_in)
    sub_codes = codes[np.ix_(ri, ci)]
    valid = sub_codes > 0
    if not valid.any():
        return None

    lat_map = _nearest_idx(lat5[ri], lat)          # (nri,)
    lon_map = _nearest_idx(lon5n[ci], lon_t)       # (nci,)
    rr, cc = np.nonzero(valid)
    cell_flat = (lat_map[rr] * nlon + lon_map[cc]).astype(np.int64)

    iso_idx_lookup = np.full(int(codes.max()) + 1, -1, dtype=np.int32)
    for num, iso in num_to_iso3.items():
        if num <= codes.max():
            iso_idx_lookup[num] = iso3_list.index(iso)
    iso_rows = iso_idx_lookup[sub_codes[rr, cc]]
    keep = iso_rows >= 0
    rr, cc, cell_flat, iso_rows = rr[keep], cc[keep], cell_flat[keep], iso_rows[keep]

    w_area = np.cos(np.deg2rad(lat5[ri][rr]))
    w_pop = pop5[np.ix_(ri, ci)][rr, cc]
    w_crop = crop5[np.ix_(ri, ci)][rr, cc]

    def mk(w):
        return sparse.coo_matrix(
            (w, (iso_rows, cell_flat)), shape=(len(iso3_list), nlat * nlon)
        ).tocsr()

    return {
        "region": region, "res": res, "lat": lat, "lon": lon,
        "ids": ids.ravel(),
        "W_area": mk(w_area), "W_pop": mk(w_pop), "W_crop": mk(w_crop),
    }


def prepass() -> None:
    """Build per-(region, res) weight matrices, owner masks, cell registry."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading 5-arcmin rasters (iso_cr + HYDE pop/crop 2000)...", flush=True)
    codes, lat5, lon5 = _read_iso_raster()
    pop5 = _hyde_2000(HYDE_POP_NC, "population")
    crop5 = _hyde_2000(HYDE_CROP_NC, "cropland")
    mp = pd.read_csv(COUNTRY_MAP_CSV).dropna(subset=["iso_num", "iso3"])
    num_to_iso3 = dict(zip(mp["iso_num"].astype(int), mp["iso3"]))
    iso3_list = sorted(set(num_to_iso3.values()))
    (REF_DIR / "iso3_list.json").write_text(json.dumps(iso3_list))

    sigs = {}
    for r in REGIONS:
        # one file from each resolution era: 1950 (1.0 deg) and 2020 (0.25 deg)
        for probe_year in (1950, 2020):
            files = sorted((ERA5_DIR / f"region={r}" / f"year={probe_year}").glob("era5_*.nc"))
            if not files:
                continue
            ds = open_month(files[0])
            if ds is None:
                continue
            sig = build_signature(r, ds, codes, lat5, lon5, pop5, crop5,
                                  num_to_iso3, iso3_list)
            ds.close()
            if sig is None:
                continue
            key = f"r{r}_res{sig['res']}"
            sigs[key] = sig
            print(f"  {key}: grid {len(sig['lat'])}x{len(sig['lon'])}, "
                  f"nnz(area)={sig['W_area'].nnz}", flush=True)

    # Owner region per canonical cell (lowest region id), per resolution era
    owner: dict[int, int] = {}
    for key in sorted(sigs, key=lambda k: sigs[k]["region"]):
        s = sigs[key]
        for cid in s["ids"]:
            c = int(cid)
            if c not in owner or s["region"] < owner[c]:
                owner[c] = s["region"]

    cells_rows = []
    for key, s in sigs.items():
        lat_g = np.repeat(s["lat"], len(s["lon"]))
        lon_g = np.tile(s["lon"], len(s["lat"]))
        own = np.array([owner[int(c)] for c in s["ids"]], dtype=np.uint8)
        mask = own == s["region"]
        land = np.asarray(s["W_area"].sum(axis=0)).ravel() > 0
        np.savez_compressed(
            REF_DIR / f"sig_{key}.npz",
            ids=s["ids"], owner_mask=mask, land_mask=land,
            lat=s["lat"], lon=s["lon"], res=s["res"],
        )
        sparse.save_npz(REF_DIR / f"W_area_{key}.npz", s["W_area"])
        sparse.save_npz(REF_DIR / f"W_pop_{key}.npz", s["W_pop"])
        sparse.save_npz(REF_DIR / f"W_crop_{key}.npz", s["W_crop"])
        cells_rows.append(pd.DataFrame({
            "cell_id": s["ids"], "lat": lat_g.astype(np.float32),
            "lon": lon_g.astype(np.float32),
            "res_deg": np.float32(s["res"]),
            "region": np.uint8(s["region"]), "owner": mask,
            "is_land": land,
        }))

    cells = pd.concat(cells_rows, ignore_index=True)
    n_regions = cells.groupby("cell_id")["region"].nunique().rename("n_regions")
    reg = cells[cells["owner"]].drop(columns=["owner"]).rename(
        columns={"region": "owner_region"})
    # cell products cover land cells only (nonzero country weight)
    reg = reg.merge(n_regions, on="cell_id")
    reg["area_km2"] = (
        (111.32 * reg["res_deg"]) ** 2 * np.cos(np.deg2rad(reg["lat"]))
    ).astype(np.float32)
    LAPTOP_DIR.joinpath("ref").mkdir(parents=True, exist_ok=True)
    reg.to_parquet(LAPTOP_DIR / "ref" / "era5_cells.parquet", index=False)

    # Long-form cell->country weights for owner cells (laptop reference)
    wrows = []
    for key, s in sigs.items():
        mask = np.load(REF_DIR / f"sig_{key}.npz")["owner_mask"]
        for wname in ("area", "pop", "crop"):
            coo = s[f"W_{wname}"].tocoo()
            keep = mask[coo.col]
            wrows.append(pd.DataFrame({
                "cell_id": s["ids"][coo.col[keep]],
                "iso3": pd.Categorical.from_codes(
                    coo.row[keep], categories=iso3_list),
                "scheme": wname,
                "weight": coo.data[keep].astype(np.float32),
            }))
    pd.concat(wrows, ignore_index=True).to_parquet(
        LAPTOP_DIR / "ref" / "era5_cell_country_weights.parquet", index=False)
    print(f"prepass done: {len(sigs)} signatures, {len(reg)} canonical cells",
          flush=True)


# ───────────────────────── sweep worker ─────────────────────────

_SIG_CACHE: dict[str, dict] = {}


def _load_sig(key: str) -> dict:
    if key not in _SIG_CACHE:
        z = np.load(REF_DIR / f"sig_{key}.npz")
        _SIG_CACHE[key] = {
            "ids": z["ids"],
            "cellmask": z["owner_mask"] & z["land_mask"],
            "lat": z["lat"], "lon": z["lon"], "res": float(z["res"]),
            "W_area": sparse.load_npz(REF_DIR / f"W_area_{key}.npz"),
            "W_pop": sparse.load_npz(REF_DIR / f"W_pop_{key}.npz"),
            "W_crop": sparse.load_npz(REF_DIR / f"W_crop_{key}.npz"),
            "iso3": json.loads((REF_DIR / "iso3_list.json").read_text()),
        }
        for w in ("W_area", "W_pop", "W_crop"):
            _SIG_CACHE[key][f"{w}_sum"] = np.asarray(
                _SIG_CACHE[key][w].sum(axis=1)).ravel()
        s = _SIG_CACHE[key]
        s["support"] = [s["W_area"].indices[
            s["W_area"].indptr[i]:s["W_area"].indptr[i + 1]]
            for i in range(s["W_area"].shape[0])]
    return _SIG_CACHE[key]


def _q16(x, scale):  # quantize to int16
    return np.clip(np.round(x * scale), -32767, 32767).astype(np.int16)


def _qu16(x, scale):
    return np.clip(np.round(x * scale), 0, 65535).astype(np.uint16)


def _max_consec(mask_h: np.ndarray) -> np.ndarray:
    """Longest True-run per cell; mask_h is (n_hours, n_cells) bool."""
    run = np.zeros(mask_h.shape[1], dtype=np.uint16)
    best = np.zeros(mask_h.shape[1], dtype=np.uint16)
    for h in range(mask_h.shape[0]):
        run = (run + mask_h[h]) * mask_h[h]
        np.maximum(best, run, out=best)
    return best


def process_month(task: tuple) -> dict:
    region, year, month, path_str = task
    t0 = time.time()
    path = Path(path_str)
    out_cd = DERIVED / "cell_daily" / f"region={region}" / f"{year}{month:02d}.parquet"
    out_cm = DERIVED / "cell_monthly" / f"region={region}" / f"{year}{month:02d}.parquet"
    out_cp = DERIVED / "_country_parts" / f"region={region}" / f"{year}{month:02d}.parquet"
    out_tb = DERIVED / "_tbin_parts" / f"region={region}" / f"{year}{month:02d}.parquet"
    if all(p.exists() for p in (out_cd, out_cm, out_cp, out_tb)):
        return {"region": region, "year": year, "month": month,
                "status": "skipped", "wall_s": 0.0}
    try:
        ds = open_month(path)
        if ds is None or "t2m" not in ds.data_vars or "tp" not in ds.data_vars:
            return {"region": region, "year": year, "month": month,
                    "status": "error", "error": "open/vars", "wall_s": 0.0}
        lat, lon, res = _detect_grid(ds)
        key = f"r{region}_res{res}"
        sig = _load_sig(key)
        if len(lat) != len(sig["lat"]) or len(lon) != len(sig["lon"]):
            ds.close()
            return {"region": region, "year": year, "month": month,
                    "status": "error", "error": "grid-mismatch", "wall_s": 0.0}

        nh = ds.sizes["valid_time"]
        nd = nh // 24
        nc = len(lat) * len(lon)

        t2m = ds["t2m"].values.astype(np.float32).reshape(nh, nc) - 273.15
        tp = ds["tp"].values.astype(np.float32).reshape(nh, nc) * 1000.0  # mm/h
        ds.close()
        np.nan_to_num(tp, copy=False, nan=0.0)
        t2m_kmin, t2m_kmax = float(np.nanmin(t2m)), float(np.nanmax(t2m))
        tp_hmax = float(tp.max())

        td = t2m[:nd * 24].reshape(nd, 24, nc)
        t_mean = td.mean(axis=1)                        # (nd, nc)
        t_min = td.min(axis=1)
        t_max = td.max(axis=1)
        pd_ = tp[:nd * 24].reshape(nd, 24, nc)
        p_day = pd_.sum(axis=1)                         # mm/day

        # rolling 1/3/6h maxima, window assigned to the day it ends in
        cs = np.vstack([np.zeros((1, nc), np.float32), np.cumsum(tp, axis=0)])
        roll = {}
        for w in (1, 3, 6):
            r = cs[w:] - cs[:-w]                        # (nh-w+1, nc)
            pad = np.vstack([np.zeros((w - 1, nc), np.float32), r])[:nd * 24]
            roll[w] = pad.reshape(nd, 24, nc).max(axis=1)

        wet_h = (pd_ >= WET_HOUR_MM).sum(axis=1).astype(np.uint8)
        hot_h = (td >= 30.0).sum(axis=1).astype(np.uint8)
        frost_h = (td <= 0.0).sum(axis=1).astype(np.uint8)

        mask = sig["cellmask"]
        days = np.repeat(np.arange(1, nd + 1, dtype=np.uint8), int(mask.sum()))
        cell_rep = np.tile(sig["ids"][mask], nd)
        pd.DataFrame({
            "day": days, "cell_id": cell_rep,
            "t2m_mean_c": _q16(t_mean[:, mask].ravel(), 100),
            "t2m_min_c": _q16(t_min[:, mask].ravel(), 100),
            "t2m_max_c": _q16(t_max[:, mask].ravel(), 100),
            "tp_mm": _qu16(p_day[:, mask].ravel(), 10),
            "tp_max1h_mm": _qu16(roll[1][:, mask].ravel(), 10),
            "tp_max3h_mm": _qu16(roll[3][:, mask].ravel(), 10),
            "tp_max6h_mm": _qu16(roll[6][:, mask].ravel(), 10),
            "wet_hours": wet_h[:, mask].ravel(),
            "hours_ge30c": hot_h[:, mask].ravel(),
            "hours_le0c": frost_h[:, mask].ravel(),
        }).to_parquet(_tmpwrite(out_cd), index=False)
        _commit(out_cd)

        # cell_monthly (owner cells only)
        wet_hourly = tp >= WET_HOUR_MM
        n_wet = wet_hourly.sum(axis=0)
        tp_wet = np.where(wet_hourly, tp, np.nan)
        with np.errstate(all="ignore"):
            p99 = np.nanpercentile(tp_wet, 99, axis=0)
            sdii = np.nanmean(tp_wet, axis=0)
        p99 = np.nan_to_num(p99, nan=0.0)
        sdii = np.nan_to_num(sdii, nan=0.0)
        pd.DataFrame({
            "cell_id": sig["ids"][mask],
            "t2m_c": _q16(t_mean[:, mask].mean(axis=0), 100),
            "tmin_mean_c": _q16(t_min[:, mask].mean(axis=0), 100),
            "tmax_mean_c": _q16(t_max[:, mask].mean(axis=0), 100),
            "t2m_min_mo_c": _q16(t_min[:, mask].min(axis=0), 100),
            "t2m_max_mo_c": _q16(t_max[:, mask].max(axis=0), 100),
            "t_day_sd_c": _q16(t_mean[:, mask].std(axis=0, ddof=0), 100),
            "tp_mm": p_day[:, mask].sum(axis=0).astype(np.float32),
            "tp_max1d_mm": _qu16(p_day[:, mask].max(axis=0), 10),
            "wet_days": (p_day[:, mask] >= 1.0).sum(axis=0).astype(np.uint8),
            "hot_days": (t_max[:, mask] >= 30.0).sum(axis=0).astype(np.uint8),
            "frost_days": (t_min[:, mask] <= 0.0).sum(axis=0).astype(np.uint8),
            "max_consec_hours_ge30c": _max_consec(t2m[:, mask] >= 30.0),
            "max_consec_hours_le0c": _max_consec(t2m[:, mask] <= 0.0),
            "tp_p99_wethour_mmh": p99[mask].astype(np.float32),
            "sdii_wethour_mmh": sdii[mask].astype(np.float32),
            "wet_hours_mo": n_wet[mask].astype(np.uint16),
        }).to_parquet(_tmpwrite(out_cm), index=False)
        _commit(out_cm)
        del tp_wet, wet_hourly, cs

        # country parts: weighted SUMS (finalize divides / blends)
        iso3 = sig["iso3"]
        fields = np.stack([t_mean, t_min, t_max, p_day,
                           wet_h.astype(np.float32),
                           hot_h.astype(np.float32),
                           frost_h.astype(np.float32)], axis=-1)  # (nd,nc,7)
        recs = {}
        for wname in ("area", "pop", "crop"):
            W = sig[f"W_{wname}"]
            vals = np.empty((W.shape[0], nd, 7), dtype=np.float64)
            for d in range(nd):
                vals[:, d, :] = W @ fields[d]
            recs[wname] = (vals, sig[f"W_{wname}_sum"])
        active = np.flatnonzero(sig["W_area_sum"] > 0)
        # pop-weighted spatial dispersion of daily mean (E[x^2] term)
        Wp = sig["W_pop"]
        t2 = np.empty((Wp.shape[0], nd), dtype=np.float64)
        for d in range(nd):
            t2[:, d] = Wp @ (t_mean[d] ** 2)
        # deluge: per-country max over supported cells of the day's max-1h rain
        deluge = np.zeros((len(iso3), nd), dtype=np.float32)
        r1h = roll[1]
        for i in active:
            supp = sig["support"][i]
            if len(supp):
                deluge[i] = r1h[:, supp].max(axis=1)
        del roll

        rows = []
        dates = pd.to_datetime(
            [f"{year}-{month:02d}-{d:02d}" for d in range(1, nd + 1)])
        names = ["t2m", "tmin", "tmax", "tp", "wet_hours", "hot_hours",
                 "frost_hours"]
        for i in active:
            df_i = pd.DataFrame({"date": dates})
            for wname, suffix in (("area", "aw"), ("pop", "pw"), ("crop", "cw")):
                vals, ws = recs[wname]
                for k, nme in enumerate(names):
                    df_i[f"s_{nme}_{suffix}"] = vals[i, :, k]
                df_i[f"ws_{suffix}"] = ws[i]
            df_i["s_t2m2_pw"] = t2[i]
            df_i["tp_max1h_cellmax"] = deluge[i]
            df_i["iso3"] = iso3[i]
            df_i["region"] = np.uint8(region)
            rows.append(df_i)
        pd.concat(rows, ignore_index=True).to_parquet(
            _tmpwrite(out_cp), index=False)
        _commit(out_cp)

        # tbins: hours of exposure in 3C bins, pop- and crop-weighted sums.
        # Per-day counts only (nc x 20) to bound memory on the big tiles.
        cell_offsets = np.arange(nc, dtype=np.int64)[None, :] * N_TBINS
        tb_rows = []
        for d in range(nd):
            sl = np.digitize(t2m[d * 24:(d + 1) * 24], TBIN_EDGES)
            counts_d = np.zeros(nc * N_TBINS, dtype=np.float32)
            np.add.at(counts_d, (cell_offsets + sl).ravel(), 1.0)
            counts_d = counts_d.reshape(nc, N_TBINS)
            for wname, Wn in (("pop", "W_pop"), ("crop", "W_crop")):
                ws = sig[f"{Wn}_sum"]
                wb = sig[Wn] @ counts_d                 # (niso, N_TBINS)
                for i in active:
                    if ws[i] <= 0:
                        continue
                    tb_rows.append(
                        (iso3[i], dates[d], wname, float(ws[i]),
                         *wb[i].astype(np.float64)))
        tb = pd.DataFrame(
            tb_rows, columns=["iso3", "date", "wscheme", "ws",
                              *[f"b{k:02d}" for k in range(N_TBINS)]])
        tb["region"] = np.uint8(region)
        tb.to_parquet(_tmpwrite(out_tb), index=False)
        _commit(out_tb)

        return {"region": region, "year": year, "month": month,
                "status": "ok", "res": res, "n_hours": nh,
                "t2m_min_c": round(t2m_kmin, 2), "t2m_max_c": round(t2m_kmax, 2),
                "tp_max_hourly_mm": round(tp_hmax, 2),
                "wall_s": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"region": region, "year": year, "month": month,
                "status": "error", "error": repr(exc)[:200],
                "wall_s": round(time.time() - t0, 2)}


def _tmpwrite(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(".tmp.parquet")


def _commit(path: Path) -> None:
    path.with_suffix(".tmp.parquet").rename(path)


# ───────────────────────── sweep driver ─────────────────────────

def sweep(workers: int = 6, regions=None, years=None) -> None:
    import multiprocessing as mp

    tasks = [(r, y, m, str(p)) for r, y, m, p in iter_raw_files(regions, years)]
    # interleave big tiles (16, 24) with the rest to smooth memory usage
    tasks.sort(key=lambda t: (t[1], t[2], t[0]))
    print(f"sweep: {len(tasks)} region-months, {workers} workers", flush=True)
    manifest = REF_DIR / "sweep_manifest.jsonl"
    done = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, open(manifest, "a") as mf:
        for res in pool.imap_unordered(process_month, tasks, chunksize=4):
            res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            mf.write(json.dumps(res) + "\n")
            mf.flush()
            done += 1
            if done % 200 == 0 or res["status"] == "error":
                rate = done / (time.time() - t0)
                eta = (len(tasks) - done) / max(rate, 1e-9) / 3600
                print(f"  {done}/{len(tasks)} ({res['status']}) "
                      f"eta {eta:.1f} h", flush=True)
    print("sweep complete", flush=True)


# ───────────────────────── finalize ─────────────────────────

def finalize() -> None:
    LAPTOP_DIR.mkdir(parents=True, exist_ok=True)

    print("finalize: blending country parts...", flush=True)
    parts = sorted((DERIVED / "_country_parts").glob("region=*/*.parquet"))
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    sums = df.drop(columns=["region"]).groupby(
        ["iso3", "date"], as_index=False, observed=True).agg(
        {c: ("max" if c == "tp_max1h_cellmax" else "sum")
         for c in df.columns if c not in ("iso3", "date", "region")})

    out = sums[["iso3", "date"]].copy()
    for suffix, tag in (("aw", ""), ("pw", "_pw"), ("cw", "_cw")):
        ws = sums[f"ws_{suffix}"]
        for nme, col in (("t2m", "t2m_c"), ("tmin", "tmin_c"),
                         ("tmax", "tmax_c"), ("tp", "tp_mm"),
                         ("wet_hours", "wet_hours"), ("hot_hours", "hot_hours"),
                         ("frost_hours", "frost_hours")):
            out[f"{col}{tag}"] = (
                sums[f"s_{nme}_{suffix}"] / ws.replace(0, np.nan)
            ).astype(np.float32)
        out[f"weight_sum{tag}"] = ws.astype(np.float32)
    m_pw = sums["s_t2m_pw"] / sums["ws_pw"].replace(0, np.nan)
    v_pw = sums["s_t2m2_pw"] / sums["ws_pw"].replace(0, np.nan) - m_pw ** 2
    out["t_sd_space_pw"] = np.sqrt(np.clip(v_pw, 0, None)).astype(np.float32)
    out["tp_max1h_cellmax"] = sums["tp_max1h_cellmax"].astype(np.float32)
    out = out.sort_values(["iso3", "date"]).reset_index(drop=True)
    out.to_parquet(LAPTOP_DIR / "era5_country_daily.parquet", index=False)
    print(f"  era5_country_daily: {len(out):,} rows "
          f"({out['iso3'].nunique()} countries, "
          f"{out['date'].min():%Y-%m-%d}..{out['date'].max():%Y-%m-%d})",
          flush=True)

    print("finalize: blending tbins...", flush=True)
    tparts = sorted((DERIVED / "_tbin_parts").glob("region=*/*.parquet"))
    tb = pd.concat((pd.read_parquet(p) for p in tparts), ignore_index=True)
    bcols = [f"b{k:02d}" for k in range(N_TBINS)]
    tsum = tb.groupby(["iso3", "date", "wscheme"], as_index=False,
                      observed=True)[["ws", *bcols]].sum()
    for c in bcols:
        tsum[c] = (tsum[c] / tsum["ws"]).astype(np.float32)
    tsum["ws"] = tsum["ws"].astype(np.float32)
    tsum = tsum.rename(columns={"ws": "weight_sum"})
    tsum.sort_values(["iso3", "date", "wscheme"]).to_parquet(
        LAPTOP_DIR / "era5_country_day_tbins.parquet", index=False)
    print(f"  era5_country_day_tbins: {len(tsum):,} rows", flush=True)

    print("finalize: monthly v2 + validation...", flush=True)
    d = out.copy()
    d["year"] = d["date"].dt.year.astype("int64")
    d["month"] = d["date"].dt.month.astype("int64")
    g = d.groupby(["iso3", "year", "month"], as_index=False, observed=True)
    v2 = g.agg(
        t2m_c=("t2m_c", "mean"), tp_mm=("tp_mm", "sum"),
        tmin_c_mean=("tmin_c", "mean"), tmax_c_mean=("tmax_c", "mean"),
        t2m_daily_sd_c=("t2m_c", "std"),
        tp_max1d_mm=("tp_mm", "max"), tp_max1h_mm=("tp_max1h_cellmax", "max"),
        wet_hours=("wet_hours", "sum"), hot_hours=("hot_hours", "sum"),
        frost_hours=("frost_hours", "sum"),
        t2m_c_pw=("t2m_c_pw", "mean"), tp_mm_pw=("tp_mm_pw", "sum"),
        t2m_c_cw=("t2m_c_cw", "mean"), tp_mm_cw=("tp_mm_cw", "sum"),
        weight_sum=("weight_sum", "mean"),
    )
    for c in v2.columns[3:]:
        v2[c] = v2[c].astype(np.float32)
    v2.to_parquet(LAPTOP_DIR / "era5_country_monthly_v2.parquet", index=False)
    print(f"  era5_country_monthly_v2: {len(v2):,} rows", flush=True)

    if EXISTING_MONTHLY.exists():
        old = pd.read_parquet(EXISTING_MONTHLY)
        cmp = old[old["year"] <= 1966].merge(
            v2[["iso3", "year", "month", "t2m_c", "tp_mm"]],
            on=["iso3", "year", "month"], suffixes=("_old", "_new"))
        if len(cmp):
            dt = (cmp["t2m_c_old"] - cmp["t2m_c_new"]).abs()
            dp = (cmp["tp_mm_old"] - cmp["tp_mm_new"]).abs()
            print(f"  validation vs {EXISTING_MONTHLY.name} (<=1966, "
                  f"{len(cmp):,} rows): |dT| median {dt.median():.4f} C, "
                  f"p99 {dt.quantile(0.99):.3f}, max {dt.max():.3f}; "
                  f"|dP| median {dp.median():.3f} mm, max {dp.max():.2f}",
                  flush=True)
    print("finalize done", flush=True)


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepass")
    sw = sub.add_parser("sweep")
    sw.add_argument("--workers", type=int, default=6)
    sw.add_argument("--regions", type=str, default=None,
                    help="comma list, e.g. 11,12")
    sw.add_argument("--years", type=str, default=None,
                    help="range, e.g. 2010-2012")
    sub.add_parser("finalize")
    args = ap.parse_args()

    if args.cmd == "prepass":
        prepass()
    elif args.cmd == "sweep":
        regions = ([int(x) for x in args.regions.split(",")]
                   if args.regions else None)
        years = None
        if args.years:
            a, _, b = args.years.partition("-")
            years = set(range(int(a), int(b or a) + 1))
        sweep(workers=args.workers, regions=regions, years=years)
    elif args.cmd == "finalize":
        finalize()


if __name__ == "__main__":
    main()
