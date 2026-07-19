"""Bulk ERA5 downloader — requests quarters (3 months) in single API calls.

Instead of one CDS API request per month (slow due to queue waits),
this batches missing months into quarterly chunks (up to 3 months per call),
reducing API calls by ~3x while staying within CDS server limits.

Fully compatible with the original era5_downloader.py — uses the same
file layout and is_downloaded() checks, so it won't re-download anything.

Usage:
    python -m analysis.shared.era5_bulk_downloader
    python -m analysis.shared.era5_bulk_downloader --workers 4
    python -m analysis.shared.era5_bulk_downloader --regions 10 11 12
    python -m analysis.shared.era5_bulk_downloader --status
"""

import argparse
import cdsapi
import ctypes
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

# ── Configuration (same as original) ─────────────────────────────────────────

ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = ["2m_temperature", "total_precipitation"]
ERA5_ROOT = Path("/Volumes/BIGDATA/HYDE35/ERA5")

YEAR_START = 1950
YEAR_END = 2025
MONTHS = list(range(1, 13))
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

# Batch size: how many months per API call (3 = quarterly)
BATCH_MONTHS = 3

MAX_RETRIES = 6
INITIAL_BACKOFF_SEC = 30
MAX_BACKOFF_SEC = 600
MIN_FILE_SIZE_BYTES = 1000

# CDS client tuning — don't let it poll forever
CDS_TIMEOUT = 300       # seconds per HTTP request
CDS_RETRY_MAX = 200     # max server-side polls (200 × 60s = ~3.3 hours max wait)
CDS_SLEEP_MAX = 60      # poll interval (seconds)


def _set_batch_size(n: int):
    global BATCH_MONTHS
    BATCH_MONTHS = n


# ── HDF5 stderr-noise suppression ────────────────────────────────────────────
# netCDF4's nc_create(NC_CLOBBER) probes the destination path with
# H5Fis_accessible(); when the file does not yet exist (the normal case for a
# fresh per-month write), HDF5 emits a multi-line stderr diagnostic. Its
# auto-error handler is per-thread, so each ThreadPoolExecutor worker prints
# the noise until we silence its handler. We resolve netCDF4's bundled
# libhdf5 dynamically and call H5Eset_auto2 from every writer thread.

_HDF5_LIB = None  # ctypes.CDLL once initialized; False if unavailable


def _silence_hdf5_thread() -> None:
    """Disable HDF5 automatic error printing in the calling thread."""
    global _HDF5_LIB
    if _HDF5_LIB is None:
        try:
            import netCDF4
            nc_dir = Path(netCDF4._netCDF4.__file__).parent
            env_lib = Path(sys.prefix) / "lib"
            cand = (
                list((nc_dir / ".dylibs").glob("libhdf5.*.dylib"))             # macOS wheel
                or list((nc_dir.parent / "netCDF4.libs").glob("libhdf5*.so*"))   # Linux wheel
                or list(env_lib.glob("libhdf5.dylib"))                           # conda macOS
                or list(env_lib.glob("libhdf5.so*"))                             # conda Linux
            )
            if not cand:
                _HDF5_LIB = False
                return
            lib = ctypes.CDLL(str(cand[0]))
            lib.H5open.argtypes = []
            lib.H5open.restype = ctypes.c_int
            lib.H5Eset_auto2.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
            lib.H5Eset_auto2.restype = ctypes.c_int
            _HDF5_LIB = lib
        except Exception:
            _HDF5_LIB = False
            return
    if _HDF5_LIB is False:
        return
    try:
        _HDF5_LIB.H5open()
        _HDF5_LIB.H5Eset_auto2(0, None, None)  # H5E_DEFAULT == 0
    except Exception:
        pass


_silence_hdf5_thread()  # silence the main thread's HDF5 handler at import


# ── Shared helpers (from original downloader) ────────────────────────────────

def load_region_bboxes() -> dict:
    """Load region bounding boxes from existing extracted NetCDF files."""
    bboxes = {}
    for reg_dir in sorted(ERA5_ROOT.iterdir()):
        if not reg_dir.name.startswith("region="):
            continue
        reg_num = int(reg_dir.name.split("=")[1])
        for yr_dir in sorted(reg_dir.iterdir()):
            ext_path = yr_dir / "_extracted"
            if not ext_path.exists():
                continue
            for f in ext_path.iterdir():
                if f.suffix == ".nc":
                    try:
                        ds = xr.open_dataset(f, engine="netcdf4")
                        bboxes[reg_num] = {
                            "north": float(ds.latitude.values.max()),
                            "south": float(ds.latitude.values.min()),
                            "east": float(ds.longitude.values.max()),
                            "west": float(ds.longitude.values.min()),
                        }
                        ds.close()
                    except Exception:
                        continue
                    break
            if reg_num in bboxes:
                break
    if not bboxes:
        raise RuntimeError("No existing ERA5 extractions found for bounding boxes.")
    return bboxes


def is_downloaded(region: int, year: int, month: int) -> bool:
    """Check if a month's per-month .nc file exists and is non-trivial.

    Post-fix: no longer trusts _extracted/ presence — that was the source of
    Phase 9's data-integrity issue (multiple zips overwrote each other in
    _extracted/, but is_downloaded() returned True after the first extraction).
    """
    fp = (
        ERA5_ROOT / f"region={region}" / f"year={year}"
        / f"era5_{region}_{year}{month:02d}.nc"
    )
    if fp.exists() and fp.stat().st_size > MIN_FILE_SIZE_BYTES:
        return True
    return False


# ── Batch download logic ─────────────────────────────────────────────────────

@dataclass
class BatchJob:
    region: int
    year: int
    months: list[int]  # 1-3 months per batch
    bbox: dict


def find_batch_jobs(
    regions: list[int],
    year_start: int,
    year_end: int,
    bboxes: dict,
) -> tuple[list[BatchJob], int, int]:
    """Find missing months, group into quarterly batches."""
    jobs = []
    skipped = 0
    total = 0
    for region in regions:
        if region not in bboxes:
            continue
        for year in range(year_start, year_end + 1):
            missing = []
            for month in MONTHS:
                total += 1
                if is_downloaded(region, year, month):
                    skipped += 1
                else:
                    missing.append(month)
            # Split missing months into chunks of BATCH_MONTHS
            for i in range(0, len(missing), BATCH_MONTHS):
                chunk = missing[i:i + BATCH_MONTHS]
                jobs.append(BatchJob(region, year, chunk, bboxes[region]))
    return jobs, skipped, total


def make_cds_client() -> cdsapi.Client:
    """Create a CDS client with tuned timeout/retry settings."""
    return cdsapi.Client(
        quiet=True,
        timeout=CDS_TIMEOUT,
        retry_max=CDS_RETRY_MAX,
        sleep_max=CDS_SLEEP_MAX,
    )


def download_batch(
    client: cdsapi.Client,
    job: BatchJob,
    logger: logging.Logger,
) -> list[dict]:
    """Download a batch of months for a region-year in one API call,
    then split into individual monthly files."""

    region, year = job.region, job.year
    months = job.months
    bbox = job.bbox

    results = []
    month_strs = [f"{m:02d}" for m in months]
    label = f"region={region} {year} [{','.join(month_strs)}]"

    logger.info(f"  {label} — {len(months)} month(s) in 1 API call")

    req = {
        "product_type": "reanalysis",
        "variable": ERA5_VARIABLES,
        "year": f"{year:04d}",
        "month": month_strs,
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
    }

    # Save zip directly into year folder so it can act as a permanent fallback
    # if the splitter fails. Cleanup logic below removes the zip only after we
    # have verified every expected per-month file is present on disk.
    out_dir = ERA5_ROOT / f"region={region}" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"_batch_{'_'.join(month_strs)}.zip"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"    {label} attempt {attempt}/{MAX_RETRIES}")
            t0 = time.time()
            r = client.retrieve(ERA5_DATASET, req)
            r.download(str(tmp_path))
            elapsed = time.time() - t0

            if not tmp_path.exists() or tmp_path.stat().st_size < MIN_FILE_SIZE_BYTES:
                logger.warning(f"    {label} file too small or missing, retrying")
                tmp_path.unlink(missing_ok=True)
                continue

            size_mb = tmp_path.stat().st_size / 1024 / 1024
            logger.info(f"  ✓ {label} downloaded ({size_mb:.1f} MB in {elapsed:.0f}s)")

            # Split into monthly files
            split_results = _split_download(tmp_path, region, year, months, logger)
            results.extend(split_results)

            # Verify every expected per-month file exists at the target size
            # before deleting the source zip. Otherwise quarantine the zip so
            # the data can be recovered without a second CDS round-trip.
            all_present = all(
                is_downloaded(region, year, m) for m in months
            )
            if all_present:
                tmp_path.unlink(missing_ok=True)
            else:
                quarantine = out_dir / "_raw_zips"
                quarantine.mkdir(exist_ok=True)
                final = quarantine / f"batch_{'_'.join(month_strs)}.zip"
                tmp_path.replace(final)
                logger.warning(
                    f"    quarantined source zip → {final} "
                    f"(per-month split incomplete; data preserved for retry)"
                )
            return results

        except Exception as e:
            wait = min(MAX_BACKOFF_SEC, INITIAL_BACKOFF_SEC * (2 ** (attempt - 1)))
            logger.warning(f"    ✗ {label} error: {e!s} — retry in {wait}s")
            tmp_path.unlink(missing_ok=True)
            time.sleep(wait)

    # All retries exhausted
    logger.error(f"  ✗✗ {label} FAILED after {MAX_RETRIES} attempts")
    for m in months:
        results.append({
            "region": region, "year": year, "month": m,
            "status": "failed", "attempts": MAX_RETRIES,
        })
    return results


def _split_download(
    path: Path,
    region: int,
    year: int,
    expected_months: list[int],
    logger: logging.Logger,
) -> list[dict]:
    """Split a downloaded file (single .nc or .zip) into per-month extracted dirs."""
    results = []

    # Ensure output dirs exist
    out_dir = ERA5_ROOT / f"region={region}" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_dir = out_dir / "_extracted"
    ext_dir.mkdir(exist_ok=True)

    # Handle zip files: extract, split by month, write per-month combined NetCDFs
    if zipfile.is_zipfile(path):
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(path) as z:
                    z.extractall(tmpdir)
                # Find the instant + accum NCs
                tmp_path = Path(tmpdir)
                instant_files = list(tmp_path.glob("*instant*.nc"))
                accum_files = list(tmp_path.glob("*accum*.nc"))
                if not instant_files or not accum_files:
                    # Some downloads come as a single .nc — fall back to multi-month split
                    logger.warning(f"    zip didn't yield instant+accum pair; trying single-ds split")
                    nc_files = list(tmp_path.glob("*.nc"))
                    if not nc_files:
                        raise RuntimeError("No .nc inside zip")
                    ds_single = xr.open_dataset(nc_files[0])
                    months_present = sorted(set(pd.to_datetime(ds_single.valid_time.values).month.astype(int).tolist()))
                    for m in months_present:
                        out_file = out_dir / f"era5_{region}_{year}{m:02d}.nc"
                        if out_file.exists() and out_file.stat().st_size > MIN_FILE_SIZE_BYTES:
                            continue
                        mask = pd.to_datetime(ds_single.valid_time.values).month == m
                        sub = ds_single.isel(valid_time=mask)
                        sub.to_netcdf(out_file)
                    ds_single.close()
                else:
                    ds_inst = xr.open_dataset(instant_files[0])
                    ds_accum = xr.open_dataset(accum_files[0])
                    months_present = sorted(set(pd.to_datetime(ds_inst.valid_time.values).month.astype(int).tolist()))
                    for m in months_present:
                        out_file = out_dir / f"era5_{region}_{year}{m:02d}.nc"
                        if out_file.exists() and out_file.stat().st_size > MIN_FILE_SIZE_BYTES:
                            # Already have this month; skip
                            continue
                        mask_inst = pd.to_datetime(ds_inst.valid_time.values).month == m
                        mask_accum = pd.to_datetime(ds_accum.valid_time.values).month == m
                        sub_inst = ds_inst.isel(valid_time=mask_inst)
                        sub_accum = ds_accum.isel(valid_time=mask_accum)
                        # Combine t2m (instant) + tp (accum) into one dataset.
                        # compat="override" silences xarray's FutureWarning and
                        # matches the future default; safe here because t2m and
                        # tp don't overlap and coords come from the same CDS source.
                        combined = xr.merge([sub_inst, sub_accum], compat="override")
                        combined.to_netcdf(out_file)
                        combined.close()
                    ds_inst.close()
                    ds_accum.close()
            logger.info(f"    extracted zip → per-month files in {out_dir}")
            for m in expected_months:
                results.append({
                    "region": region, "year": year, "month": m,
                    "status": "downloaded", "attempts": 1,
                })
            return results
        except Exception as e:
            logger.error(f"    zip extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            for m in expected_months:
                results.append({
                    "region": region, "year": year, "month": m,
                    "status": "failed", "attempts": 1,
                })
            return results

    # Single NetCDF — if only 1 month, just move it
    if len(expected_months) == 1:
        m = expected_months[0]
        out_file = ext_dir / f"era5_{region}_{year}{m:02d}.nc"
        try:
            path.rename(out_file) if not out_file.exists() else None
            # If rename failed because cross-device, copy
            if not out_file.exists():
                import shutil
                shutil.copy2(path, out_file)
            size_mb = out_file.stat().st_size / 1024 / 1024
            logger.info(f"    → region={region} {year}-{m:02d} ({size_mb:.1f} MB)")
            results.append({
                "region": region, "year": year, "month": m,
                "status": "downloaded", "attempts": 1, "size_mb": size_mb,
            })
        except Exception as e:
            logger.warning(f"    move error {year}-{m:02d}: {e}")
            results.append({
                "region": region, "year": year, "month": m,
                "status": "failed", "attempts": 1,
            })
        return results

    # Multi-month NetCDF — split by time dimension
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception as e:
        logger.error(f"    Cannot open file: {e}")
        for m in expected_months:
            results.append({
                "region": region, "year": year, "month": m,
                "status": "failed", "attempts": 1,
            })
        return results

    # Find the time dimension name
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"

    for month in expected_months:
        out_file = ext_dir / f"era5_{region}_{year}{month:02d}.nc"
        try:
            month_ds = ds.sel({time_dim: ds[time_dim].dt.month == month})
            if month_ds.sizes.get(time_dim, 0) == 0:
                logger.warning(f"    No data for {year}-{month:02d}")
                results.append({
                    "region": region, "year": year, "month": month,
                    "status": "failed", "attempts": 1,
                })
                continue
            month_ds.to_netcdf(out_file)
            size_mb = out_file.stat().st_size / 1024 / 1024
            logger.info(f"    split → region={region} {year}-{month:02d} ({size_mb:.1f} MB)")
            results.append({
                "region": region, "year": year, "month": month,
                "status": "downloaded", "attempts": 1, "size_mb": size_mb,
            })
        except Exception as e:
            logger.warning(f"    split error {year}-{month:02d}: {e}")
            results.append({
                "region": region, "year": year, "month": month,
                "status": "failed", "attempts": 1,
            })

    ds.close()
    return results


# ── Orchestration ────────────────────────────────────────────────────────────

class _RobustFileHandler(logging.FileHandler):
    """FileHandler that reopens the log file after a transient OSError.

    The default FileHandler keeps writing to a stale FD after an EIO on
    /Volumes/BIGDATA (USB/APFS hiccup), so every subsequent emit fails. We
    catch the OSError, close + reopen the file, and retry once. handleError
    runs only if the reopen also fails.
    """

    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            try:
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except OSError:
                        pass
                self.stream = self._open()
                super().emit(record)
            except Exception:
                self.handleError(record)


def run_batch_downloads(
    regions: list[int],
    year_start: int,
    year_end: int,
    max_workers: int,
    dry_run: bool = False,
):
    """Main batch download orchestrator."""
    logger = logging.getLogger("era5_batch")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        fh = _RobustFileHandler(ERA5_ROOT / "_bulk_download.log")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("ERA5 Batch Download Manager (quarterly)")
    logger.info(f"Regions: {regions}")
    logger.info(f"Years: {year_start}-{year_end}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Batch size: {BATCH_MONTHS} months per API call")
    logger.info("=" * 60)

    bboxes = load_region_bboxes()
    logger.info(f"Loaded {len(bboxes)} region bounding boxes")

    jobs, skipped, total = find_batch_jobs(regions, year_start, year_end, bboxes)
    total_missing = sum(len(j.months) for j in jobs)

    logger.info(f"Total files: {total}")
    logger.info(f"Already downloaded: {skipped}")
    logger.info(f"Missing files: {total_missing} in {len(jobs)} API calls")
    logger.info(f"  (vs {total_missing} with per-month downloader)")

    if dry_run:
        logger.info("\nDRY RUN — showing planned requests:")
        for j in jobs[:40]:
            ms = [f"{m:02d}" for m in j.months]
            logger.info(f"  region={j.region} {j.year} months=[{','.join(ms)}]")
        if len(jobs) > 40:
            logger.info(f"  ... and {len(jobs) - 40} more requests")
        return

    if not jobs:
        logger.info("Nothing to download — all files present!")
        return

    logger.info("")

    completed = 0
    failed = 0
    jobs_done = 0

    def worker_fn(job: BatchJob) -> list[dict]:
        client = make_cds_client()
        return download_batch(client, job, logger)

    with ThreadPoolExecutor(
        max_workers=max_workers, initializer=_silence_hdf5_thread
    ) as pool:
        futures = {pool.submit(worker_fn, job): job for job in jobs}

        for future in as_completed(futures):
            jobs_done += 1
            job = futures[future]
            try:
                results = future.result()
            except Exception as e:
                logger.error(
                    f"Unhandled error for region={job.region} {job.year}: {e}"
                )
                results = [{
                    "region": job.region, "year": job.year, "month": m,
                    "status": "error", "attempts": 0,
                } for m in job.months]

            for r in results:
                if r["status"] == "downloaded":
                    completed += 1
                elif r["status"] in ("failed", "error"):
                    failed += 1

            logger.info(
                f"Progress: {completed + skipped}/{total} "
                f"({completed} new, {skipped} existed, {failed} failed) "
                f"[{jobs_done}/{len(jobs)} API calls done]"
            )

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH DOWNLOAD COMPLETE")
    logger.info(f"  New downloads:   {completed}")
    logger.info(f"  Already existed: {skipped}")
    logger.info(f"  Failed:          {failed}")
    logger.info(f"  Total coverage:  {completed + skipped}/{total}")
    logger.info("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch ERA5 downloader — quarterly chunks per API call"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--year-start", type=int, default=YEAR_START)
    parser.add_argument("--year-end", type=int, default=YEAR_END)
    parser.add_argument("--regions", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_MONTHS,
                        help="Months per API call (default: 3)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()
    regions = args.regions or list(range(1, 26))

    _set_batch_size(args.batch_size)

    if args.status:
        from analysis.shared.era5_downloader import show_status
        show_status(regions, args.year_start, args.year_end)
        return

    run_batch_downloads(
        regions=regions,
        year_start=args.year_start,
        year_end=args.year_end,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
