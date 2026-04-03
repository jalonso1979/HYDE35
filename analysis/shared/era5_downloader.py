"""Robust, parallel ERA5 downloader with resume capability.

Downloads ERA5 reanalysis data (2m temperature + total precipitation)
for all 25 HYDE regions, 1950-2025, with:
- Resume from last successful download (skips existing files)
- Exponential backoff retry on network failures
- Parallel downloads across regions (configurable concurrency)
- Checkpointing and progress logging
- Automatic extraction of zip archives

Usage:
    python -m analysis.shared.era5_downloader
    python -m analysis.shared.era5_downloader --workers 4 --year-start 1968 --year-end 2025
    python -m analysis.shared.era5_downloader --regions 1 2 3 --dry-run
"""

import argparse
import cdsapi
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

# ── Configuration ────────────────────────────────────────────────────────────

ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = ["2m_temperature", "total_precipitation"]
ERA5_ROOT = Path("/Volumes/BIGDATA/HYDE35/ERA5")
CHECKPOINT_FILE = ERA5_ROOT / "_download_checkpoint.csv"

YEAR_START = 1950
YEAR_END = 2025
MONTHS = list(range(1, 13))
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

MAX_RETRIES = 8
INITIAL_BACKOFF_SEC = 10
MAX_BACKOFF_SEC = 900
MIN_FILE_SIZE_BYTES = 1000  # reject suspiciously small downloads

BBOX_PAD_DEG = 0.5


# ── Region definitions ───────────────────────────────────────────────────────

def load_region_bboxes() -> dict:
    """Load region bounding boxes from existing extracted NetCDF files,
    or compute from HYDE grid if no extractions exist."""
    bboxes = {}

    # First try: read from existing extracted files
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
        raise RuntimeError(
            "No existing ERA5 extractions found. Cannot determine region bounding boxes. "
            "Run the HYDE_and_ERA notebook first to establish region definitions."
        )

    return bboxes


# ── Download logic ───────────────────────────────────────────────────────────

def file_path(region: int, year: int, month: int) -> Path:
    """Canonical file path for a region-year-month download."""
    return (
        ERA5_ROOT
        / f"region={region}"
        / f"year={year}"
        / f"era5_{region}_{year}{month:02d}.nc"
    )


def is_downloaded(region: int, year: int, month: int) -> bool:
    """Check if a file already exists and is large enough to be valid."""
    fp = file_path(region, year, month)
    if fp.exists() and fp.stat().st_size > MIN_FILE_SIZE_BYTES:
        return True
    # Also check if extracted version exists
    ext_dir = fp.parent / "_extracted"
    if ext_dir.exists():
        extracted = [f for f in ext_dir.iterdir() if f.suffix == ".nc"]
        if len(extracted) >= 1:
            return True
    return False


def download_one(
    client: cdsapi.Client,
    region: int,
    year: int,
    month: int,
    bbox: dict,
    logger: logging.Logger,
) -> dict:
    """Download a single region-year-month file with retries."""

    fp = file_path(region, year, month)
    result = {
        "region": region,
        "year": year,
        "month": month,
        "status": "skipped",
        "attempts": 0,
        "file": str(fp),
    }

    if is_downloaded(region, year, month):
        return result

    # Ensure directory exists
    fp.parent.mkdir(parents=True, exist_ok=True)

    req = {
        "product_type": "reanalysis",
        "variable": ERA5_VARIABLES,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        result["attempts"] = attempt
        try:
            logger.info(
                f"  region={region} {year}-{month:02d} attempt {attempt}/{MAX_RETRIES}"
            )
            r = client.retrieve(ERA5_DATASET, req)
            r.download(str(fp))

            if fp.exists() and fp.stat().st_size > MIN_FILE_SIZE_BYTES:
                result["status"] = "downloaded"
                result["size_mb"] = fp.stat().st_size / 1024 / 1024
                logger.info(
                    f"  ✓ region={region} {year}-{month:02d} "
                    f"({result['size_mb']:.1f} MB)"
                )

                # Extract if it's a zip
                _extract_if_zip(fp, logger)
                return result
            else:
                logger.warning(
                    f"  ✗ region={region} {year}-{month:02d} "
                    f"file too small ({fp.stat().st_size} bytes), retrying"
                )
                fp.unlink(missing_ok=True)

        except Exception as e:
            wait = min(MAX_BACKOFF_SEC, INITIAL_BACKOFF_SEC * (2 ** (attempt - 1)))
            logger.warning(
                f"  ✗ region={region} {year}-{month:02d} "
                f"error: {e!s:.100s}... retry in {wait}s"
            )
            time.sleep(wait)

    result["status"] = "failed"
    logger.error(f"  ✗✗ region={region} {year}-{month:02d} FAILED after {MAX_RETRIES} attempts")
    return result


def _extract_if_zip(fp: Path, logger: logging.Logger):
    """If the downloaded file is a zip archive, extract it."""
    try:
        if zipfile.is_zipfile(fp):
            ext_dir = fp.parent / "_extracted"
            ext_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(fp) as z:
                z.extractall(ext_dir)
            logger.info(f"    extracted zip → {ext_dir}")
    except Exception as e:
        logger.warning(f"    zip extraction failed: {e}")


# ── Orchestration ────────────────────────────────────────────────────────────

@dataclass
class DownloadJob:
    region: int
    year: int
    month: int
    bbox: dict


def build_job_queue(
    regions: list[int],
    year_start: int,
    year_end: int,
    bboxes: dict,
) -> list[DownloadJob]:
    """Build the list of downloads needed, skipping already-completed ones."""
    jobs = []
    skipped = 0
    for region in regions:
        if region not in bboxes:
            continue
        bbox = bboxes[region]
        for year in range(year_start, year_end + 1):
            for month in MONTHS:
                if is_downloaded(region, year, month):
                    skipped += 1
                else:
                    jobs.append(DownloadJob(region, year, month, bbox))
    return jobs, skipped


def save_checkpoint(results: list[dict]):
    """Append results to checkpoint CSV."""
    import csv

    file_exists = CHECKPOINT_FILE.exists()
    with open(CHECKPOINT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["region", "year", "month", "status", "attempts", "file"]
        )
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})


def run_downloads(
    regions: list[int],
    year_start: int,
    year_end: int,
    max_workers: int,
    dry_run: bool = False,
):
    """Main download orchestrator."""
    logger = logging.getLogger("era5_dl")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        # Also log to file
        fh = logging.FileHandler(ERA5_ROOT / "_download.log")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("ERA5 Download Manager")
    logger.info(f"Regions: {regions}")
    logger.info(f"Years: {year_start}-{year_end}")
    logger.info(f"Workers: {max_workers}")
    logger.info("=" * 60)

    # Load region bounding boxes
    bboxes = load_region_bboxes()
    logger.info(f"Loaded {len(bboxes)} region bounding boxes")

    # Build job queue (skips existing)
    jobs, skipped = build_job_queue(regions, year_start, year_end, bboxes)
    total_possible = len(regions) * (year_end - year_start + 1) * 12
    logger.info(f"Total files: {total_possible}")
    logger.info(f"Already downloaded: {skipped}")
    logger.info(f"Remaining: {len(jobs)}")

    if dry_run:
        logger.info("DRY RUN — no downloads will be performed")
        for j in jobs[:20]:
            logger.info(f"  Would download: region={j.region} {j.year}-{j.month:02d}")
        if len(jobs) > 20:
            logger.info(f"  ... and {len(jobs) - 20} more")
        return

    if not jobs:
        logger.info("Nothing to download — all files present!")
        return

    # Estimate time
    est_sec_per_file = 30  # rough estimate
    est_total = len(jobs) * est_sec_per_file / max_workers
    logger.info(f"Estimated time: {est_total / 3600:.1f} hours at ~{est_sec_per_file}s/file")
    logger.info("")

    # Download with thread pool
    # Each worker gets its own CDS client to avoid thread-safety issues
    completed = 0
    failed = 0
    batch_results = []

    def worker_fn(job: DownloadJob) -> dict:
        client = cdsapi.Client(quiet=True)
        return download_one(client, job.region, job.year, job.month, job.bbox, logger)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker_fn, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "region": job.region,
                    "year": job.year,
                    "month": job.month,
                    "status": "error",
                    "attempts": 0,
                    "file": str(file_path(job.region, job.year, job.month)),
                }
                logger.error(f"Unhandled error for region={job.region} {job.year}-{job.month:02d}: {e}")

            batch_results.append(result)
            if result["status"] == "downloaded":
                completed += 1
            elif result["status"] == "failed":
                failed += 1

            # Checkpoint every 50 files
            if len(batch_results) >= 50:
                save_checkpoint(batch_results)
                batch_results = []
                done = completed + failed + skipped
                logger.info(
                    f"Progress: {done}/{total_possible} "
                    f"({completed} new, {skipped} existed, {failed} failed)"
                )

    # Final checkpoint
    if batch_results:
        save_checkpoint(batch_results)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"  New downloads:  {completed}")
    logger.info(f"  Already existed: {skipped}")
    logger.info(f"  Failed:          {failed}")
    logger.info(f"  Total coverage:  {completed + skipped}/{total_possible}")
    logger.info("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 climate data for HYDE regions"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel download threads (default: 3, max recommended: 5)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=YEAR_START,
        help=f"Start year (default: {YEAR_START})",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=YEAR_END,
        help=f"End year (default: {YEAR_END})",
    )
    parser.add_argument(
        "--regions",
        type=int,
        nargs="+",
        default=None,
        help="Specific region numbers to download (default: all 25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status and exit",
    )

    args = parser.parse_args()

    # Determine regions
    if args.regions:
        regions = args.regions
    else:
        regions = list(range(1, 26))

    if args.status:
        show_status(regions, args.year_start, args.year_end)
        return

    run_downloads(
        regions=regions,
        year_start=args.year_start,
        year_end=args.year_end,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )


def show_status(regions: list[int], year_start: int, year_end: int):
    """Show current download status."""
    print("ERA5 Download Status")
    print("=" * 60)

    total = 0
    downloaded = 0
    by_region = {}

    for region in regions:
        reg_total = 0
        reg_done = 0
        for year in range(year_start, year_end + 1):
            for month in MONTHS:
                reg_total += 1
                total += 1
                if is_downloaded(region, year, month):
                    reg_done += 1
                    downloaded += 1
        by_region[region] = (reg_done, reg_total)

    print(f"\nOverall: {downloaded}/{total} ({downloaded / total * 100:.1f}%)")
    print(f"\nBy region:")
    for region in regions:
        done, tot = by_region[region]
        pct = done / tot * 100 if tot > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  Region {region:>2d}: {bar} {done:>4d}/{tot:>4d} ({pct:.0f}%)")

    remaining = total - downloaded
    est_hours = remaining * 30 / 3 / 3600  # 30s per file, 3 workers
    print(f"\nRemaining: {remaining} files (~{est_hours:.1f} hours at 3 workers)")


if __name__ == "__main__":
    main()
